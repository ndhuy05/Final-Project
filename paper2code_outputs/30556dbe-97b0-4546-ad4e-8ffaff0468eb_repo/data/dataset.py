"""
ECL-YOLOv11 Dataset Module

This module implements the dataset loader for ECL-YOLOv11, handling YOLO-format
annotations, weather augmentation integration, and data preprocessing.

Based on the paper: "Robust Object Detection in Adverse Weather Conditions: 
ECL-YOLOv11 for Automotive Vision Systems"

Author: ECL-YOLOv11 Reproduction Team
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import defaultdict
import random
import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent))

# Try to import weather augmentation
try:
    from data.weather_augmentation import WeatherAugmentation
except ImportError:
    try:
        from .weather_augmentation import WeatherAugmentation
    except ImportError:
        # Fallback: Define a stub class if weather augmentation is not available
        class WeatherAugmentation:
            def __init__(self, *args, **kwargs):
                pass
            def apply_weather(self, image, weather_type, **kwargs):
                return image

# Try to import configuration
try:
    from utils.config import get_config_manager
except ImportError:
    # Fallback configuration functions
    def get_config_manager():
        return None


# =============================================================================
# Constants and Default Values
# =============================================================================

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default class names (from paper)
DEFAULT_CLASS_NAMES = [
    'car', 'person', 'bus', 'bicycle', 
    'truck', 'train', 'motorcycle'
]

# Default dataset paths
DEFAULT_DATA_ROOT = "./data"
DEFAULT_TRAIN_DIR = "train"
DEFAULT_VAL_DIR = "val"
DEFAULT_TEST_DIR = "test"


# =============================================================================
# Utility Functions
# =============================================================================

def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: Image in BGR format (H, W, 3)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image using OpenCV (BGR format)
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def load_yolo_annotations(label_path: Union[str, Path]) -> List[List[float]]:
    """
    Load YOLO-format annotations from a label file.
    
    The YOLO format stores annotations as:
    class_id x_center y_center width height
    
    All values are normalized to [0, 1].
    
    Args:
        label_path: Path to the label file
        
    Returns:
        List of annotations, each as [class_id, x_center, y_center, width, height]
        
    Raises:
        FileNotFoundError: If label file doesn't exist
    """
    label_path = Path(label_path)
    
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    annotations = []
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse YOLO format: class_id x_center y_center width height
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Validate and clamp values to [0, 1]
                class_id = max(0, class_id)
                x_center = np.clip(x_center, 0.0, 1.0)
                y_center = np.clip(y_center, 0.0, 1.0)
                width = np.clip(width, 0.0, 1.0)
                height = np.clip(height, 0.0, 1.0)
                
                annotations.append([class_id, x_center, y_center, width, height])
            except ValueError:
                # Skip malformed lines
                continue
    
    return annotations


def yolo_to_xyxy(
    annotations: List[List[float]], 
    image_width: int, 
    image_height: int
) -> np.ndarray:
    """
    Convert YOLO format annotations to xyxy (corner) format.
    
    Args:
        annotations: List of [class_id, x_center, y_center, width, height]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        np.ndarray: Array of shape (N, 6) with [class_id, x1, y1, x2, y2]
    """
    if len(annotations) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    
    result = []
    
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        
        # Convert from center format to corner format
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        width_px = width * image_width
        height_px = height * image_height
        
        # Calculate corners
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2
        
        # Clamp to image boundaries
        x1 = np.clip(x1, 0, image_width)
        y1 = np.clip(y1, 0, image_height)
        x2 = np.clip(x2, 0, image_width)
        y2 = np.clip(y2, 0, image_height)
        
        result.append([class_id, x1, y1, x2, y2])
    
    return np.array(result, dtype=np.float32)


def xyxy_to_yolo(
    boxes: np.ndarray, 
    image_width: int, 
    image_height: int
) -> List[List[float]]:
    """
    Convert xyxy (corner) format to YOLO format.
    
    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        List of [x_center, y_center, width, height] normalized to [0, 1]
    """
    if len(boxes) == 0:
        return []
    
    result = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Calculate center and dimensions
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        # Clamp to [0, 1]
        x_center = np.clip(x_center, 0.0, 1.0)
        y_center = np.clip(y_center, 0.0, 1.0)
        width = np.clip(width, 0.0, 1.0)
        height = np.clip(height, 0.0, 1.0)
        
        result.append([x_center, y_center, width, height])
    
    return result


def letterbox_resize(
    image: np.ndarray, 
    target_size: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with padding to maintain aspect ratio (letterbox).
    
    Args:
        image: Input image (H, W, C) in BGR format
        target_size: Target (width, height)
        color: Padding color (B, G, R)
        auto: Minimum rectangle padding
        scale_fill: Stretch to fill
        scaleup: Allow scaling up
        
    Returns:
        Tuple of (resized_image, scale_factor, padding)
    """
    shape = image.shape[:2]  # current shape [height, width]
    
    # New shape [height, width]
    new_shape = (target_size[1], target_size[0])
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    
    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    
    if auto:
        # Minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scale_fill:
        # Stretch
        dw, dh = 0, 0
        new_unpad = (new_shape[0], new_shape[1])
        r = new_shape[0] / shape[0], new_shape[1] / shape[1]
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=color
    )
    
    return image, r, (dw, dh)


def normalize_image(
    image: np.ndarray, 
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD
) -> np.ndarray:
    """
    Normalize image using ImageNet statistics.
    
    Args:
        image: Input image in range [0, 255] (uint8)
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image in range [-2, 2] approximately
    """
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply mean and std normalization
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    
    image = (image - mean) / std
    
    return image


def normalize_image_simple(image: np.ndarray) -> np.ndarray:
    """
    Simple normalization to [0, 1] range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


# =============================================================================
# Main Dataset Class
# =============================================================================

class YOLODataset(Dataset):
    """
    YOLO-format Dataset for ECL-YOLOv11 training and evaluation.
    
    This dataset handles:
    - Loading images from YOLO-format directory structure
    - Parsing YOLO-format annotations
    - Applying weather augmentation during training
    - Preprocessing images (resize, normalize, convert to tensor)
    
    Attributes:
        root_dir: Root directory of the dataset
        split: Dataset split ('train', 'val', 'test')
        image_size: Target image size (default: 640)
        augment: Whether to apply data augmentation
        weather_types: List of weather types for augmentation
        cache_images: Whether to cache images in memory
        
    Input:
        Root directory should have the structure:
        root_dir/
        ├── train/
        │   ├── images/
        │   │   └── *.jpg
        │   └── labels/
        │       └── *.txt
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path] = DEFAULT_DATA_ROOT,
        split: str = 'train',
        image_size: int = 640,
        augment: bool = True,
        weather_types: Optional[List[str]] = None,
        cache_images: bool = False,
        class_names: Optional[List[str]] = None,
        use_weather_prob: float = 0.5,
        config: Optional[Dict] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize YOLODataset.
        
        Args:
            root_dir: Root directory of the YOLO-format dataset
            split: Dataset split ('train', 'val', or 'test')
            image_size: Target image size for resizing
            augment: Whether to apply weather augmentation
            weather_types: List of weather types to apply during training
            cache_images: Whether to cache loaded images in memory
            class_names: List of class names
            use_weather_prob: Probability of applying weather augmentation
            config: Configuration dictionary
            transform: Optional custom transform function
        """
        super().__init__()
        
        # Load configuration if provided
        if config is None:
            try:
                config_manager = get_config_manager()
                if config_manager is not None:
                    config = config_manager.get_config_dict()
            except:
                config = {}
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.cache_images = cache_images
        self.use_weather_prob = use_weather_prob
        self.transform = transform
        
        # Set class names
        if class_names is None:
            class_names = DEFAULT_CLASS_NAMES
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(class_names)}
        
        # Set weather types
        if weather_types is None:
            weather_types = ['fog', 'rain', 'snow']
        self.weather_types = weather_types
        
        # Initialize weather augmentation
        if augment:
            # Get weather parameters from config
            weather_params = {}
            if config:
                weather_params = config.get('data', {}).get('weather_params', {})
            
            self.weather_aug = WeatherAugmentation(
                fog_density=weather_params.get('fog_density', 0.5),
                rain_intensity=weather_params.get('rain_intensity', 0.5),
                snow_intensity=weather_params.get('snow_intensity', 0.5),
                seed=None  # Random seed per call
            )
        else:
            self.weather_aug = None
        
        # Image cache
        self.image_cache = {} if cache_images else None
        
        # Find image and label files
        self.image_paths = []
        self.label_paths = []
        self.image_ids = []
        
        self._find_files()
        
        # Print dataset info
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def _find_files(self) -> None:
        """
        Find all image and label files in the dataset directory.
        """
        # Define split directory
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            # Try alternative structure
            for subdir in ['train', 'val', 'test']:
                alt_dir = self.root_dir / subdir
                if alt_dir.exists():
                    split_dir = alt_dir
                    break
        
        image_dir = split_dir / 'images'
        label_dir = split_dir / 'labels'
        
        # Check if directories exist
        if not image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {image_dir}")
        
        if not label_dir.exists():
            print(f"Warning: Labels directory not found: {label_dir}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        for img_file in sorted(image_dir.iterdir()):
            if img_file.suffix.lower() in image_extensions:
                self.image_paths.append(img_file)
                
                # Find corresponding label file
                label_file = label_dir / f"{img_file.stem}.txt"
                self.label_paths.append(label_file if label_file.exists() else None)
                
                # Generate image ID
                self.image_ids.append(f"{self.split}_{img_file.stem}")
        
        # Validate
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # Check for missing labels
        missing_labels = sum(1 for lp in self.label_paths if lp is None)
        if missing_labels > 0:
            print(f"Warning: {missing_labels} images have no corresponding label files")
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of images in the dataset
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List, str]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'image': Image tensor of shape (3, H, W)
                - 'targets': List of target tensors [class_id, x1, y1, x2, y2]
                - 'image_id': String identifier for the image
                - 'weather': Applied weather type (or 'none')
                - 'orig_shape': Original image shape (H, W)
        """
        # Load image
        image = self._load_image(idx)
        
        # Get original shape
        orig_h, orig_w = image.shape[:2]
        
        # Apply weather augmentation (during training)
        weather_type = 'none'
        if self.augment and self.weather_aug is not None:
            if random.random() < self.use_weather_prob:
                weather_type = random.choice(self.weather_types)
                
                # Apply random intensity based on weather type
                intensity = random.uniform(0.3, 0.8)
                
                if weather_type == 'fog':
                    image = self.weather_aug.add_fog(image, density=intensity)
                elif weather_type == 'rain':
                    image = self.weather_aug.add_rain(image, intensity=intensity)
                elif weather_type == 'snow':
                    image = self.weather_aug.add_snow(image, intensity=intensity)
        
        # Resize image with letterbox
        image_resized, scale, padding = letterbox_resize(
            image, 
            (self.image_size, self.image_size),
            color=(114, 114, 114)
        )
        
        # Normalize image
        image_normalized = normalize_image(image_resized)
        
        # Apply custom transform if provided
        if self.transform is not None:
            image_normalized = self.transform(image_normalized)
        
        # Convert to CHW format (PyTorch format)
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        
        # Load annotations
        annotations = self._load_annotations(idx)
        
        # Convert to xyxy format and scale to resized image
        targets = self._prepare_targets(annotations, scale, padding)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_chw).float()
        
        # Convert targets to tensor (or list of tensors)
        if len(targets) > 0:
            targets_tensor = torch.from_numpy(targets).float()
        else:
            # Empty tensor for no objects
            targets_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'targets': targets_tensor,
            'image_id': self.image_ids[idx],
            'weather': weather_type,
            'orig_shape': (orig_h, orig_w)
        }
    
    def _load_image(self, idx: int) -> np.ndarray:
        """
        Load image from file or cache.
        
        Args:
            idx: Index of the image
            
        Returns:
            Image in BGR format (H, W, 3)
        """
        image_path = self.image_paths[idx]
        
        # Check cache
        if self.image_cache is not None and idx in self.image_cache:
            return self.image_cache[idx].copy()
        
        # Load image
        image = load_image(image_path)
        
        # Cache if enabled
        if self.image_cache is not None:
            self.image_cache[idx] = image.copy()
        
        return image
    
    def _load_annotations(self, idx: int) -> List[List[float]]:
        """
        Load annotations for an image.
        
        Args:
            idx: Index of the image
            
        Returns:
            List of annotations in YOLO format
        """
        label_path = self.label_paths[idx]
        
        if label_path is None or not label_path.exists():
            return []
        
        return load_yolo_annotations(label_path)
    
    def _prepare_targets(
        self, 
        annotations: List[List[float]], 
        scale: float,
        padding: Tuple[int, int]
    ) -> np.ndarray:
        """
        Prepare target tensor from annotations.
        
        Args:
            annotations: List of YOLO-format annotations
            scale: Scale factor from letterbox resizing
            padding: Padding values from letterbox
            
        Returns:
            Array of shape (N, 5) with [class_id, x1, y1, x2, y2] in normalized coordinates
        """
        if len(annotations) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        
        # Get original image dimensions from cache or file
        # For now, use target size since we're working with normalized coordinates
        
        # Convert to xyxy in normalized coordinates [0, 1]
        targets = []
        
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            
            # Calculate corners in normalized coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Clamp to [0, 1]
            x1 = np.clip(x1, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            y2 = np.clip(y2, 0.0, 1.0)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            targets.append([class_id, x1, y1, x2, y2])
        
        if len(targets) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        
        return np.array(targets, dtype=np.float32)
    
    def get_class_counts(self) -> Dict[str, int]:
        """
        Get the count of samples per class.
        
        Returns:
            Dictionary mapping class names to counts
        """
        class_counts = defaultdict(int)
        
        for idx in range(len(self)):
            annotations = self._load_annotations(idx)
            for ann in annotations:
                class_id = int(ann[0])
                class_name = self.idx_to_class.get(class_id, f"class_{class_id}")
                class_counts[class_name] += 1
        
        return dict(class_counts)
    
    def get_weather_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of weather types in the dataset.
        
        Note: This returns the configuration, not actual applied weather.
        
        Returns:
            Dictionary mapping weather types to counts (for reference)
        """
        return {
            'fog': len(self) // len(self.weather_types),
            'rain': len(self) // len(self.weather_types),
            'snow': len(self) // len(self.weather_types),
            'none': len(self) // len(self.weather_types)
        }
    
    def get_sample_by_id(self, image_id: str) -> Dict:
        """
        Get a sample by its image ID.
        
        Args:
            image_id: Image identifier string
            
        Returns:
            Sample dictionary
            
        Raises:
            ValueError: If image_id not found
        """
        try:
            idx = self.image_ids.index(image_id)
            return self[idx]
        except ValueError:
            raise ValueError(f"Image ID not found: {image_id}")


# =============================================================================
# DataLoader Factory Functions
# =============================================================================

def create_dataloader(
    root_dir: Union[str, Path],
    split: str = 'train',
    batch_size: int = 16,
    image_size: int = 640,
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    class_names: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for YOLO-format dataset.
    
    Args:
        root_dir: Root directory of the dataset
        split: Dataset split ('train', 'val', 'test')
        batch_size: Number of samples per batch
        image_size: Target image size
        augment: Whether to apply data augmentation
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop last incomplete batch
        class_names: List of class names
        config: Configuration dictionary
        **dataset_kwargs: Additional arguments for YOLODataset
        
    Returns:
        DataLoader: Configured PyTorch DataLoader
    """
    # Create dataset
    dataset = YOLODataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
        augment=augment,
        class_names=class_names,
        config=config,
        **dataset_kwargs
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    
    return dataloader


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for YOLO dataset.
    
    This function handles batching of variable-length target lists.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    # Stack images
    images = torch.stack([item['image'] for item in batch], dim=0)
    
    # Collect targets (keep as list for variable lengths)
    targets = [item['targets'] for item in batch]
    
    # Collect other metadata
    image_ids = [item['image_id'] for item in batch]
    weathers = [item['weather'] for item in batch]
    orig_shapes = [item['orig_shape'] for item in batch]
    
    return {
        'images': images,
        'targets': targets,
        'image_ids': image_ids,
        'weathers': weathers,
        'orig_shapes': orig_shapes
    }


# =============================================================================
# Dataset Splitting Utilities
# =============================================================================

def split_dataset(
    image_dir: Path,
    label_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_root: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, List[Path]]:
    """
    Split a YOLO-format dataset into train/val/test sets.
    
    Args:
        image_dir: Source images directory
        label_dir: Source labels directory
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        output_root: Output root directory (if None, modifies in place)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing lists of image paths
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths = sorted([
        f for f in image_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    # Shuffle
    random.shuffle(image_paths)
    
    # Calculate split indices
    n_train = int(len(image_paths) * train_ratio)
    n_val = int(len(image_paths) * val_ratio)
    
    # Split
    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:n_train + n_val]
    test_paths = image_paths[n_train + n_val:]
    
    # Create output directories
    if output_root is None:
        output_root = image_dir.parent
    
    splits = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }
    
    # Copy files to output directories
    for split_name, paths in splits.items():
        split_image_dir = output_root / split_name / 'images'
        split_label_dir = output_root / split_name / 'labels'
        
        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in paths:
            # Copy image
            import shutil
            dst_image = split_image_dir / img_path.name
            if not dst_image.exists():
                shutil.copy(img_path, dst_image)
            
            # Copy label
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = split_label_dir / label_path.name
                if not dst_label.exists():
                    shutil.copy(label_path, dst_label)
    
    return splits


# =============================================================================
# Data Analysis Utilities
# =============================================================================

def analyze_dataset(dataset: YOLODataset) -> Dict:
    """
    Analyze dataset statistics.
    
    Args:
        dataset: YOLODataset instance
        
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'num_images': len(dataset),
        'split': dataset.split,
        'image_size': dataset.image_size,
        'class_counts': dataset.get_class_counts(),
        'weather_types': dataset.weather_types,
        'augment_enabled': dataset.augment
    }
    
    # Calculate class distribution
    total_objects = sum(stats['class_counts'].values())
    if total_objects > 0:
        stats['class_distribution'] = {
            cls: count / total_objects 
            for cls, count in stats['class_counts'].items()
        }
    else:
        stats['class_distribution'] = {}
    
    return stats


# =============================================================================
# Test/Verification Code
# =============================================================================

if __name__ == "__main__":
    # Test dataset module
    print("Testing YOLODataset Implementation")
    print("=" * 50)
    
    # Create a simple test with synthetic data
    import tempfile
    import shutil
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dataset structure
        train_dir = tmpdir / 'train'
        train_images = train_dir / 'images'
        train_labels = train_dir / 'labels'
        train_images.mkdir(parents=True)
        train_labels.mkdir(parents=True)
        
        # Create sample images and labels
        for i in range(10):
            # Create a sample image (640x480)
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img_path = train_images / f"img_{i:04d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Create YOLO format label
            # class_id x_center y_center width height (all normalized)
            annotations = [
                f"0 0.5 0.5 0.3 0.3\n",  # car
                f"1 0.3 0.7 0.1 0.2\n",  # person
            ]
            
            label_path = train_labels / f"img_{i:04d}.txt"
            with open(label_path, 'w') as f:
                f.writelines(annotations)
        
        # Test 1: Create dataset
        print("\n1. Testing dataset creation:")
        dataset = YOLODataset(
            root_dir=tmpdir,
            split='train',
            image_size=640,
            augment=True,
            weather_types=['fog', 'rain', 'snow']
        )
        print(f"   Dataset size: {len(dataset)}")
        
        # Test 2: Get sample
        print("\n2. Testing __getitem__:")
        sample = dataset[0]
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Targets shape: {sample['targets'].shape}")
        print(f"   Image ID: {sample['image_id']}")
        print(f"   Weather: {sample['weather']}")
        print(f"   Original shape: {sample['orig_shape']}")
        
        # Test 3: Iterate through dataset
        print("\n3. Testing dataset iteration:")
        for i, sample in enumerate(dataset):
            if i >= 3:
                break
            print(f"   Sample {i}: image={sample['image'].shape}, targets={sample['targets'].shape}")
        
        # Test 4: Create dataloader
        print("\n4. Testing DataLoader:")
        dataloader = create_dataloader(
            root_dir=tmpdir,
            split='train',
            batch_size=4,
            image_size=640,
            augment=True,
            shuffle=True,
            num_workers=0,  # Use 0 for testing
            pin_memory=False
        )
        
        for batch in dataloader:
            print(f"   Batch images shape: {batch['images'].shape}")
            print(f"   Batch targets: {len(batch['targets'])} samples")
            print(f"   Batch image IDs: {batch['image_ids']}")
            break
        
        # Test 5: Test without augmentation
        print("\n5. Testing without augmentation:")
        dataset_no_aug = YOLODataset(
            root_dir=tmpdir,
            split='train',
            image_size=640,
            augment=False
        )
        sample = dataset_no_aug[0]
        print(f"   Weather: {sample['weather']}")
        
        # Test 6: Analyze dataset
        print("\n6. Testing dataset analysis:")
        stats = analyze_dataset(dataset)
        print(f"   Number of images: {stats['num_images']}")
        print(f"   Class counts: {stats['class_counts']}")
        print(f"   Class distribution: {stats['class_distribution']}")
        
        # Test 7: Test with different image sizes
        print("\n7. Testing with different image sizes:")
        for size in [320, 416, 640, 1280]:
            dataset_test = YOLODataset(
                root_dir=tmpdir,
                split='train',
                image_size=size,
                augment=False
            )
            sample = dataset_test[0]
            print(f"   Size {size}: image shape {sample['image'].shape}")
        
        # Test 8: Test error handling
        print("\n8. Testing error handling:")
        try:
            dataset_empty = YOLODataset(
                root_dir=tmpdir,
                split='nonexistent',
                image_size=640
            )
        except Exception as e:
            print(f"   Expected error for invalid split: {type(e).__name__}")
        
        # Test 9: Test weather augmentation
        print("\n9. Testing weather augmentation integration:")
        weather = WeatherAugmentation(fog_density=0.5)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        fog_img = weather.add_fog(img)
        rain_img = weather.add_rain(img)
        snow_img = weather.add_snow(img)
        
        print(f"   Original: {img.shape}, mean={img.mean():.2f}")
        print(f"   Fog: {fog_img.shape}, mean={fog_img.mean():.2f}")
        print(f"   Rain: {rain_img.shape}, mean={rain_img.mean():.2f}")
        print(f"   Snow: {snow_img.shape}, mean={snow_img.mean():.2f}")
        
        print("\n" + "=" * 50)
        print("Dataset module test completed!")
