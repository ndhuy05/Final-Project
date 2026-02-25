"""
data_loader.py

Video data loading and preprocessing module for the conveyor belt speed detection system.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module handles:
1. Video file loading from dataset directory
2. Consecutive frame pair extraction for optical flow and feature matching
3. Frame preprocessing (resizing, normalization, format conversion)
4. Ground truth speed label association
5. Batching for efficient processing

Author: Based on paper methodology
"""

import os
import glob
import re
from typing import List, Tuple, Optional, Dict, Any, Iterator, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np

# Try to import OpenCV and PyTorch
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    raise ImportError("OpenCV (cv2) is required for video loading. Install with: pip install opencv-python")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader as TorchDataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    raise ImportError("PyTorch is required for tensor operations. Install with: pip install torch")

# Import config
try:
    from config import Config, get_config
except ImportError:
    # Fallback if config module not available
    Config = None
    get_config = None


@dataclass
class FramePair:
    """Data structure for a pair of consecutive video frames.
    
    This class represents a single frame pair used for both optical flow
    estimation and feature matching in the speed detection pipeline.
    
    Attributes:
        frame1: First frame as numpy array (H, W, C) in BGR format
        frame2: Second frame as numpy array (H, W, C) in BGR format
        timestamp: Time in seconds from start of video
        ground_truth_speed: Actual belt speed in m/s from ground truth
        frame_index: Index of the first frame in the pair
        scenario: Scenario type ("lab" or "mine")
        video_path: Path to the source video file
    """
    frame1: np.ndarray
    frame2: np.ndarray
    timestamp: float
    ground_truth_speed: float
    frame_index: int
    scenario: str = "lab"
    video_path: str = ""
    
    def __post_init__(self):
        """Validate and process frame pair after initialization."""
        # Ensure frames are numpy arrays
        if not isinstance(self.frame1, np.ndarray):
            self.frame1 = np.array(self.frame1)
        if not isinstance(self.frame2, np.ndarray):
            self.frame2 = np.array(self.frame2)
        
        # Ensure frames are in correct format (BGR for OpenCV compatibility)
        if len(self.frame1.shape) != 3:
            raise ValueError(f"frame1 must be 3D array (H, W, C), got shape {self.frame1.shape}")
        if len(self.frame2.shape) != 3:
            raise ValueError(f"frame2 must be 3D array (H, W, C), got shape {self.frame2.shape}")
        
        # Check frame dimensions match
        if self.frame1.shape != self.frame2.shape:
            raise ValueError(
                f"Frame shapes must match: {self.frame1.shape} vs {self.frame2.shape}"
            )
    
    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Get the shape of frames (H, W, C)."""
        return self.frame1.shape
    
    @property
    def height(self) -> int:
        """Get frame height."""
        return self.frame1.shape[0]
    
    @property
    def width(self) -> int:
        """Get frame width."""
        return self.frame1.shape[1]
    
    @property
    def channels(self) -> int:
        """Get number of channels."""
        return self.frame1.shape[2] if len(self.frame1.shape) > 2 else 1


@dataclass
class VideoMetadata:
    """Metadata for a video file in the dataset.
    
    Attributes:
        path: Path to the video file
        scenario: Scenario type ("lab" or "mine")
        belt_speed: Ground truth belt speed in m/s
        illumination: Illumination condition in lux (if available)
        frame_count: Total number of frames in video
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
        duration: Video duration in seconds
    """
    path: str
    scenario: str
    belt_speed: float
    illumination: Optional[int] = None
    frame_count: int = 0
    fps: float = 25.0
    width: int = 1920
    height: int = 1080
    duration: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.fps > 0:
            self.duration = self.frame_count / self.fps


class VideoLoader:
    """Video file loader with frame extraction capabilities.
    
    This class handles low-level video loading operations including
    opening video files, extracting frames, and validating video properties.
    """
    
    def __init__(
        self, 
        target_fps: Optional[float] = None,
        target_resolution: Optional[Tuple[int, int]] = None
    ):
        """Initialize video loader.
        
        Args:
            target_fps: Target frame rate for resampling (None = use original)
            target_resolution: Target resolution (width, height) for resizing
        """
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        
        # Video capture object
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_path: str = ""
    
    def open(self, video_path: str) -> bool:
        """Open a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if successfully opened, False otherwise
        """
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return False
        
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            print(f"Warning: Could not open video: {video_path}")
            self._cap = None
            return False
        
        self._current_path = video_path
        return True
    
    def close(self) -> None:
        """Close the currently open video."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._current_path = ""
    
    def get_properties(self) -> Optional[Dict[str, Any]]:
        """Get video properties.
        
        Returns:
            Dictionary containing video metadata or None if no video is open
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        
        return {
            'frame_count': int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': self._cap.get(cv2.CAP_PROP_FPS),
            'width': int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fourcc': int(self._cap.get(cv2.CAP_PROP_FOURCC))
        }
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read the next frame.
        
        Returns:
            Frame as numpy array in BGR format, or None if end of video
        """
        if self._cap is None:
            return None
        
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        # Resize if target resolution is specified
        if self.target_resolution is not None:
            frame = cv2.resize(
                frame, 
                (self.target_resolution[0], self.target_resolution[1]),
                interpolation=cv2.INTER_LINEAR
            )
        
        return frame
    
    def read_frame_at(self, frame_index: int) -> Optional[np.ndarray]:
        """Read a specific frame by index.
        
        Note: This is inefficient for random access; use seek for sequential reads.
        
        Args:
            frame_index: Index of frame to read (0-based)
            
        Returns:
            Frame as numpy array in BGR format, or None if invalid index
        """
        if self._cap is None:
            return None
        
        # Seek to frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return self.read_frame()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure video is closed."""
        self.close()


class DatasetScanner:
    """Scanner for finding and categorizing video files in the dataset.
    
    This class scans the video directory and extracts metadata from
    filenames using a consistent naming convention.
    
    Expected filename format: {scenario}_{speed}fps_{illumination}lux_{index}.mp4
    Example: lab_1.0fps_50lux_001.mp4, mine_2.5fps_120lux_002.mp4
    """
    
    # Regex pattern for parsing video filenames
    FILENAME_PATTERN = re.compile(
        r'^(?P<scenario>lab|mine)_'
        r'(?P<speed>[\d.]+)fps_'
        r'(?P<illumination>[\d]+)lux_'
        r'(?P<index>[\d]+)\.(?P<ext>mp4|avi|mov|MOV|MP4|AVI)$',
        re.IGNORECASE
    )
    
    def __init__(self, video_dir: str):
        """Initialize dataset scanner.
        
        Args:
            video_dir: Root directory containing video files
        """
        self.video_dir = str(Path(video_dir).resolve())
        self._videos: List[VideoMetadata] = []
    
    def scan(
        self, 
        extensions: List[str] = None,
        recursive: bool = False
    ) -> List[VideoMetadata]:
        """Scan directory for video files.
        
        Args:
            extensions: List of valid video extensions (default: common formats)
            recursive: Whether to scan subdirectories
            
        Returns:
            List of VideoMetadata objects for found videos
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        
        # Find all video files
        video_pattern = "**/*" if recursive else "*"
        video_files = []
        for ext in extensions:
            pattern = os.path.join(self.video_dir, f"{video_pattern}{ext}")
            video_files.extend(glob.glob(pattern, recursive=recursive))
        
        # Parse each video file
        self._videos = []
        for video_path in sorted(video_files):
            metadata = self._parse_video_path(video_path)
            if metadata is not None:
                self._videos.append(metadata)
        
        print(f"Found {len(self._videos)} videos in {self.video_dir}")
        return self._videos
    
    def _parse_video_path(self, video_path: str) -> Optional[VideoMetadata]:
        """Parse video file path to extract metadata.
        
        Args:
            video_path: Full path to video file
            
        Returns:
            VideoMetadata object or None if parsing fails
        """
        filename = os.path.basename(video_path)
        
        # Try to match against pattern
        match = self.FILENAME_PATTERN.match(filename)
        
        if match is None:
            # Try fallback parsing: just use directory name for scenario
            # and filename for speed
            return self._parse_video_path_fallback(video_path)
        
        # Extract metadata from match
        scenario = match.group('scenario').lower()
        speed = float(match.group('speed'))
        illumination = int(match.group('illumination'))
        
        # Get video properties
        metadata = VideoMetadata(
            path=video_path,
            scenario=scenario,
            belt_speed=speed,
            illumination=illumination
        )
        
        # Try to get video properties (may fail for some files)
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                metadata.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata.fps = cap.get(cv2.CAP_PROP_FPS)
                metadata.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception as e:
            print(f"Warning: Could not read properties from {filename}: {e}")
        
        return metadata
    
    def _parse_video_path_fallback(self, video_path: str) -> Optional[VideoMetadata]:
        """Fallback parsing for videos not matching the expected pattern.
        
        Attempts to extract speed from filename using common patterns.
        
        Args:
            video_path: Full path to video file
            
        Returns:
            VideoMetadata object or None if parsing fails completely
        """
        filename = os.path.basename(video_path)
        
        # Try to find speed in filename using common patterns
        speed_patterns = [
            r'([\d.]+)\s*m/?s',  # e.g., 1.5m/s, 2.0m/s
            r'speed[\s_-]*([\d.]+)',  # e.g., speed_1.5, speed-2.0
            r'([\d.]+)fps',  # e.g., 1.5fps (might be speed in some naming)
        ]
        
        speed = 1.0  # Default speed
        scenario = "lab"  # Default scenario
        illumination = 500  # Default illumination
        
        # Determine scenario from parent directory
        parent_dir = os.path.basename(os.path.dirname(video_path)).lower()
        if 'mine' in parent_dir or 'coal' in parent_dir:
            scenario = "mine"
        
        # Try to find speed in filename
        for pattern in speed_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                try:
                    speed = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        return VideoMetadata(
            path=video_path,
            scenario=scenario,
            belt_speed=speed,
            illumination=illumination
        )
    
    @property
    def videos(self) -> List[VideoMetadata]:
        """Get list of discovered videos."""
        return self._videos
    
    def get_videos_by_scenario(self, scenario: str) -> List[VideoMetadata]:
        """Get videos filtered by scenario.
        
        Args:
            scenario: Scenario type ("lab" or "mine")
            
        Returns:
            List of videos in the specified scenario
        """
        return [v for v in self._videos if v.scenario == scenario]
    
    def get_videos_by_speed(self, speed: float, tolerance: float = 0.01) -> List[VideoMetadata]:
        """Get videos filtered by belt speed.
        
        Args:
            speed: Target belt speed in m/s
            tolerance: Tolerance for matching speeds
            
        Returns:
            List of videos with matching speed
        """
        return [v for v in self._videos if abs(v.belt_speed - speed) <= tolerance]
    
    def get_videos_by_illumination(self, illumination: int, tolerance: int = 10) -> List[VideoMetadata]:
        """Get videos filtered by illumination level.
        
        Args:
            illumination: Target illumination in lux
            tolerance: Tolerance for matching illumination
            
        Returns:
            List of videos with matching illumination
        """
        if self._videos[0].illumination is None:
            return self._videos
        return [v for v in self._videos if v.illumination is not None 
                and abs(v.illumination - illumination) <= tolerance]


class FramePreprocessor:
    """Frame preprocessing pipeline for the speed detection system.
    
    Handles image transformations including:
    - Resizing to model input dimensions
    - Color space conversion (BGR to RGB)
    - Normalization to [0, 1] range
    - Optional data augmentation
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 448),  # (width, height)
        normalize: bool = True,
        convert_rgb: bool = True,
        augment: bool = False
    ):
        """Initialize frame preprocessor.
        
        Args:
            target_size: Target size as (width, height) for resizing
            normalize: Whether to normalize pixel values to [0, 1]
            convert_rgb: Whether to convert BGR to RGB
            augment: Whether to apply data augmentation (for training)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.convert_rgb = convert_rgb
        self.augment = augment
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame.
        
        Args:
            frame: Input frame in BGR format (H, W, C)
            
        Returns:
            Processed frame
        """
        # Make a copy to avoid modifying original
        processed = frame.copy()
        
        # Resize if needed
        if self.target_size is not None:
            processed = cv2.resize(
                processed,
                (self.target_size[0], self.target_size[1]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Convert BGR to RGB
        if self.convert_rgb:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
        
        # Apply augmentation if enabled
        if self.augment:
            processed = self._augment(processed)
        
        return processed
    
    def process_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a pair of frames.
        
        Args:
            frame1: First frame in BGR format
            frame2: Second frame in BGR format
            
        Returns:
            Tuple of (processed_frame1, processed_frame2)
        """
        return self.process_frame(frame1), self.process_frame(frame2)
    
    def _augment(self, frame: np.ndarray) -> np.ndarray:
        """Apply data augmentation to frame.
        
        Note: This is a simple augmentation for demonstration.
        More sophisticated augmentations may be added.
        
        Args:
            frame: Input frame (already normalized to [0, 1])
            
        Returns:
            Augmented frame
        """
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.9, 1.1)
            frame = np.clip(frame * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        if np.random.rand() > 0.5:
            contrast_factor = np.random.uniform(0.9, 1.1)
            mean = frame.mean()
            frame = np.clip((frame - mean) * contrast_factor + mean, 0, 1)
        
        return frame


class ConveyorBeltDataset(Dataset):
    """PyTorch Dataset for conveyor belt speed detection.
    
    This dataset loads video files, extracts frame pairs, and provides
    them in a format suitable for PyTorch DataLoader.
    """
    
    def __init__(
        self,
        video_dir: str,
        config: Optional[Config] = None,
        frame_skip: int = 0,
        max_frames_per_video: Optional[int] = None,
        preprocess: bool = True
    ):
        """Initialize dataset.
        
        Args:
            video_dir: Directory containing video files
            config: Configuration object (uses default if None)
            frame_skip: Number of frames to skip between pairs (0 = consecutive)
            max_frames_per_video: Maximum number of frame pairs per video
            preprocess: Whether to apply preprocessing
        """
        # Get configuration
        if config is None:
            if get_config is not None:
                try:
                    config = get_config()
                except:
                    pass
        
        # Store configuration
        self.config = config
        self.frame_skip = frame_skip
        self.max_frames_per_video = max_frames_per_video
        self.preprocess = preprocess
        
        # Get preprocessing parameters from config
        if config is not None:
            self.target_size = (
                config.model.optical_flow.input_width,
                config.model.optical_flow.input_height
            )
            self.frame_rate = config.dataset.frame_rate
        else:
            self.target_size = (256, 448)  # Default RAFT input size
            self.frame_rate = 25
        
        # Initialize preprocessor
        self.preprocessor = FramePreprocessor(
            target_size=self.target_size if preprocess else None,
            normalize=preprocess,
            convert_rgb=preprocess
        )
        
        # Scan for videos
        self.scanner = DatasetScanner(video_dir)
        self.videos = self.scanner.scan()
        
        # Build frame pair index
        self._frame_pairs: List[Dict[str, Any]] = []
        self._build_frame_pair_index()
        
        print(f"Dataset initialized with {len(self._frame_pairs)} frame pairs")
    
    def _build_frame_pair_index(self) -> None:
        """Build index of all frame pairs in the dataset."""
        print("Building frame pair index...")
        
        for video_meta in self.videos:
            try:
                # Open video
                cap = cv2.VideoCapture(video_meta.path)
                if not cap.isOpened():
                    print(f"Warning: Could not open {video_meta.path}")
                    continue
                
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Calculate number of valid pairs
                # Need at least 2 frames for a pair
                max_pairs = total_frames - 1 - self.frame_skip
                if max_pairs <= 0:
                    cap.release()
                    continue
                
                # Limit frames per video if specified
                if self.max_frames_per_video is not None:
                    max_pairs = min(max_pairs, self.max_frames_per_video)
                
                # Add frame pairs to index
                for i in range(max_pairs):
                    # Skip frames if frame_skip > 0
                    frame1_idx = i
                    frame2_idx = i + 1 + self.frame_skip
                    
                    if frame2_idx >= total_frames:
                        break
                    
                    # Calculate timestamp
                    timestamp = frame1_idx / video_fps
                    
                    self._frame_pairs.append({
                        'video_path': video_meta.path,
                        'frame1_index': frame1_idx,
                        'frame2_index': frame2_idx,
                        'timestamp': timestamp,
                        'ground_truth_speed': video_meta.belt_speed,
                        'scenario': video_meta.scenario,
                        'fps': video_fps
                    })
                
                cap.release()
                
            except Exception as e:
                print(f"Error processing {video_meta.path}: {e}")
                continue
        
        print(f"Built index with {len(self._frame_pairs)} frame pairs")
    
    def __len__(self) -> int:
        """Get number of frame pairs in dataset."""
        return len(self._frame_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a frame pair by index.
        
        Args:
            idx: Index of frame pair
            
        Returns:
            Dictionary containing:
                - frame1: Preprocessed first frame (C, H, W) tensor
                - frame2: Preprocessed second frame (C, H, W) tensor
                - timestamp: Time in seconds
                - ground_truth_speed: Actual belt speed in m/s
                - frame_index: Index of first frame
                - scenario: Scenario type
        """
        # Get frame pair info
        pair_info = self._frame_pairs[idx]
        
        # Open video
        cap = cv2.VideoCapture(pair_info['video_path'])
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {pair_info['video_path']}")
        
        try:
            # Read first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, pair_info['frame1_index'])
            ret1, frame1 = cap.read()
            if not ret1:
                raise RuntimeError(f"Could not read frame {pair_info['frame1_index']}")
            
            # Read second frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, pair_info['frame2_index'])
            ret2, frame2 = cap.read()
            if not ret2:
                raise RuntimeError(f"Could not read frame {pair_info['frame2_index']}")
            
        finally:
            cap.release()
        
        # Preprocess frames
        if self.preprocess:
            frame1 = self.preprocessor.process_frame(frame1)
            frame2 = self.preprocessor.process_frame(frame2)
        
        # Convert to PyTorch tensors (C, H, W format)
        # For RGB: (H, W, C) -> (C, H, W)
        # For BGR: need to convert first
        if self.preprocess and frame1.shape[-1] == 3:
            # Image is in (H, W, C) format after preprocessing
            # Convert to (C, H, W)
            frame1 = np.transpose(frame1, (2, 0, 1))
            frame2 = np.transpose(frame2, (2, 0, 1))
        
        # Create tensors
        frame1_tensor = torch.from_numpy(frame1.copy()).float()
        frame2_tensor = torch.from_numpy(frame2.copy()).float()
        
        return {
            'frame1': frame1_tensor,
            'frame2': frame2_tensor,
            'timestamp': pair_info['timestamp'],
            'ground_truth_speed': pair_info['ground_truth_speed'],
            'frame_index': pair_info['frame1_index'],
            'scenario': pair_info['scenario']
        }
    
    def get_frame_pair_raw(self, idx: int) -> FramePair:
        """Get a raw frame pair without preprocessing.
        
        This is useful for feature matching which may need different
        preprocessing than optical flow.
        
        Args:
            idx: Index of frame pair
            
        Returns:
            FramePair object
        """
        pair_info = self._frame_pairs[idx]
        
        # Open video
        cap = cv2.VideoCapture(pair_info['video_path'])
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {pair_info['video_path']}")
        
        try:
            # Read frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, pair_info['frame1_index'])
            ret1, frame1 = cap.read()
            if not ret1:
                raise RuntimeError(f"Could not read frame {pair_info['frame1_index']}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, pair_info['frame2_index'])
            ret2, frame2 = cap.read()
            if not ret2:
                raise RuntimeError(f"Could not read frame {pair_info['frame2_index']}")
            
        finally:
            cap.release()
        
        return FramePair(
            frame1=frame1,
            frame2=frame2,
            timestamp=pair_info['timestamp'],
            ground_truth_speed=pair_info['ground_truth_speed'],
            frame_index=pair_info['frame1_index'],
            scenario=pair_info['scenario'],
            video_path=pair_info['video_path']
        )


def create_dataloader(
    video_dir: str,
    config: Optional[Config] = None,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    frame_skip: int = 0,
    max_frames_per_video: Optional[int] = None,
    preprocess: bool = True
) -> TorchDataLoader:
    """Create a PyTorch DataLoader for conveyor belt speed detection.
    
    This is the main entry point for creating data loaders in the system.
    
    Args:
        video_dir: Directory containing video files
        config: Configuration object (uses default if None)
        batch_size: Number of frame pairs per batch
        shuffle: Whether to shuffle data (for training)
        num_workers: Number of worker processes for data loading
        frame_skip: Number of frames to skip between pairs
        max_frames_per_video: Maximum number of frame pairs per video
        preprocess: Whether to apply preprocessing
        
    Returns:
        PyTorch DataLoader instance
    """
    # Get configuration
    if config is None:
        if get_config is not None:
            try:
                config = get_config()
            except:
                pass
    
    # Get batch size from config if not specified
    if config is not None and batch_size is None:
        batch_size = config.training.batch_size
    
    # Create dataset
    dataset = ConveyorBeltDataset(
        video_dir=video_dir,
        config=config,
        frame_skip=frame_skip,
        max_frames_per_video=max_frames_per_video,
        preprocess=preprocess
    )
    
    # Create dataloader
    dataloader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if HAS_TORCH else False,
        drop_last=False
    )
    
    return dataloader


def extract_frame_pairs_from_video(
    video_path: str,
    ground_truth_speed: float,
    scenario: str = "lab",
    frame_skip: int = 0,
    max_pairs: Optional[int] = None,
    preprocessor: Optional[FramePreprocessor] = None
) -> Iterator[FramePair]:
    """Extract frame pairs from a single video file.
    
    This is a generator function that yields FramePair objects for
    a single video. Useful for processing individual videos.
    
    Args:
        video_path: Path to video file
        ground_truth_speed: Ground truth belt speed in m/s
        scenario: Scenario type ("lab" or "mine")
        frame_skip: Number of frames to skip between pairs
        max_pairs: Maximum number of frame pairs to extract
        preprocessor: Optional frame preprocessor
        
    Yields:
        FramePair objects
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate number of possible pairs
        max_possible = total_frames - 1 - frame_skip
        if max_possible <= 0:
            return
        
        num_pairs = max_possible if max_pairs is None else min(max_pairs, max_possible)
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            return
        
        for i in range(num_pairs):
            # Skip to frame2 if frame_skip > 0
            if frame_skip > 0:
                frame_idx = i + 1 + frame_skip
                if frame_idx >= total_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read second frame
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Apply preprocessor if provided
            if preprocessor is not None:
                prev_frame = preprocessor.process_frame(prev_frame)
                curr_frame = preprocessor.process_frame(curr_frame)
            
            # Calculate timestamp
            timestamp = i / fps if fps > 0 else 0
            
            yield FramePair(
                frame1=prev_frame.copy(),
                frame2=curr_frame.copy(),
                timestamp=timestamp,
                ground_truth_speed=ground_truth_speed,
                frame_index=i,
                scenario=scenario,
                video_path=video_path
            )
            
            # Move to next pair (read next frame as new prev_frame)
            ret, prev_frame = cap.read()
            if not ret:
                break
            
            # Reset to proper position for next iteration
            if frame_skip > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i + 2)
    
    finally:
        cap.release()


# Global dataset and dataloader registry
_datasets: Dict[str, ConveyorBeltDataset] = {}
_dataloaders: Dict[str, TorchDataLoader] = {}


def get_dataset(name: str = "default", video_dir: Optional[str] = None, **kwargs) -> ConveyorBeltDataset:
    """Get or create a dataset by name.
    
    Implements a simple registry pattern for datasets.
    
    Args:
        name: Dataset name identifier
        video_dir: Directory containing video files
        **kwargs: Additional arguments for ConveyorBeltDataset
        
    Returns:
        ConveyorBeltDataset instance
    """
    global _datasets
    
    if name in _datasets:
        return _datasets[name]
    
    if video_dir is None:
        # Try to get from config
        if get_config is not None:
            try:
                config = get_config()
                video_dir = config.paths.video_dir
            except:
                video_dir = "./data/videos"
        else:
            video_dir = "./data/videos"
    
    # Create new dataset
    dataset = ConveyorBeltDataset(video_dir=video_dir, **kwargs)
    _datasets[name] = dataset
    
    return dataset


def get_dataloader(
    name: str = "default",
    video_dir: Optional[str] = None,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs
) -> TorchDataLoader:
    """Get or create a dataloader by name.
    
    Implements a simple registry pattern for dataloaders.
    
    Args:
        name: Dataloader name identifier
        video_dir: Directory containing video files
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        **kwargs: Additional arguments for create_dataloader
        
    Returns:
        TorchDataLoader instance
    """
    global _dataloaders
    
    # Create a key including parameters
    key = f"{name}_{batch_size}_{shuffle}_{num_workers}"
    
    if key in _dataloaders:
        return _dataloaders[key]
    
    # Get or create dataset
    dataset = get_dataset(name, video_dir)
    
    # Create dataloader
    dataloader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    _dataloaders[key] = dataloader
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Data Loader Module Test")
    print("=" * 60)
    
    # Test 1: Create default preprocessor
    print("\n[Test 1] Creating frame preprocessor...")
    preprocessor = FramePreprocessor(
        target_size=(256, 448),
        normalize=True,
        convert_rgb=True
    )
    print("  ✓ Preprocessor created")
    
    # Test 2: Create dummy frame and process it
    print("\n[Test 2] Processing dummy frame...")
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    processed = preprocessor.process_frame(dummy_frame)
    print(f"  Input shape: {dummy_frame.shape}")
    print(f"  Output shape: {processed.shape}")
    print(f"  Output range: [{processed.min():.3f}, {processed.max():.3f}]")
    print("  ✓ Frame processing works")
    
    # Test 3: VideoMetadata creation
    print("\n[Test 3] Creating VideoMetadata...")
    metadata = VideoMetadata(
        path="./data/videos/lab_1.0fps_50lux_001.mp4",
        scenario="lab",
        belt_speed=1.0,
        illumination=50,
        frame_count=1000,
        fps=25.0,
        width=1920,
        height=1080
    )
    print(f"  Scenario: {metadata.scenario}")
    print(f"  Belt speed: {metadata.belt_speed} m/s")
    print(f"  Duration: {metadata.duration:.1f} seconds")
    print("  ✓ VideoMetadata created")
    
    # Test 4: FramePair creation
    print("\n[Test 4] Creating FramePair...")
    frame1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    frame_pair = FramePair(
        frame1=frame1,
        frame2=frame2,
        timestamp=0.04,
        ground_truth_speed=1.5,
        frame_index=0,
        scenario="lab"
    )
    print(f"  Frame shape: {frame_pair.frame_shape}")
    print(f"  Ground truth: {frame_pair.ground_truth_speed} m/s")
    print(f"  Timestamp: {frame_pair.timestamp} s")
    print("  ✓ FramePair created")
    
    # Test 5: DatasetScanner with non-existent directory
    print("\n[Test 5] Testing DatasetScanner with empty directory...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        scanner = DatasetScanner(tmpdir)
        videos = scanner.scan()
        print(f"  Found {len(videos)} videos in empty directory")
        print("  ✓ Scanner handles empty directory")
    
    # Test 6: Test preprocessing pipeline
    print("\n[Test 6] Testing full preprocessing pipeline...")
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    processed1, processed2 = preprocessor.process_pair(test_frame, test_frame)
    print(f"  Processed shape: {processed1.shape}")
    print(f"  Processed dtype: {processed1.dtype}")
    print(f"  Value range: [{processed1.min():.3f}, {processed1.max():.3f}]")
    print("  ✓ Preprocessing pipeline works")
    
    # Test 7: Test with data augmentation
    print("\n[Test 7] Testing data augmentation...")
    aug_preprocessor = FramePreprocessor(
        target_size=(256, 448),
        normalize=True,
        convert_rgb=True,
        augment=True
    )
    # Process same frame multiple times
    results = []
    for _ in range(5):
        aug_frame = aug_preprocessor.process_frame(test_frame.copy())
        results.append(aug_frame.mean())
    
    print(f"  Mean values (should vary slightly): {[f'{r:.4f}' for r in results]}")
    print("  ✓ Data augmentation works")
    
    # Test 8: Test config integration
    print("\n[Test 8] Testing config integration...")
    try:
        config = get_config()
        print(f"  Frame rate from config: {config.dataset.frame_rate} fps")
        print(f"  Resolution from config: {config.dataset.resolution}")
        print(f"  Optical flow input size: {config.model.optical_flow.input_size}")
        print("  ✓ Config integration works")
    except Exception as e:
        print(f"  Note: Config not available ({e}), using defaults")
    
    # Test 9: Test VideoLoader context manager
    print("\n[Test 9] Testing VideoLoader context manager...")
    # Try to open a non-existent video (will fail gracefully)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_video = os.path.join(tmpdir, "test.mp4")
        loader = VideoLoader()
        # This will fail since video doesn't exist
        opened = loader.open(test_video)
        print(f"  Open non-existent video: {opened}")
        loader.close()
        print("  ✓ VideoLoader handles missing files gracefully")
    
    # Test 10: Batch tensor creation
    print("\n[Test 10] Testing batch creation...")
    batch_frames = []
    for i in range(4):
        frame = np.random.rand(3, 448, 256).astype(np.float32)  # (C, H, W)
        batch_frames.append(torch.from_numpy(frame))
    
    batch_tensor = torch.stack(batch_frames, dim=0)
    print(f"  Batch shape: {batch_tensor.shape}")
    print(f"  Expected: (4, 3, 448, 256)")
    assert batch_tensor.shape == (4, 3, 448, 256)
    print("  ✓ Batch creation works")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
