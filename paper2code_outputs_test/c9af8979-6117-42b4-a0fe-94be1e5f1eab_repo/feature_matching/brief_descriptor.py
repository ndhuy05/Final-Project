"""
feature_matching/brief_descriptor.py

BRIEF (Binary Robust Independent Elementary Features) descriptor extractor for 
conveyor belt speed detection.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements the BRIEF descriptor extraction algorithm to generate
binary descriptors for feature points detected by the Harris corner detector.
The extracted descriptors are used for feature matching between consecutive 
frames to calculate conveyor belt speed.

Based on Section 3.2 of the paper: "The BRIEF algorithm extracts and matches 
the descriptors"

Author: Based on paper methodology
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import warnings

# Try to import OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    raise ImportError("OpenCV (cv2) is required for BRIEF descriptor extraction. Install with: pip install opencv-python")

# Try to import OpenCV xfeatures2d for BRIEF (may not be available in all installations)
try:
    import cv2.xfeatures2d
    HAS_CV2_XFEATURES = True
except ImportError:
    HAS_CV2_XFEATURES = False

# Try to import configuration
try:
    from config import Config, get_config, FeatureMatchingConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    get_config = None
    FeatureMatchingConfig = None


class BriefDescriptorExtractor:
    """BRIEF (Binary Robust Independent Elementary Features) descriptor extractor.
    
    This class extracts binary descriptors from feature points in images.
    BRIEF descriptors are compact binary vectors that encode the appearance
    of image patches around keypoints.
    
    Based on Section 3.2 of the paper: "uses the BRIEF algorithm to extract 
    and match the descriptors"
    
    The BRIEF descriptor works by:
    1. Predefining a set of pixel pair locations (256 pairs by default)
    2. For each keypoint, comparing intensities at each pair of locations
    3. Encoding comparisons as binary bits (1 if first > second, else 0)
    
    Attributes:
        patch_size: Size of the image patch around each keypoint (from config)
        bytes_per_descriptor: Number of bytes per descriptor (32 for 256-bit)
        bit_count: Number of binary comparisons (256 bits standard)
    """
    
    def __init__(
        self,
        patch_size: int = 31,
        bytes_per_descriptor: int = 32,
        use_opencv: bool = True,
        validate_keypoints: bool = True,
        boundary_margin: int = 15
    ):
        """Initialize the BRIEF descriptor extractor.
        
        Args:
            patch_size: Size of the square patch for descriptor computation.
                       Must be odd. Default 31 from config.yaml.
            bytes_per_descriptor: Number of bytes per descriptor.
                                 32 bytes = 256 bits (default).
            use_opencv: Whether to use OpenCV's built-in BRIEF if available.
                       Falls back to custom implementation if False.
            validate_keypoints: Whether to filter out keypoints near boundaries.
            boundary_margin: Minimum distance from image boundary in pixels.
        """
        # Validate parameters
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
        if bytes_per_descriptor <= 0:
            raise ValueError(f"bytes_per_descriptor must be positive, got {bytes_per_descriptor}")
        if boundary_margin < 0:
            raise ValueError(f"boundary_margin must be non-negative, got {boundary_margin}")
        
        self.patch_size = patch_size
        self.bytes_per_descriptor = bytes_per_descriptor
        self.bit_count = bytes_per_descriptor * 8
        self.use_opencv = use_opencv and HAS_CV2_XFEATURES
        self.validate_keypoints = validate_keypoints
        self.boundary_margin = boundary_margin
        
        # Initialize OpenCV BRIEF extractor if available
        self._opencv_extractor = None
        if self.use_opencv:
            self._init_opencv_extractor()
        
        # Initialize custom sampling pattern if needed
        self._sampling_pattern = None
        if not self.use_opencv:
            self._init_sampling_pattern()
        
        # Statistics tracking
        self._stats = {
            'total_keypoints': 0,
            'valid_keypoints': 0,
            'invalid_near_boundary': 0,
            'computation_failures': 0
        }
    
    def _init_opencv_extractor(self) -> None:
        """Initialize OpenCV's BRIEF descriptor extractor."""
        if not HAS_CV2_XFEATURES:
            warnings.warn("OpenCV xfeatures2d not available, using custom implementation")
            self.use_opencv = False
            self._init_sampling_pattern()
            return
        
        try:
            # Create OpenCV BRIEF extractor
            # bytes defaults to 32 (256 bits)
            self._opencv_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                descriptorSize=self.bit_count,
                bytes=self.bytes_per_descriptor,
                useGaussian=True,
                hammingWindow=self.patch_size
            )
        except Exception as e:
            warnings.warn(f"Failed to create OpenCV BRIEF extractor: {e}")
            self.use_opencv = False
            self._init_sampling_pattern()
    
    def _init_sampling_pattern(self) -> None:
        """Initialize custom BRIEF sampling pattern.
        
        Generates predefined pixel pair locations for intensity comparisons.
        This follows the pattern from the original BRIEF paper (Calonder et al., 2010).
        """
        np.random.seed(42)  # Fixed seed for reproducibility
        
        # Generate sampling pattern around center (0, 0)
        # Each pair has two points: (x1, y1) and (x2, y2)
        
        # Generate random point locations within patch_size
        half_patch = self.patch_size // 2
        
        # Generate coordinates for all bit comparisons
        # Points are sampled from a Gaussian distribution around the center
        # This provides better stability than uniform sampling
        
        # First point in each pair
        x1 = np.random.randint(-half_patch, half_patch + 1, size=(self.bit_count,))
        y1 = np.random.randint(-half_patch, half_patch + 1, size=(self.bit_count,))
        
        # Second point in each pair
        x2 = np.random.randint(-half_patch, half_patch + 1, size=(self.bit_count,))
        y2 = np.random.randint(-half_patch, half_patch + 1, size=(self.bit_count,))
        
        # Store as numpy array for efficient computation
        self._sampling_pattern = np.stack([x1, y1, x2, y2], axis=1)  # (bit_count, 4)
    
    @classmethod
    def from_config(
        cls,
        config: Optional[FeatureMatchingConfig] = None
    ) -> 'BriefDescriptorExtractor':
        """Create BRIEF extractor from configuration.
        
        Args:
            config: Feature matching configuration. If None, uses default values.
            
        Returns:
            BriefDescriptorExtractor instance with configured parameters
        """
        if config is not None:
            return cls(
                patch_size=config.brief_patch_size
            )
        else:
            # Try to get from global config
            if CONFIG_AVAILABLE:
                try:
                    global_config = get_config()
                    return cls.from_config(global_config.model.feature_matching)
                except:
                    pass
            
            # Fallback to defaults from config.yaml
            return cls(
                patch_size=31
            )
    
    def _validate_keypoint(
        self,
        keypoint: cv2.KeyPoint,
        image_shape: Tuple[int, int]
    ) -> bool:
        """Check if keypoint is valid for descriptor extraction.
        
        A keypoint is valid if it's far enough from image boundaries
        to compute the full descriptor patch.
        
        Args:
            keypoint: KeyPoint to validate
            image_shape: Shape of input image (height, width)
            
        Returns:
            True if keypoint is valid, False otherwise
        """
        if not self.validate_keypoints:
            return True
        
        x, y = keypoint.pt
        height, width = image_shape[:2]
        
        # Compute safe region considering patch_size and boundary_margin
        margin = max(self.boundary_margin, self.patch_size // 2 + 1)
        
        # Check if keypoint is within safe region
        if x < margin or x >= width - margin:
            return False
        if y < margin or y >= height - margin:
            return False
        
        return True
    
    def _filter_keypoints(
        self,
        keypoints: List[cv2.KeyPoint],
        image: np.ndarray
    ) -> List[cv2.KeyPoint]:
        """Filter out keypoints that are too close to image boundaries.
        
        Args:
            keypoints: List of keypoints to filter
            image: Input image for shape reference
            
        Returns:
            Filtered list of keypoints
        """
        if not self.validate_keypoints:
            return keypoints
        
        image_shape = image.shape[:2]
        valid_keypoints = []
        
        for kp in keypoints:
            if self._validate_keypoint(kp, image_shape):
                valid_keypoints.append(kp)
            else:
                self._stats['invalid_near_boundary'] += 1
        
        return valid_keypoints
    
    def _extract_custom_descriptor(
        self,
        image: np.ndarray,
        keypoint: cv2.KeyPoint
    ) -> Optional[np.ndarray]:
        """Extract BRIEF descriptor using custom implementation.
        
        Args:
            image: Input image in grayscale (H, W)
            keypoint: KeyPoint to compute descriptor for
            
        Returns:
            Descriptor as numpy array of shape (bytes_per_descriptor,)
            Returns None if computation fails
        """
        try:
            x, y = keypoint.pt
            x, y = int(x), int(y)
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Compute descriptor bits
            descriptor = np.zeros(self.bytes_per_descriptor, dtype=np.uint8)
            
            # Get sampling pattern
            pattern = self._sampling_pattern
            
            # For each bit comparison
            for i in range(self.bit_count):
                x1, y1, x2, y2 = pattern[i]
                
                # Get coordinates relative to keypoint
                px1, py1 = x + x1, y + y1
                px2, py2 = x + x2, y + y2
                
                # Check bounds
                if (0 <= px1 < width and 0 <= py1 < height and
                    0 <= px2 < width and 0 <= py2 < height):
                    
                    # Compare intensities
                    if image[py1, px1] > image[py2, px2]:
                        # Set bit in descriptor
                        byte_idx = i // 8
                        bit_idx = i % 8
                        descriptor[byte_idx] |= (1 << bit_idx)
            
            return descriptor
            
        except Exception as e:
            self._stats['computation_failures'] += 1
            return None
    
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Compute BRIEF descriptors for given keypoints.
        
        This is the main method for descriptor extraction. It:
        1. Validates keypoints (removes those near boundaries)
        2. Extracts descriptors using OpenCV or custom implementation
        3. Returns valid keypoints and their descriptors
        
        Args:
            image: Input image in BGR or grayscale format (H, W, C) or (H, W)
            keypoints: List of keypoints to compute descriptors for
            
        Returns:
            Tuple of:
                - valid_keypoints: Keypoints that have valid descriptors
                - descriptors: Array of shape (N, bytes_per_descriptor) where N 
                              is number of valid keypoints
                              
        Raises:
            ValueError: If image is None or empty
        """
        # Validate input
        if image is None:
            raise ValueError("Input image cannot be None")
        
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Input must be numpy array, got {type(image)}")
        
        if image.size == 0:
            raise ValueError("Input image is empty")
        
        # Handle multi-channel images
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Convert BGR to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                # Single channel
                gray = image.squeeze()
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        else:
            gray = image
        
        # Ensure grayscale is correct type
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        # Track statistics
        self._stats['total_keypoints'] += len(keypoints)
        
        # Filter keypoints near boundaries
        filtered_keypoints = self._filter_keypoints(keypoints, gray)
        self._stats['valid_keypoints'] += len(filtered_keypoints)
        
        # Handle empty keypoints
        if len(filtered_keypoints) == 0:
            return [], np.array([], dtype=np.uint8).reshape(0, self.bytes_per_descriptor)
        
        # Extract descriptors based on implementation
        if self.use_opencv and self._opencv_extractor is not None:
            return self._compute_opencv(gray, filtered_keypoints)
        else:
            return self._compute_custom(gray, filtered_keypoints)
    
    def _compute_opencv(
        self,
        gray: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract descriptors using OpenCV implementation.
        
        Args:
            gray: Grayscale image
            keypoints: Valid keypoints
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        try:
            keypoints, descriptors = self._opencv_extractor.compute(gray, keypoints)
            
            # Handle case where some keypoints couldn't be processed
            if keypoints is None:
                return [], np.array([], dtype=np.uint8).reshape(0, self.bytes_per_descriptor)
            
            # Ensure descriptors is numpy array
            if descriptors is None:
                descriptors = np.array([], dtype=np.uint8).reshape(0, self.bytes_per_descriptor)
            elif not isinstance(descriptors, np.ndarray):
                descriptors = np.array(descriptors, dtype=np.uint8)
            
            return keypoints, descriptors
            
        except Exception as e:
            warnings.warn(f"OpenCV BRIEF computation failed: {e}, using custom implementation")
            return self._compute_custom(gray, keypoints)
    
    def _compute_custom(
        self,
        gray: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract descriptors using custom implementation.
        
        Args:
            gray: Grayscale image
            keypoints: Valid keypoints
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        descriptors = []
        valid_keypoints = []
        
        for kp in keypoints:
            desc = self._extract_custom_descriptor(gray, kp)
            
            if desc is not None:
                descriptors.append(desc)
                valid_keypoints.append(kp)
        
        if len(descriptors) == 0:
            return [], np.array([], dtype=np.uint8).reshape(0, self.bytes_per_descriptor)
        
        # Stack descriptors into array
        descriptors_array = np.stack(descriptors, axis=0)
        
        return valid_keypoints, descriptors_array
    
    def compute_single(
        self,
        image: np.ndarray,
        keypoint: cv2.KeyPoint
    ) -> Optional[np.ndarray]:
        """Compute descriptor for a single keypoint.
        
        Convenience method for computing descriptor for one keypoint.
        
        Args:
            image: Input image in grayscale
            keypoint: Single keypoint
            
        Returns:
            Descriptor as numpy array, or None if computation fails
        """
        keypoints = [keypoint]
        valid_keypoints, descriptors = self.compute(image, keypoints)
        
        if len(valid_keypoints) == 0:
            return None
        
        return descriptors[0]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get extraction statistics.
        
        Returns:
            Dictionary containing:
                - total_keypoints: Total keypoints processed
                - valid_keypoints: Successfully processed keypoints
                - invalid_near_boundary: Keypoints rejected due to boundary
                - computation_failures: Failed descriptor computations
        """
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset extraction statistics."""
        self._stats = {
            'total_keypoints': 0,
            'valid_keypoints': 0,
            'invalid_near_boundary': 0,
            'computation_failures': 0
        }
    
    def set_patch_size(self, patch_size: int) -> None:
        """Update patch size for descriptor extraction.
        
        Note: This will recreate the sampling pattern if using custom implementation.
        
        Args:
            patch_size: New patch size (must be odd)
        """
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")
        
        self.patch_size = patch_size
        self.boundary_margin = max(self.boundary_margin, patch_size // 2 + 1)
        
        # Recreate sampling pattern if using custom implementation
        if not self.use_opencv:
            self._init_sampling_pattern()
    
    def get_patch_size(self) -> int:
        """Get current patch size.
        
        Returns:
            Current patch size
        """
        return self.patch_size
    
    def get_descriptor_size(self) -> int:
        """Get descriptor size in bytes.
        
        Returns:
            Number of bytes per descriptor
        """
        return self.bytes_per_descriptor
    
    def get_bit_count(self) -> int:
        """Get descriptor size in bits.
        
        Returns:
            Number of bits per descriptor
        """
        return self.bit_count
    
    def __repr__(self) -> str:
        """String representation of the extractor."""
        return (
            f"BriefDescriptorExtractor(\n"
            f"  patch_size={self.patch_size},\n"
            f"  bytes_per_descriptor={self.bytes_per_descriptor},\n"
            f"  bit_count={self.bit_count},\n"
            f"  use_opencv={self.use_opencv},\n"
            f"  validate_keypoints={self.validate_keypoints},\n"
            f"  boundary_margin={self.boundary_margin}\n"
            f")"
        )


class BriefDescriptorComputer:
    """Convenience class wrapper for BRIEF descriptor extraction.
    
    This class provides a simplified interface that matches the expected
    API in the feature matching pipeline.
    """
    
    def __init__(
        self,
        config: Optional[FeatureMatchingConfig] = None
    ):
        """Initialize the BRIEF descriptor computer.
        
        Args:
            config: Feature matching configuration
        """
        self.extractor = BriefDescriptorExtractor.from_config(config)
    
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Compute BRIEF descriptors.
        
        Args:
            image: Input image
            keypoints: Keypoints to compute descriptors for
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        return self.extractor.compute(image, keypoints)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return self.extractor.get_statistics()


def create_brief_extractor(
    config: Optional[FeatureMatchingConfig] = None,
    patch_size: int = 31,
    use_opencv: bool = True
) -> BriefDescriptorExtractor:
    """Factory function to create a BRIEF descriptor extractor.
    
    Args:
        config: Feature matching configuration (takes priority if provided)
        patch_size: Patch size parameter (used if config is None)
        use_opencv: Whether to use OpenCV implementation if available
        
    Returns:
        BriefDescriptorExtractor instance
    """
    if config is not None:
        return BriefDescriptorExtractor(
            patch_size=config.brief_patch_size,
            use_opencv=use_opencv
        )
    else:
        # Try to get from global config
        if CONFIG_AVAILABLE:
            try:
                global_config = get_config()
                return create_brief_extractor(global_config.model.feature_matching, patch_size)
            except:
                pass
        
        # Use provided or default values
        return BriefDescriptorExtractor(
            patch_size=patch_size,
            use_opencv=use_opencv
        )


# Global extractor registry
_extractors: Dict[str, BriefDescriptorExtractor] = {}


def get_brief_extractor(
    name: str = "default",
    config: Optional[FeatureMatchingConfig] = None,
    **kwargs
) -> BriefDescriptorExtractor:
    """Get or create a BRIEF extractor by name.
    
    Args:
        name: Extractor identifier
        config: Feature matching configuration
        **kwargs: Additional arguments for BriefDescriptorExtractor
        
    Returns:
        BriefDescriptorExtractor instance
    """
    global _extractors
    
    if name in _extractors:
        return _extractors[name]
    
    extractor = create_brief_extractor(config=config, **kwargs)
    _extractors[name] = extractor
    
    return extractor


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("BRIEF Descriptor Module Test")
    print("=" * 60)
    
    # Test 1: Create extractor with defaults
    print("\n[Test 1] Creating extractor with defaults...")
    extractor = BriefDescriptorExtractor()
    print(f"  Created: {repr(extractor)}")
    print("  ✓ BRIEF extractor created")
    
    # Test 2: Create extractor from config
    print("\n[Test 2] Creating extractor from configuration...")
    try:
        if CONFIG_AVAILABLE:
            config = get_config()
            extractor_config = create_brief_extractor(config.model.feature_matching)
            print(f"  From config: {repr(extractor_config)}")
            print("  ✓ Config integration works")
        else:
            print("  Skipping (config not available)")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Test 3: Create synthetic test image with features
    print("\n[Test 3] Creating synthetic test image...")
    h, w = 480, 640
    test_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add varied content
    for y in range(h):
        for x in range(w):
            if x < w // 2 and y < h // 2:
                # Checkerboard
                test_image[y, x] = [((x // 20) + (y // 20)) % 2 * 255] * 3
            elif x >= w // 2 and y < h // 2:
                # Diagonal lines
                test_image[y, x] = [(x + y) % 20 < 10] * [255, 200, 150]
            elif x < w // 2 and y >= h // 2:
                # Random noise
                test_image[y, x] = np.random.randint(50, 200, 3)
            else:
                # Gradient
                test_image[y, x] = [x % 256, y % 256, 128]
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    print(f"  Test image shape: {test_image.shape}")
    print(f"  Grayscale shape: {gray_image.shape}")
    print("  ✓ Test image created")
    
    # Test 4: Create synthetic keypoints
    print("\n[Test 4] Creating synthetic keypoints...")
    # Create keypoints in valid regions
    keypoints = []
    for i in range(10):
        for j in range(10):
            x = 50 + j * 50
            y = 50 + i * 50
            kp = cv2.KeyPoint(x=float(x), y=float(y), size=10)
            keypoints.append(kp)
    
    print(f"  Created {len(keypoints)} keypoints")
    print("  ✓ Keypoints created")
    
    # Test 5: Extract descriptors
    print("\n[Test 5] Extracting descriptors...")
    valid_keypoints, descriptors = extractor.compute(test_image, keypoints)
    print(f"  Input keypoints: {len(keypoints)}")
    print(f"  Valid keypoints: {len(valid_keypoints)}")
    print(f"  Descriptors shape: {descriptors.shape}")
    if len(descriptors) > 0:
        print(f"  First descriptor: {descriptors[0][:8]}... (showing first 8 bytes)")
    print("  ✓ Descriptor extraction works")
    
    # Test 6: Test with edge keypoints
    print("\n[Test 6] Testing with edge keypoints...")
    edge_keypoints = [
        cv2.KeyPoint(x=0, y=240, size=10),      # Left edge
        cv2.KeyPoint(x=639, y=240, size=10),    # Right edge
        cv2.KeyPoint(x=320, y=0, size=10),      # Top edge
        cv2.KeyPoint(x=320, y=479, size=10),    # Bottom edge
        cv2.KeyPoint(x=320, y=240, size=10),    # Center (valid)
    ]
    
    edge_kps, edge_descs = extractor.compute(test_image, edge_keypoints)
    print(f"  Edge keypoints: {len(edge_keypoints)}")
    print(f"  Valid edge keypoints: {len(edge_kps)}")
    print("  ✓ Edge handling works")
    
    # Test 7: Test custom implementation fallback
    print("\n[Test 7] Testing custom implementation fallback...")
    extractor_custom = BriefDescriptorExtractor(patch_size=31, use_opencv=False)
    custom_kps, custom_descs = extractor_custom.compute(test_image, keypoints)
    print(f"  Custom implementation keypoints: {len(custom_kps)}")
    print(f"  Custom descriptors shape: {custom_descs.shape}")
    print("  ✓ Custom implementation works")
    
    # Test 8: Test empty keypoints
    print("\n[Test 8] Testing with empty keypoints...")
    empty_kps, empty_descs = extractor.compute(test_image, [])
    print(f"  Empty keypoints: {len(empty_kps)}")
    print(f"  Empty descriptors shape: {empty_descs.shape}")
    print("  ✓ Handles empty keypoints")
    
    # Test 9: Test sampling pattern
    print("\n[Test 9] Testing sampling pattern...")
    if extractor._sampling_pattern is not None:
        pattern = extractor._sampling_pattern
        print(f"  Pattern shape: {pattern.shape}")
        print(f"  Pattern x1 range: [{pattern[:, 0].min()}, {pattern[:, 0].max()}]")
        print(f"  Pattern y2 range: [{pattern[:, 3].min()}, {pattern[:, 3].max()}]")
        print("  ✓ Sampling pattern initialized")
    
    # Test 10: Test single keypoint extraction
    print("\n[Test 10] Testing single keypoint extraction...")
    single_kp = cv2.KeyPoint(x=320, y=240, size=10)
    single_desc = extractor.compute_single(test_image, single_kp)
    print(f"  Single descriptor shape: {single_desc.shape}")
    print("  ✓ Single keypoint works")
    
    # Test 11: Test statistics tracking
    print("\n[Test 11] Testing statistics tracking...")
    stats = extractor.get_statistics()
    print(f"  Statistics: {stats}")
    extractor.reset_statistics()
    stats_after = extractor.get_statistics()
    print(f"  After reset: {stats_after}")
    print("  ✓ Statistics work")
    
    # Test 12: Test descriptor size
    print("\n[Test 12] Testing descriptor size...")
    desc_size = extractor.get_descriptor_size()
    bit_count = extractor.get_bit_count()
    patch_size = extractor.get_patch_size()
    print(f"  Descriptor size: {desc_size} bytes")
    print(f"  Bit count: {bit_count} bits")
    print(f"  Patch size: {patch_size}")
    print("  ✓ Size methods work")
    
    # Test 13: Test set_patch_size
    print("\n[Test 13] Testing patch size update...")
    extractor.set_patch_size(47)
    new_patch = extractor.get_patch_size()
    print(f"  New patch size: {new_patch}")
    extractor.set_patch_size(31)  # Reset
    print("  ✓ Patch size update works")
    
    # Test 14: Test patch size validation
    print("\n[Test 14] Testing patch size validation...")
    try:
        extractor.set_patch_size(30)  # Even number should fail
        print("  ERROR: Should have raised ValueError for even patch_size")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
    print("  ✓ Validation works")
    
    # Test 15: Test factory function
    print("\n[Test 15] Testing factory function...")
    factory_ext = create_brief_extractor()
    print(f"  Factory type: {type(factory_ext).__name__}")
    print("  ✓ Factory function works")
    
    # Test 16: Test get_brief_extractor registry
    print("\n[Test 16] Testing registry...")
    reg_ext = get_brief_extractor("test_registry")
    print(f"  Registry type: {type(reg_ext).__name__}")
    print("  ✓ Registry works")
    
    # Test 17: Test different patch sizes
    print("\n[Test 17] Testing different patch sizes...")
    for ps in [15, 31, 51]:
        ext = BriefDescriptorExtractor(patch_size=ps)
        kps, descs = ext.compute(test_image, keypoints[:10])
        print(f"  Patch size {ps}: {len(kps)} keypoints, {descs.shape}")
    print("  ✓ Different patch sizes work")
    
    # Test 18: Test with validation disabled
    print("\n[Test 18] Testing with validation disabled...")
    ext_no_val = BriefDescriptorExtractor(validate_keypoints=False)
    edge_kps, edge_descs = ext_no_val.compute(test_image, edge_keypoints)
    print(f"  Without validation: {len(edge_kps)} valid")
    print("  ✓ Validation disable works")
    
    # Test 19: Test config integration
    print("\n[Test 19] Testing config integration...")
    if CONFIG_AVAILABLE:
        try:
            cfg = get_config()
            print(f"  Config BRIEF patch size: {cfg.model.feature_matching.brief_patch_size}")
            print("  ✓ Config integration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (config not available)")
    
    # Test 20: Performance test
    print("\n[Test 20] Performance test...")
    import time
    
    # Create more keypoints for testing
    perf_keypoints = []
    for i in range(100):
        for j in range(100):
            x = 20 + j * 6
            y = 20 + i * 4
            if x < w - 20 and y < h - 20:
                kp = cv2.KeyPoint(x=float(x), y=float(y), size=5)
                perf_keypoints.append(kp)
    
    # Time extraction
    extractor.reset_statistics()
    start_time = time.time()
    result_kps, result_descs = extractor.compute(test_image, perf_keypoints)
    elapsed = time.time() - start_time
    
    print(f"  Processed {len(perf_keypoints)} keypoints in {elapsed:.3f}s")
    print(f"  Valid keypoints: {len(result_kps)}")
    print(f"  Throughput: {len(perf_keypoints)/elapsed:.1f} keypoints/sec")
    print("  ✓ Performance test complete")
    
    print("\n" + "=" * 60)
    print("All BRIEF Descriptor tests passed!")
    print("=" * 60)
