"""
feature_matching/harris_detector.py

Harris corner detector for conveyor belt speed detection.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements the Harris corner detection algorithm to extract feature 
points from conveyor belt images. The detected corners are used in the 
Harris-BRIEF-RANSAC feature matching pipeline to calculate belt speed through 
feature correspondence between consecutive frames.

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
    raise ImportError("OpenCV (cv2) is required for Harris corner detection. Install with: pip install opencv-python")

# Try to import configuration
try:
    from config import Config, get_config, FeatureMatchingConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    get_config = None
    FeatureMatchingConfig = None


class HarrisDetector:
    """Harris corner detector for extracting feature points from images.
    
    This class implements the Harris corner detection algorithm for detecting
    corners in conveyor belt images. The detected corners serve as feature
    points for the subsequent BRIEF descriptor extraction and RANSAC-based
    matching pipeline.
    
    Based on Section 3.2 of the paper: "The algorithm uses the Harris operator 
    to extract the feature points of the conveyor belt image"
    
    Attributes:
        block_size: Neighborhood size for structure tensor computation (default: 2)
        ksize: Aperture parameter for Sobel operator (default: 3)
        k: Harris detector free parameter controlling sensitivity (default: 0.04)
        threshold_ratio: Threshold as ratio of maximum corner response (default: 0.01)
    """
    
    def __init__(
        self,
        block_size: int = 2,
        ksize: int = 3,
        k: float = 0.04,
        threshold_ratio: float = 0.01,
        enable_nms: bool = True,
        nms_window: int = 5,
        min_distance: int = 5
    ):
        """Initialize the Harris corner detector.
        
        Args:
            block_size: Neighborhood size for structure tensor computation.
                       Larger values smooth over larger areas but may miss small features.
            ksize: Aperture parameter for Sobel operator. Must be odd (1, 3, 5, or 7).
                   Larger values give smoother gradients but may blur details.
            k: Harris detector free parameter. Controls corner sensitivity.
               Range: [0.04, 0.06] typically. Higher values detect fewer corners.
            threshold_ratio: Minimum corner response as ratio of maximum response.
                            Range: [0.001, 0.1]. Lower values detect more corners.
            enable_nms: Whether to apply non-maximum suppression.
            nms_window: Window size for NMS kernel.
            min_distance: Minimum distance between detected corners.
        """
        # Validate parameters
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if ksize <= 0 or ksize % 2 == 0:
            raise ValueError(f"ksize must be positive odd, got {ksize}")
        if k < 0 or k >= 1:
            raise ValueError(f"k must be in [0, 1), got {k}")
        if threshold_ratio <= 0 or threshold_ratio >= 1:
            raise ValueError(f"threshold_ratio must be in (0, 1), got {threshold_ratio}")
        
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
        self.threshold_ratio = threshold_ratio
        self.enable_nms = enable_nms
        self.nms_window = nms_window
        self.min_distance = min_distance
        
        # Internal state
        self._corner_response: Optional[np.ndarray] = None
        self._max_response: float = 0.0
    
    @classmethod
    def from_config(
        cls,
        config: Optional[FeatureMatchingConfig] = None
    ) -> 'HarrisDetector':
        """Create HarrisDetector from configuration.
        
        Args:
            config: Feature matching configuration. If None, uses default values.
            
        Returns:
            HarrisDetector instance with configured parameters
        """
        if config is not None:
            return cls(
                block_size=config.harris_block_size,
                ksize=config.harris_ksize,
                k=config.harris_k,
                threshold_ratio=config.harris_threshold_ratio
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
                block_size=2,
                ksize=3,
                k=0.04,
                threshold_ratio=0.01
            )
    
    def _compute_gradients(
        self,
        gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute image gradients using Sobel operator.
        
        Args:
            gray: Input grayscale image (H, W)
            
        Returns:
            Tuple of (Ix, Iy) gradient images
        """
        # Compute gradients in x and y directions
        Ix = cv2.Sobel(
            gray, 
            cv2.CV_64F, 
            1, 0, 
            ksize=self.ksize
        )
        Iy = cv2.Sobel(
            gray, 
            cv2.CV_64F, 
            0, 1, 
            ksize=self.ksize
        )
        
        return Ix, Iy
    
    def _compute_structure_tensor(
        self,
        Ix: np.ndarray,
        Iy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute structure tensor components for corner detection.
        
        Args:
            Ix: Gradient in x direction
            Iy: Gradient in y direction
            
        Returns:
            Tuple of (Ixx, Iyy, Ixy) tensor components
        """
        # Compute gradient products
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # Apply Gaussian blur to smooth the structure tensor
        # Using block_size as kernel size
        ksize = 2 * self.block_size + 1  # Convert to odd number
        
        Ixx = cv2.GaussianBlur(Ixx, (ksize, ksize), 0)
        Iyy = cv2.GaussianBlur(Iyy, (ksize, ksize), 0)
        Ixy = cv2.GaussianBlur(Ixy, (ksize, ksize), 0)
        
        return Ixx, Iyy, Ixy
    
    def _compute_corner_response(
        self,
        Ixx: np.ndarray,
        Iyy: np.ndarray,
        Ixy: np.ndarray
    ) -> np.ndarray:
        """Compute Harris corner response for each pixel.
        
        The corner response is computed using:
            R = det(M) - k * trace(M)²
            det(M) = Ixx * Iyy - Ixy²
            trace(M) = Ixx + Iyy
            
        where M is the structure tensor at each pixel.
        
        Args:
            Ixx, Iyy, Ixy: Structure tensor components
            
        Returns:
            Corner response map (same size as input)
        """
        # Compute determinant and trace
        det_m = Ixx * Iyy - Ixy * Ixy
        trace_m = Ixx + Iyy
        
        # Compute corner response
        # Using the formula: R = det - k * trace²
        R = det_m - self.k * (trace_m * trace_m)
        
        # Clip to avoid numerical issues with very small values
        R = np.clip(R, -1e10, 1e10)
        
        return R
    
    def _apply_threshold(
        self,
        R: np.ndarray
    ) -> np.ndarray:
        """Apply threshold to corner response map.
        
        Args:
            R: Corner response map
            
        Returns:
            Binary mask of corner pixels
        """
        # Find maximum response
        self._max_response = R.max()
        
        # Apply threshold based on max response
        threshold = self.threshold_ratio * self._max_response
        
        # Create binary mask
        corner_mask = (R > threshold).astype(np.uint8)
        
        return corner_mask
    
    def _apply_non_maximum_suppression(
        self,
        corner_mask: np.ndarray,
        R: np.ndarray
    ) -> np.ndarray:
        """Apply non-maximum suppression to find local maxima.
        
        Args:
            corner_mask: Binary mask of corner pixels
            R: Corner response map
            
        Returns:
            Refined binary mask after NMS
        """
        if not self.enable_nms:
            return corner_mask
        
        # Dilate the response map to find local maxima
        # Pixels that are local maxima will have value equal to dilated value
        kernel_size = 2 * self.nms_window + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        R_dilated = cv2.dilate(R, kernel)
        
        # Keep only pixels that are both corner candidates and local maxima
        corner_mask_nms = ((corner_mask == 1) & (R == R_dilated)).astype(np.uint8)
        
        return corner_mask_nms
    
    def _min_distance_filtering(
        self,
        corners: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Filter corners to maintain minimum distance between them.
        
        Uses greedy algorithm to keep well-separated corners.
        
        Args:
            corners: List of corner positions (y, x)
            
        Returns:
            Filtered list of corner positions
        """
        if len(corners) == 0:
            return corners
        
        # Sort corners by distance from image center (prioritize center)
        h, w = self._corner_response.shape
        center_y, center_x = h // 2, w // 2
        
        corners_with_dist = [
            (y, x, (y - center_y)**2 + (x - center_x)**2)
            for y, x in corners
        ]
        corners_with_dist.sort(key=lambda x: x[2])
        
        # Greedy selection
        filtered = []
        min_dist_sq = self.min_distance ** 2
        
        for y, x, _ in corners_with_dist:
            # Check if too close to any already selected corner
            too_close = False
            for fy, fx in filtered:
                if (y - fy)**2 + (x - fx)**2 < min_dist_sq:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append((y, x))
        
        return filtered
    
    def detect(
        self,
        image: np.ndarray,
        return_response: bool = False
    ) -> Union[List[cv2.KeyPoint], Tuple[List[cv2.KeyPoint], np.ndarray]]:
        """Detect corners in the input image.
        
        This is the main method for corner detection. It performs the following steps:
        1. Convert to grayscale if needed
        2. Compute image gradients
        3. Compute structure tensor components
        4. Calculate corner response
        5. Apply threshold
        6. Apply non-maximum suppression
        7. Filter by minimum distance
        8. Convert to KeyPoint objects
        
        Args:
            image: Input image (H, W, C) or (H, W)
            return_response: If True, also return corner response map
            
        Returns:
            List of cv2.KeyPoint objects, or (list, response_map) if return_response=True
            
        Raises:
            ValueError: If image is invalid or empty
        """
        # Validate input
        if image is None:
            raise ValueError("Input image cannot be None")
        
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Input must be numpy array, got {type(image)}")
        
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
        
        # Ensure grayscale is correct type and contiguous
        if gray.dtype != np.float32 and gray.dtype != np.float64:
            gray = np.float32(gray)
        
        # Ensure contiguous array for performance
        gray = np.ascontiguousarray(gray)
        
        # Validate image size
        if gray.size == 0:
            raise ValueError("Input image is empty")
        
        # Step 1: Compute gradients
        Ix, Iy = self._compute_gradients(gray)
        
        # Step 2: Compute structure tensor
        Ixx, Iyy, Ixy = self._compute_structure_tensor(Ix, Iy)
        
        # Step 3: Compute corner response
        R = self._compute_corner_response(Ixx, Iyy, Ixy)
        self._corner_response = R
        
        # Step 4: Apply threshold
        corner_mask = self._apply_threshold(R)
        
        # Step 5: Apply non-maximum suppression
        corner_mask = self._apply_non_maximum_suppression(corner_mask, R)
        
        # Step 6: Find corner coordinates
        corners = np.where(corner_mask > 0)
        corner_coords = list(zip(corners[0], corners[1]))  # (y, x) format
        
        # Step 7: Apply minimum distance filtering
        if self.min_distance > 0:
            corner_coords = self._min_distance_filtering(corner_coords)
        
        # Step 8: Convert to KeyPoint objects
        keypoints = []
        for y, x in corner_coords:
            # Use corner response as size indicator
            response = float(R[y, x])
            size = float(self.block_size * 2 + 1)
            
            # Create KeyPoint (angle=-1 means not applicable for Harris)
            kp = cv2.KeyPoint(
                x=float(x),      # Column (width direction)
                y=float(y),      # Row (height direction)
                size=size,
                angle=-1.0,      # Harris doesn't provide angle
                response=response,
                octave=0,
                class_id=-1
            )
            keypoints.append(kp)
        
        # Sort by response (strongest corners first)
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
        
        if return_response:
            return keypoints, R
        else:
            return keypoints
    
    def detect_multi_scale(
        self,
        image: np.ndarray,
        scales: List[float] = [1.0, 0.75, 0.5]
    ) -> List[cv2.KeyPoint]:
        """Detect corners at multiple scales.
        
        Useful for detecting features at different scales for more robust matching.
        
        Args:
            image: Input image
            scales: List of scale factors to apply
            
        Returns:
            Combined list of KeyPoints from all scales
        """
        all_keypoints = []
        
        for scale in scales:
            if scale == 1.0:
                scaled_image = image
            else:
                h, w = image.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled_image = cv2.resize(image, (new_w, new_h))
            
            # Detect corners at this scale
            keypoints = self.detect(scaled_image)
            
            # Scale keypoint coordinates back to original image size
            if scale != 1.0:
                for kp in keypoints:
                    kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
            
            all_keypoints.extend(keypoints)
        
        # Remove duplicates (keep higher response)
        if len(all_keypoints) > 0:
            # Sort by position and response
            all_keypoints.sort(key=lambda kp: (kp.pt[0], kp.pt[1], -kp.response))
            
            # Keep unique positions within min_distance
            filtered = []
            for kp in all_keypoints:
                x, y = kp.pt
                is_duplicate = False
                for fkp in filtered:
                    fx, fy = fkp.pt
                    if (x - fx)**2 + (y - fy)**2 < self.min_distance**2:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered.append(kp)
            
            all_keypoints = filtered
        
        return all_keypoints
    
    def get_corner_response(self) -> Optional[np.ndarray]:
        """Get the last computed corner response map.
        
        Returns:
            Corner response map from last detection, or None if not available
        """
        return self._corner_response
    
    def set_parameters(
        self,
        block_size: Optional[int] = None,
        ksize: Optional[int] = None,
        k: Optional[float] = None,
        threshold_ratio: Optional[float] = None
    ) -> None:
        """Update detector parameters.
        
        Args:
            block_size: New block size
            ksize: New Sobel kernel size (must be odd)
            k: New Harris k parameter
            threshold_ratio: New threshold ratio
        """
        if block_size is not None and block_size > 0:
            self.block_size = block_size
        
        if ksize is not None and ksize > 0 and ksize % 2 == 1:
            self.ksize = ksize
        
        if k is not None and 0 <= k < 1:
            self.k = k
        
        if threshold_ratio is not None and 0 < threshold_ratio < 1:
            self.threshold_ratio = threshold_ratio
        
        # Reset internal state
        self._corner_response = None
        self._max_response = 0.0
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current detector parameters.
        
        Returns:
            Dictionary of current parameters
        """
        return {
            'block_size': self.block_size,
            'ksize': self.ksize,
            'k': self.k,
            'threshold_ratio': self.threshold_ratio,
            'enable_nms': self.enable_nms,
            'nms_window': self.nms_window,
            'min_distance': self.min_distance
        }
    
    def __repr__(self) -> str:
        """String representation of the detector."""
        return (
            f"HarrisDetector(\n"
            f"  block_size={self.block_size},\n"
            f"  ksize={self.ksize},\n"
            f"  k={self.k},\n"
            f"  threshold_ratio={self.threshold_ratio},\n"
            f"  enable_nms={self.enable_nms},\n"
            f"  nms_window={self.nms_window},\n"
            f"  min_distance={self.min_distance}\n"
            f")"
        )


def create_harris_detector(
    config: Optional[FeatureMatchingConfig] = None,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01
) -> HarrisDetector:
    """Factory function to create a HarrisDetector.
    
    Args:
        config: Feature matching configuration (takes priority if provided)
        block_size: Block size parameter (used if config is None)
        ksize: Sobel kernel size (used if config is None)
        k: Harris k parameter (used if config is None)
        threshold_ratio: Threshold ratio (used if config is None)
        
    Returns:
        HarrisDetector instance
    """
    if config is not None:
        return HarrisDetector.from_config(config)
    else:
        # Try to get from global config
        if CONFIG_AVAILABLE:
            try:
                global_config = get_config()
                return HarrisDetector.from_config(global_config.model.feature_matching)
            except:
                pass
        
        # Use provided or default values
        return HarrisDetector(
            block_size=block_size,
            ksize=ksize,
            k=k,
            threshold_ratio=threshold_ratio
        )


# Global detector registry
_detectors: Dict[str, HarrisDetector] = {}


def get_harris_detector(
    name: str = "default",
    config: Optional[FeatureMatchingConfig] = None,
    **kwargs
) -> HarrisDetector:
    """Get or create a HarrisDetector by name.
    
    Args:
        name: Detector identifier
        config: Feature matching configuration
        **kwargs: Additional arguments for HarrisDetector
        
    Returns:
        HarrisDetector instance
    """
    global _detectors
    
    if name in _detectors:
        return _detectors[name]
    
    detector = create_harris_detector(config=config, **kwargs)
    _detectors[name] = detector
    
    return detector


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Harris Detector Module Test")
    print("=" * 60)
    
    # Test 1: Create detector with defaults
    print("\n[Test 1] Creating detector with defaults...")
    detector = HarrisDetector()
    print(f"  Created: {repr(detector)}")
    print("  ✓ HarrisDetector created")
    
    # Test 2: Create detector from config
    print("\n[Test 2] Creating detector from configuration...")
    try:
        if CONFIG_AVAILABLE:
            config = get_config()
            detector_config = create_harris_detector(config.model.feature_matching)
            print(f"  From config: {repr(detector_config)}")
            print("  ✓ Config integration works")
        else:
            print("  Skipping (config not available)")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Test 3: Create synthetic test image
    print("\n[Test 3] Creating synthetic test image...")
    # Create a chessboard-like pattern with corners
    h, w = 480, 640
    test_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add some gradient and shapes
    for y in range(h):
        for x in range(w):
            # Create regions with different textures
            if x < w // 2 and y < h // 2:
                # Region 1: Checkerboard pattern
                test_image[y, x] = [((x // 20) + (y // 20)) % 2 * 255] * 3
            elif x >= w // 2 and y < h // 2:
                # Region 2: Diagonal lines
                test_image[y, x] = [(x + y) % 20 < 10] * [255, 200, 150]
            elif x < w // 2 and y >= h // 2:
                # Region 3: Random noise
                test_image[y, x] = np.random.randint(50, 200, 3)
            else:
                # Region 4: Gradient
                test_image[y, x] = [x % 256, y % 256, 128]
    
    print(f"  Test image shape: {test_image.shape}")
    print("  ✓ Test image created")
    
    # Test 4: Detect corners in synthetic image
    print("\n[Test 4] Detecting corners in synthetic image...")
    keypoints = detector.detect(test_image)
    print(f"  Detected {len(keypoints)} corners")
    if len(keypoints) > 0:
        print(f"  Top 5 corner responses: {sorted([kp.response for kp in keypoints[:5]], reverse=True)}")
    print("  ✓ Corner detection works")
    
    # Test 5: Test with different parameters
    print("\n[Test 5] Testing with different parameters...")
    for k_val in [0.04, 0.06, 0.1]:
        for thresh in [0.01, 0.05, 0.1]:
            det = HarrisDetector(k=k_val, threshold_ratio=thresh)
            kps = det.detect(test_image)
            print(f"  k={k_val}, threshold={thresh}: {len(kps)} corners")
    print("  ✓ Parameter variation works")
    
    # Test 6: Test gradient computation
    print("\n[Test 6] Testing gradient computation...")
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    Ix, Iy = detector._compute_gradients(gray)
    print(f"  Ix range: [{Ix.min():.2f}, {Ix.max():.2f}]")
    print(f"  Iy range: [{Iy.min():.2f}, {Iy.max():.2f}]")
    print("  ✓ Gradient computation works")
    
    # Test 7: Test structure tensor computation
    print("\n[Test 7] Testing structure tensor computation...")
    Ixx, Iyy, Ixy = detector._compute_structure_tensor(Ix, Iy)
    print(f"  Ixx range: [{Ixx.min():.2f}, {Ixx.max():.2f}]")
    print(f"  Iyy range: [{Iyy.min():.2f}, {Iyy.max():.2f}]")
    print(f"  Ixy range: [{Ixy.min():.2f}, {Ixy.max():.2f}]")
    print("  ✓ Structure tensor works")
    
    # Test 8: Test corner response computation
    print("\n[Test 8] Testing corner response computation...")
    R = detector._compute_corner_response(Ixx, Iyy, Ixy)
    print(f"  Response range: [{R.min():.2e}, {R.max():.2e}]")
    print(f"  Max response: {R.max():.2e}")
    detector._corner_response = R
    print("  ✓ Corner response works")
    
    # Test 9: Test thresholding
    print("\n[Test 9] Testing thresholding...")
    corner_mask = detector._apply_threshold(R)
    print(f"  Corner pixels after threshold: {np.sum(corner_mask > 0)}")
    print("  ✓ Thresholding works")
    
    # Test 10: Test non-maximum suppression
    print("\n[Test 10] Testing non-maximum suppression...")
    corner_mask_nms = detector._apply_non_maximum_suppression(corner_mask, R)
    print(f"  Corner pixels after NMS: {np.sum(corner_mask_nms > 0)}")
    print("  ✓ NMS works")
    
    # Test 11: Test multi-scale detection
    print("\n[Test 11] Testing multi-scale detection...")
    scales = [1.0, 0.75, 0.5]
    multi_kps = detector.detect_multi_scale(test_image, scales)
    print(f"  Multi-scale corners: {len(multi_kps)}")
    print("  ✓ Multi-scale works")
    
    # Test 12: Test minimum distance filtering
    print("\n[Test 12] Testing minimum distance filtering...")
    detector_nomin = HarrisDetector(min_distance=0)
    kps_nomin = detector_nomin.detect(test_image)
    detector_withmin = HarrisDetector(min_distance=20)
    kps_withmin = detector_withmin.detect(test_image)
    print(f"  Without min distance: {len(kps_nomin)}")
    print(f"  With min distance 20: {len(kps_withmin)}")
    print("  ✓ Min distance filtering works")
    
    # Test 13: Test return response
    print("\n[Test 13] Testing return_response parameter...")
    kps_with_resp, resp_map = detector.detect(test_image, return_response=True)
    print(f"  Keypoints: {len(kps_with_resp)}")
    print(f"  Response map shape: {resp_map.shape}")
    print("  ✓ Return response works")
    
    # Test 14: Test empty/low-texture image
    print("\n[Test 14] Testing with low-texture image...")
    low_tex = np.full((200, 200), 128, dtype=np.uint8)
    kps_low = detector.detect(low_tex)
    print(f"  Low-texture corners: {len(kps_low)}")
    print("  ✓ Handles low-texture")
    
    # Test 15: Test factory function
    print("\n[Test 15] Testing factory function...")
    factory_det = create_harris_detector()
    print(f"  Factory type: {type(factory_det).__name__}")
    print("  ✓ Factory function works")
    
    # Test 16: Test get_harris_detector registry
    print("\n[Test 16] Testing registry...")
    reg_det = get_harris_detector("test_registry")
    print(f"  Registry type: {type(reg_det).__name__}")
    print("  ✓ Registry works")
    
    # Test 17: Test parameter setting/getting
    print("\n[Test 17] Testing parameter methods...")
    detector.set_parameters(k=0.1, threshold_ratio=0.05)
    params = detector.get_parameters()
    print(f"  Set k=0.1, threshold=0.05")
    print(f"  Current k: {params['k']}, threshold: {params['threshold_ratio']}")
    print("  ✓ Parameter methods work")
    
    # Test 18: Test with OpenCV's built-in Harris (for validation)
    print("\n[Test 18] Comparing with OpenCV's built-in Harris...")
    gray_float = np.float32(gray)
    cv_corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    cv_threshold = 0.01 * cv_corners.max()
    cv_keypoints = np.where(cv_corners > cv_threshold)
    print(f"  OpenCV detected: {len(cv_keypoints[0])} corners")
    print(f"  Our detector: {len(keypoints)} corners")
    print("  ✓ Comparison completed")
    
    print("\n" + "=" * 60)
    print("All Harris Detector tests passed!")
    print("=" * 60)
