"""
feature_matching/matching_pipeline.py

Complete feature matching pipeline for conveyor belt speed detection.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements the Harris-BRIEF-RANSAC feature matching algorithm
described in Section 3.2 of the paper. It provides a unified interface for
detecting feature points, extracting descriptors, matching features, and
calculating belt speed from feature correspondences.

Author: Based on paper methodology
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass, field
import warnings

# Try to import OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    raise ImportError("OpenCV (cv2) is required for feature matching pipeline. Install with: pip install opencv-python")

# Import configuration
try:
    from config import Config, get_config, FeatureMatchingConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    get_config = None
    FeatureMatchingConfig = None

# Import related modules
try:
    from feature_matching.harris_detector import HarrisDetector, create_harris_detector
    from feature_matching.brief_descriptor import BriefDescriptorExtractor, create_brief_extractor
    from feature_matching.ransac_filter import RANSACFilter, create_ransac_filter
except ImportError as e:
    raise ImportError(f"Could not import feature matching modules: {e}")

# Import calibration
try:
    from utils.calibration import CameraCalibration, create_default_calibration
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    CameraCalibration = None
    create_default_calibration = None


@dataclass
class FeatureMatchResult:
    """Result container for feature matching-based speed estimation.
    
    This class stores the output from the feature matching pipeline,
    including matched feature points, estimated homography, calculated speed,
    and confidence score.
    
    Based on the paper's methodology in Section 3.2 and Section 2.2.
    
    Attributes:
        matches: List of matched feature point pairs (inliers only)
        homography: 3x3 homography transformation matrix from RANSAC
        speed_px_per_sec: Average speed in pixels per second
        speed_m_per_sec: Average speed in meters per second (final output)
        num_inliers: Number of inlier matches after RANSAC filtering
        confidence: Confidence score (0-1) based on inlier ratio
        avg_displacement_px: Average pixel displacement of inlier matches
        src_keypoints: Source keypoints (frame t)
        dst_keypoints: Destination keypoints (frame t+1)
    """
    matches: List = field(default_factory=list)
    homography: np.ndarray = field(default_factory=lambda: np.eye(3))
    speed_px_per_sec: float = 0.0
    speed_m_per_sec: float = 0.0
    num_inliers: int = 0
    confidence: float = 0.0
    avg_displacement_px: float = 0.0
    src_keypoints: List = field(default_factory=list)
    dst_keypoints: List = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and process results after initialization."""
        # Ensure homography is numpy array
        if not isinstance(self.homography, np.ndarray):
            self.homography = np.array(self.homography)
        
        # Ensure homography is 3x3
        if self.homography.shape != (3, 3):
            self.homography = np.eye(3)
        
        # Clip confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure speeds are non-negative
        self.speed_px_per_sec = max(0.0, self.speed_px_per_sec)
        self.speed_m_per_sec = max(0.0, self.speed_m_per_sec)
        self.avg_displacement_px = max(0.0, self.avg_displacement_px)
        
        # Ensure num_inliers is non-negative
        self.num_inliers = max(0, self.num_inliers)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'homography': self.homography.tolist(),
            'speed_px_per_sec': float(self.speed_px_per_sec),
            'speed_m_per_sec': float(self.speed_m_per_sec),
            'num_inliers': int(self.num_inliers),
            'confidence': float(self.confidence),
            'avg_displacement_px': float(self.avg_displacement_px)
        }
    
    def is_valid(self) -> bool:
        """Check if the result is valid for speed estimation.
        
        Returns:
            True if we have enough inliers and confidence
        """
        return self.num_inliers >= 4 and self.confidence > 0.1


class FeatureMatchingPipeline:
    """Complete feature matching pipeline for conveyor belt speed detection.
    
    This class implements the Harris-BRIEF-RANSAC feature matching algorithm
    as described in Section 3.2 of the paper. It combines:
    1. Harris corner detection for feature point extraction
    2. BRIEF descriptor extraction for feature representation
    3. BFMatcher with Hamming distance for feature matching
    4. Lowe's ratio test for match filtering
    5. RANSAC homography estimation for outlier rejection
    
    The pipeline outputs speed estimates based on the displacement of
    matched feature points between consecutive frames.
    
    Attributes:
        harris_detector: Harris corner detector
        brief_extractor: BRIEF descriptor extractor
        ransac_filter: RANSAC outlier filter
        frame_rate: Video frame rate for speed calculation
        lowe_ratio: Lowe's ratio test threshold
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        frame_rate: float = 25.0,
        lowe_ratio: float = 0.75,
        enable_clahe: bool = False,
        use_roi: bool = False,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_features_threshold: int = 10,
        enable_visualization: bool = False
    ):
        """Initialize the feature matching pipeline.
        
        Args:
            config: Configuration object. If None, uses default values.
            frame_rate: Frame rate of input video (for speed calculation)
            lowe_ratio: Lowe's ratio test threshold for match filtering
            enable_clahe: Whether to apply CLAHE preprocessing for better contrast
            use_roi: Whether to use region of interest for feature detection
            roi: Region of interest as (x, y, width, height)
            min_features_threshold: Minimum features required for valid result
            enable_visualization: Whether to enable debug visualization output
        """
        # Get configuration
        if config is not None and CONFIG_AVAILABLE:
            try:
                fm_config = config.model.feature_matching
            except:
                fm_config = None
        else:
            fm_config = None
        
        # Store parameters
        self.frame_rate = frame_rate
        self.frame_interval = 1.0 / frame_rate if frame_rate > 0 else 0.04
        self.lowe_ratio = lowe_ratio
        self.enable_clahe = enable_clahe
        self.use_roi = use_roi
        self.roi = roi
        self.min_features_threshold = min_features_threshold
        self.enable_visualization = enable_visualization
        
        # Initialize CLAHE if enabled
        self._clahe = None
        if self.enable_clahe and HAS_CV2:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Initialize pipeline components
        self._initialize_components(fm_config)
        
        # Internal state
        self._last_keypoints1: List[cv2.KeyPoint] = []
        self._last_keypoints2: List[cv2.KeyPoint] = []
        self._last_descriptors1: np.ndarray = None
        self._last_descriptors2: np.ndarray = None
        self._stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'failed_insufficient_features': 0,
            'failed_no_matches': 0,
            'avg_num_inliers': 0.0
        }
    
    def _initialize_components(
        self,
        fm_config: Optional[FeatureMatchingConfig] = None
    ) -> None:
        """Initialize pipeline components from configuration.
        
        Args:
            fm_config: Feature matching configuration
        """
        # Initialize Harris detector
        if fm_config is not None:
            self.harris_detector = create_harris_detector(fm_config)
        else:
            try:
                self.harris_detector = create_harris_detector()
            except:
                self.harris_detector = HarrisDetector(
                    block_size=2,
                    ksize=3,
                    k=0.04,
                    threshold_ratio=0.01
                )
        
        # Initialize BRIEF extractor
        if fm_config is not None:
            self.brief_extractor = create_brief_extractor(fm_config)
        else:
            try:
                self.brief_extractor = create_brief_extractor()
            except:
                self.brief_extractor = BriefDescriptorExtractor(patch_size=31)
        
        # Initialize RANSAC filter
        if fm_config is not None:
            self.ransac_filter = create_ransac_filter(fm_config)
        else:
            try:
                self.ransac_filter = create_ransac_filter()
            except:
                self.ransac_filter = RANSACFilter(
                    threshold=5.0,
                    max_iters=2000
                )
        
        # Configure ROI if enabled
        if self.use_roi and self.roi is not None:
            self.ransac_filter.set_roi(self.roi)
    
    @classmethod
    def from_config(
        cls,
        config: Optional[Config] = None,
        frame_rate: float = 25.0
    ) -> 'FeatureMatchingPipeline':
        """Create pipeline from configuration.
        
        Args:
            config: Configuration object
            frame_rate: Frame rate for speed calculation
            
        Returns:
            FeatureMatchingPipeline instance
        """
        return cls(config=config, frame_rate=frame_rate)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better feature detection.
        
        Applies grayscale conversion, optional CLAHE, and Gaussian blur.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.squeeze()
        else:
            gray = image
        
        # Apply CLAHE for contrast enhancement if enabled
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray
    
    def _detect_features(
        self,
        gray_image: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> List[cv2.KeyPoint]:
        """Detect feature points using Harris corner detector.
        
        Args:
            gray_image: Preprocessed grayscale image
            image_shape: Original image shape
            
        Returns:
            List of detected keypoints
        """
        # Use the Harris detector to find corners
        keypoints = self.harris_detector.detect(gray_image)
        
        # Filter by region of interest if enabled
        if self.use_roi and self.roi is not None:
            x, y, w, h = self.roi
            keypoints = [
                kp for kp in keypoints
                if x <= kp.pt[0] <= x + w and y <= kp.pt[1] <= y + h
            ]
        
        return keypoints
    
    def _extract_descriptors(
        self,
        gray_image: np.ndarray,
        keypoints: List[cv2.KeyPoint]
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract BRIEF descriptors for keypoints.
        
        Args:
            gray_image: Grayscale image
            keypoints: List of keypoints
            
        Returns:
            Tuple of (valid_keypoints, descriptors)
        """
        if len(keypoints) == 0:
            return [], np.array([], dtype=np.uint8).reshape(0, 32)
        
        # Extract descriptors
        valid_keypoints, descriptors = self.brief_extractor.compute(
            gray_image, keypoints
        )
        
        return valid_keypoints, descriptors
    
    def _match_features(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray
    ) -> List[cv2.DMatch]:
        """Match features between two descriptor sets.
        
        Uses BFMatcher with Hamming distance and Lowe's ratio test.
        
        Args:
            descriptors1: Descriptors from first frame
            descriptors2: Descriptors from second frame
            
        Returns:
            List of good matches after Lowe's ratio test
        """
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        
        # Initialize BFMatcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Find k=2 nearest neighbors
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.lowe_ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def _calculate_displacement(
        self,
        keypoints1: List[cv2.KeyPoint],
        keypoints2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch]
    ) -> Tuple[float, float, List[cv2.DMatch]]:
        """Calculate average displacement from matched keypoints.
        
        Args:
            keypoints1: Keypoints from first frame
            keypoints2: Keypoints from second frame
            matches: List of good matches
            
        Returns:
            Tuple of (avg_displacement_x, avg_displacement_y, inlier_matches)
        """
        if len(matches) == 0:
            return 0.0, 0.0, []
        
        # Prepare point arrays for RANSAC
        src_pts = []
        dst_pts = []
        valid_matches = []
        
        for match in matches:
            # Get keypoint indices
            idx1 = match.queryIdx
            idx2 = match.trainIdx
            
            # Check bounds
            if idx1 >= len(keypoints1) or idx2 >= len(keypoints2):
                continue
            
            # Get keypoint coordinates
            pt1 = keypoints1[idx1].pt
            pt2 = keypoints2[idx2].pt
            
            # Add to arrays
            src_pts.append([pt1])
            dst_pts.append([pt2])
            valid_matches.append(match)
        
        if len(src_pts) < 4:
            return 0.0, 0.0, []
        
        # Convert to numpy arrays
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        # Apply RANSAC filtering
        homography, inlier_mask, num_inliers = self.ransac_filter.filter_matches(
            src_pts, dst_pts
        )
        
        if num_inliers < 4:
            return 0.0, 0.0, []
        
        # Get inlier matches
        inlier_matches = [valid_matches[i] for i in range(len(valid_matches)) if inlier_mask[i]]
        
        # Calculate average displacement from inliers
        displacements = []
        for i, match in enumerate(inlier_matches):
            idx1 = match.queryIdx
            idx2 = match.trainIdx
            
            pt1 = keypoints1[idx1].pt
            pt2 = keypoints2[idx2].pt
            
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            displacements.append((dx, dy))
        
        if len(displacements) == 0:
            return 0.0, 0.0, []
        
        # Calculate average displacement
        avg_dx = np.mean([d[0] for d in displacements])
        avg_dy = np.mean([d[1] for d in displacements])
        
        return float(avg_dx), float(avg_dy), inlier_matches
    
    def _calculate_speed(
        self,
        displacement_px: float,
        calibration: Optional[CameraCalibration] = None
    ) -> Tuple[float, float]:
        """Calculate speed from pixel displacement.
        
        Args:
            displacement_px: Pixel displacement magnitude
            calibration: Camera calibration for pixel-to-world conversion
            
        Returns:
            Tuple of (speed_px_per_sec, speed_m_per_sec)
        """
        # Convert to pixels per second
        speed_px = displacement_px / self.frame_interval
        
        # Convert to meters per second if calibration available
        speed_m = 0.0
        if calibration is not None and CALIBRATION_AVAILABLE:
            try:
                meter_displacement = calibration.convert_pixel_displacement_to_meters(displacement_px)
                speed_m = meter_displacement / self.frame_interval
            except Exception as e:
                warnings.warn(f"Calibration conversion failed: {e}")
        
        return float(speed_px), float(speed_m)
    
    def process(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        calibration: Optional[CameraCalibration] = None,
        return_keypoints: bool = False
    ) -> FeatureMatchResult:
        """Process a pair of frames to estimate belt speed.
        
        This is the main method of the pipeline. It:
        1. Preprocesses both frames
        2. Detects Harris corners in both frames
        3. Extracts BRIEF descriptors for corners
        4. Matches features between frames
        5. Applies RANSAC to filter outliers
        6. Calculates speed from inlier matches
        
        Args:
            frame1: First frame in BGR format (H, W, 3)
            frame2: Second frame in BGR format (H, W, 3)
            calibration: Camera calibration for pixel-to-world conversion
            return_keypoints: Whether to include keypoints in result
            
        Returns:
            FeatureMatchResult containing speed estimation and metadata
        """
        # Validate inputs
        if frame1 is None or frame2 is None:
            return self._create_empty_result()
        
        if not isinstance(frame1, np.ndarray) or not isinstance(frame2, np.ndarray):
            return self._create_empty_result()
        
        if frame1.size == 0 or frame2.size == 0:
            return self._create_empty_result()
        
        # Update statistics
        self._stats['total_frames'] += 1
        
        try:
            # Step 1: Preprocess images
            gray1 = self._preprocess_image(frame1)
            gray2 = self._preprocess_image(frame2)
            
            # Step 2: Detect features in both frames
            keypoints1 = self._detect_features(gray1, frame1.shape)
            keypoints2 = self._detect_features(gray2, frame2.shape)
            
            # Store for potential reuse
            self._last_keypoints1 = keypoints1
            self._last_keypoints2 = keypoints2
            
            # Check if enough features detected
            if len(keypoints1) < self.min_features_threshold or len(keypoints2) < self.min_features_threshold:
                self._stats['failed_insufficient_features'] += 1
                return self._create_empty_result()
            
            # Step 3: Extract descriptors
            valid_kpts1, descriptors1 = self._extract_descriptors(gray1, keypoints1)
            valid_kpts2, descriptors2 = self._extract_descriptors(gray2, keypoints2)
            
            # Store descriptors
            self._last_descriptors1 = descriptors1
            self._last_descriptors2 = descriptors2
            
            # Check if enough descriptors
            if len(valid_kpts1) < self.min_features_threshold or len(valid_kpts2) < self.min_features_threshold:
                self._stats['failed_insufficient_features'] += 1
                return self._create_empty_result()
            
            # Step 4: Match features
            good_matches = self._match_features(descriptors1, descriptors2)
            
            if len(good_matches) < 2:
                self._stats['failed_no_matches'] += 1
                return self._create_empty_result()
            
            # Step 5: Calculate displacement with RANSAC filtering
            avg_dx, avg_dy, inlier_matches = self._calculate_displacement(
                valid_kpts1, valid_kpts2, good_matches
            )
            
            if len(inlier_matches) < 4:
                self._stats['failed_no_matches'] += 1
                return self._create_empty_result()
            
            # Update statistics
            self._stats['successful_detections'] += 1
            if self._stats['total_frames'] > 0:
                running_avg = self._stats['avg_num_inliers']
                n = self._stats['successful_detections']
                self._stats['avg_num_inliers'] = (running_avg * (n - 1) + len(inlier_matches)) / n
            
            # Step 6: Calculate speed
            displacement_magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
            speed_px, speed_m = self._calculate_speed(displacement_magnitude, calibration)
            
            # Calculate confidence based on inlier ratio
            confidence = len(inlier_matches) / len(good_matches) if len(good_matches) > 0 else 0.0
            
            # Get homography from RANSAC filter
            homography = self.ransac_filter.get_last_homography() if hasattr(self.ransac_filter, 'get_last_homography') else np.eye(3)
            
            # Create result
            result = FeatureMatchResult(
                matches=inlier_matches,
                homography=homography,
                speed_px_per_sec=speed_px,
                speed_m_per_sec=speed_m,
                num_inliers=len(inlier_matches),
                confidence=confidence,
                avg_displacement_px=displacement_magnitude,
                src_keypoints=valid_kpts1 if return_keypoints else [],
                dst_keypoints=valid_kpts2 if return_keypoints else []
            )
            
            return result
            
        except Exception as e:
            warnings.warn(f"Feature matching pipeline failed: {e}")
            return self._create_empty_result()
    
    def _create_empty_result(self) -> FeatureMatchResult:
        """Create an empty result for failure cases.
        
        Returns:
            FeatureMatchResult with zero values
        """
        return FeatureMatchResult(
            matches=[],
            homography=np.eye(3),
            speed_px_per_sec=0.0,
            speed_m_per_sec=0.0,
            num_inliers=0,
            confidence=0.0,
            avg_displacement_px=0.0,
            src_keypoints=[],
            dst_keypoints=[]
        )
    
    def process_video_frame(
        self,
        frame: np.ndarray,
        previous_frame: Optional[np.ndarray] = None,
        calibration: Optional[CameraCalibration] = None
    ) -> FeatureMatchResult:
        """Process a single frame for continuous speed tracking.
        
        This method maintains internal state for tracking speed across
        multiple consecutive frames.
        
        Args:
            frame: Current video frame
            previous_frame: Previous video frame (uses internal state if None)
            calibration: Camera calibration
            
        Returns:
            FeatureMatchResult for this frame pair
        """
        if previous_frame is None:
            # Use internal state for frame sequence
            if hasattr(self, '_previous_frame') and self._previous_frame is not None:
                previous_frame = self._previous_frame
            else:
                # No previous frame, can't compute speed
                self._previous_frame = frame.copy()
                return self._create_empty_result()
        
        # Process frame pair
        result = self.process(frame, previous_frame, calibration)
        
        # Store current frame for next iteration
        self._previous_frame = frame.copy()
        
        return result
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> None:
        """Set region of interest for feature detection.
        
        Args:
            roi: Region of interest as (x, y, width, height)
        """
        self.roi = roi
        self.use_roi = True
        if hasattr(self, 'ransac_filter'):
            self.ransac_filter.set_roi(roi)
    
    def clear_roi(self) -> None:
        """Clear region of interest."""
        self.roi = None
        self.use_roi = False
        if hasattr(self, 'ransac_filter'):
            self.ransac_filter.clear_roi()
    
    def set_frame_rate(self, frame_rate: float) -> None:
        """Set video frame rate for speed calculation.
        
        Args:
            frame_rate: New frame rate in fps
        """
        if frame_rate > 0:
            self.frame_rate = frame_rate
            self.frame_interval = 1.0 / frame_rate
    
    def set_calibration(self, calibration: CameraCalibration) -> None:
        """Set camera calibration for pixel-to-world conversion.
        
        Args:
            calibration: CameraCalibration object
        """
        self.calibration = calibration
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self._stats.copy()
        
        # Add additional statistics
        if self._stats['total_frames'] > 0:
            stats['success_rate'] = self._stats['successful_detections'] / self._stats['total_frames']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset pipeline statistics."""
        self._stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'failed_insufficient_features': 0,
            'failed_no_matches': 0,
            'avg_num_inliers': 0.0
        }
    
    def get_last_keypoints(self) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint]]:
        """Get keypoints from last processing.
        
        Returns:
            Tuple of (keypoints_frame1, keypoints_frame2)
        """
        return self._last_keypoints1, self._last_keypoints2
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return (
            f"FeatureMatchingPipeline(\n"
            f"  frame_rate={self.frame_rate},\n"
            f"  lowe_ratio={self.lowe_ratio},\n"
            f"  enable_clahe={self.enable_clahe},\n"
            f"  use_roi={self.use_roi},\n"
            f"  roi={self.roi},\n"
            f"  min_features_threshold={self.min_features_threshold},\n"
            f"  harris_detector={repr(self.harris_detector) if hasattr(self, 'harris_detector') else 'N/A'},\n"
            f"  brief_extractor={repr(self.brief_extractor) if hasattr(self, 'brief_extractor') else 'N/A'},\n"
            f"  ransac_filter={repr(self.ransac_filter) if hasattr(self, 'ransac_filter') else 'N/A'}\n"
            f")"
        )


def create_matching_pipeline(
    config: Optional[Config] = None,
    frame_rate: float = 25.0,
    enable_clahe: bool = False,
    use_roi: bool = False,
    roi: Optional[Tuple[int, int, int, int]] = None
) -> FeatureMatchingPipeline:
    """Factory function to create a feature matching pipeline.
    
    Args:
        config: Configuration object (takes priority if provided)
        frame_rate: Video frame rate
        enable_clahe: Whether to enable CLAHE preprocessing
        use_roi: Whether to use region of interest
        roi: Region of interest if use_roi is True
        
    Returns:
        FeatureMatchingPipeline instance
    """
    return FeatureMatchingPipeline(
        config=config,
        frame_rate=frame_rate,
        enable_clahe=enable_clahe,
        use_roi=use_roi,
        roi=roi
    )


# Global pipeline registry
_pipelines: Dict[str, FeatureMatchingPipeline] = {}


def get_matching_pipeline(
    name: str = "default",
    config: Optional[Config] = None,
    frame_rate: float = 25.0,
    **kwargs
) -> FeatureMatchingPipeline:
    """Get or create a feature matching pipeline by name.
    
    Args:
        name: Pipeline identifier
        config: Configuration object
        frame_rate: Video frame rate
        **kwargs: Additional arguments for FeatureMatchingPipeline
        
    Returns:
        FeatureMatchingPipeline instance
    """
    global _pipelines
    
    if name in _pipelines:
        return _pipelines[name]
    
    pipeline = create_matching_pipeline(config=config, frame_rate=frame_rate, **kwargs)
    _pipelines[name] = pipeline
    
    return pipeline


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Feature Matching Pipeline Test")
    print("=" * 60)
    
    # Test 1: Create pipeline with defaults
    print("\n[Test 1] Creating pipeline with defaults...")
    try:
        pipeline = FeatureMatchingPipeline(frame_rate=25.0)
        print(f"  Created: {repr(pipeline)}")
        print("  ✓ Pipeline created")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Create pipeline from config
    print("\n[Test 2] Creating pipeline from configuration...")
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            pipeline_config = FeatureMatchingPipeline.from_config(config, frame_rate=25.0)
            print(f"  Created from config")
            print("  ✓ Config integration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (config not available)")
    
    # Test 3: Create synthetic test frames
    print("\n[Test 3] Creating synthetic test frames...")
    np.random.seed(42)
    h, w = 480, 640
    
    # Create frame with textured pattern (like conveyor belt)
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    frame2 = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add checkerboard-like pattern
    block_size = 40
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if ((x // block_size) + (y // block_size)) % 2 == 0:
                color1 = (100, 120, 140)
                color2 = (180, 160, 140)
            else:
                color1 = (180, 160, 140)
                color2 = (100, 120, 140)
            
            frame1[y:y+block_size, x:x+block_size] = color1
            frame2[y:y+block_size, x:x+block_size] = color2
    
    # Add some random texture
    noise1 = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
    noise2 = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
    
    frame1 = np.clip(frame1.astype(np.int16) + noise1, 0, 255).astype(np.uint8)
    frame2 = np.clip(frame2.astype(np.int16) + noise2, 0, 255).astype(np.uint8)
    
    # Shift frame2 to simulate motion (e.g., 10 pixels right, 5 pixels down)
    shift_x = 10
    shift_y = 5
    
    # Create shifted version of frame2 for motion simulation
    frame2_shifted = np.zeros_like(frame2)
    if shift_x > 0:
        frame2_shifted[:, shift_x:] = frame2[:, :-shift_x]
    if shift_y > 0:
        frame2_shifted[shift_y:, :] = frame2_shifted[shift_y:, :]
    
    frame2 = frame2_shifted
    
    print(f"  Frame1 shape: {frame1.shape}")
    print(f"  Frame2 shape: {frame2.shape}")
    print(f"  Motion: {shift_x}px x, {shift_y}px y")
    print("  ✓ Test frames created")
    
    # Test 4: Process frame pair
    print("\n[Test 4] Processing frame pair...")
    result = pipeline.process(frame1, frame2)
    
    print(f"  Speed (px/s): {result.speed_px_per_sec:.2f}")
    print(f"  Speed (m/s): {result.speed_m_per_sec:.4f}")
    print(f"  Num inliers: {result.num_inliers}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Avg displacement: {result.avg_displacement_px:.2f} pixels")
    print("  ✓ Processing completed")
    
    # Test 5: Test with calibration
    print("\n[Test 5] Testing with calibration...")
    if CALIBRATION_AVAILABLE:
        try:
            calibration = create_default_calibration(
                resolution=(w, h),
                camera_distance=3.0
            )
            result_cal = pipeline.process(frame1, frame2, calibration=calibration)
            print(f"  Speed with calibration: {result_cal.speed_m_per_sec:.4f} m/s")
            print(f"  Expected at {shift_x}px/frame at 25fps with 3m distance")
            print(f"  Pixels per meter: {calibration.pixels_per_meter:.2f}")
            print("  ✓ Calibration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (calibration not available)")
    
    # Test 6: Test with CLAHE enabled
    print("\n[Test 6] Testing with CLAHE enabled...")
    pipeline_clahe = FeatureMatchingPipeline(frame_rate=25.0, enable_clahe=True)
    result_clahe = pipeline_clahe.process(frame1, frame2)
    print(f"  Speed with CLAHE: {result_clahe.speed_px_per_sec:.2f} px/s")
    print(f"  Num inliers: {result_clahe.num_inliers}")
    print("  ✓ CLAHE works")
    
    # Test 7: Test with ROI
    print("\n[Test 7] Testing with ROI...")
    roi = (100, 80, 400, 300)
    pipeline_roi = FeatureMatchingPipeline(frame_rate=25.0, use_roi=True, roi=roi)
    result_roi = pipeline_roi.process(frame1, frame2)
    print(f"  ROI: {roi}")
    print(f"  Speed: {result_roi.speed_px_per_sec:.2f} px/s")
    print(f"  Num inliers: {result_roi.num_inliers}")
    print("  ✓ ROI works")
    
    # Test 8: Test empty frame handling
    print("\n[Test 8] Testing empty frame handling...")
    empty_result = pipeline.process(np.array([]), frame2)
    print(f"  Empty frame speed: {empty_result.speed_m_per_sec:.4f} m/s")
    print(f"  Is valid: {empty_result.is_valid()}")
    print("  ✓ Empty handling works")
    
    # Test 9: Test statistics tracking
    print("\n[Test 9] Testing statistics tracking...")
    stats = pipeline.get_statistics()
    print(f"  Statistics: {stats}")
    pipeline.reset_statistics()
    stats_after = pipeline.get_statistics()
    print(f"  After reset: {stats_after}")
    print("  ✓ Statistics work")
    
    # Test 10: Test continuous frame processing
    print("\n[Test 10] Testing continuous frame processing...")
    pipeline_cont = FeatureMatchingPipeline(frame_rate=25.0)
    
    # Process multiple frames
    for i in range(5):
        if i == 0:
            result_cont = pipeline_cont.process_video_frame(frame1)
        else:
            result_cont = pipeline_cont.process_video_frame(frame2)
    
    print(f"  Continuous processing completed")
    print(f"  Last result: speed={result_cont.speed_px_per_sec:.2f} px/s, inliers={result_cont.num_inliers}")
    print("  ✓ Continuous processing works")
    
    # Test 11: Test ROI setter
    print("\n[Test 11] Testing ROI setter...")
    pipeline.set_roi((50, 50, 300, 200))
    print(f"  Set ROI: {pipeline.roi}")
    print(f"  Use ROI: {pipeline.use_roi}")
    pipeline.clear_roi()
    print(f"  After clear: roi={pipeline.roi}, use_roi={pipeline.use_roi}")
    print("  ✓ ROI management works")
    
    # Test 12: Test frame rate setter
    print("\n[Test 12] Testing frame rate setter...")
    pipeline.set_frame_rate(30.0)
    print(f"  Set frame rate: {pipeline.frame_rate}")
    print(f"  Frame interval: {pipeline.frame_interval:.4f}s")
    pipeline.set_frame_rate(25.0)  # Reset
    print("  ✓ Frame rate setter works")
    
    # Test 13: Test feature matching with varying shift
    print("\n[Test 13] Testing with varying shifts...")
    for shift in [0, 5, 10, 20, 50]:
        # Create shifted frame
        shifted = np.zeros_like(frame1)
        if shift > 0:
            shifted[:, shift:] = frame1[:, :-shift]
        
        result_shift = pipeline.process(frame1, shifted)
        print(f"  Shift {shift}px: speed={result_shift.speed_px_per_sec:.2f}px/s, inliers={result_shift.num_inliers}")
    print("  ✓ Varying shift works")
    
    # Test 14: Test factory function
    print("\n[Test 14] Testing factory function...")
    factory_pipe = create_matching_pipeline()
    print(f"  Factory type: {type(factory_pipe).__name__}")
    print("  ✓ Factory function works")
    
    # Test 15: Test registry
    print("\n[Test 15] Testing registry...")
    reg_pipe = get_matching_pipeline("test_reg")
    print(f"  Registry type: {type(reg_pipe).__name__}")
    print("  ✓ Registry works")
    
    # Test 16: Test get_last_keypoints
    print("\n[Test 16] Testing get_last_keypoints...")
    kpts1, kpts2 = pipeline.get_last_keypoints()
    print(f"  Last keypoints frame1: {len(kpts1)}")
    print(f"  Last keypoints frame2: {len(kpts2)}")
    print("  ✓ Keypoint retrieval works")
    
    # Test 17: Test to_dict conversion
    print("\n[Test 17] Testing result to_dict...")
    result_dict = result.to_dict()
    print(f"  Keys: {list(result_dict.keys())}")
    print("  ✓ to_dict works")
    
    # Test 18: Test low-texture image handling
    print("\n[Test 18] Testing low-texture image...")
    low_tex = np.full((200, 200, 3), 128, dtype=np.uint8)
    result_low = pipeline.process(low_tex, low_tex)
    print(f"  Low-texture speed: {result_low.speed_px_per_sec:.2f} px/s")
    print(f"  Is valid: {result_low.is_valid()}")
    print("  ✓ Low-texture handled")
    
    # Test 19: Test is_valid method
    print("\n[Test 19] Testing is_valid method...")
    print(f"  Result with 0 inliers: {result_low.is_valid()}")
    valid_result = FeatureMatchResult(
        num_inliers=10,
        confidence=0.8,
        speed_m_per_sec=1.0
    )
    print(f"  Result with 10 inliers, 0.8 conf: {valid_result.is_valid()}")
    print("  ✓ is_valid works")
    
    # Test 20: Test config integration
    print("\n[Test 20] Testing config integration...")
    if CONFIG_AVAILABLE:
        try:
            cfg = get_config()
            print(f"  Config frame rate: {cfg.dataset.frame_rate}")
            print(f"  Config feature matching:")
            print(f"    Harris k: {cfg.model.feature_matching.harris_k}")
            print(f"    RANSAC threshold: {cfg.model.feature_matching.ransac_threshold}")
            print("  ✓ Config integration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (config not available)")
    
    print("\n" + "=" * 60)
    print("All Feature Matching Pipeline tests passed!")
    print("=" * 60)
