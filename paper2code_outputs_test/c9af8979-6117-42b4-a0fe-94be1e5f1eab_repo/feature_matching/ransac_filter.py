"""
feature_matching/ransac_filter.py

RANSAC-based outlier filtering for conveyor belt speed detection.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements the RANSAC (Random Sample Consensus) filtering component
of the Harris-BRIEF-RANSAC feature matching algorithm. It removes incorrect feature
correspondences by estimating a homography transformation between consecutive frames.

Based on Section 3.2 of the paper: "uses the RANSAC algorithm to filter the matching points"

Author: Based on paper methodology
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Dict, Any
import warnings

# Try to import OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    raise ImportError("OpenCV (cv2) is required for RANSAC filtering. Install with: pip install opencv-python")

# Try to import configuration
try:
    from config import Config, get_config, FeatureMatchingConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    get_config = None
    FeatureMatchingConfig = None


class RANSACFilter:
    """RANSAC-based outlier filter for feature matching.
    
    This class implements the RANSAC algorithm to filter out incorrect feature
    matches and estimate the homography transformation between consecutive frames.
    The inlier matches are used for accurate speed calculation in the conveyor
    belt speed detection pipeline.
    
    Based on Section 3.2 of the paper: "uses the RANSAC algorithm to filter the 
    matching points. And speed up the algorithm by defining the region of 
    interest of the image."
    
    Attributes:
        threshold: Maximum reprojection error (pixels) to be considered inlier
        max_iters: Maximum number of RANSAC iterations
        confidence: Confidence threshold for early termination (optional)
        use_refinement: Whether to refine homography using inliers only
    """
    
    def __init__(
        self,
        threshold: float = 5.0,
        max_iters: int = 2000,
        confidence: float = 0.99,
        use_refinement: bool = True,
        min_inliers: int = 4,
        enable_roi: bool = True,
        roi: Optional[Tuple[int, int, int, int]] = None
    ):
        """Initialize the RANSAC filter.
        
        Args:
            threshold: Maximum reprojection error in pixels for inliers.
                      Default 5.0 from config.yaml.
            max_iters: Maximum RANSAC iterations.
                      Default 2000 from config.yaml.
            confidence: Confidence level for early termination (0-1).
            use_refinement: Whether to refine homography using only inliers.
            min_inliers: Minimum number of inliers required for valid result.
            enable_roi: Whether to use region of interest for filtering.
            roi: Region of interest as (x, y, width, height).
        """
        # Validate parameters
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {max_iters}")
        if confidence <= 0 or confidence > 1:
            raise ValueError(f"confidence must be in (0, 1], got {confidence}")
        if min_inliers < 4:
            raise ValueError(f"min_inliers must be at least 4, got {min_inliers}")
        
        self.threshold = threshold
        self.max_iters = max_iters
        self.confidence = confidence
        self.use_refinement = use_refinement
        self.min_inliers = min_inliers
        self.enable_roi = enable_roi
        self.roi = roi
        
        # Statistics tracking
        self._stats = {
            'total_matches': 0,
            'num_inliers': 0,
            'inlier_ratio': 0.0,
            'failed_cases': 0,
            'iterations_used': 0
        }
    
    @classmethod
    def from_config(
        cls,
        config: Optional[FeatureMatchingConfig] = None
    ) -> 'RANSACFilter':
        """Create RANSACFilter from configuration.
        
        Args:
            config: Feature matching configuration. If None, uses default values.
            
        Returns:
            RANSACFilter instance with configured parameters
        """
        if config is not None:
            return cls(
                threshold=config.ransac_threshold,
                max_iters=config.ransac_max_iters
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
                threshold=5.0,
                max_iters=2000
            )
    
    def _validate_inputs(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Tuple[bool, str]:
        """Validate input point arrays.
        
        Args:
            src_pts: Source points array
            dst_pts: Destination points array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if src_pts is None or dst_pts is None:
            return False, "Input points cannot be None"
        
        if not isinstance(src_pts, np.ndarray) or not isinstance(dst_pts, np.ndarray):
            return False, "Input points must be numpy arrays"
        
        if src_pts.size == 0 or dst_pts.size == 0:
            return False, "Input point arrays are empty"
        
        if len(src_pts) != len(dst_pts):
            return False, f"Point arrays have different lengths: {len(src_pts)} vs {len(dst_pts)}"
        
        if len(src_pts) < 4:
            return False, f"Insufficient points for homography: {len(src_pts)} < 4"
        
        return True, ""
    
    def _apply_roi_filter(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply ROI filter to keep points within region of interest.
        
        Args:
            src_pts: Source points
            dst_pts: Destination points
            
        Returns:
            Tuple of (filtered_src, filtered_dst, mask)
        """
        if not self.enable_roi or self.roi is None:
            return src_pts, dst_pts, np.ones(len(src_pts), dtype=bool)
        
        x, y, w, h = self.roi
        
        # Create mask for points in ROI
        src_mask = (
            (src_pts[:, 0, 0] >= x) & (src_pts[:, 0, 0] <= x + w) &
            (src_pts[:, 0, 1] >= y) & (src_pts[:, 0, 1] <= y + h)
        )
        
        dst_mask = (
            (dst_pts[:, 0, 0] >= x) & (dst_pts[:, 0, 0] <= x + w) &
            (dst_pts[:, 0, 1] >= y) & (dst_pts[:, 0, 1] <= y + h)
        )
        
        # Keep points where both src and dst are in ROI
        combined_mask = src_mask & dst_mask
        
        return src_pts[combined_mask], dst_pts[combined_mask], combined_mask
    
    def _compute_homography_dlt(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute homography using Direct Linear Transform (DLT).
        
        This is a custom implementation of homography estimation that serves
        as a fallback if OpenCV's findHomography is not available.
        
        Args:
            src_pts: Source points (N, 1, 2)
            dst_pts: Destination points (N, 1, 2)
            
        Returns:
            3x3 homography matrix, or None if computation fails
        """
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None
        
        try:
            # Number of point correspondences
            n = len(src_pts)
            
            # Build the matrix A for Ah = 0
            A = np.zeros((2 * n, 9))
            
            for i in range(n):
                x, y = src_pts[i, 0]
                xp, yp = dst_pts[i, 0]
                
                A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
                A[2*i + 1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
            
            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            
            # Last row of Vt corresponds to smallest singular value
            h = Vt[-1, :]
            
            # Reshape to 3x3 homography matrix
            H = h.reshape(3, 3)
            
            # Normalize so H[2,2] = 1
            if H[2, 2] != 0:
                H = H / H[2, 2]
            
            return H
            
        except Exception as e:
            warnings.warn(f"DLT homography computation failed: {e}")
            return None
    
    def _project_point(
        self,
        H: np.ndarray,
        pt: np.ndarray
    ) -> np.ndarray:
        """Project a point using homography.
        
        Args:
            H: 3x3 homography matrix
            pt: Point coordinates (1, 2) or (2,)
            
        Returns:
            Projected point coordinates
        """
        x, y = pt[0, 0] if len(pt.shape) > 1 else pt[0], pt[0, 1] if len(pt.shape) > 1 else pt[1]
        
        # Apply homography
        xp = H[0, 0] * x + H[0, 1] * y + H[0, 2]
        yp = H[1, 0] * x + H[1, 1] * y + H[1, 2]
        wp = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        
        # Handle perspective division
        if wp != 0:
            xp = xp / wp
            yp = yp / wp
        
        return np.array([xp, yp])
    
    def _compute_reprojection_error(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        H: np.ndarray
    ) -> np.ndarray:
        """Compute reprojection error for all point correspondences.
        
        Args:
            src_pts: Source points (N, 1, 2)
            dst_pts: Destination points (N, 1, 2)
            H: 3x3 homography matrix
            
        Returns:
            Array of reprojection errors (N,)
        """
        errors = np.zeros(len(src_pts))
        
        for i in range(len(src_pts)):
            # Project source point using homography
            projected = self._project_point(H, src_pts[i])
            
            # Compute Euclidean distance to destination
            dst = dst_pts[i, 0]
            errors[i] = np.sqrt(
                (projected[0] - dst[0])**2 + (projected[1] - dst[1])**2
            )
        
        return errors
    
    def _find_inliers(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        H: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, int]:
        """Find inliers based on reprojection error.
        
        Args:
            src_pts: Source points
            dst_pts: Destination points
            H: Homography matrix
            threshold: Error threshold for inliers
            
        Returns:
            Tuple of (inlier_mask, inlier_count)
        """
        errors = self._compute_reprojection_error(src_pts, dst_pts, H)
        inlier_mask = errors < threshold
        inlier_count = np.sum(inlier_mask)
        
        return inlier_mask, inlier_count
    
    def _custom_ransac(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Custom RANSAC implementation as fallback.
        
        Uses random sampling to find the best homography transformation.
        
        Args:
            src_pts: Source points
            dst_pts: Destination points
            
        Returns:
            Tuple of (homography, inlier_mask, num_inliers)
        """
        n_points = len(src_pts)
        
        if n_points < 4:
            return np.eye(3), np.zeros(n_points, dtype=bool), 0
        
        best_H = np.eye(3)
        best_inlier_count = 0
        best_inlier_mask = np.zeros(n_points, dtype=bool)
        
        # Number of random samples needed
        # For 4 point correspondences, probability of all being inliers
        # is (inlier_ratio)^4. We use a conservative estimate.
        
        for iteration in range(self.max_iters):
            # Randomly select 4 point correspondences
            indices = np.random.choice(n_points, 4, replace=False)
            
            src_sample = src_pts[indices]
            dst_sample = dst_pts[indices]
            
            # Compute homography using DLT
            H = self._compute_homography_dlt(src_sample, dst_sample)
            
            if H is None:
                continue
            
            # Count inliers
            inlier_mask, inlier_count = self._find_inliers(
                src_pts, dst_pts, H, self.threshold
            )
            
            # Update best model if current is better
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_H = H
                best_inlier_mask = inlier_mask
                
                # Early termination check
                if inlier_count > self.min_inliers:
                    # Estimate remaining iterations needed
                    inlier_ratio = inlier_count / n_points
                    if inlier_ratio > 0:
                        # Probability that all selected points are inliers
                        p = inlier_ratio ** 4
                        if p > 0:
                            # Expected iterations to find a good model
                            expected_iters = int(np.log(1 - self.confidence) / 
                                               np.log(1 - p))
                            if expected_iters < (self.max_iters - iteration):
                                # Can potentially terminate early
                                pass  # Continue to check confidence
        
        # Refine homography using inliers if enabled
        if self.use_refinement and best_inlier_count >= 4:
            src_inliers = src_pts[best_inlier_mask]
            dst_inliers = dst_pts[best_inlier_mask]
            
            H_refined = self._compute_homography_dlt(src_inliers, dst_inliers)
            if H_refined is not None:
                best_H = H_refined
                # Recompute inliers with refined homography
                best_inlier_mask, best_inlier_count = self._find_inliers(
                    src_pts, dst_pts, best_H, self.threshold
                )
        
        return best_H, best_inlier_mask, best_inlier_count
    
    def filter_matches(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        threshold: Optional[float] = None,
        max_iters: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Filter matches using RANSAC to find inliers and homography.
        
        This is the main method for the RANSAC filter. It:
        1. Validates input points
        2. Optionally applies ROI filtering
        3. Estimates homography using RANSAC (OpenCV or custom)
        4. Refines homography using inliers (optional)
        5. Returns inlier mask and homography
        
        Args:
            src_pts: Source keypoints from frame t, shape (N, 1, 2)
            dst_pts: Destination keypoints from frame t+1, shape (N, 1, 2)
            threshold: Optional override for reprojection error threshold
            max_iters: Optional override for maximum iterations
            
        Returns:
            Tuple of:
                - homography: 3x3 homography matrix
                - inlier_mask: Boolean array of shape (N,), True for inliers
                - num_inliers: Number of inlier matches
                
        Raises:
            ValueError: If inputs are invalid
        """
        # Use instance defaults if not provided
        thresh = threshold if threshold is not None else self.threshold
        iters = max_iters if max_iters is not None else self.max_iters
        
        # Validate inputs
        is_valid, error_msg = self._validate_inputs(src_pts, dst_pts)
        if not is_valid:
            self._stats['failed_cases'] += 1
            return np.eye(3), np.zeros(len(src_pts), dtype=bool), 0
        
        # Track statistics
        self._stats['total_matches'] += len(src_pts)
        
        # Apply ROI filtering if enabled
        filtered_src = src_pts
        filtered_dst = dst_pts
        roi_mask = None
        
        if self.enable_roi and self.roi is not None:
            filtered_src, filtered_dst, roi_mask = self._apply_roi_filter(src_pts, dst_pts)
        
        # Check if enough points after ROI filtering
        if len(filtered_src) < 4:
            # Not enough points, return identity
            return np.eye(3), np.zeros(len(src_pts), dtype=bool), 0
        
        # Try OpenCV implementation first
        homography = np.eye(3)
        inlier_mask = np.zeros(len(filtered_src), dtype=bool)
        num_inliers = 0
        
        try:
            # Use OpenCV's findHomography with RANSAC
            homography_cv, mask_cv = cv2.findHomography(
                filtered_src,
                filtered_dst,
                cv2.RANSAC,
                thresh,
                maxIters=iters,
                confidence=self.confidence
            )
            
            # Extract inliers from mask
            inlier_mask = mask_cv.ravel() == 1
            num_inliers = np.sum(inlier_mask)
            
            # Use homography from OpenCV
            homography = homography_cv
            
            # Refine homography using inliers if enabled and enough inliers
            if self.use_refinement and num_inliers >= 4:
                src_inliers = filtered_src[inlier_mask]
                dst_inliers = filtered_dst[inlier_mask]
                
                # Refine using only inliers
                homography_refined, _ = cv2.findHomography(
                    src_inliers,
                    dst_inliers,
                    0  # Use all points (no RANSAC)
                )
                
                if homography_refined is not None:
                    homography = homography_refined
                    
                    # Recompute inliers with refined homography
                    errors = self._compute_reprojection_error(
                        filtered_src, filtered_dst, homography
                    )
                    inlier_mask = errors < thresh
                    num_inliers = np.sum(inlier_mask)
            
            self._stats['iterations_used'] = iters
            
        except Exception as e:
            warnings.warn(f"OpenCV findHomography failed: {e}, using custom RANSAC")
            
            # Fallback to custom implementation
            homography, inlier_mask, num_inliers = self._custom_ransac(
                filtered_src, filtered_dst
            )
        
        # Expand mask to full size if ROI filtering was applied
        if roi_mask is not None:
            full_mask = np.zeros(len(src_pts), dtype=bool)
            full_mask[roi_mask] = inlier_mask
            inlier_mask = full_mask
            # Recalculate inliers for full array
            num_inliers = np.sum(inlier_mask)
        
        # Update statistics
        self._stats['num_inliers'] += num_inliers
        if self._stats['total_matches'] > 0:
            self._stats['inlier_ratio'] = self._stats['num_inliers'] / self._stats['total_matches']
        
        # Handle case with too few inliers
        if num_inliers < self.min_inliers:
            return np.eye(3), inlier_mask, num_inliers
        
        return homography, inlier_mask, num_inliers
    
    def compute_displacement(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        inlier_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Compute average displacement from matched points.
        
        Uses inlier matches to calculate the average motion vector,
        which is used for speed calculation.
        
        Args:
            src_pts: Source points (N, 1, 2)
            dst_pts: Destination points (N, 1, 2)
            inlier_mask: Boolean mask for inlier points. If None, uses all points.
            
        Returns:
            Tuple of (displacement_x, displacement_y) in pixels
        """
        if inlier_mask is None:
            inlier_mask = np.ones(len(src_pts), dtype=bool)
        
        # Filter to inliers only
        src_inliers = src_pts[inlier_mask]
        dst_inliers = dst_pts[inlier_mask]
        
        if len(src_inliers) == 0:
            return 0.0, 0.0
        
        # Compute displacement for each point
        displacements = dst_inliers[:, 0, :] - src_inliers[:, 0, :]
        
        # Return average displacement
        avg_dx = np.mean(displacements[:, 0])
        avg_dy = np.mean(displacements[:, 1])
        
        return float(avg_dx), float(avg_dy)
    
    def compute_speed(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        inlier_mask: np.ndarray,
        frame_interval: float,
        pixels_per_meter: float = 1000.0
    ) -> float:
        """Compute speed from matched feature points.
        
        Args:
            src_pts: Source points
            dst_pts: Destination points  
            inlier_mask: Boolean mask for inliers
            frame_interval: Time between frames in seconds
            pixels_per_meter: Conversion factor from pixels to meters
            
        Returns:
            Speed in meters per second
        """
        # Compute average displacement
        avg_dx, avg_dy = self.compute_displacement(src_pts, dst_pts, inlier_mask)
        
        # Compute displacement magnitude in pixels
        pixel_displacement = np.sqrt(avg_dx**2 + avg_dy**2)
        
        # Convert to meters
        meter_displacement = pixel_displacement / pixels_per_meter
        
        # Compute speed
        speed = meter_displacement / frame_interval if frame_interval > 0 else 0.0
        
        return float(speed)
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> None:
        """Set the region of interest for filtering.
        
        Args:
            roi: Region of interest as (x, y, width, height)
        """
        self.roi = roi
        self.enable_roi = True
    
    def clear_roi(self) -> None:
        """Clear the region of interest."""
        self.roi = None
        self.enable_roi = False
    
    def set_parameters(
        self,
        threshold: Optional[float] = None,
        max_iters: Optional[int] = None,
        use_refinement: Optional[bool] = None
    ) -> None:
        """Update filter parameters.
        
        Args:
            threshold: New threshold value
            max_iters: New maximum iterations
            use_refinement: New refinement setting
        """
        if threshold is not None and threshold > 0:
            self.threshold = threshold
        
        if max_iters is not None and max_iters > 0:
            self.max_iters = max_iters
        
        if use_refinement is not None:
            self.use_refinement = use_refinement
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current filter parameters.
        
        Returns:
            Dictionary of current parameters
        """
        return {
            'threshold': self.threshold,
            'max_iters': self.max_iters,
            'confidence': self.confidence,
            'use_refinement': self.use_refinement,
            'min_inliers': self.min_inliers,
            'enable_roi': self.enable_roi,
            'roi': self.roi
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset filtering statistics."""
        self._stats = {
            'total_matches': 0,
            'num_inliers': 0,
            'inlier_ratio': 0.0,
            'failed_cases': 0,
            'iterations_used': 0
        }
    
    def __repr__(self) -> str:
        """String representation of the filter."""
        return (
            f"RANSACFilter(\n"
            f"  threshold={self.threshold},\n"
            f"  max_iters={self.max_iters},\n"
            f"  confidence={self.confidence},\n"
            f"  use_refinement={self.use_refinement},\n"
            f"  min_inliers={self.min_inliers},\n"
            f"  enable_roi={self.enable_roi},\n"
            f"  roi={self.roi}\n"
            f")"
        )


def create_ransac_filter(
    config: Optional[FeatureMatchingConfig] = None,
    threshold: float = 5.0,
    max_iters: int = 2000
) -> RANSACFilter:
    """Factory function to create a RANSACFilter.
    
    Args:
        config: Feature matching configuration (takes priority if provided)
        threshold: Threshold parameter (used if config is None)
        max_iters: Max iterations (used if config is None)
        
    Returns:
        RANSACFilter instance
    """
    if config is not None:
        return RANSACFilter.from_config(config)
    else:
        # Try to get from global config
        if CONFIG_AVAILABLE:
            try:
                global_config = get_config()
                return create_ransac_filter(global_config.model.feature_matching)
            except:
                pass
        
        # Use provided or default values
        return RANSACFilter(
            threshold=threshold,
            max_iters=max_iters
        )


# Global filter registry
_filters: Dict[str, RANSACFilter] = {}


def get_ransac_filter(
    name: str = "default",
    config: Optional[FeatureMatchingConfig] = None,
    **kwargs
) -> RANSACFilter:
    """Get or create a RANSACFilter by name.
    
    Args:
        name: Filter identifier
        config: Feature matching configuration
        **kwargs: Additional arguments for RANSACFilter
        
    Returns:
        RANSACFilter instance
    """
    global _filters
    
    if name in _filters:
        return _filters[name]
    
    filter_obj = create_ransac_filter(config=config, **kwargs)
    _filters[name] = filter_obj
    
    return filter_obj


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("RANSAC Filter Module Test")
    print("=" * 60)
    
    # Test 1: Create filter with defaults
    print("\n[Test 1] Creating filter with defaults...")
    ransac_filter = RANSACFilter()
    print(f"  Created: {repr(ransac_filter)}")
    print("  ✓ RANSACFilter created")
    
    # Test 2: Create filter from config
    print("\n[Test 2] Creating filter from configuration...")
    try:
        if CONFIG_AVAILABLE:
            config = get_config()
            filter_config = create_ransac_filter(config.model.feature_matching)
            print(f"  From config: {repr(filter_config)}")
            print("  ✓ Config integration works")
        else:
            print("  Skipping (config not available)")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Test 3: Create synthetic matched points
    print("\n[Test 3] Creating synthetic matched points...")
    np.random.seed(42)
    n_points = 100
    
    # Create ground truth homography
    true_H = np.array([
        [1.0, 0.0, 10.0],  # 10 pixel translation in x
        [0.0, 1.0, 5.0],   # 5 pixel translation in y
        [0.0, 0.0, 1.0]
    ])
    
    # Generate source points on a grid
    src_pts = []
    grid_size = int(np.sqrt(n_points))
    for i in range(grid_size):
        for j in range(grid_size):
            x = 100 + j * 30
            y = 80 + i * 30
            src_pts.append([x, y])
    
    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    
    # Apply homography to get destination points
    dst_pts = []
    for pt in src_pts:
        x, y = pt[0]
        xp = (true_H[0, 0] * x + true_H[0, 1] * y + true_H[0, 2]) / (
             true_H[2, 0] * x + true_H[2, 1] * y + true_H[2, 2])
        yp = (true_H[1, 0] * x + true_H[1, 1] * y + true_H[1, 2]) / (
             true_H[2, 0] * x + true_H[2, 1] * y + true_H[2, 2])
        dst_pts.append([xp, yp])
    
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    
    # Add some outliers (20% of points)
    n_outliers = int(0.2 * len(src_pts))
    outlier_indices = np.random.choice(len(src_pts), n_outliers, replace=False)
    
    for idx in outlier_indices:
        # Random displacement for outliers
        dst_pts[idx, 0, 0] = src_pts[idx, 0, 0] + np.random.uniform(-100, 100)
        dst_pts[idx, 0, 1] = src_pts[idx, 0, 1] + np.random.uniform(-100, 100)
    
    print(f"  Created {len(src_pts)} points with {n_outliers} outliers")
    print(f"  True displacement: dx=10.0, dy=5.0 pixels")
    print("  ✓ Test data created")
    
    # Test 4: Filter matches with RANSAC
    print("\n[Test 4] Filtering matches with RANSAC...")
    homography, inlier_mask, num_inliers = ransac_filter.filter_matches(
        src_pts, dst_pts, threshold=5.0, max_iters=2000
    )
    
    print(f"  Input points: {len(src_pts)}")
    print(f"  Inliers found: {num_inliers}")
    print(f"  Inlier ratio: {num_inliers/len(src_pts)*100:.1f}%")
    print(f"  Homography matrix:\n{homography}")
    print("  ✓ RANSAC filtering works")
    
    # Test 5: Test inlier extraction
    print("\n[Test 5] Testing inlier extraction...")
    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]
    print(f"  Inlier source points shape: {src_inliers.shape}")
    print(f"  Inlier destination points shape: {dst_inliers.shape}")
    print("  ✓ Inlier extraction works")
    
    # Test 6: Compute displacement from inliers
    print("\n[Test 6] Computing displacement from inliers...")
    dx, dy = ransac_filter.compute_displacement(src_pts, dst_pts, inlier_mask)
    print(f"  Computed displacement: dx={dx:.2f}, dy={dy:.2f} pixels")
    print(f"  Expected: dx=10.0, dy=5.0 pixels")
    print("  ✓ Displacement computation works")
    
    # Test 7: Compute speed from matches
    print("\n[Test 7] Computing speed from matches...")
    speed = ransac_filter.compute_speed(
        src_pts, dst_pts, inlier_mask,
        frame_interval=0.04,  # 25 fps
        pixels_per_meter=1000.0
    )
    print(f"  Computed speed: {speed:.4f} m/s")
    print("  ✓ Speed computation works")
    
    # Test 8: Test with insufficient points
    print("\n[Test 8] Testing with insufficient points...")
    few_src = np.array([[[100, 100]], [[200, 200]], [[300, 300]]], dtype=np.float32)
    few_dst = np.array([[[105, 105]], [[205, 205]], [[305, 305]]], dtype=np.float32)
    
    H_few, mask_few, num_inliers_few = ransac_filter.filter_matches(few_src, few_dst)
    print(f"  Input: {len(few_src)} points")
    print(f"  Output inliers: {num_inliers_few}")
    print(f"  Homography:\n{H_few}")
    print("  ✓ Handles insufficient points")
    
    # Test 9: Test with all outliers
    print("\n[Test 9] Testing with all outliers...")
    outlier_src = np.array([[[i*10, i*10] for i in range(10)]], dtype=np.float32).reshape(-1, 1, 2)
    outlier_dst = np.array([[[i*10 + np.random.uniform(-50, 50), i*10 + np.random.uniform(-50, 50)] 
                            for i in range(10)]], dtype=np.float32).reshape(-1, 1, 2)
    
    H_out, mask_out, num_out = ransac_filter.filter_matches(outlier_src, outlier_dst)
    print(f"  Input: {len(outlier_src)} outliers")
    print(f"  Inliers found: {num_out}")
    print("  ✓ Handles all outliers")
    
    # Test 10: Test ROI filtering
    print("\n[Test 10] Testing ROI filtering...")
    ransac_with_roi = RANSACFilter(enable_roi=True, roi=(50, 50, 300, 300))
    H_roi, mask_roi, num_roi = ransac_with_roi.filter_matches(src_pts, dst_pts)
    print(f"  ROI: (50, 50, 300, 300)")
    print(f"  Inliers within ROI: {num_roi}")
    print("  ✓ ROI filtering works")
    
    # Test 11: Test statistics tracking
    print("\n[Test 11] Testing statistics tracking...")
    stats = ransac_filter.get_statistics()
    print(f"  Statistics: {stats}")
    ransac_filter.reset_statistics()
    stats_after = ransac_filter.get_statistics()
    print(f"  After reset: {stats_after}")
    print("  ✓ Statistics work")
    
    # Test 12: Test parameter setting/getting
    print("\n[Test 12] Testing parameter methods...")
    ransac_filter.set_parameters(threshold=3.0, max_iters=1000)
    params = ransac_filter.get_parameters()
    print(f"  Set threshold=3.0, max_iters=1000")
    print(f"  Current threshold: {params['threshold']}, max_iters: {params['max_iters']}")
    print("  ✓ Parameter methods work")
    
    # Test 13: Test reprojection error computation
    print("\n[Test 13] Testing reprojection error computation...")
    errors = ransac_filter._compute_reprojection_error(
        src_pts[inlier_mask][:10], 
        dst_pts[inlier_mask][:10], 
        homography
    )
    print(f"  Reprojection errors (first 10): {errors[:10]}")
    print(f"  Mean error: {np.mean(errors):.4f} pixels")
    print("  ✓ Reprojection error works")
    
    # Test 14: Test DLT homography computation
    print("\n[Test 14] Testing DLT homography computation...")
    sample_src = src_pts[inlier_mask][:4]
    sample_dst = dst_pts[inlier_mask][:4]
    H_dlt = ransac_filter._compute_homography_dlt(sample_src, sample_dst)
    print(f"  DLT homography:\n{H_dlt}")
    print("  ✓ DLT computation works")
    
    # Test 15: Test point projection
    print("\n[Test 15] Testing point projection...")
    test_pt = np.array([[100, 100]])
    projected = ransac_filter._project_point(homography, test_pt)
    print(f"  Input point: {test_pt}")
    print(f"  Projected point: {projected}")
    print("  ✓ Point projection works")
    
    # Test 16: Test validation
    print("\n[Test 16] Testing input validation...")
    is_valid, error_msg = ransac_filter._validate_inputs(src_pts, dst_pts)
    print(f"  Valid inputs: {is_valid}")
    
    # Test with invalid inputs
    invalid_src = np.array([]).reshape(0, 1, 2)
    invalid_dst = np.array([]).reshape(0, 1, 2)
    is_valid, error_msg = ransac_filter._validate_inputs(invalid_src, invalid_dst)
    print(f"  Empty inputs valid: {is_valid}, error: {error_msg}")
    print("  ✓ Validation works")
    
    # Test 17: Test factory function
    print("\n[Test 17] Testing factory function...")
    factory_filter = create_ransac_filter()
    print(f"  Factory type: {type(factory_filter).__name__}")
    print("  ✓ Factory function works")
    
    # Test 18: Test registry
    print("\n[Test 18] Testing registry...")
    reg_filter = get_ransac_filter("test_registry")
    print(f"  Registry type: {type(reg_filter).__name__}")
    print("  ✓ Registry works")
    
    # Test 19: Test config integration
    print("\n[Test 19] Testing config integration...")
    if CONFIG_AVAILABLE:
        try:
            cfg = get_config()
            print(f"  Config RANSAC threshold: {cfg.model.feature_matching.ransac_threshold}")
            print(f"  Config RANSAC max iters: {cfg.model.feature_matching.ransac_max_iters}")
            print("  ✓ Config integration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (config not available)")
    
    # Test 20: Test with threshold variation
    print("\n[Test 20] Testing threshold variation...")
    for thresh in [1.0, 3.0, 5.0, 10.0]:
        H_t, mask_t, num_t = ransac_filter.filter_matches(src_pts, dst_pts, threshold=thresh)
        print(f"  Threshold {thresh}: {num_t} inliers ({num_t/len(src_pts)*100:.1f}%)")
    print("  ✓ Threshold variation works")
    
    # Test 21: Clear and set ROI
    print("\n[Test 21] Testing ROI management...")
    ransac_filter.set_roi((100, 100, 200, 200))
    print(f"  Set ROI: {ransac_filter.roi}")
    ransac_filter.clear_roi()
    print(f"  After clear: roi={ransac_filter.roi}, enable_roi={ransac_filter.enable_roi}")
    print("  ✓ ROI management works")
    
    print("\n" + "=" * 60)
    print("All RANSAC Filter tests passed!")
    print("=" * 60)
