"""
utils/calibration.py

Camera calibration utilities for the conveyor belt speed detection system.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module handles:
1. Camera intrinsic/extrinsic parameter calibration
2. Pixel-to-world coordinate transformation
3. Pixel displacement to real-world speed conversion

Author: Based on paper methodology
"""

import os
import yaml
import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

# Try to import OpenCV, fall back to numpy-only implementation if unavailable
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Using simplified calibration.")


@dataclass
class CameraCalibration:
    """Camera calibration parameters for pixel-to-world conversion.
    
    This class encapsulates all camera parameters needed to convert pixel-level
    measurements (optical flow displacement, feature point displacement) into
    real-world physical units (meters).
    
    Based on the paper methodology:
    - The mapping between image coordinate system and world coordinate system
    - Physical size represented by each pixel in the image
    - Actual displacement and speed calculation
    
    Attributes:
        fx: Focal length in pixels (x-axis)
        fy: Focal length in pixels (y-axis)
        cx: Principal point x-coordinate (optical center)
        cy: Principal point y-coordinate (optical center)
        dist_coeffs: Camera distortion coefficients [k1, k2, p1, p2, k3]
        rotation_matrix: 3x3 rotation matrix for extrinsic parameters
        translation_vector: 3x1 translation vector for extrinsic parameters
        pixels_per_meter: Critical conversion factor - pixels per meter
        image_width: Calibrated image width
        image_height: Calibrated image height
        camera_distance: Distance from camera to conveyor belt (meters)
    """
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 960.0
    cy: float = 540.0
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pixels_per_meter: float = 1000.0
    image_width: int = 1920
    image_height: int = 1080
    camera_distance: float = 3.0
    
    def __post_init__(self):
        """Validate and process calibration parameters after initialization."""
        # Ensure dist_coeffs is a numpy array with correct shape
        if isinstance(self.dist_coeffs, list):
            self.dist_coeffs = np.array(self.dist_coeffs, dtype=np.float64)
        
        # Ensure rotation matrix is numpy array
        if not isinstance(self.rotation_matrix, np.ndarray):
            self.rotation_matrix = np.array(self.rotation_matrix, dtype=np.float64)
        
        if not isinstance(self.translation_vector, np.ndarray):
            self.translation_vector = np.array(self.translation_vector, dtype=np.float64)
        
        # Ensure proper shapes
        if self.dist_coeffs.shape != (5,):
            # Pad or truncate to 5 coefficients
            coeffs = np.zeros(5)
            min_len = min(5, len(self.dist_coeffs))
            coeffs[:min_len] = self.dist_coeffs[:min_len]
            self.dist_coeffs = coeffs
        
        if self.rotation_matrix.shape != (3, 3):
            self.rotation_matrix = np.eye(3)
        
        if self.translation_vector.shape != (3,):
            self.translation_vector = np.zeros(3)
    
    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Get the camera intrinsic matrix.
        
        Returns:
            3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get image resolution as (width, height)."""
        return (self.image_width, self.image_height)
    
    def validate(self) -> bool:
        """Validate calibration parameters.
        
        Returns:
            True if calibration parameters are valid
            
        Raises:
            ValueError: If validation fails
        """
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be positive: fx={self.fx}, fy={self.fy}")
        
        if self.pixels_per_meter <= 0:
            raise ValueError(f"pixels_per_meter must be positive: {self.pixels_per_meter}")
        
        if self.camera_distance <= 0:
            raise ValueError(f"camera_distance must be positive: {self.camera_distance}")
        
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(f"Image dimensions must be positive: {self.image_width}x{self.image_height}")
        
        # Check if cx, cy are within image bounds
        if not (0 <= self.cx <= self.image_width and 0 <= self.cy <= self.image_height):
            raise ValueError(f"Principal point outside image bounds: cx={self.cx}, cy={self.cy}")
        
        return True
    
    def pixel_to_world(self, u: float, v: float, Z: float = 0.0) -> Tuple[float, float, float]:
        """Convert pixel coordinates to world coordinates.
        
        Based on the pinhole camera model:
        - Image coordinates (u, v) are converted to normalized camera coordinates
        - Then transformed to world coordinates using extrinsic parameters
        
        Args:
            u: Pixel x-coordinate (column)
            v: Pixel y-coordinate (row)
            Z: World Z coordinate (default 0 for belt surface)
            
        Returns:
            Tuple of (X, Y, Z) world coordinates in meters
        """
        # Convert to normalized camera coordinates
        x_normalized = (u - self.cx) / self.fx
        y_normalized = (v - self.cy) / self.fy
        
        # Apply extrinsic transformation (simplified for Z=0 plane)
        # For the conveyor belt scenario, we assume the belt is parallel to the image plane
        # at distance camera_distance from the camera
        
        # Simple case: no rotation, camera looking perpendicular to belt
        X = x_normalized * self.camera_distance
        Y = y_normalized * self.camera_distance
        Z_actual = Z
        
        return (X, Y, Z_actual)
    
    def world_to_pixel(self, X: float, Y: float, Z: float = 0.0) -> Tuple[float, float]:
        """Convert world coordinates to pixel coordinates.
        
        Args:
            X: World X coordinate in meters
            Y: World Y coordinate in meters
            Z: World Z coordinate (default 0 for belt surface)
            
        Returns:
            Tuple of (u, v) pixel coordinates
        """
        # For simple case with no rotation
        u = self.fx * (X / self.camera_distance) + self.cx
        v = self.fy * (Y / self.camera_distance) + self.cy
        
        return (u, v)
    
    def convert_pixel_displacement_to_meters(self, pixel_displacement: float) -> float:
        """Convert pixel displacement to meters.
        
        This is the critical conversion function for speed measurement.
        Based on the paper: "the physical size represented by each pixel 
        in the image can be obtained"
        
        Args:
            pixel_displacement: Displacement in pixels
            
        Returns:
            Displacement in meters
        """
        return pixel_displacement / self.pixels_per_meter
    
    def convert_meters_to_pixels(self, meter_displacement: float) -> float:
        """Convert meter displacement to pixels.
        
        Args:
            meter_displacement: Displacement in meters
            
        Returns:
            Displacement in pixels
        """
        return meter_displacement * self.pixels_per_meter
    
    def calculate_speed_from_displacement(
        self, 
        pixel_displacement: float, 
        frame_interval: float
    ) -> float:
        """Calculate speed in m/s from pixel displacement between frames.
        
        Based on the paper methodology:
        - Pixel displacement is converted to real-world distance
        - Speed = distance / time between frames
        
        Args:
            pixel_displacement: Average pixel displacement between consecutive frames
            frame_interval: Time between frames in seconds (1/frame_rate)
            
        Returns:
            Speed in meters per second
        """
        # Convert pixel displacement to meters
        meter_displacement = self.convert_pixel_displacement_to_meters(pixel_displacement)
        
        # Calculate speed: distance / time
        speed = meter_displacement / frame_interval
        
        return speed
    
    def calculate_speed_from_flow(
        self, 
        flow_magnitude: float, 
        frame_interval: float
    ) -> float:
        """Calculate speed from optical flow magnitude.
        
        This is a convenience method for the optical flow branch.
        
        Args:
            flow_magnitude: Average optical flow magnitude in pixels
            frame_interval: Time between frames in seconds
            
        Returns:
            Speed in meters per second
        """
        return self.calculate_speed_from_displacement(flow_magnitude, frame_interval)
    
    def calculate_speed_from_matches(
        self, 
        match_displacement: float, 
        frame_interval: float
    ) -> float:
        """Calculate speed from feature match displacement.
        
        This is a convenience method for the feature matching branch.
        
        Args:
            match_displacement: Average feature point displacement in pixels
            frame_interval: Time between frames in seconds
            
        Returns:
            Speed in meters per second
        """
        return self.calculate_speed_from_displacement(match_displacement, frame_interval)
    
    def to_dict(self) -> dict:
        """Convert calibration to dictionary for serialization.
        
        Returns:
            Dictionary representation of calibration
        """
        return {
            'fx': float(self.fx),
            'fy': float(self.fy),
            'cx': float(self.cx),
            'cy': float(self.cy),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'rotation_matrix': self.rotation_matrix.tolist(),
            'translation_vector': self.translation_vector.tolist(),
            'pixels_per_meter': float(self.pixels_per_meter),
            'image_width': int(self.image_width),
            'image_height': int(self.image_height),
            'camera_distance': float(self.camera_distance)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CameraCalibration':
        """Create calibration from dictionary.
        
        Args:
            data: Dictionary containing calibration parameters
            
        Returns:
            CameraCalibration instance
        """
        return cls(
            fx=data.get('fx', 1000.0),
            fy=data.get('fy', 1000.0),
            cx=data.get('cx', 960.0),
            cy=data.get('cy', 540.0),
            dist_coeffs=np.array(data.get('dist_coeffs', [0, 0, 0, 0, 0])),
            rotation_matrix=np.array(data.get('rotation_matrix', np.eye(3).tolist())),
            translation_vector=np.array(data.get('translation_vector', [0, 0, 0])),
            pixels_per_meter=data.get('pixels_per_meter', 1000.0),
            image_width=data.get('image_width', 1920),
            image_height=data.get('image_height', 1080),
            camera_distance=data.get('camera_distance', 3.0)
        )
    
    def save(self, filepath: str) -> None:
        """Save calibration to YAML file.
        
        Args:
            filepath: Path to save calibration file
        """
        filepath = str(Path(filepath).resolve())
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'CameraCalibration':
        """Load calibration from YAML file.
        
        Args:
            filepath: Path to calibration file
            
        Returns:
            CameraCalibration instance
            
        Raises:
            FileNotFoundError: If calibration file doesn't exist
        """
        filepath = str(Path(filepath).resolve())
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)


def calibrate_camera(
    chessboard_images: List[np.ndarray],
    chessboard_size: Tuple[int, int] = (9, 6),
    square_size: float = 0.025
) -> CameraCalibration:
    """Perform camera calibration using chessboard images.
    
    This function uses OpenCV's camera calibration to compute intrinsic
    and extrinsic camera parameters from multiple views of a chessboard pattern.
    
    Based on the paper methodology for camera calibration.
    
    Args:
        chessboard_images: List of chessboard calibration images
        chessboard_size: Number of inner corners (columns, rows)
        square_size: Size of each square in meters (default 25mm)
        
    Returns:
        CameraCalibration instance with computed parameters
        
    Note:
        Requires OpenCV to be installed
    """
    if not HAS_CV2:
        raise ImportError("OpenCV is required for camera calibration")
    
    # Prepare object points
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []
    imgpoints = []
    
    # Find corners in each image
    for img in chessboard_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(
                gray, 
                corners, 
                (11, 11), 
                (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners_refined)
    
    if len(objpoints) == 0:
        raise ValueError("No valid chessboard corners found in images")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, 
        imgpoints, 
        gray.shape[::-1], 
        None, 
        None
    )
    
    # Get image dimensions
    h, w = gray.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # Extract intrinsic parameters
    fx = newcameramtx[0, 0]
    fy = newcameramtx[1, 1]
    cx = newcameramtx[0, 2]
    cy = newcameramtx[1, 2]
    
    # Calculate pixels_per_meter based on focal length and assumed distance
    # Using the formula: pixels_per_meter = focal_length_pixels / distance_meters
    # We use an estimate of 3m based on the paper
    camera_distance = 3.0
    pixels_per_meter_estimated = fx / camera_distance
    
    return CameraCalibration(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dist_coeffs=dist.flatten(),
        pixels_per_meter=pixels_per_meter_estimated,
        image_width=w,
        image_height=h,
        camera_distance=camera_distance
    )


def create_default_calibration(
    resolution: Tuple[int, int] = (1920, 1080),
    camera_distance: float = 3.0,
    focal_length_mm: float = 25.0,
    sensor_width_mm: float = 36.0
) -> CameraCalibration:
    """Create a default calibration with estimated parameters.
    
    This function creates a reasonable default calibration when no
    calibration data is available. Uses typical values from the paper:
    - Resolution: 1920×1080 (or 1280×720)
    - Camera distance: 3m
    - Typical industrial camera settings
    
    Args:
        resolution: Image resolution (width, height)
        camera_distance: Distance from camera to conveyor belt in meters
        focal_length_mm: Lens focal length in millimeters
        sensor_width_mm: Camera sensor width in millimeters
        
    Returns:
        CameraCalibration instance with estimated parameters
    """
    width, height = resolution
    
    # Calculate focal length in pixels
    # pixels_per_mm = sensor_width_pixels / sensor_width_mm
    # focal_length_pixels = focal_length_mm * pixels_per_mm
    sensor_width_pixels = width
    pixels_per_mm = sensor_width_pixels / sensor_width_mm
    fx = focal_length_mm * pixels_per_mm
    fy = fx  # Assume square pixels
    
    # Principal point at image center
    cx = width / 2
    cy = height / 2
    
    # Calculate pixels per meter
    # At distance D, the field of view width in meters is:
    # FOV_width_m = (sensor_width_mm / 1000) * (camera_distance / focal_length_mm)
    # pixels_per_meter = sensor_width_pixels / FOV_width_m
    # Simplified: pixels_per_meter = focal_length_pixels / camera_distance
    pixels_per_meter = fx / camera_distance
    
    return CameraCalibration(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dist_coeffs=np.zeros(5),
        pixels_per_meter=pixels_per_meter,
        image_width=width,
        image_height=height,
        camera_distance=camera_distance
    )


def set_calibration_from_known_distance(
    resolution: Tuple[int, int],
    known_distance_mm: float,
    known_pixels: float,
    camera_distance: float = 3.0
) -> CameraCalibration:
    """Create calibration from known distance measurements.
    
    This method is useful when you have a reference object of known size
    in the image and want to determine the pixel-to-meter ratio.
    
    Args:
        resolution: Image resolution (width, height)
        known_distance_mm: Known physical distance in millimeters
        known_pixels: Measured pixel distance in the image
        camera_distance: Distance from camera to object in meters
        
    Returns:
        CameraCalibration instance
    """
    # Calculate pixels per meter
    known_distance_m = known_distance_mm / 1000.0
    pixels_per_meter = known_pixels / known_distance_m
    
    # Estimate focal length from the ratio
    # pixels_per_meter = focal_length_pixels / camera_distance
    focal_length_pixels = pixels_per_meter * camera_distance
    
    # Create calibration
    width, height = resolution
    return CameraCalibration(
        fx=focal_length_pixels,
        fy=focal_length_pixels,
        cx=width / 2,
        cy=height / 2,
        dist_coeffs=np.zeros(5),
        pixels_per_meter=pixels_per_meter,
        image_width=width,
        image_height=height,
        camera_distance=camera_distance
    )


def load_or_create_calibration(
    calibration_path: str,
    resolution: Tuple[int, int] = (1920, 1080),
    camera_distance: float = 3.0
) -> CameraCalibration:
    """Load calibration from file or create default.
    
    This is the main entry point for getting a calibration object.
    It tries to load from file, and if that fails, creates a default calibration.
    
    Args:
        calibration_path: Path to calibration YAML file
        resolution: Default resolution if creating new calibration
        camera_distance: Default camera distance if creating new calibration
        
    Returns:
        CameraCalibration instance
    """
    try:
        # Try to load existing calibration
        calibration = CameraCalibration.load(calibration_path)
        print(f"Loaded calibration from {calibration_path}")
        return calibration
    except FileNotFoundError:
        # Create default calibration
        print(f"No calibration file found at {calibration_path}")
        print(f"Creating default calibration for {resolution[0]}x{resolution[1]}")
        
        calibration = create_default_calibration(
            resolution=resolution,
            camera_distance=camera_distance
        )
        
        # Optionally save the default calibration
        try:
            calibration.save(calibration_path)
            print(f"Saved default calibration to {calibration_path}")
        except Exception as e:
            print(f"Warning: Could not save calibration: {e}")
        
        return calibration


class CalibrationManager:
    """Manager class for handling camera calibration operations.
    
    This class provides a higher-level interface for calibration management,
    including validation, comparison, and batch operations.
    """
    
    def __init__(self, calibration: CameraCalibration):
        """Initialize calibration manager.
        
        Args:
            calibration: CameraCalibration instance
        """
        self.calibration = calibration
        self._validate()
    
    def _validate(self) -> None:
        """Validate the calibration."""
        self.calibration.validate()
    
    def get_roi_mask(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Create a region-of-interest mask for speed measurement.
        
        The paper mentions using a region of interest (ROI) on the conveyor belt
        for more accurate speed measurement.
        
        Args:
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            Binary mask with ROI set to 1
        """
        mask = np.zeros((self.calibration.image_height, self.calibration.image_width), dtype=np.uint8)
        x, y, w, h = roi
        mask[y:y+h, x:x+w] = 1
        return mask
    
    def calculate_scale_factor(self, new_resolution: Tuple[int, int]) -> float:
        """Calculate scale factor for different resolution.
        
        Useful when processing videos at different resolutions than
        the calibrated resolution.
        
        Args:
            new_resolution: New resolution (width, height)
            
        Returns:
            Scale factor to apply to pixels_per_meter
        """
        new_width, new_height = new_resolution
        orig_width = self.calibration.image_width
        orig_height = self.calibration.image_height
        
        # Average scale factor
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height
        scale_factor = (scale_x + scale_y) / 2
        
        return scale_factor
    
    def get_calibration_for_resolution(
        self, 
        resolution: Tuple[int, int]
    ) -> CameraCalibration:
        """Get calibration adjusted for different resolution.
        
        Args:
            resolution: New resolution (width, height)
            
        Returns:
            New CameraCalibration adjusted for the resolution
        """
        scale_factor = self.calculate_scale_factor(resolution)
        
        return CameraCalibration(
            fx=self.calibration.fx * scale_factor,
            fy=self.calibration.fy * scale_factor,
            cx=resolution[0] / 2,
            cy=resolution[1] / 2,
            dist_coeffs=self.calibration.dist_coeffs.copy(),
            rotation_matrix=self.calibration.rotation_matrix.copy(),
            translation_vector=self.calibration.translation_vector.copy(),
            pixels_per_meter=self.calibration.pixels_per_meter * scale_factor,
            image_width=resolution[0],
            image_height=resolution[1],
            camera_distance=self.calibration.camera_distance
        )
    
    def print_summary(self) -> None:
        """Print a summary of calibration parameters."""
        print("=" * 50)
        print("Camera Calibration Summary")
        print("=" * 50)
        print(f"Resolution: {self.calibration.image_width}x{self.calibration.image_height}")
        print(f"Focal Length: fx={self.calibration.fx:.2f}, fy={self.calibration.fy:.2f}")
        print(f"Principal Point: cx={self.calibration.cx:.2f}, cy={self.calibration.cy:.2f}")
        print(f"Camera Distance: {self.calibration.camera_distance} m")
        print(f"Pixels per Meter: {self.calibration.pixels_per_meter:.2f} px/m")
        print(f"Distortion Coefficients: {self.calibration.dist_coeffs}")
        print("=" * 50)


# Global calibration instance
_default_calibration: Optional[CameraCalibration] = None


def get_default_calibration() -> CameraCalibration:
    """Get the global default calibration instance.
    
    Returns:
        Global CameraCalibration instance
    """
    global _default_calibration
    if _default_calibration is None:
        # Try to load from default path
        try:
            _default_calibration = CameraCalibration.load("./calibration.yaml")
        except FileNotFoundError:
            # Create default calibration
            _default_calibration = create_default_calibration()
    return _default_calibration


def set_default_calibration(calibration: CameraCalibration) -> None:
    """Set the global default calibration.
    
    Args:
        calibration: CameraCalibration instance to use as default
    """
    global _default_calibration
    _default_calibration = calibration


def reset_default_calibration() -> None:
    """Reset the global default calibration to newly created default."""
    global _default_calibration
    _default_calibration = create_default_calibration()


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Camera Calibration Module Test")
    print("=" * 60)
    
    # Test 1: Create default calibration (1920x1080)
    print("\n[Test 1] Creating default calibration for 1920x1080...")
    cal_1080p = create_default_calibration(
        resolution=(1920, 1080),
        camera_distance=3.0
    )
    cal_1080p.validate()
    print(f"  ✓ Pixels per meter: {cal_1080p.pixels_per_meter:.2f}")
    
    # Test 2: Create default calibration (1280x720)
    print("\n[Test 2] Creating default calibration for 1280x720...")
    cal_720p = create_default_calibration(
        resolution=(1280, 720),
        camera_distance=3.0
    )
    print(f"  ✓ Pixels per meter: {cal_720p.pixels_per_meter:.2f}")
    
    # Test 3: Test pixel displacement conversion
    print("\n[Test 3] Testing pixel displacement conversion...")
    test_pixel_displacement = 100.0  # pixels
    meter_displacement = cal_1080p.convert_pixel_displacement_to_meters(test_pixel_displacement)
    print(f"  {test_pixel_displacement} pixels = {meter_displacement:.4f} meters")
    
    # Test 4: Test speed calculation (25 fps = 0.04s frame interval)
    print("\n[Test 4] Testing speed calculation...")
    frame_interval = 1.0 / 25.0  # 0.04 seconds
    speed = cal_1080p.calculate_speed_from_displacement(test_pixel_displacement, frame_interval)
    print(f"  At {test_pixel_displacement} px displacement per {frame_interval*1000:.1f}ms:")
    print(f"  Speed = {speed:.4f} m/s")
    
    # Test 5: Test calibration save/load
    print("\n[Test 5] Testing calibration save/load...")
    test_path = "./test_calibration.yaml"
    cal_1080p.save(test_path)
    loaded_cal = CameraCalibration.load(test_path)
    assert loaded_cal.pixels_per_meter == cal_1080p.pixels_per_meter
    print(f"  ✓ Saved and loaded calibration successfully")
    
    # Clean up test file
    import os
    if os.path.exists(test_path):
        os.remove(test_path)
    
    # Test 6: Test resolution scaling
    print("\n[Test 6] Testing resolution scaling...")
    manager = CalibrationManager(cal_1080p)
    cal_720p_scaled = manager.get_calibration_for_resolution((1280, 720))
    print(f"  Original 1920x1080: {cal_1080p.pixels_per_meter:.2f} px/m")
    print(f"  Scaled for 1280x720: {cal_720p_scaled.pixels_per_meter:.2f} px/m")
    
    # Test 7: Test ROI mask creation
    print("\n[Test 7] Testing ROI mask creation...")
    roi = (100, 200, 500, 300)  # x, y, width, height
    roi_mask = manager.get_roi_mask(roi)
    print(f"  ROI: {roi}")
    print(f"  Mask shape: {roi_mask.shape}")
    print(f"  Non-zero pixels in ROI: {np.sum(roi_mask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])}")
    
    # Test 8: Test with known distance
    print("\n[Test 8] Testing calibration from known distance...")
    cal_known = set_calibration_from_known_distance(
        resolution=(1920, 1080),
        known_distance_mm=1000.0,  # 1 meter reference object
        known_pixels=300.0,        # Measures 300 pixels in image
        camera_distance=3.0
    )
    print(f"  Known: 1000mm = 300px at 3m distance")
    print(f"  Calculated pixels_per_meter: {cal_known.pixels_per_meter:.2f}")
    
    # Test 9: Print calibration summary
    print("\n[Test 9] Printing calibration summary...")
    manager.print_summary()
    
    # Test 10: Test bidirectional conversion
    print("\n[Test 10] Testing bidirectional conversion...")
    original_meters = 0.5  # 50 cm
    pixels = cal_1080p.convert_meters_to_pixels(original_meters)
    back_to_meters = cal_1080p.convert_pixel_displacement_to_meters(pixels)
    print(f"  {original_meters} m -> {pixels:.2f} px -> {back_to_meters:.4f} m")
    assert abs(original_meters - back_to_meters) < 1e-6
    print("  ✓ Bidirectional conversion verified")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
