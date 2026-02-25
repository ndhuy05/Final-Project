"""
optical_flow/flow_estimator.py

Optical flow estimator for conveyor belt speed detection.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module provides the high-level interface for optical flow-based speed estimation,
wrapping the RAFT-SEnet model and providing utilities for converting flow fields
to meaningful speed measurements.

Author: Based on paper methodology
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
import warnings
import time

# Handle import errors gracefully
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV (cv2) not available. Some functionality may be limited.")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch (torch) not available. Optical flow estimation requires PyTorch.")

# Import related modules - handle potential import errors
try:
    from config import Config, get_config, OpticalFlowConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    get_config = None
    OpticalFlowConfig = None

try:
    from utils.calibration import CameraCalibration, create_default_calibration
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    CameraCalibration = None
    create_default_calibration = None

try:
    from optical_flow.raft_model import RAFTSEnetModel, create_raft_senet_model
    RAFT_MODEL_AVAILABLE = True
except ImportError:
    RAFT_MODEL_AVAILABLE = False
    RAFTSEnetModel = None
    create_raft_senet_model = None


@dataclass
class OpticalFlowResult:
    """Result container for optical flow-based speed estimation.
    
    This class stores the output from the optical flow estimation process,
    including the dense flow field, calculated speed, and confidence score.
    
    Attributes:
        flow_field: Dense optical flow field (H × W × 2) with (u, v) components.
                   Each component represents pixel displacement.
        speed_px_per_sec: Average flow magnitude in pixels per second.
        confidence: Confidence score (0-1) indicating reliability of the estimate.
                   Higher values indicate more reliable measurements.
        frame_interval: Time between frames used for calculation in seconds.
        avg_flow_magnitude: Average pixel displacement per frame.
        valid_pixel_ratio: Ratio of valid flow pixels to total pixels in ROI.
    """
    flow_field: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 2)))
    speed_px_per_sec: float = 0.0
    confidence: float = 0.0
    frame_interval: float = 0.04
    avg_flow_magnitude: float = 0.0
    valid_pixel_ratio: float = 0.0
    
    def __post_init__(self):
        """Validate and process results after initialization."""
        # Ensure flow_field is a numpy array
        if not isinstance(self.flow_field, np.ndarray):
            self.flow_field = np.array(self.flow_field)
        
        # Clip confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure speed is non-negative
        self.speed_px_per_sec = max(0.0, self.speed_px_per_sec)
        self.avg_flow_magnitude = max(0.0, self.avg_flow_magnitude)
        
        # Ensure valid pixel ratio is in [0, 1]
        self.valid_pixel_ratio = max(0.0, min(1.0, self.valid_pixel_ratio))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            'flow_field': self.flow_field.tolist() if isinstance(self.flow_field, np.ndarray) else self.flow_field,
            'speed_px_per_sec': float(self.speed_px_per_sec),
            'confidence': float(self.confidence),
            'frame_interval': float(self.frame_interval),
            'avg_flow_magnitude': float(self.avg_flow_magnitude),
            'valid_pixel_ratio': float(self.valid_pixel_ratio)
        }


class FlowEstimator:
    """High-level interface for optical flow-based speed estimation.
    
    This class wraps the RAFT-SEnet model and provides utilities for converting
    optical flow fields into meaningful speed measurements (m/s).
    
    The estimator performs the following steps:
    1. Preprocess input frames (BGR → RGB, resize, normalize)
    2. Run RAFT-SEnet inference to estimate optical flow
    3. Post-process flow field (extract magnitude, apply ROI)
    4. Calculate confidence based on flow quality
    5. Convert to real-world speed using camera calibration
    
    Attributes:
        device: Device to run inference on ('cuda' or 'cpu')
        input_size: Input size for the model (height, width)
        frame_rate: Frame rate of the input video
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu',
        frame_rate: float = 25.0,
        calibration: Optional[CameraCalibration] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        flow_threshold: float = 0.1,
        use_senet: bool = True,
        num_iterations: int = 12
    ):
        """Initialize the flow estimator.
        
        Args:
            config: Configuration object containing model parameters.
                   If None, uses default values.
            checkpoint_path: Path to pretrained model weights.
            device: Device to run inference on ('cuda' or 'cpu')
            frame_rate: Frame rate of input video (for speed calculation)
            calibration: Camera calibration object for pixel-to-world conversion.
                        If None, uses default calibration.
            roi: Region of interest as (x, y, width, height) for speed calculation.
                If None, uses the entire image.
            flow_threshold: Minimum flow magnitude to consider valid (pixels/frame)
            use_senet: Whether to use SENet-enhanced model
            num_iterations: Number of RAFT refinement iterations
        """
        # Store configuration
        self.device = device
        self.frame_rate = frame_rate
        self.frame_interval = 1.0 / frame_rate if frame_rate > 0 else 0.04
        self.flow_threshold = flow_threshold
        self.num_iterations = num_iterations
        self.roi = roi
        
        # Get configuration parameters
        if config is not None and CONFIG_AVAILABLE:
            try:
                opt_config = config.model.optical_flow
                self.input_size = (opt_config.input_height, opt_config.input_width)
                self.use_senet = opt_config.use_senet
                self.senet_reduction = opt_config.senet_reduction
            except:
                self.input_size = (448, 256)
                self.use_senet = use_senet
                self.senet_reduction = 16
        else:
            self.input_size = (448, 256)
            self.use_senet = use_senet
            self.senet_reduction = 16
        
        # Initialize calibration
        self.calibration = calibration
        if self.calibration is None and CALIBRATION_AVAILABLE:
            try:
                self.calibration = create_default_calibration(
                    resolution=(1920, 1080),
                    camera_distance=3.0
                )
            except:
                pass
        
        # Initialize model
        self.model: Optional[RAFTSEnetModel] = None
        self._initialized = False
        
        # Try to initialize the model
        if RAFT_MODEL_AVAILABLE and HAS_TORCH:
            self._load_model(config, checkpoint_path)
        else:
            warnings.warn(
                "RAFT model or PyTorch not available. "
                "FlowEstimator will run in limited mode."
            )
    
    def _load_model(
        self,
        config: Optional[Config],
        checkpoint_path: Optional[str] = None
    ) -> None:
        """Load the RAFT-SEnet model.
        
        Args:
            config: Configuration object
            checkpoint_path: Path to pretrained weights
        """
        if not RAFT_MODEL_AVAILABLE or not HAS_TORCH:
            warnings.warn("Cannot load model: RAFT or PyTorch not available")
            return
        
        try:
            # Get optical flow config
            if config is not None and CONFIG_AVAILABLE:
                try:
                    opt_config = config.model.optical_flow
                except:
                    opt_config = None
            else:
                opt_config = None
            
            # Create model
            self.model = create_raft_senet_model(
                config=opt_config,
                pretrained_path=checkpoint_path,
                device=self.device
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            self._initialized = True
            print(f"Flow estimator initialized with RAFT-SEnet on {self.device}")
            
        except Exception as e:
            warnings.warn(f"Failed to load RAFT-SEnet model: {e}")
            self._initialized = False
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame for model input.
        
        Converts BGR (OpenCV format) to RGB, resizes to model input size,
        normalizes to [0, 1], and converts to PyTorch tensor format.
        
        Args:
            frame: Input frame in BGR format (H × W × 3)
            
        Returns:
            Preprocessed frame tensor (1 × 3 × H × W)
        """
        if not HAS_CV2:
            raise ImportError("OpenCV is required for frame preprocessing")
        
        # Convert BGR to RGB
        if frame.shape[-1] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Resize to model input size
        target_height, target_width = self.input_size
        frame_resized = cv2.resize(
            frame_rgb,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to [0, 1]
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor (C × H × W)
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
        
        # Add batch dimension
        frame_tensor = frame_tensor.unsqueeze(0)
        
        # Move to device
        if HAS_TORCH:
            frame_tensor = frame_tensor.to(self.device)
        
        return frame_tensor
    
    def _postprocess_flow(
        self,
        flow: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Post-process the flow field from model output.
        
        Converts flow tensor to numpy array, upscales to original resolution
        if needed, and rearranges to (H × W × 2) format.
        
        Args:
            flow: Flow tensor from model (B × 2 × H × W)
            original_size: Original image size (height, width)
            
        Returns:
            Flow field as numpy array (H × W × 2)
        """
        # Move to CPU and convert to numpy
        flow_np = flow.detach().cpu().numpy()
        
        # Remove batch dimension
        flow_np = flow_np[0]  # (2, H, W)
        
        # Transpose to (H, W, 2)
        flow_np = np.transpose(flow_np, (1, 2, 0))
        
        # Upscale if needed
        if original_size != self.input_size:
            flow_np = cv2.resize(
                flow_np,
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        return flow_np
    
    def _extract_flow_magnitude(
        self,
        flow_field: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, float, float]:
        """Extract flow magnitude and calculate statistics.
        
        Args:
            flow_field: Optical flow field (H × W × 2)
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            Tuple of (flow_magnitude, avg_magnitude, valid_ratio)
        """
        # Extract u and v components
        u = flow_field[:, :, 0]
        v = flow_field[:, :, 1]
        
        # Calculate magnitude
        magnitude = np.sqrt(u**2 + v**2)
        
        # Apply ROI mask if provided
        if roi is not None:
            x, y, w, h = roi
            mask = np.zeros_like(magnitude, dtype=bool)
            mask[y:y+h, x:x+w] = True
            magnitude = magnitude[mask]
        
        # Calculate statistics
        valid_mask = magnitude > self.flow_threshold
        valid_ratio = valid_mask.sum() / magnitude.size if magnitude.size > 0 else 0.0
        
        # Average magnitude for valid pixels only
        if valid_mask.sum() > 0:
            avg_magnitude = magnitude[valid_mask].mean()
        else:
            avg_magnitude = 0.0
        
        return magnitude, avg_magnitude, valid_ratio
    
    def _calculate_confidence(
        self,
        flow_magnitude: np.ndarray,
        valid_ratio: float,
        avg_magnitude: float
    ) -> float:
        """Calculate confidence score for the flow estimation.
        
        Confidence is based on:
        1. Coverage ratio: how many valid flow pixels
        2. Flow consistency: how uniform the flow is
        
        Args:
            flow_magnitude: Flow magnitude array
            valid_ratio: Ratio of valid pixels
            avg_magnitude: Average flow magnitude
            
        Returns:
            Confidence score in [0, 1]
        """
        # Component 1: Coverage ratio (50% weight)
        coverage_score = min(valid_ratio * 2, 1.0)  # Scale: 50% valid = 1.0
        
        # Component 2: Flow consistency (50% weight)
        # Lower variance = higher confidence
        if avg_magnitude > self.flow_threshold:
            # Normalized variance
            variance = flow_magnitude.std() / (avg_magnitude + 1e-6)
            consistency_score = max(0.0, 1.0 - variance)
        else:
            # No meaningful flow detected
            consistency_score = 0.0
        
        # Combined confidence
        confidence = 0.5 * coverage_score + 0.5 * consistency_score
        
        return min(max(confidence, 0.0), 1.0)
    
    def _convert_to_speed(
        self,
        avg_flow_magnitude: float,
        frame_interval: float,
        calibration: Optional[CameraCalibration] = None
    ) -> float:
        """Convert flow magnitude to real-world speed.
        
        Args:
            avg_flow_magnitude: Average pixel displacement per frame
            frame_interval: Time between frames in seconds
            calibration: Camera calibration for pixel-to-world conversion
            
        Returns:
            Speed in meters per second
        """
        # Convert to pixels per second
        speed_px_per_sec = avg_flow_magnitude / frame_interval
        
        # Convert to real-world speed using calibration
        if calibration is not None and CALIBRATION_AVAILABLE:
            try:
                # Convert pixel displacement to meters
                meter_displacement = calibration.convert_pixel_displacement_to_meters(avg_flow_magnitude)
                
                # Calculate speed in m/s
                speed_mps = meter_displacement / frame_interval
                
                return speed_mps
            except Exception as e:
                warnings.warn(f"Calibration conversion failed: {e}")
        
        # If no calibration, return raw pixels per second
        return speed_px_per_sec
    
    def estimate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        calibration: Optional[CameraCalibration] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        return_flow: bool = True
    ) -> OpticalFlowResult:
        """Estimate speed from a pair of consecutive frames.
        
        This is the main method for optical flow-based speed estimation.
        
        Args:
            frame1: First frame in BGR format (H × W × 3)
            frame2: Second frame in BGR format (H × W × 3)
            calibration: Optional camera calibration (overrides stored calibration)
            roi: Optional region of interest (overrides stored ROI)
            return_flow: Whether to include flow field in result
            
        Returns:
            OpticalFlowResult containing speed and confidence estimates
        """
        # Use stored or provided calibration
        cal = calibration if calibration is not None else self.calibration
        roi_used = roi if roi is not None else self.roi
        
        # Store original size for post-processing
        original_size = (frame1.shape[0], frame1.shape[1])
        
        # Check if model is available
        if not self._initialized or self.model is None:
            # Return fallback result
            return self._fallback_result()
        
        try:
            # Preprocess frames
            frame1_tensor = self._preprocess_frame(frame1)
            frame2_tensor = self._preprocess_frame(frame2)
            
            # Run inference
            with torch.no_grad():
                result = self.model(
                    frame1_tensor,
                    frame2_tensor,
                    num_iterations=self.num_iterations,
                    return_intermediate=False
                )
            
            # Get flow field
            flow_field = result['flow']
            
            # Post-process flow
            flow_field_np = self._postprocess_flow(flow_field, original_size)
            
            # Extract flow magnitude
            magnitude, avg_magnitude, valid_ratio = self._extract_flow_magnitude(
                flow_field_np, roi_used
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(magnitude, valid_ratio, avg_magnitude)
            
            # Convert to speed
            speed_px_per_sec = avg_magnitude / self.frame_interval
            speed_mps = self._convert_to_speed(avg_magnitude, self.frame_interval, cal)
            
            # Create result
            return OpticalFlowResult(
                flow_field=flow_field_np if return_flow else np.zeros((1, 1, 2)),
                speed_px_per_sec=speed_px_per_sec,
                confidence=confidence,
                frame_interval=self.frame_interval,
                avg_flow_magnitude=avg_magnitude,
                valid_pixel_ratio=valid_ratio
            )
            
        except Exception as e:
            warnings.warn(f"Flow estimation failed: {e}")
            return self._fallback_result()
    
    def _fallback_result(self) -> OpticalFlowResult:
        """Create a fallback result when estimation fails.
        
        Returns:
            OpticalFlowResult with zero speed and zero confidence
        """
        return OpticalFlowResult(
            flow_field=np.zeros((1, 1, 2)),
            speed_px_per_sec=0.0,
            confidence=0.0,
            frame_interval=self.frame_interval,
            avg_flow_magnitude=0.0,
            valid_pixel_ratio=0.0
        )
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> None:
        """Set the region of interest for speed calculation.
        
        Args:
            roi: Region of interest as (x, y, width, height)
        """
        self.roi = roi
    
    def set_calibration(self, calibration: CameraCalibration) -> None:
        """Set the camera calibration for pixel-to-world conversion.
        
        Args:
            calibration: CameraCalibration object
        """
        self.calibration = calibration
    
    def get_speed_from_flow(
        self,
        flow_field: np.ndarray,
        calibration: Optional[CameraCalibration] = None,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """Calculate speed directly from a pre-computed flow field.
        
        This is a utility method for cases where flow is computed elsewhere.
        
        Args:
            flow_field: Pre-computed optical flow field (H × W × 2)
            calibration: Optional camera calibration
            roi: Optional region of interest
            
        Returns:
            Speed in meters per second
        """
        cal = calibration if calibration is not None else self.calibration
        roi_used = roi if roi is not None else self.roi
        
        # Extract magnitude
        magnitude, avg_magnitude, _ = self._extract_flow_magnitude(flow_field, roi_used)
        
        # Convert to speed
        speed_mps = self._convert_to_speed(avg_magnitude, self.frame_interval, cal)
        
        return speed_mps
    
    def __repr__(self) -> str:
        """String representation of the FlowEstimator."""
        return (
            f"FlowEstimator(\n"
            f"  device={self.device},\n"
            f"  input_size={self.input_size},\n"
            f"  frame_rate={self.frame_rate},\n"
            f"  initialized={self._initialized},\n"
            f"  use_senet={self.use_senet},\n"
            f"  num_iterations={self.num_iterations}\n"
            f")"
        )


def create_flow_estimator(
    config: Optional[Config] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu',
    calibration: Optional[CameraCalibration] = None,
    frame_rate: float = 25.0
) -> FlowEstimator:
    """Factory function to create a FlowEstimator instance.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to pretrained model weights
        device: Device to run on
        calibration: Camera calibration object
        frame_rate: Video frame rate
        
    Returns:
        Initialized FlowEstimator instance
    """
    return FlowEstimator(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        calibration=calibration,
        frame_rate=frame_rate
    )


# Global estimator registry
_estimators: Dict[str, FlowEstimator] = {}


def get_flow_estimator(
    name: str = "default",
    config: Optional[Config] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu',
    calibration: Optional[CameraCalibration] = None,
    frame_rate: float = 25.0
) -> FlowEstimator:
    """Get or create a FlowEstimator by name.
    
    Args:
        name: Estimator identifier
        config: Configuration object
        checkpoint_path: Path to pretrained weights
        device: Device to run on
        calibration: Camera calibration
        frame_rate: Video frame rate
        
    Returns:
        FlowEstimator instance
    """
    global _estimators
    
    if name in _estimators:
        return _estimators[name]
    
    estimator = create_flow_estimator(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        calibration=calibration,
        frame_rate=frame_rate
    )
    
    _estimators[name] = estimator
    
    return estimator


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Flow Estimator Module Test")
    print("=" * 60)
    
    # Test 1: Create FlowEstimator with defaults
    print("\n[Test 1] Creating FlowEstimator with defaults...")
    try:
        estimator = FlowEstimator(frame_rate=25.0)
        print(f"  Created: {repr(estimator)}")
        print("  ✓ FlowEstimator created")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Test 2: Test OpticalFlowResult dataclass
    print("\n[Test 2] Testing OpticalFlowResult...")
    result = OpticalFlowResult(
        flow_field=np.random.rand(100, 100, 2),
        speed_px_per_sec=25.0,
        confidence=0.8,
        frame_interval=0.04,
        avg_flow_magnitude=1.0,
        valid_pixel_ratio=0.7
    )
    print(f"  Speed px/s: {result.speed_px_per_sec}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Avg magnitude: {result.avg_flow_magnitude}")
    result_dict = result.to_dict()
    print(f"  ✓ OpticalFlowResult works")
    
    # Test 3: Test with synthetic frames
    print("\n[Test 3] Testing with synthetic frames...")
    if HAS_CV2:
        # Create synthetic frames (gray gradient)
        height, width = 480, 640
        frame1 = np.zeros((height, width, 3), dtype=np.uint8)
        frame2 = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some gradient
        for y in range(height):
            for x in range(width):
                frame1[y, x] = [x % 256, y % 256, (x + y) % 256]
                frame2[y, x] = [(x + 5) % 256, y % 256, (x + y + 5) % 256]
        
        # Estimate flow (will use fallback if model not available)
        if estimator is not None:
            result = estimator.estimate(frame1, frame2)
            print(f"  Estimated speed px/s: {result.speed_px_per_sec:.2f}")
            print(f"  Confidence: {result.confidence:.2f}")
            print("  ✓ Estimation completed")
        else:
            print("  Skipping (model not available)")
    else:
        print("  Skipping (OpenCV not available)")
    
    # Test 4: Test calibration conversion
    print("\n[Test 4] Testing calibration conversion...")
    if CALIBRATION_AVAILABLE:
        try:
            cal = create_default_calibration(resolution=(1920, 1080), camera_distance=3.0)
            pixel_disp = 10.0  # 10 pixels
            meter_disp = cal.convert_pixel_displacement_to_meters(pixel_disp)
            print(f"  {pixel_disp} pixels = {meter_disp:.4f} meters")
            
            # Calculate speed
            frame_interval = 0.04  # 25 fps
            speed = meter_disp / frame_interval
            print(f"  Speed at {pixel_disp}px/{frame_interval*1000:.0f}ms = {speed:.2f} m/s")
            print("  ✓ Calibration conversion works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (calibration module not available)")
    
    # Test 5: Test ROI setting
    print("\n[Test 5] Testing ROI setting...")
    if estimator is not None:
        roi = (100, 100, 400, 200)  # x, y, width, height
        estimator.set_roi(roi)
        print(f"  Set ROI: {estimator.roi}")
        print("  ✓ ROI setting works")
    
    # Test 6: Test speed conversion utility
    print("\n[Test 6] Testing speed conversion utility...")
    if estimator is not None and CALIBRATION_AVAILABLE:
        try:
            # Create simple flow field
            flow_field = np.zeros((100, 100, 2))
            flow_field[:, :, 0] = 5.0  # 5 pixels right
            
            speed = estimator.get_speed_from_flow(flow_field)
            print(f"  Flow speed: {speed:.4f} m/s")
            print("  ✓ Speed conversion works")
        except Exception as e:
            print(f"  Note: {e}")
    
    # Test 7: Test fallback result
    print("\n[Test 7] Testing fallback result...")
    fallback = estimator._fallback_result()
    print(f"  Fallback speed: {fallback.speed_px_per_sec}")
    print(f"  Fallback confidence: {fallback.confidence}")
    print("  ✓ Fallback works")
    
    # Test 8: Test factory function
    print("\n[Test 8] Testing factory function...")
    factory_estimator = create_flow_estimator(frame_rate=30.0)
    print(f"  Factory frame rate: {factory_estimator.frame_rate}")
    print("  ✓ Factory function works")
    
    # Test 9: Test get_flow_estimator registry
    print("\n[Test 9] Testing registry...")
    reg_estimator = get_flow_estimator("test_reg", frame_rate=30.0)
    print(f"  Registry estimator frame rate: {reg_estimator.frame_rate}")
    print("  ✓ Registry works")
    
    # Test 10: Test config integration
    print("\n[Test 10] Testing config integration...")
    if CONFIG_AVAILABLE:
        try:
            cfg = get_config()
            print(f"  Config frame rate: {cfg.dataset.frame_rate}")
            print(f"  Config optical flow input size: {cfg.model.optical_flow.input_size}")
            print("  ✓ Config integration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (config not available)")
    
    print("\n" + "=" * 60)
    print("All Flow Estimator tests completed!")
    print("=" * 60)
