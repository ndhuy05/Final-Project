"""
main.py

Main entry point for the conveyor belt speed detection system.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module orchestrates the entire pipeline:
1. Loads configuration from config.yaml
2. Initializes camera calibration
3. Sets up optical flow estimator (RAFT-SEnet)
4. Sets up feature matching pipeline (Harris-BRIEF-RANSAC)
5. Processes video frames through both branches
6. Performs Bayesian decision fusion
7. Evaluates results and compares with baselines

Author: Based on paper methodology
"""

import os
import sys
import time
import glob
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import json
import yaml

# Handle imports with fallbacks
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV (cv2) not available. Install with: pip install opencv-python")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch (torch) not available. Install with: pip install torch")

# Import local modules
try:
    from config import Config, get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    get_config = None

# Try to import processing modules
OPTICAL_FLOW_AVAILABLE = False
FEATURE_MATCHING_AVAILABLE = False
FUSION_AVAILABLE = False
CALIBRATION_AVAILABLE = False
METRICS_AVAILABLE = False

try:
    from optical_flow.flow_estimator import FlowEstimator, create_flow_estimator
    from optical_flow.raft_model import RAFTSEnetModel
    OPTICAL_FLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import optical flow modules: {e}")

try:
    from feature_matching.matching_pipeline import FeatureMatchingPipeline, create_matching_pipeline
    FEATURE_MATCHING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import feature matching modules: {e}")

try:
    from fusion.bayesian_fusion import BayesianFusion, create_bayesian_fusion
    from fusion.image_quality import ImageQualityAnalyzer
    FUSION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import fusion modules: {e}")

try:
    from utils.calibration import CameraCalibration, create_default_calibration, load_or_create_calibration
    CALIBRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import calibration modules: {e}")

try:
    from utils.metrics import calculate_mae, calculate_rmse, calculate_error_percentage, evaluate_speed_detection, EvaluationResult
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import metrics modules: {e}")


@dataclass
class ProcessingResult:
    """Container for processing results of a single frame pair."""
    frame_index: int
    timestamp: float
    ground_truth_speed: float
    optical_flow_speed: float = 0.0
    feature_matching_speed: float = 0.0
    fused_speed: float = 0.0
    weight_optical: float = 0.5
    weight_feature: float = 0.5
    confidence_optical: float = 0.0
    confidence_feature: float = 0.0
    brightness: float = 127.5
    contrast: float = 50.0
    num_optical_pixels: int = 0
    num_feature_matches: int = 0
    processing_time_ms: float = 0.0
    is_valid: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'frame_index': self.frame_index,
            'timestamp': float(self.timestamp),
            'ground_truth_speed': float(self.ground_truth_speed),
            'optical_flow_speed': float(self.optical_flow_speed),
            'feature_matching_speed': float(self.feature_matching_speed),
            'fused_speed': float(self.fused_speed),
            'weight_optical': float(self.weight_optical),
            'weight_feature': float(self.weight_feature),
            'confidence_optical': float(self.confidence_optical),
            'confidence_feature': float(self.confidence_feature),
            'brightness': float(self.brightness),
            'contrast': float(self.contrast),
            'num_optical_pixels': self.num_optical_pixels,
            'num_feature_matches': self.num_feature_matches,
            'processing_time_ms': float(self.processing_time_ms),
            'is_valid': self.is_valid,
            'error_message': str(self.error_message)
        }


@dataclass
class VideoResult:
    """Container for processing results of an entire video."""
    video_path: str
    video_name: str
    ground_truth_speed: float
    results: List[ProcessingResult] = field(default_factory=list)
    total_frames: int = 0
    valid_frames: int = 0
    avg_processing_time_ms: float = 0.0
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'video_path': self.video_path,
            'video_name': self.video_name,
            'ground_truth_speed': float(self.ground_truth_speed),
            'total_frames': self.total_frames,
            'valid_frames': self.valid_frames,
            'avg_processing_time_ms': float(self.avg_processing_time_ms),
            'error_message': str(self.error_message),
            'results': [r.to_dict() for r in self.results]
        }
    
    def get_estimated_speeds(self) -> List[float]:
        """Get list of fused speed estimates."""
        return [r.fused_speed for r in self.results if r.is_valid]
    
    def get_ground_truth_speeds(self) -> List[float]:
        """Get list of ground truth speeds."""
        return [r.ground_truth_speed for r in self.results if r.is_valid]


class ConveyorBeltSpeedDetector:
    """Main class for conveyor belt speed detection.
    
    This class orchestrates the entire detection pipeline including:
    - Optical flow estimation (RAFT-SEnet)
    - Feature matching (Harris-BRIEF-RANSAC)
    - Bayesian decision fusion
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        device: str = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu',
        enable_optical_flow: bool = True,
        enable_feature_matching: bool = True,
        enable_fusion: bool = True,
        calibration_path: Optional[str] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        verbose: bool = True
    ):
        """Initialize the conveyor belt speed detector.
        
        Args:
            config: Configuration object. If None, uses default config.
            device: Device for inference ('cuda' or 'cpu')
            enable_optical_flow: Whether to enable optical flow branch
            enable_feature_matching: Whether to enable feature matching branch
            enable_fusion: Whether to enable fusion (for single-branch modes)
            calibration_path: Path to camera calibration file
            roi: Region of interest as (x, y, width, height)
            verbose: Whether to print progress information
        """
        # Store configuration
        self.config = config
        self.device = device
        self.enable_optical_flow = enable_optical_flow
        self.enable_feature_matching = enable_feature_matching
        self.enable_fusion = enable_fusion
        self.roi = roi
        self.verbose = verbose
        
        # Get configuration values
        if config is not None:
            self.frame_rate = config.dataset.frame_rate
            self.frame_interval = config.dataset.frame_interval
        else:
            self.frame_rate = 25.0
            self.frame_interval = 1.0 / self.frame_rate
        
        # Initialize components
        self._calibration: Optional[CameraCalibration] = None
        self._flow_estimator: Optional[FlowEstimator] = None
        self._feature_matcher: Optional[FeatureMatchingPipeline] = None
        self._fusion_module: Optional[BayesianFusion] = None
        self._quality_analyzer: Optional[ImageQualityAnalyzer] = None
        
        # Initialize all components
        self._initialize_components(calibration_path)
        
        # Statistics
        self._stats = {
            'total_videos': 0,
            'total_frames': 0,
            'valid_frames': 0,
            'failed_frames': 0,
            'total_processing_time_ms': 0.0
        }
        
        if self.verbose:
            print(f"ConveyorBeltSpeedDetector initialized on device: {self.device}")
            print(f"  Optical flow: {'enabled' if enable_optical_flow else 'disabled'}")
            print(f"  Feature matching: {'enabled' if enable_feature_matching else 'disabled'}")
            print(f"  Fusion: {'enabled' if enable_fusion else 'disabled'}")
    
    def _initialize_components(self, calibration_path: Optional[str]) -> None:
        """Initialize all processing components."""
        
        # Initialize calibration
        if CALIBRATION_AVAILABLE:
            if calibration_path is None and self.config is not None:
                calibration_path = self.config.paths.calibration_file
            
            try:
                self._calibration = load_or_create_calibration(
                    calibration_path or "./calibration.yaml",
                    resolution=(1920, 1080),
                    camera_distance=3.0
                )
                if self.verbose:
                    print(f"  Calibration loaded: {self._calibration.pixels_per_meter:.2f} pixels/meter")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not load calibration: {e}")
                self._calibration = create_default_calibration()
        else:
            # Create basic calibration
            self._calibration = create_default_calibration(
                resolution=(1920, 1080),
                camera_distance=3.0
            )
        
        # Initialize optical flow estimator
        if self.enable_optical_flow and OPTICAL_FLOW_AVAILABLE:
            try:
                self._flow_estimator = create_flow_estimator(
                    config=self.config,
                    frame_rate=self.frame_rate,
                    calibration=self._calibration
                )
                if self.verbose:
                    print(f"  Optical flow estimator initialized")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not initialize optical flow: {e}")
                self.enable_optical_flow = False
        
        # Initialize feature matching pipeline
        if self.enable_feature_matching and FEATURE_MATCHING_AVAILABLE:
            try:
                self._feature_matcher = create_matching_pipeline(
                    config=self.config,
                    frame_rate=self.frame_rate,
                    enable_clahe=False,
                    use_roi=self.roi is not None,
                    roi=self.roi
                )
                if self.verbose:
                    print(f"  Feature matching pipeline initialized")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not initialize feature matching: {e}")
                self.enable_feature_matching = False
        
        # Initialize fusion module
        if self.enable_fusion and FUSION_AVAILABLE:
            try:
                self._fusion_module = create_bayesian_fusion(
                    config=self.config,
                    enable_adaptive_fusion=True
                )
                self._quality_analyzer = ImageQualityAnalyzer()
                if self.verbose:
                    print(f"  Bayesian fusion module initialized")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not initialize fusion: {e}")
                self.enable_fusion = False
    
    def _extract_speed_from_filename(self, filename: str) -> float:
        """Extract ground truth speed from video filename.
        
        Args:
            filename: Video filename
            
        Returns:
            Ground truth speed in m/s (default 1.0 if not found)
        """
        # Try to extract speed from filename patterns
        import re
        
        # Pattern 1: speed followed by 'fps' or 'mps' or 'ms'
        patterns = [
            r'(\d+\.?\d*)\s*(?:fps|mps|ms)',  # e.g., 1.5fps, 2.0ms
            r'speed[_-](\d+\.?\d*)',            # e.g., speed_1.5
            r'_(\d+\.?\d*)\s*fps',              # e.g., _1.5fps
            r'_(\d+\.?\d*)\s*mps',              # e.g., _1.5mps
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Default speed if not found
        return 1.0
    
    def _calculate_speed_from_flow_result(
        self,
        flow_result: Any,
        calibration: CameraCalibration
    ) -> float:
        """Calculate speed in m/s from optical flow result.
        
        Args:
            flow_result: OpticalFlowResult object
            calibration: Camera calibration
            
        Returns:
            Speed in m/s
        """
        try:
            # Check if result has the right attributes
            if hasattr(flow_result, 'speed_px_per_sec'):
                speed_px = flow_result.speed_px_per_sec
            elif hasattr(flow_result, 'speed_m_per_sec'):
                return flow_result.speed_m_per_sec
            else:
                return 0.0
            
            # Convert from px/s to m/s
            if speed_px > 0:
                # Speed = pixels_per_second / pixels_per_meter
                speed_mps = speed_px / calibration.pixels_per_meter
                return float(speed_mps)
            
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_speed_from_match_result(
        self,
        match_result: Any,
        calibration: CameraCalibration
    ) -> float:
        """Calculate speed in m/s from feature matching result.
        
        Args:
            match_result: FeatureMatchResult object
            calibration: Camera calibration
            
        Returns:
            Speed in m/s
        """
        try:
            if hasattr(match_result, 'speed_m_per_sec'):
                return float(match_result.speed_m_per_sec)
            elif hasattr(match_result, 'speed_px_per_sec'):
                speed_px = match_result.speed_px_per_sec
                if speed_px > 0:
                    return speed_px / calibration.pixels_per_meter
            return 0.0
        except Exception:
            return 0.0
    
    def process_frame_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
        ground_truth_speed: float = 1.0
    ) -> ProcessingResult:
        """Process a single frame pair through the complete pipeline.
        
        Args:
            frame1: First frame (BGR format)
            frame2: Second frame (BGR format)
            frame_index: Index of the frame in the video
            timestamp: Timestamp in seconds
            ground_truth_speed: Known ground truth speed (for evaluation)
            
        Returns:
            ProcessingResult with fused speed and metadata
        """
        start_time = time.time()
        
        result = ProcessingResult(
            frame_index=frame_index,
            timestamp=timestamp,
            ground_truth_speed=ground_truth_speed,
            is_valid=False
        )
        
        try:
            # Validate frames
            if frame1 is None or frame2 is None:
                result.error_message = "Frame is None"
                return result
            
            if frame1.size == 0 or frame2.size == 0:
                result.error_message = "Frame is empty"
                return result
            
            # Step 1: Image quality assessment
            if FUSION_AVAILABLE and self._quality_analyzer is not None:
                try:
                    quality = self._quality_analyzer.analyze(frame1)
                    result.brightness = quality.brightness
                    result.contrast = quality.contrast
                except Exception as e:
                    result.brightness = 127.5
                    result.contrast = 50.0
            else:
                # Fallback: simple calculation
                gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                result.brightness = float(np.mean(gray))
                result.contrast = float(np.std(gray))
            
            # Step 2: Optical flow branch
            optical_speed = 0.0
            optical_confidence = 0.0
            num_optical_pixels = 0
            
            if self.enable_optical_flow and self._flow_estimator is not None:
                try:
                    flow_result = self._flow_estimator.estimate(
                        frame1,
                        frame2,
                        calibration=self._calibration,
                        roi=self.roi
                    )
                    
                    if flow_result is not None:
                        optical_speed = self._calculate_speed_from_flow_result(
                            flow_result, self._calibration
                        )
                        optical_confidence = flow_result.confidence if hasattr(flow_result, 'confidence') else 0.5
                        num_optical_pixels = flow_result.valid_pixel_ratio if hasattr(flow_result, 'valid_pixel_ratio') else 0
                
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Optical flow estimation failed: {e}")
            
            result.optical_flow_speed = optical_speed
            result.confidence_optical = optical_confidence
            result.num_optical_pixels = int(num_optical_pixels * 1000)  # Approximate
            
            # Step 3: Feature matching branch
            feature_speed = 0.0
            feature_confidence = 0.0
            num_feature_matches = 0
            
            if self.enable_feature_matching and self._feature_matcher is not None:
                try:
                    match_result = self._feature_matcher.process(
                        frame1,
                        frame2,
                        calibration=self._calibration
                    )
                    
                    if match_result is not None:
                        feature_speed = self._calculate_speed_from_match_result(
                            match_result, self._calibration
                        )
                        feature_confidence = match_result.confidence if hasattr(match_result, 'confidence') else 0.5
                        num_feature_matches = match_result.num_inliers if hasattr(match_result, 'num_inliers') else 0
                
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Feature matching failed: {e}")
            
            result.feature_matching_speed = feature_speed
            result.confidence_feature = feature_confidence
            result.num_feature_matches = num_feature_matches
            
            # Step 4: Bayesian fusion
            if self.enable_fusion and self._fusion_module is not None:
                try:
                    # Calculate raw weights based on image quality
                    brightness = result.brightness / 255.0
                    contrast = min(result.contrast / 128.0, 1.0)
                    
                    weight_a = 0.5 if self.config is None else self.config.fusion.weight_a
                    weight_b = 0.5 if self.config is None else self.config.fusion.weight_b
                    
                    w_optical_raw = weight_a * brightness + weight_b * contrast
                    w_feature_raw = weight_a * brightness + weight_b * contrast
                    
                    # Adjust by confidence
                    w_optical_adj = w_optical_raw * (0.5 + 0.5 * optical_confidence)
                    w_feature_adj = w_feature_raw * (0.5 + 0.5 * feature_confidence)
                    
                    # Normalize weights
                    total = w_optical_adj + w_feature_adj
                    if total > 0:
                        result.weight_optical = w_optical_adj / total
                        result.weight_feature = w_feature_adj / total
                    else:
                        result.weight_optical = 0.5
                        result.weight_feature = 0.5
                    
                    # Calculate fused speed
                    result.fused_speed = (
                        result.weight_optical * optical_speed +
                        result.weight_feature * feature_speed
                    )
                    
                except Exception as e:
                    if self.verbose:
                        warnings.warn(f"Fusion failed: {e}")
                    # Fallback: simple average
                    if optical_speed > 0 and feature_speed > 0:
                        result.fused_speed = (optical_speed + feature_speed) / 2
                    else:
                        result.fused_speed = max(optical_speed, feature_speed)
            else:
                # No fusion - use single branch result
                result.weight_optical = 1.0 if optical_speed > 0 else 0.0
                result.weight_feature = 1.0 if feature_speed > 0 else 0.0
                result.fused_speed = optical_speed if optical_speed > 0 else feature_speed
            
            # Mark as valid if we got any speed estimate
            result.is_valid = result.fused_speed > 0
            
        except Exception as e:
            result.error_message = str(e)
            if self.verbose:
                warnings.warn(f"Frame processing failed: {e}")
        
        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        frame_skip: int = 0
    ) -> VideoResult:
        """Process an entire video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            frame_skip: Number of frames to skip between pairs
            
        Returns:
            VideoResult containing all processing results
        """
        video_name = os.path.basename(video_path)
        ground_truth_speed = self._extract_speed_from_filename(video_name)
        
        result = VideoResult(
            video_path=video_path,
            video_name=video_name,
            ground_truth_speed=ground_truth_speed
        )
        
        if not HAS_CV2:
            result.error_message = "OpenCV not available"
            return result
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result.error_message = f"Could not open video: {video_path}"
                return result
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            result.total_frames = total_frames
            
            # Determine frames to process
            num_pairs = total_frames - 1 - frame_skip
            if num_pairs <= 0:
                result.error_message = "Insufficient frames in video"
                cap.release()
                return result
            
            if max_frames is not None:
                num_pairs = min(num_pairs, max_frames)
            
            # Read first frame
            ret, prev_frame = cap.read()
            if not ret:
                result.error_message = "Could not read first frame"
                cap.release()
                return result
            
            # Process frame pairs
            processing_times = []
            
            with tqdm(total=num_pairs, desc=f"Processing {video_name}", 
                     disable=not self.verbose) as pbar:
                for i in range(num_pairs):
                    # Skip frames if frame_skip > 0
                    if frame_skip > 0:
                        frame_idx = i + 1 + frame_skip
                        if frame_idx >= total_frames:
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    
                    # Read second frame
                    ret, curr_frame = cap.read()
                    if not ret:
                        break
                    
                    # Calculate timestamp
                    timestamp = i / fps if fps > 0 else i * self.frame_interval
                    
                    # Process frame pair
                    frame_result = self.process_frame_pair(
                        prev_frame,
                        curr_frame,
                        frame_index=i,
                        timestamp=timestamp,
                        ground_truth_speed=ground_truth_speed
                    )
                    
                    result.results.append(frame_result)
                    processing_times.append(frame_result.processing_time_ms)
                    
                    if frame_result.is_valid:
                        result.valid_frames += 1
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Move to next pair
                    prev_frame = curr_frame.copy()
            
            cap.release()
            
            # Calculate statistics
            if processing_times:
                result.avg_processing_time_ms = np.mean(processing_times)
            
            # Update global statistics
            self._stats['total_videos'] += 1
            self._stats['total_frames'] += result.total_frames
            self._stats['valid_frames'] += result.valid_frames
            self._stats['failed_frames'] += (result.total_frames - result.valid_frames)
            self._stats['total_processing_time_ms'] += sum(processing_times)
            
        except Exception as e:
            result.error_message = str(e)
            if self.verbose:
                warnings.warn(f"Video processing failed: {e}")
        
        return result
    
    def process_video_directory(
        self,
        video_dir: str,
        max_videos: Optional[int] = None,
        max_frames_per_video: Optional[int] = None,
        frame_skip: int = 0,
        extensions: List[str] = None
    ) -> List[VideoResult]:
        """Process all videos in a directory.
        
        Args:
            video_dir: Directory containing video files
            max_videos: Maximum number of videos to process
            max_frames_per_video: Maximum frames per video
            frame_skip: Frames to skip between pairs
            extensions: Valid video file extensions
            
        Returns:
            List of VideoResult objects
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        
        # Find all video files
        video_files = []
        for ext in extensions:
            pattern = os.path.join(video_dir, f"*{ext}")
            video_files.extend(glob.glob(pattern))
        
        video_files = sorted(video_files)
        
        if max_videos is not None:
            video_files = video_files[:max_videos]
        
        results = []
        
        if self.verbose:
            print(f"\nFound {len(video_files)} videos in {video_dir}")
        
        for video_path in tqdm(video_files, desc="Processing videos", disable=not self.verbose):
            result = self.process_video(
                video_path,
                max_frames=max_frames_per_video,
                frame_skip=frame_skip
            )
            results.append(result)
        
        return results
    
    def evaluate_results(
        self,
        results: List[VideoResult]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate results using metrics.
        
        Args:
            results: List of VideoResult objects
            
        Returns:
            Dictionary mapping method names to EvaluationResult
        """
        if not METRICS_AVAILABLE:
            return {}
        
        # Collect all speed estimates and ground truth
        all_fused = []
        all_optical = []
        all_feature = []
        all_ground_truth = []
        
        for video_result in results:
            for frame_result in video_result.results:
                if frame_result.is_valid:
                    all_fused.append(frame_result.fused_speed)
                    all_optical.append(frame_result.optical_flow_speed)
                    all_feature.append(frame_result.feature_matching_speed)
                    all_ground_truth.append(frame_result.ground_truth_speed)
        
        if not all_fused:
            return {}
        
        # Convert to numpy arrays
        fused_array = np.array(all_fused)
        optical_array = np.array(all_optical)
        feature_array = np.array(all_feature)
        ground_truth_array = np.array(all_ground_truth)
        
        # Evaluate each method
        evaluation_results = {}
        
        evaluation_results['fused'] = evaluate_speed_detection(fused_array, ground_truth_array)
        evaluation_results['optical_flow'] = evaluate_speed_detection(optical_array, ground_truth_array)
        evaluation_results['feature_matching'] = evaluate_speed_detection(feature_array, ground_truth_array)
        
        return evaluation_results
    
    def print_evaluation_summary(
        self,
        evaluation_results: Dict[str, EvaluationResult]
    ) -> None:
        """Print evaluation summary.
        
        Args:
            evaluation_results: Dictionary of method evaluations
        """
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\n{'Method':<20} {'MAE (m/s)':<15} {'RMSE (m/s)':<15} {'Error %':<15}")
        print("-" * 70)
        
        for method_name, result in evaluation_results.items():
            print(f"{method_name:<20} {result.mae:<15.4f} {result.rmse:<15.4f} {result.error_percentage:<15.2f}")
        
        print("=" * 70)
    
    def save_results(
        self,
        results: List[VideoResult],
        output_path: str,
        evaluation_results: Optional[Dict[str, EvaluationResult]] = None
    ) -> None:
        """Save processing results to file.
        
        Args:
            results: List of VideoResult objects
            output_path: Path to save results
            evaluation_results: Optional evaluation results
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert results to serializable format
        output_data = {
            'video_results': [v.to_dict() for v in results],
            'statistics': self._stats.copy()
        }
        
        # Add evaluation results if provided
        if evaluation_results is not None:
            output_data['evaluation'] = {
                method_name: result.to_dict()
                for method_name, result in evaluation_results.items()
            }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self._stats.copy()
        
        if stats['total_frames'] > 0:
            stats['success_rate'] = stats['valid_frames'] / stats['total_frames']
            stats['avg_processing_time_ms'] = (
                stats['total_processing_time_ms'] / stats['valid_frames']
                if stats['valid_frames'] > 0 else 0.0
            )
        
        return stats


def load_configuration(config_path: str = "./config.yaml") -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Config object with loaded settings
    """
    try:
        # Try to load from YAML
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create Config object from dictionary
        if config_dict is not None:
            # Extract sections
            dataset_cfg = config_dict.get('dataset', {})
            training_cfg = config_dict.get('training', {})
            model_cfg = config_dict.get('model', {})
            fusion_cfg = config_dict.get('fusion', {})
            paths_cfg = config_dict.get('paths', {})
            
            # Map to Config object
            config = Config()
            
            # Update dataset settings
            if 'resolution' in dataset_cfg:
                config.dataset.resolution = tuple(dataset_cfg['resolution'])
            if 'frame_rate' in dataset_cfg:
                config.dataset.frame_rate = dataset_cfg['frame_rate']
            if 'lab_belt_speeds' in dataset_cfg:
                config.dataset.lab_belt_speeds = dataset_cfg['lab_belt_speeds']
            
            # Update training settings
            if 'learning_rate' in training_cfg:
                config.training.learning_rate = training_cfg['learning_rate']
            if 'batch_size' in training_cfg:
                config.training.batch_size = training_cfg['batch_size']
            if 'num_epochs' in training_cfg:
                config.training.num_epochs = training_cfg['num_epochs']
            
            # Update model settings
            if 'optical_flow' in model_cfg:
                of_cfg = model_cfg['optical_flow']
                config.model.optical_flow.input_height = of_cfg.get('input_height', 448)
                config.model.optical_flow.input_width = of_cfg.get('input_width', 256)
                config.model.optical_flow.use_senet = of_cfg.get('use_senet', True)
                config.model.optical_flow.senet_reduction = of_cfg.get('senet_reduction', 16)
            
            if 'feature_matching' in model_cfg:
                fm_cfg = model_cfg['feature_matching']
                config.model.feature_matching.harris_k = fm_cfg.get('harris_k', 0.04)
                config.model.feature_matching.ransac_threshold = fm_cfg.get('ransac_threshold', 5.0)
                config.model.feature_matching.ransac_max_iters = fm_cfg.get('ransac_max_iters', 2000)
            
            # Update fusion settings
            if 'weight_a' in fusion_cfg:
                config.fusion.weight_a = fusion_cfg['weight_a']
            if 'weight_b' in fusion_cfg:
                config.fusion.weight_b = fusion_cfg['weight_b']
            
            # Update paths
            if 'data_dir' in paths_cfg:
                config.paths.data_dir = paths_cfg['data_dir']
            if 'video_dir' in paths_cfg:
                config.paths.video_dir = paths_cfg['video_dir']
            if 'output_dir' in paths_cfg:
                config.paths.output_dir = paths_cfg['output_dir']
            
            return config
        
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
    
    # Return default config
    return Config()


def run_detection(
    video_dir: str = None,
    config_path: str = "./config.yaml",
    output_dir: str = "./output",
    max_videos: int = None,
    max_frames_per_video: int = None,
    device: str = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu',
    save_results: bool = True
) -> Tuple[ConveyorBeltSpeedDetector, List[VideoResult], Dict[str, EvaluationResult]]:
    """Run the complete detection pipeline.
    
    Args:
        video_dir: Directory containing video files. If None, uses config path.
        config_path: Path to configuration file
        output_dir: Directory for output files
        max_videos: Maximum number of videos to process
        max_frames_per_video: Maximum frames per video
        device: Device for inference
        save_results: Whether to save results to file
        
    Returns:
        Tuple of (detector, results, evaluations)
    """
    # Load configuration
    config = load_configuration(config_path)
    
    # Use config paths if not provided
    if video_dir is None:
        video_dir = config.paths.video_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    print("Initializing conveyor belt speed detector...")
    detector = ConveyorBeltSpeedDetector(
        config=config,
        device=device,
        enable_optical_flow=OPTICAL_FLOW_AVAILABLE,
        enable_feature_matching=FEATURE_MATCHING_AVAILABLE,
        enable_fusion=FUSION_AVAILABLE,
        verbose=True
    )
    
    # Process videos
    print(f"\nProcessing videos from: {video_dir}")
    results = detector.process_video_directory(
        video_dir,
        max_videos=max_videos,
        max_frames_per_video=max_frames_per_video
    )
    
    # Evaluate results
    print("\nEvaluating results...")
    evaluations = detector.evaluate_results(results)
    
    # Print summary
    detector.print_evaluation_summary(evaluations)
    
    # Save results if requested
    if save_results:
        output_path = os.path.join(output_dir, "detection_results.json")
        detector.save_results(results, output_path, evaluations)
    
    return detector, results, evaluations


def main():
    """Main entry point for the detection system."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Conveyor Belt Speed Detection via Optical Flow and Feature Matching Fusion"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='./config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--video_dir', '-v',
        type=str,
        default=None,
        help='Directory containing video files'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max_videos', '-n',
        type=int,
        default=None,
        help='Maximum number of videos to process'
    )
    parser.add_argument(
        '--max_frames', '-f',
        type=int,
        default=None,
        help='Maximum frames per video'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save results to file'
    )
    
    args = parser.parse_args()
    
    # Run detection
    detector, results, evaluations = run_detection(
        video_dir=args.video_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames,
        device=args.device,
        save_results=not args.no_save
    )
    
    # Print final statistics
    stats = detector.get_statistics()
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"Total videos processed: {stats['total_videos']}")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Valid frames: {stats['valid_frames']}")
    print(f"Failed frames: {stats['failed_frames']}")
    if stats['total_frames'] > 0:
        print(f"Success rate: {stats['success_rate']*100:.1f}%")
    if stats['valid_frames'] > 0:
        print(f"Avg processing time: {stats['avg_processing_time_ms']:.1f} ms/frame")
    print("=" * 70)


if __name__ == "__main__":
    main()
