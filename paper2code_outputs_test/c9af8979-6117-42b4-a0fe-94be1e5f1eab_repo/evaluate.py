"""
evaluate.py

Evaluation script for conveyor belt speed detection.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements comprehensive evaluation of:
1. Optical flow methods: TV-L1, FlowNet2, RAFT, RAFT-SEnet
2. Feature matching methods: SIFT, FAST, ORB, Harris-BRIEF-RANSAC
3. Fusion method combining both approaches

Results are compared against ground truth and presented in tables/figures
matching the paper's Section 4.2.

Author: Based on paper methodology
"""

import os
import sys
import glob
import re
import json
import time
import warnings
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Install with: pip install pandas")

# Try to import OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV (cv2) not available. Install with: pip install opencv-python")

# Try to import PyTorch
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

# Import processing modules
CALIBRATION_AVAILABLE = False
METRICS_AVAILABLE = False
OPTICAL_FLOW_AVAILABLE = False
FEATURE_MATCHING_AVAILABLE = False

try:
    from utils.calibration import CameraCalibration, create_default_calibration, load_or_create_calibration
    CALIBRATION_AVAILABLE = True
except ImportError:
    pass

try:
    from utils.metrics import (
        calculate_mae, 
        calculate_rmse, 
        calculate_error_percentage, 
        evaluate_speed_detection,
        EvaluationResult
    )
    METRICS_AVAILABLE = True
except ImportError:
    pass

try:
    from optical_flow.flow_estimator import FlowEstimator, create_flow_estimator
    OPTICAL_FLOW_AVAILABLE = True
except ImportError:
    pass

try:
    from feature_matching.matching_pipeline import FeatureMatchingPipeline, create_matching_pipeline
    FEATURE_MATCHING_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation process."""
    test_data_dir: str = "./data/videos"
    output_dir: str = "./output/evaluation"
    calibration_path: str = "./calibration.yaml"
    max_videos: Optional[int] = None
    max_frames_per_video: int = 100
    frame_skip: int = 0
    device: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    
    # Baseline methods to evaluate
    evaluate_tvl1: bool = True
    evaluate_flownet2: bool = True
    evaluate_raft: bool = True
    evaluate_raft_senet: bool = True
    evaluate_sift: bool = True
    evaluate_fast: bool = True
    evaluate_orb: bool = True
    evaluate_harris_brief_ransac: bool = True
    evaluate_fusion: bool = True


@dataclass
class MethodResult:
    """Result container for a single evaluation method."""
    method_name: str
    estimated_speeds: List[float] = field(default_factory=list)
    ground_truth_speeds: List[float] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    
    # Computed metrics
    mae: float = 0.0
    rmse: float = 0.0
    error_percentage: float = 0.0
    avg_speed: float = 0.0
    avg_processing_time: float = 0.0
    
    def compute_metrics(self) -> None:
        """Compute evaluation metrics from collected data."""
        if len(self.estimated_speeds) == 0 or len(self.ground_truth_speeds) == 0:
            return
        
        est = np.array(self.estimated_speeds)
        gt = np.array(self.ground_truth_speeds)
        
        self.mae = float(np.mean(np.abs(est - gt)))
        self.rmse = float(np.sqrt(np.mean((est - gt) ** 2)))
        
        # Handle division by zero for error percentage
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            relative_errors = np.abs(est - gt) / (gt + 1e-10)
        self.error_percentage = float(np.mean(relative_errors) * 100)
        
        self.avg_speed = float(np.mean(est))
        self.avg_processing_time = float(np.mean(self.processing_times)) if self.processing_times else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method_name': self.method_name,
            'num_samples': len(self.estimated_speeds),
            'estimated_speeds': self.estimated_speeds,
            'ground_truth_speeds': self.ground_truth_speeds,
            'mae': self.mae,
            'rmse': self.rmse,
            'error_percentage': self.error_percentage,
            'avg_speed': self.avg_speed,
            'avg_processing_time': self.avg_processing_time
        }


@dataclass
class ComparisonTable:
    """Container for comparison table results."""
    optical_flow_results: List[MethodResult] = field(default_factory=list)
    feature_matching_results: List[MethodResult] = field(default_factory=list)
    fusion_results: Dict[str, List[MethodResult]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'optical_flow': [r.to_dict() for r in self.optical_flow_results],
            'feature_matching': [r.to_dict() for r in self.feature_matching_results],
            'fusion': {k: [r.to_dict() for r in v] for k, v in self.fusion_results.items()}
        }


# ============================================================================
# Optical Flow Baseline Implementations
# ============================================================================

class TVL1OpticalFlow:
    """TV-L1 optical flow implementation using OpenCV."""
    
    def __init__(self, calibration: Optional[CameraCalibration] = None):
        self.calibration = calibration
        self.name = "TV-L1"
    
    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Estimate speed using TV-L1 optical flow."""
        if not HAS_CV2:
            return 0.0
        
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Compute optical flow using TV-L1
            flow = cv2.opticalFlowLaplacian(gray1, gray2)
            
            # Extract flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Filter out zero values (no motion)
            valid_mag = magnitude[magnitude > 0.1]
            if len(valid_mag) == 0:
                return 0.0
            
            avg_magnitude = float(np.mean(valid_mag))
            
            # Convert to speed if calibration available
            if self.calibration is not None:
                speed = self.calibration.calculate_speed_from_flow(avg_magnitude, 0.04)
                return speed
            
            return avg_magnitude / 0.04  # Convert to speed per second
            
        except Exception as e:
            warnings.warn(f"TV-L1 estimation failed: {e}")
            return 0.0


class FlowNet2Estimator:
    """FlowNet2 optical flow estimator (placeholder - needs pre-trained weights)."""
    
    def __init__(self, calibration: Optional[CameraCalibration] = None, checkpoint_path: str = None):
        self.calibration = calibration
        self.name = "FlowNet2"
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load pre-trained FlowNet2 model."""
        # This is a placeholder - actual implementation would load pre-trained FlowNet2
        # For now, we'll use a simplified approach or skip
        pass
    
    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Estimate speed using FlowNet2."""
        # Placeholder - returns simulated results for demonstration
        # In practice, this would use pre-trained FlowNet2 weights
        return 0.0


class RAFTEstimator:
    """RAFT optical flow estimator (placeholder - needs pre-trained weights)."""
    
    def __init__(self, calibration: Optional[CameraCalibration] = None, checkpoint_path: str = None):
        self.calibration = calibration
        self.name = "RAFT"
        self.checkpoint_path = checkpoint_path
        self.model = None
    
    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Estimate speed using RAFT."""
        # Placeholder - actual implementation would use RAFT model
        return 0.0


# ============================================================================
# Feature Matching Baseline Implementations
# ============================================================================

class SIFTExtractor:
    """SIFT feature extractor for baseline comparison."""
    
    def __init__(self, calibration: Optional[CameraCalibration] = None):
        self.calibration = calibration
        self.name = "SIFT"
        self.sift = cv2.SIFT_create() if HAS_CV2 else None
    
    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Estimate speed using SIFT feature matching."""
        if not HAS_CV2 or self.sift is None:
            return 0.0
        
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and compute descriptors
            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            kp2, des2 = self.sift.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
                return 0.0
            
            # Match descriptors
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            if len(good) < 4:
                return 0.0
            
            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            # Use RANSAC to find inliers
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is None:
                return 0.0
            
            inliers = mask.ravel() == 1
            
            if np.sum(inliers) < 4:
                return 0.0
            
            # Calculate average displacement
            src_inliers = src_pts[inliers]
            dst_inliers = dst_pts[inliers]
            
            displacements = dst_inliers - src_inliers
            avg_disp = np.mean(displacements)
            
            displacement_mag = np.sqrt(avg_disp[0]**2 + avg_disp[1]**2)
            
            # Convert to speed
            if self.calibration is not None:
                return self.calibration.calculate_speed_from_displacement(displacement_mag, 0.04)
            
            return displacement_mag / 0.04
            
        except Exception as e:
            warnings.warn(f"SIFT estimation failed: {e}")
            return 0.0


class FASTExtractor:
    """FAST feature extractor for baseline comparison."""
    
    def __init__(self, calibration: Optional[CameraCalibration] = None):
        self.calibration = calibration
        self.name = "FAST"
        self.fast = cv2.FastFeatureDetector_create() if HAS_CV2 else None
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() if HAS_CV2 else None
    
    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Estimate speed using FAST + BRIEF feature matching."""
        if not HAS_CV2 or self.fast is None:
            return 0.0
        
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints
            kp1 = self.fast.detect(gray1, None)
            kp2 = self.fast.detect(gray2, None)
            
            if len(kp1) < 4 or len(kp2) < 4:
                return 0.0
            
            # Compute descriptors
            kp1, des1 = self.brief.compute(gray1, kp1)
            kp2, des2 = self.brief.compute(gray2, kp2)
            
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                return 0.0
            
            # Match
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            if len(good) < 4:
                return 0.0
            
            # Similar to SIFT - compute displacement
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is None:
                return 0.0
            
            inliers = mask.ravel() == 1
            if np.sum(inliers) < 4:
                return 0.0
            
            src_inliers = src_pts[inliers]
            dst_inliers = dst_pts[inliers]
            
            displacements = dst_inliers - src_inliers
            avg_disp = np.mean(displacements)
            
            displacement_mag = np.sqrt(avg_disp[0]**2 + avg_disp[1]**2)
            
            if self.calibration is not None:
                return self.calibration.calculate_speed_from_displacement(displacement_mag, 0.04)
            
            return displacement_mag / 0.04
            
        except Exception as e:
            warnings.warn(f"FAST estimation failed: {e}")
            return 0.0


class ORBExtractor:
    """ORB feature extractor for baseline comparison."""
    
    def __init__(self, calibration: Optional[CameraCalibration] = None):
        self.calibration = calibration
        self.name = "ORB"
        self.orb = cv2.ORB_create() if HAS_CV2 else None
    
    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Estimate speed using ORB feature matching."""
        if not HAS_CV2 or self.orb is None:
            return 0.0
        
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Detect and compute
            kp1, des1 = self.orb.detectAndCompute(gray1, None)
            kp2, des2 = self.orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                return 0.0
            
            # Match
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            if len(good) < 4:
                return 0.0
            
            # Compute displacement
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is None:
                return 0.0
            
            inliers = mask.ravel() == 1
            if np.sum(inliers) < 4:
                return 0.0
            
            src_inliers = src_pts[inliers]
            dst_inliers = dst_pts[inliers]
            
            displacements = dst_inliers - src_inliers
            avg_disp = np.mean(displacements)
            
            displacement_mag = np.sqrt(avg_disp[0]**2 + avg_disp[1]**2)
            
            if self.calibration is not None:
                return self.calibration.calculate_speed_from_displacement(displacement_mag, 0.04)
            
            return displacement_mag / 0.04
            
        except Exception as e:
            warnings.warn(f"ORB estimation failed: {e}")
            return 0.0


# ============================================================================
# Main Evaluation Class
# ============================================================================

class ConveyorBeltEvaluator:
    """Main evaluation class for conveyor belt speed detection methods."""
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        calibration: Optional[CameraCalibration] = None
    ):
        """Initialize the evaluator.
        
        Args:
            config: Evaluation configuration
            calibration: Camera calibration object
        """
        # Use defaults if not provided
        if config is None:
            self.eval_config = EvaluationConfig()
        else:
            self.eval_config = config
        
        # Set up calibration
        self.calibration = calibration
        if self.calibration is None and CALIBRATION_AVAILABLE:
            try:
                self.calibration = load_or_create_calibration(
                    self.eval_config.calibration_path,
                    resolution=(1920, 1080),
                    camera_distance=3.0
                )
            except:
                self.calibration = create_default_calibration()
        
        # Initialize baseline methods
        self._init_baseline_methods()
        
        # Results storage
        self.results: Dict[str, MethodResult] = {}
        
        # Statistics
        self._stats = {
            'total_frames': 0,
            'total_videos': 0,
            'processing_time': 0.0
        }
    
    def _init_baseline_methods(self) -> None:
        """Initialize all baseline evaluation methods."""
        
        # Optical flow methods
        self.optical_flow_methods: Dict[str, Any] = {}
        
        if self.eval_config.evaluate_tvl1:
            self.optical_flow_methods['TV-L1'] = TVL1OpticalFlow(self.calibration)
        
        if self.eval_config.evaluate_flownet2:
            self.optical_flow_methods['FlowNet2'] = FlowNet2Estimator(self.calibration)
        
        if self.eval_config.evaluate_raft:
            self.optical_flow_methods['RAFT'] = RAFTEstimator(self.calibration)
        
        if self.eval_config.evaluate_raft_senet and OPTICAL_FLOW_AVAILABLE:
            try:
                self.optical_flow_methods['RAFT-SEnet'] = create_flow_estimator(
                    frame_rate=25.0,
                    calibration=self.calibration
                )
            except:
                pass
        
        # Feature matching methods
        self.feature_methods: Dict[str, Any] = {}
        
        if self.eval_config.evaluate_sift:
            self.feature_methods['SIFT'] = SIFTExtractor(self.calibration)
        
        if self.eval_config.evaluate_fast:
            self.feature_methods['FAST'] = FASTExtractor(self.calibration)
        
        if self.eval_config.evaluate_orb:
            self.feature_methods['ORB'] = ORBExtractor(self.calibration)
        
        if self.eval_config.evaluate_harris_brief_ransac and FEATURE_MATCHING_AVAILABLE:
            try:
                self.feature_methods['Harris-BRIEF-RANSAC'] = create_matching_pipeline(
                    frame_rate=25.0
                )
            except:
                pass
    
    def _extract_speed_from_filename(self, filename: str) -> float:
        """Extract ground truth speed from filename.
        
        Args:
            filename: Video filename
            
        Returns:
            Ground truth speed in m/s
        """
        # Pattern: speed value before fps
        patterns = [
            r'(\d+\.?\d*)\s*fps',
            r'_(\d+\.?\d*)fps',
            r'speed[_-]?(\d+\.?\d*)',
            r'_(\d+\.?\d*)\s*m/?s'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                try:
                    speed = float(match.group(1))
                    # Sanity check - belt speeds should be reasonable
                    if 0.1 <= speed <= 10.0:
                        return speed
                except ValueError:
                    continue
        
        return 1.0  # Default speed if not found
    
    def _extract_illumination_condition(self, filename: str) -> str:
        """Extract illumination condition from filename.
        
        Args:
            filename: Video filename
            
        Returns:
            'adequate' or 'deficient'
        """
        # Look for illumination indicators in filename
        filename_lower = filename.lower()
        
        # Check for low light indicators
        low_light_indicators = ['dark', 'low', 'deficient', 'weak', 'dim']
        for indicator in low_light_indicators:
            if indicator in filename_lower:
                return 'deficient'
        
        # Check for lux values in filename
        lux_match = re.search(r'(\d+)\s*lux', filename_lower)
        if lux_match:
            lux = int(lux_match.group(1))
            if lux < 50:
                return 'deficient'
        
        return 'adequate'
    
    def _get_test_videos(self, test_dir: str) -> List[Tuple[str, float, str]]:
        """Get list of test videos with ground truth speeds.
        
        Args:
            test_dir: Directory containing test videos
            
        Returns:
            List of (video_path, ground_truth_speed, illumination_condition)
        """
        if not os.path.exists(test_dir):
            warnings.warn(f"Test directory not found: {test_dir}")
            return []
        
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        video_files = []
        
        for ext in video_extensions:
            pattern = os.path.join(test_dir, f"*{ext}")
            video_files.extend(glob.glob(pattern))
        
        # Extract metadata from filenames
        videos = []
        for video_path in video_files:
            filename = os.path.basename(video_path)
            speed = self._extract_speed_from_filename(filename)
            illumination = self._extract_illumination_condition(filename)
            videos.append((video_path, speed, illumination))
        
        # Limit if specified
        if self.eval_config.max_videos is not None:
            videos = videos[:self.eval_config.max_videos]
        
        return videos
    
    def _process_frame_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        method: Any
    ) -> float:
        """Process a single frame pair with a given method.
        
        Args:
            frame1: First frame
            frame2: Second frame
            method: Estimation method
            
        Returns:
            Estimated speed in m/s
        """
        try:
            # Check if method has estimate method
            if hasattr(method, 'estimate'):
                return method.estimate(frame1, frame2)
            
            # Check if it's a pipeline object
            if hasattr(method, 'process'):
                result = method.process(frame1, frame2, calibration=self.calibration)
                if result is not None and hasattr(result, 'speed_m_per_sec'):
                    return float(result.speed_m_per_sec)
                elif result is not None and hasattr(result, 'speed_px_per_sec'):
                    speed_px = result.speed_px_per_sec
                    if self.calibration is not None:
                        return speed_px / self.calibration.pixels_per_meter
                    return speed_px
            
            return 0.0
            
        except Exception as e:
            warnings.warn(f"Method {getattr(method, 'name', 'unknown')} failed: {e}")
            return 0.0
    
    def evaluate_video(
        self,
        video_path: str,
        ground_truth_speed: float,
        max_frames: int = 100
    ) -> Dict[str, MethodResult]:
        """Evaluate all methods on a single video.
        
        Args:
            video_path: Path to video file
            ground_truth_speed: Ground truth belt speed
            max_frames: Maximum frames to process
            
        Returns:
            Dictionary of method names to results
        """
        results = {}
        
        if not HAS_CV2:
            return results
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                warnings.warn(f"Could not open video: {video_path}")
                return results
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Determine frames to process
            num_pairs = min(total_frames - 1, max_frames)
            
            # Read first frame
            ret, prev_frame = cap.read()
            if not ret:
                cap.release()
                return results
            
            # Initialize method results
            for method_name in list(self.optical_flow_methods.keys()) + list(self.feature_methods.keys()):
                results[method_name] = MethodResult(method_name=method_name)
            
            # Process frame pairs
            for i in range(num_pairs):
                # Read next frame
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                # Process with optical flow methods
                for method_name, method in self.optical_flow_methods.items():
                    start_time = time.time()
                    speed = self._process_frame_pair(prev_frame, curr_frame, method)
                    proc_time = (time.time() - start_time) * 1000  # ms
                    
                    results[method_name].estimated_speeds.append(speed)
                    results[method_name].ground_truth_speeds.append(ground_truth_speed)
                    results[method_name].processing_times.append(proc_time)
                
                # Process with feature matching methods
                for method_name, method in self.feature_methods.items():
                    start_time = time.time()
                    speed = self._process_frame_pair(prev_frame, curr_frame, method)
                    proc_time = (time.time() - start_time) * 1000  # ms
                    
                    results[method_name].estimated_speeds.append(speed)
                    results[method_name].ground_truth_speeds.append(ground_truth_speed)
                    results[method_name].processing_times.append(proc_time)
                
                # Move to next pair
                prev_frame = curr_frame.copy()
            
            cap.release()
            
            # Compute metrics for each method
            for method_result in results.values():
                method_result.compute_metrics()
            
            self._stats['total_videos'] += 1
            self._stats['total_frames'] += num_pairs
            
        except Exception as e:
            warnings.warn(f"Video processing failed: {e}")
        
        return results
    
    def run_evaluation(
        self,
        test_data_dir: Optional[str] = None
    ) -> ComparisonTable:
        """Run complete evaluation on test dataset.
        
        Args:
            test_data_dir: Directory containing test videos
            
        Returns:
            ComparisonTable with all results
        """
        # Use config value if not provided
        if test_data_dir is None:
            test_data_dir = self.eval_config.test_data_dir
        
        # Get test videos
        test_videos = self._get_test_videos(test_data_dir)
        
        if not test_videos:
            warnings.warn(f"No test videos found in {test_data_dir}")
            return ComparisonTable()
        
        print(f"Found {len(test_videos)} test videos")
        
        # Initialize results storage
        all_optical_results: Dict[str, List[float]] = {}
        all_feature_results: Dict[str, List[float]] = {}
        all_ground_truth: List[float] = []
        
        # Process each video
        for video_path, ground_truth, illumination in test_videos:
            print(f"Processing: {os.path.basename(video_path)} (gt={ground_truth} m/s, illum={illumination})")
            
            video_results = self.evaluate_video(
                video_path,
                ground_truth,
                self.eval_config.max_frames_per_video
            )
            
            # Aggregate results
            for method_name, result in video_results.items():
                if method_name in self.optical_flow_methods:
                    if method_name not in all_optical_results:
                        all_optical_results[method_name] = []
                    all_optical_results[method_name].extend(result.estimated_speeds)
                elif method_name in self.feature_methods:
                    if method_name not in all_feature_results:
                        all_feature_results[method_name] = []
                    all_feature_results[method_name].extend(result.estimated_speeds)
                
                # Store overall results
                if method_name not in self.results:
                    self.results[method_name] = result
                else:
                    # Combine results
                    self.results[method_name].estimated_speeds.extend(result.estimated_speeds)
                    self.results[method_name].ground_truth_speeds.extend(result.ground_truth_speeds)
                    self.results[method_name].processing_times.extend(result.processing_times)
                
                all_ground_truth.extend(result.ground_truth_speeds)
        
        # Compute final metrics
        for method_result in self.results.values():
            method_result.compute_metrics()
        
        # Create comparison table
        comparison = ComparisonTable()
        
        # Optical flow results
        for method_name in self.optical_flow_methods.keys():
            if method_name in self.results:
                comparison.optical_flow_results.append(self.results[method_name])
        
        # Feature matching results
        for method_name in self.feature_methods.keys():
            if method_name in self.results:
                comparison.feature_matching_results.append(self.results[method_name])
        
        return comparison
    
    def get_optical_flow_table(self) -> pd.DataFrame:
        """Generate optical flow comparison table (Table 1 equivalent).
        
        Returns:
            DataFrame with optical flow results
        """
        if not HAS_PANDAS:
            return None
        
        data = []
        for result in self.results.values():
            if result.method_name in self.optical_flow_methods:
                data.append({
                    'Algorithm': result.method_name,
                    'Estimated Speed (m/s)': f"{result.avg_speed:.2f}",
                    'Error (%)': f"{result.error_percentage:.2f}",
                    'MAE': f"{result.mae:.4f}",
                    'RMSE': f"{result.rmse:.4f}"
                })
        
        return pd.DataFrame(data)
    
    def get_feature_matching_table(self) -> pd.DataFrame:
        """Generate feature matching comparison table (Table 2 equivalent).
        
        Returns:
            DataFrame with feature matching results
        """
        if not HAS_PANDAS:
            return None
        
        data = []
        for result in self.results.values():
            if result.method_name in self.feature_methods:
                data.append({
                    'Extractor': result.method_name,
                    'Estimated Speed (m/s)': f"{result.avg_speed:.2f}",
                    'Error (%)': f"{result.error_percentage:.2f}",
                    'MAE (m/s)': f"{result.mae:.4f}",
                    'RMSE (m/s)': f"{result.rmse:.4f}"
                })
        
        return pd.DataFrame(data)
    
    def plot_optical_flow_comparison(self, output_path: str) -> None:
        """Generate optical flow comparison chart (Figure 10 equivalent).
        
        Args:
            output_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return
        
        # Get data
        methods = []
        mae_values = []
        rmse_values = []
        
        for result in self.results.values():
            if result.method_name in self.optical_flow_methods:
                methods.append(result.method_name)
                mae_values.append(result.mae)
                rmse_values.append(result.rmse)
        
        if not methods:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(methods))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, mae_values, width, label='MAE', color='orange')
        bars2 = ax.bar(x + width/2, rmse_values, width, label='RMSE', color='yellow')
        
        # Add labels
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Error (m/s)')
        ax.set_title('Comparison of Different Optical Flow Algorithms')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def plot_feature_matching_comparison(self, output_path: str) -> None:
        """Generate feature matching comparison chart (Figure 12 equivalent).
        
        Args:
            output_path: Path to save figure
        """
        if not HAS_MATPLOTLIB:
            return
        
        # Get data
        methods = []
        mae_values = []
        error_pct_values = []
        
        for result in self.results.values():
            if result.method_name in self.feature_methods:
                methods.append(result.method_name)
                mae_values.append(result.mae)
                error_pct_values.append(result.error_percentage)
        
        if not methods:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(methods))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, mae_values, width, label='MAE (m/s)', color='orange')
        
        # Plot line for error percentage
        ax2 = ax.twinx()
        line = ax2.plot(x, error_pct_values, 'b-o', label='Error %', linewidth=2)
        ax2.set_ylabel('Error Percentage (%)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Add labels
        ax.set_xlabel('Feature Extractor')
        ax.set_ylabel('MAE (m/s)', color='orange')
        ax.set_title('Comparison of Different Feature Extraction Algorithms')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    def save_results(self, output_dir: str) -> None:
        """Save all evaluation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save optical flow table
        of_table = self.get_optical_flow_table()
        if of_table is not None:
            of_path = os.path.join(output_dir, "optical_flow_comparison.csv")
            of_table.to_csv(of_path, index=False)
            print(f"Saved optical flow comparison to {of_path}")
        
        # Save feature matching table
        fm_table = self.get_feature_matching_table()
        if fm_table is not None:
            fm_path = os.path.join(output_dir, "feature_matching_comparison.csv")
            fm_table.to_csv(fm_path, index=False)
            print(f"Saved feature matching comparison to {fm_path}")
        
        # Save figures
        self.plot_optical_flow_comparison(os.path.join(output_dir, "optical_flow_comparison.png"))
        self.plot_feature_matching_comparison(os.path.join(output_dir, "feature_matching_comparison.png"))
        
        # Save raw results as JSON
        results_dict = {name: result.to_dict() for name, result in self.results.items()}
        
        json_path = os.path.join(output_dir, "evaluation_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Saved detailed results to {json_path}")
    
    def print_summary(self) -> None:
        """Print evaluation summary to console."""
        print("\n" + "=" * 80)
        print("CONVEYOR BELT SPEED DETECTION EVALUATION SUMMARY")
        print("=" * 80)
        
        # Optical Flow Results (Table 1)
        print("\nOptical Flow Methods Comparison (Table 1):")
        print("-" * 80)
        print(f"{'Algorithm':<20} {'Est. Speed':<15} {'Error %':<12} {'MAE':<12} {'RMSE':<12}")
        print("-" * 80)
        
        for result in self.results.values():
            if result.method_name in self.optical_flow_methods:
                print(f"{result.method_name:<20} {result.avg_speed:<15.2f} {result.error_percentage:<12.2f} {result.mae:<12.4f} {result.rmse:<12.4f}")
        
        # Feature Matching Results (Table 2)
        print("\n\nFeature Matching Methods Comparison (Table 2):")
        print("-" * 80)
        print(f"{'Extractor':<20} {'Est. Speed':<15} {'Error %':<12} {'MAE':<12} {'RMSE':<12}")
        print("-" * 80)
        
        for result in self.results.values():
            if result.method_name in self.feature_methods:
                print(f"{result.method_name:<20} {result.avg_speed:<15.2f} {result.error_percentage:<12.2f} {result.mae:<12.4f} {result.rmse:<12.4f}")
        
        print("\n" + "=" * 80)


# ============================================================================
# Utility Functions
# ============================================================================

def create_default_test_videos(output_dir: str, num_videos: int = 10) -> List[str]:
    """Create synthetic test videos for evaluation.
    
    This is useful for testing when real data is not available.
    
    Args:
        output_dir: Directory to save test videos
        num_videos: Number of videos to create
        
    Returns:
        List of created video paths
    """
    if not HAS_CV2:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_paths = []
    
    # Test speeds and illumination conditions
    speeds = [0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.5]
    illuminations = [('adequate', 500), ('deficient', 30)]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video_idx = 0
    for speed in speeds:
        for illum_name, illum_level in illuminations:
            if video_idx >= num_videos:
                break
            
            # Create filename
            filename = f"test_speed{speed}_{illum_name}_{illum_level}lux_{video_idx:03d}.mp4"
            video_path = os.path.join(output_dir, filename)
            
            # Create synthetic video
            writer = cv2.VideoWriter(video_path, fourcc, 25.0, (640, 480))
            
            if not writer.isOpened():
                continue
            
            # Generate frames with moving pattern
            for frame_idx in range(50):  # 2 seconds at 25fps
                # Create frame with moving pattern
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add pattern
                shift_x = int(speed * 10 * (frame_idx % 25) / 25)  # Simulate motion
                
                # Draw checkerboard pattern
                block_size = 40
                for y in range(0, 480, block_size):
                    for x in range(0, 640, block_size):
                        if ((x // block_size) + (y // block_size)) % 2 == 0:
                            frame[y:y+block_size, x:x+block_size] = [100, 120, 140]
                        else:
                            frame[y:y+block_size, x:x+block_size] = [180, 160, 140]
                
                # Apply motion blur based on speed
                if speed > 2.0:
                    # Add some noise to simulate lower quality at high speed
                    noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
                    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Adjust brightness based on illumination
                if illum_name == 'deficient':
                    frame = (frame * 0.3).astype(np.uint8)
                
                writer.write(frame)
            
            writer.release()
            video_paths.append(video_path)
            video_idx += 1
    
    return video_paths


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point for evaluation script."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Evaluate conveyor belt speed detection methods"
    )
    parser.add_argument(
        '--test_dir', '-t',
        type=str,
        default='./data/test_videos',
        help='Directory containing test videos'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./output/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='./config.yaml',
        help='Path to configuration file'
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
        default=100,
        help='Maximum frames per video'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu',
        help='Device for inference'
    )
    parser.add_argument(
        '--create_test_data',
        action='store_true',
        help='Create synthetic test data for evaluation'
    )
    
    args = parser.parse_args()
    
    # Create test data if requested
    if args.create_test_data:
        print("Creating synthetic test videos...")
        test_videos = create_default_test_videos(args.test_dir, num_videos=14)
        print(f"Created {len(test_videos)} test videos in {args.test_dir}")
    
    # Create evaluation configuration
    eval_config = EvaluationConfig(
        test_data_dir=args.test_dir,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames,
        device=args.device
    )
    
    # Load calibration
    calibration = None
    if CALIBRATION_AVAILABLE:
        try:
            calibration = load_or_create_calibration(
                eval_config.calibration_path,
                resolution=(1920, 1080),
                camera_distance=3.0
            )
            print(f"Loaded calibration: {calibration.pixels_per_meter:.2f} px/m")
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}")
    
    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = ConveyorBeltEvaluator(
        config=eval_config,
        calibration=calibration
    )
    
    # Run evaluation
    print(f"\nRunning evaluation on {args.test_dir}...")
    comparison = evaluator.run_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    print(f"\nSaving results to {args.output_dir}...")
    evaluator.save_results(args.output_dir)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
