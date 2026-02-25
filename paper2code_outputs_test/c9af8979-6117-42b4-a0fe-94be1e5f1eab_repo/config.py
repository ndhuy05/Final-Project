"""
config.py

Central configuration management module for the conveyor belt speed detection system.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module encapsulates all hyperparameters, path configurations, and model settings
referenced throughout the codebase.

Author: Based on paper methodology
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from pathlib import Path


@dataclass
class DatasetConfig:
    """Dataset configuration parameters.
    
    Attributes:
        resolution: Video resolution (width, height) in pixels
        frame_rate: Video frame rate in frames per second
        lab_illumination: Laboratory illumination conditions in lux
        lab_belt_speeds: Laboratory belt speed settings in m/s
        mine_illumination: Coal mine illumination range in lux
        mine_belt_speeds: Coal mine belt speed range in m/s
    """
    resolution: Tuple[int, int] = (1920, 1080)
    frame_rate: int = 25
    lab_illumination: List[int] = field(default_factory=lambda: [50, 1000])
    lab_belt_speeds: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.5])
    mine_illumination: List[int] = field(default_factory=lambda: [10, 500])
    mine_belt_speeds: List[float] = field(default_factory=lambda: [1.0, 5.0])
    
    @property
    def width(self) -> int:
        """Get resolution width."""
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        """Get resolution height."""
        return self.resolution[1]
    
    @property
    def frame_interval(self) -> float:
        """Get time interval between frames in seconds."""
        return 1.0 / self.frame_rate


@dataclass
class TrainingConfig:
    """Training configuration for RAFT-SEnet model.
    
    Based on Section 4.2 of the paper:
    - OS: WIN10
    - GPU: RTX3090
    - CPU: i9-14900k
    - Optimizer: Adam
    - Initial learning rate: 0.0001
    - Weight decay: 1e-4
    - Momentum: 0.9
    - Batch size: 4
    - Training rounds: 500
    - Luminosity loss weight: 1.0
    - Smoothing loss weight: 0.01
    """
    os: str = "WIN10"
    gpu: str = "RTX3090"
    cpu: str = "i9-14900k"
    optimizer: str = "Adam"
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001  # 1e-4
    momentum: float = 0.9
    batch_size: int = 4
    num_epochs: int = 500
    luminosity_loss_weight: float = 1.0
    smoothing_loss_weight: float = 0.01
    
    def __post_init__(self):
        """Validate training parameters."""
        if self.learning_rate <= 0 or self.learning_rate >= 1:
            raise ValueError(f"Learning rate must be in (0, 1), got {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {self.num_epochs}")
        if self.luminosity_loss_weight < 0 or self.smoothing_loss_weight < 0:
            raise ValueError("Loss weights must be non-negative")


@dataclass
class OpticalFlowConfig:
    """Optical flow (RAFT-SEnet) model configuration.
    
    Attributes:
        input_height: Input image height for the model
        input_width: Input image width for the model
        use_senet: Whether to use SENet attention module
        senet_reduction: SENet reduction ratio for channel compression
    """
    input_height: int = 448
    input_width: int = 256
    use_senet: bool = True
    senet_reduction: int = 16
    
    @property
    def input_size(self) -> Tuple[int, int]:
        """Get input size as (height, width)."""
        return (self.input_height, self.input_width)
    
    def __post_init__(self):
        """Validate optical flow parameters."""
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("Input dimensions must be positive")
        if self.senet_reduction <= 0:
            raise ValueError(f"SENet reduction must be positive, got {self.senet_reduction}")


@dataclass
class FeatureMatchingConfig:
    """Feature matching (Harris-BRIEF-RANSAC) configuration.
    
    Based on Section 3.2 of the paper:
    - Harris k parameter: 0.04
    - Harris block size: 2
    - BRIEF patch size: 31
    - RANSAC threshold: 5.0 pixels
    - RANSAC iterations: 2000
    
    Attributes:
        harris_block_size: Block size for Harris corner detection
        harris_ksize: Aperture parameter for Sobel operator
        harris_k: Harris detector free parameter (k)
        harris_threshold_ratio: Ratio for thresholding Harris response
        brief_patch_size: Patch size for BRIEF descriptor
        ransac_threshold: RANSAC inlier threshold in pixels
        ransac_max_iters: Maximum RANSAC iterations
    """
    harris_block_size: int = 2
    harris_ksize: int = 3
    harris_k: float = 0.04
    harris_threshold_ratio: float = 0.01
    brief_patch_size: int = 31
    ransac_threshold: float = 5.0
    ransac_max_iters: int = 2000
    
    def __post_init__(self):
        """Validate feature matching parameters."""
        if self.harris_block_size <= 0:
            raise ValueError(f"Harris block size must be positive, got {self.harris_block_size}")
        if self.harris_k < 0 or self.harris_k >= 1:
            raise ValueError(f"Harris k must be in [0, 1), got {self.harris_k}")
        if self.harris_threshold_ratio <= 0 or self.harris_threshold_ratio >= 1:
            raise ValueError(f"Harris threshold ratio must be in (0, 1), got {self.harris_threshold_ratio}")
        if self.brief_patch_size <= 0:
            raise ValueError(f"BRIEF patch size must be positive, got {self.brief_patch_size}")
        if self.ransac_threshold <= 0:
            raise ValueError(f"RANSAC threshold must be positive, got {self.ransac_threshold}")
        if self.ransac_max_iters <= 0:
            raise ValueError(f"RANSAC max iterations must be positive, got {self.ransac_max_iters}")


@dataclass
class ModelConfig:
    """Complete model configuration container."""
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    feature_matching: FeatureMatchingConfig = field(default_factory=FeatureMatchingConfig)


@dataclass
class FusionConfig:
    """Bayesian fusion configuration.
    
    Based on Section 3.3 of the paper:
    The weight calculation formula: w = aL + bC
    where L is brightness and C is contrast.
    
    Note: Parameters a and b are "adjusted according to the actual field situation"
    (quoted from paper). Default values are placeholders requiring tuning.
    
    Attributes:
        weight_a: Weight parameter a for brightness (Equation 8)
        weight_b: Weight parameter b for contrast (Equation 8)
    """
    weight_a: float = 0.5
    weight_b: float = 0.5
    
    def __post_init__(self):
        """Validate fusion parameters."""
        if self.weight_a < 0 or self.weight_a > 1:
            raise ValueError(f"Weight a must be in [0, 1], got {self.weight_a}")
        if self.weight_b < 0 or self.weight_b > 1:
            raise ValueError(f"Weight b must be in [0, 1], got {self.weight_b}")


@dataclass
class EvaluationConfig:
    """Evaluation metrics and baseline configuration.
    
    Metrics based on Section 4.2 of the paper:
    - MAE: Mean Absolute Error (Equation 11)
    - RMSE: Root Mean Square Error (Equation 12)
    - Error percentage: Relative error in percentage
    
    Attributes:
        metrics: List of metric names to compute
        optical_flow_baselines: Baseline optical flow methods for comparison
        feature_matching_baselines: Baseline feature matching methods for comparison
    """
    metrics: List[str] = field(default_factory=lambda: ["MAE", "RMSE", "error_percentage"])
    optical_flow_baselines: List[str] = field(default_factory=lambda: ["TV-L1", "FlowNet2", "RAFT"])
    feature_matching_baselines: List[str] = field(default_factory=lambda: ["SIFT", "FAST", "ORB"])


@dataclass
class PathsConfig:
    """File system paths configuration.
    
    Attributes:
        data_dir: Root data directory
        video_dir: Directory containing input videos
        output_dir: Directory for output results
        model_checkpoint_dir: Directory for model checkpoints
        calibration_file: Path to camera calibration file
    """
    data_dir: str = "./data"
    video_dir: str = "./data/videos"
    output_dir: str = "./output"
    model_checkpoint_dir: str = "./checkpoints"
    calibration_file: str = "./calibration.yaml"
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir = str(Path(self.data_dir).resolve())
        self.video_dir = str(Path(self.video_dir).resolve())
        self.output_dir = str(Path(self.output_dir).resolve())
        self.model_checkpoint_dir = str(Path(self.model_checkpoint_dir).resolve())
        self.calibration_file = str(Path(self.calibration_file).resolve())
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.video_dir, self.output_dir, self.model_checkpoint_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    @property
    def checkpoint_dir(self) -> str:
        """Alias for model_checkpoint_dir for convenience."""
        return self.model_checkpoint_dir


@dataclass
class Config:
    """Main configuration container for the entire system.
    
    This class aggregates all configuration sections and provides a unified
    interface for accessing hyperparameters throughout the codebase.
    
    Usage:
        cfg = Config()
        frame_rate = cfg.dataset.frame_rate
        learning_rate = cfg.training.learning_rate
        ransac_threshold = cfg.model.feature_matching.ransac_threshold
    """
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.
        
        Returns:
            Dictionary representation of all configuration values
        """
        return {
            'dataset': {
                'resolution': self.dataset.resolution,
                'frame_rate': self.dataset.frame_rate,
                'frame_interval': self.dataset.frame_interval,
                'lab_illumination': self.dataset.lab_illumination,
                'lab_belt_speeds': self.dataset.lab_belt_speeds,
                'mine_illumination': self.dataset.mine_illumination,
                'mine_belt_speeds': self.dataset.mine_belt_speeds,
            },
            'training': {
                'optimizer': self.training.optimizer,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'momentum': self.training.momentum,
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'luminosity_loss_weight': self.training.luminosity_loss_weight,
                'smoothing_loss_weight': self.training.smoothing_loss_weight,
            },
            'model': {
                'optical_flow': {
                    'input_height': self.model.optical_flow.input_height,
                    'input_width': self.model.optical_flow.input_width,
                    'use_senet': self.model.optical_flow.use_senet,
                    'senet_reduction': self.model.optical_flow.senet_reduction,
                },
                'feature_matching': {
                    'harris_block_size': self.model.feature_matching.harris_block_size,
                    'harris_ksize': self.model.feature_matching.harris_ksize,
                    'harris_k': self.model.feature_matching.harris_k,
                    'harris_threshold_ratio': self.model.feature_matching.harris_threshold_ratio,
                    'brief_patch_size': self.model.feature_matching.brief_patch_size,
                    'ransac_threshold': self.model.feature_matching.ransac_threshold,
                    'ransac_max_iters': self.model.feature_matching.ransac_max_iters,
                },
            },
            'fusion': {
                'weight_a': self.fusion.weight_a,
                'weight_b': self.fusion.weight_b,
            },
            'evaluation': {
                'metrics': self.evaluation.metrics,
                'optical_flow_baselines': self.evaluation.optical_flow_baselines,
                'feature_matching_baselines': self.evaluation.feature_matching_baselines,
            },
            'paths': {
                'data_dir': self.paths.data_dir,
                'video_dir': self.paths.video_dir,
                'output_dir': self.paths.output_dir,
                'model_checkpoint_dir': self.paths.model_checkpoint_dir,
                'calibration_file': self.paths.calibration_file,
            },
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Create Config from YAML file.
        
        Note: This is a placeholder. Full YAML loading would require
        pyyaml or similar library.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config instance with values from YAML
        """
        # Placeholder - actual implementation would parse YAML
        # For now, return default configuration
        import warnings
        warnings.warn("YAML loading not fully implemented, using default config")
        return cls()
    
    def validate(self) -> bool:
        """Validate all configuration parameters.
        
        Returns:
            True if all validations pass
            
        Raises:
            ValueError: If any validation fails
        """
        # Validate dataset
        if self.dataset.frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive: {self.dataset.frame_rate}")
        
        # Validate training
        self.training.__post_init__()
        
        # Validate model
        self.model.optical_flow.__post_init__()
        self.model.feature_matching.__post_init__()
        
        # Validate fusion
        self.fusion.__post_init__()
        
        return True


# Global configuration instance
_default_config: Config = None


def get_config() -> Config:
    """Get the global configuration instance (singleton pattern).
    
    Returns:
        The global Config instance
    """
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def reset_config() -> Config:
    """Reset the global configuration to default values.
    
    Returns:
        New Config instance with default values
    """
    global _default_config
    _default_config = Config()
    return _default_config


def set_config(config: Config) -> None:
    """Set the global configuration instance.
    
    Args:
        config: Config instance to use as global configuration
    """
    global _default_config
    _default_config = config


# Convenience accessors for commonly used parameters
def get_frame_rate() -> int:
    """Get the configured frame rate."""
    return get_config().dataset.frame_rate


def get_frame_interval() -> float:
    """Get the time interval between frames."""
    return get_config().dataset.frame_interval


def get_calibration_path() -> str:
    """Get the camera calibration file path."""
    return get_config().paths.calibration_file


def get_output_dir() -> str:
    """Get the output directory path."""
    return get_config().paths.output_dir


def get_checkpoint_dir() -> str:
    """Get the model checkpoint directory path."""
    return get_config().paths.model_checkpoint_dir


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    cfg = Config()
    
    # Print configuration summary
    print("=" * 60)
    print("Conveyor Belt Speed Detection Configuration")
    print("=" * 60)
    
    print("\n[Dataset]")
    print(f"  Resolution: {cfg.dataset.width}x{cfg.dataset.height}")
    print(f"  Frame Rate: {cfg.dataset.frame_rate} fps")
    print(f"  Frame Interval: {cfg.dataset.frame_interval:.4f} s")
    print(f"  Lab Illumination: {cfg.dataset.lab_illumination} lux")
    print(f"  Lab Belt Speeds: {cfg.dataset.lab_belt_speeds} m/s")
    
    print("\n[Training]")
    print(f"  Optimizer: {cfg.training.optimizer}")
    print(f"  Learning Rate: {cfg.training.learning_rate}")
    print(f"  Weight Decay: {cfg.training.weight_decay}")
    print(f"  Batch Size: {cfg.training.batch_size}")
    print(f"  Epochs: {cfg.training.num_epochs}")
    
    print("\n[Model - Optical Flow]")
    print(f"  Input Size: {cfg.model.optical_flow.input_size}")
    print(f"  Use SENet: {cfg.model.optical_flow.use_senet}")
    print(f"  SENet Reduction: {cfg.model.optical_flow.senet_reduction}")
    
    print("\n[Model - Feature Matching]")
    print(f"  Harris k: {cfg.model.feature_matching.harris_k}")
    print(f"  Harris Block Size: {cfg.model.feature_matching.harris_block_size}")
    print(f"  BRIEF Patch Size: {cfg.model.feature_matching.brief_patch_size}")
    print(f"  RANSAC Threshold: {cfg.model.feature_matching.ransac_threshold} px")
    print(f"  RANSAC Max Iterations: {cfg.model.feature_matching.ransac_max_iters}")
    
    print("\n[Fusion]")
    print(f"  Weight a (brightness): {cfg.fusion.weight_a}")
    print(f"  Weight b (contrast): {cfg.fusion.weight_b}")
    
    print("\n[Evaluation]")
    print(f"  Metrics: {cfg.evaluation.metrics}")
    print(f"  Optical Flow Baselines: {cfg.evaluation.optical_flow_baselines}")
    print(f"  Feature Matching Baselines: {cfg.evaluation.feature_matching_baselines}")
    
    print("\n[Paths]")
    print(f"  Data Directory: {cfg.paths.data_dir}")
    print(f"  Video Directory: {cfg.paths.video_dir}")
    print(f"  Output Directory: {cfg.paths.output_dir}")
    print(f"  Checkpoint Directory: {cfg.paths.model_checkpoint_dir}")
    print(f"  Calibration File: {cfg.paths.calibration_file}")
    
    # Validate configuration
    print("\n[Validation]")
    try:
        cfg.validate()
        print("  ✓ Configuration is valid")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")
    
    # Test dictionary export
    print("\n[Dictionary Export]")
    config_dict = cfg.to_dict()
    print(f"  ✓ Exported {len(config_dict)} top-level sections")
    
    print("\n" + "=" * 60)
    print("Configuration loaded successfully!")
    print("=" * 60)
