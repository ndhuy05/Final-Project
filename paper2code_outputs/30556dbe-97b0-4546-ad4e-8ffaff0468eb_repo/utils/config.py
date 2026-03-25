"""
ECL-YOLOv11 Configuration Module

This module provides centralized configuration management for the ECL-YOLOv11
object detection framework. It handles loading, parsing, validation, and
accessing configuration parameters across all project modules.

Based on the paper: "Robust Object Detection in Adverse Weather Conditions: 
ECL-YOLOv11 for Automotive Vision Systems"

Author: ECL-YOLOv11 Reproduction Team
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # Hardware Configuration
    "hardware": {
        "cpu": "Intel Core i5-12400F",
        "gpu": "NVIDIA GeForce RTX 3060",
        "gpu_memory_gb": 12,
        "os": "Windows 11"
    },
    
    # Software Environment
    "software": {
        "python": "3.10.15",
        "pytorch": "2.5.0",
        "cuda": "12.1",
        "framework": "Ultralytics 8.3.9"
    },
    
    # Training Configuration
    "training": {
        "epochs": 600,
        "batch_size": 16,
        "image_size": 640,
        "early_stopping": {
            "patience": 20,
            "monitor": "val_mAP50",
            "mode": "max"
        },
        "learning_rate_schedule": {
            "type": "cosine_annealing",
            "lr0": 0.01,
            "lrf": 0.01
        },
        "optimizer": {
            "type": "SGD",
            "momentum": 0.937,
            "weight_decay": 0.0005
        },
        "regularization": {
            "label_smoothing": 0.1,
            "dropout": 0.0
        },
        "device": "cuda",
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1
    },
    
    # Model Configuration
    "model": {
        "name": "ECL-YOLOv11",
        "num_classes": 7,
        "classes": ["car", "person", "bus", "bicycle", "truck", "train", "motorcycle"],
        "input_channels": 3,
        "reg_max": 16,
        # Module-specific configurations
        "ce_module": {
            "enabled": True,
            "sobel_kernel_size": 3,
            "use_residual": True,
            "activation": "SiLU"
        },
        "aenet": {
            "enabled": True,
            "pyramid_channels": [256, 512, 1024],
            "rcm_stages": 2,
            "use_dif": True,
            "use_fbm": True
        },
        "ldhead": {
            "enabled": True,
            "shared_conv": True,
            "use_groupnorm": True,
            "groupnorm_groups": 32,
            "depthwise_separable": True
        }
    },
    
    # Data Configuration
    "data": {
        "format": "yolo",
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "data_root": "./data",
        "train_images": "./data/images/train",
        "val_images": "./data/images/val",
        "test_images": "./data/images/test",
        "train_labels": "./data/labels/train",
        "val_labels": "./data/labels/val",
        "test_labels": "./data/labels/test",
        "augmentation": {
            "weather": ["fog", "rain", "snow"],
            "geometric": ["flip", "rotate", "scale"],
            "photometric": ["brightness", "contrast", "saturation"]
        },
        "weather_params": {
            "fog_density": 0.5,
            "rain_intensity": 0.5,
            "snow_intensity": 0.5
        }
    },
    
    # Evaluation Configuration
    "evaluation": {
        "metrics": ["mAP@50", "mAP@50-95", "Precision", "Recall", "FPS"],
        "iou_thresholds": {
            "map50": 0.50,
            "map50_95_start": 0.50,
            "map50_95_end": 0.95,
            "steps": 10
        },
        "save_predictions": True,
        "visualize_results": True
    },
    
    # Inference Configuration
    "inference": {
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "input_size": 640,
        "half_precision": False,
        "batch_size": 1
    },
    
    # Logging and Checkpointing
    "logging": {
        "log_dir": "./runs",
        "save_dir": "./weights",
        "log_interval": 10,
        "save_interval": 50,
        "tensorboard": True,
        "verbose": True
    },
    
    # Loss Function Weights (based on typical YOLO training)
    "loss": {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5
    }
}


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class HardwareConfig:
    """Hardware configuration container"""
    cpu: str = "Intel Core i5-12400F"
    gpu: str = "NVIDIA GeForce RTX 3060"
    gpu_memory_gb: int = 12
    os: str = "Windows 11"


@dataclass
class TrainingConfig:
    """Training configuration container"""
    epochs: int = 600
    batch_size: int = 16
    image_size: int = 640
    device: str = "cuda"
    early_stopping_patience: int = 20
    early_stopping_monitor: str = "val_mAP50"
    learning_rate_schedule_type: str = "cosine_annealing"
    lr0: float = 0.01
    lrf: float = 0.01
    optimizer_type: str = "SGD"
    momentum: float = 0.937
    weight_decay: float = 0.0005
    label_smoothing: float = 0.1
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1


@dataclass
class ModelConfig:
    """Model configuration container"""
    name: str = "ECL-YOLOv11"
    num_classes: int = 7
    classes: List[str] = field(default_factory=lambda: ["car", "person", "bus", "bicycle", "truck", "train", "motorcycle"])
    input_channels: int = 3
    reg_max: int = 16


@dataclass
class DataConfig:
    """Data configuration container"""
    format: str = "yolo"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    data_root: str = "./data"
    augmentation_weather: List[str] = field(default_factory=lambda: ["fog", "rain", "snow"])


@dataclass
class EvaluationConfig:
    """Evaluation configuration container"""
    metrics: List[str] = field(default_factory=lambda: ["mAP@50", "mAP@50-95", "Precision", "Recall", "FPS"])
    map50_threshold: float = 0.50
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45


@dataclass
class InferenceConfig:
    """Inference configuration container"""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    input_size: int = 640
    half_precision: bool = False
    batch_size: int = 1


# =============================================================================
# Configuration Manager Class
# =============================================================================

class ConfigManager:
    """
    Singleton Configuration Manager for ECL-YOLOv11.
    
    This class handles loading, parsing, validation, and accessing 
    configuration parameters throughout the project.
    
    Attributes:
        _config (Dict[str, Any]): The loaded configuration dictionary
        _device (Optional[torch.device]): Cached device object
        _class_to_idx (Dict[str, int]): Mapping from class names to indices
        _idx_to_class (Dict[int, str]): Mapping from indices to class names
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Dict[str, Any] = {}
    _device: Optional[torch.device] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ConfigManager':
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the configuration manager"""
        if not ConfigManager._initialized:
            self._config = DEFAULT_CONFIG.copy()
            ConfigManager._initialized = True
    
    def load_from_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            # Try to find config.yaml in common locations
            possible_paths = [
                Path("./config.yaml"),
                Path("../config.yaml"),
                Path(__file__).parent.parent / "config.yaml",
                Path.cwd() / "config.yaml"
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                print(f"Warning: Config file not found at {config_path}, using defaults")
                return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
            
            if loaded_config:
                # Merge loaded config with defaults
                self._merge_config(loaded_config)
                print(f"Successfully loaded configuration from {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML config: {e}")
    
    def _merge_config(self, loaded_config: Dict[str, Any]) -> None:
        """
        Recursively merge loaded config with default config.
        
        Args:
            loaded_config: Configuration dictionary loaded from YAML
        """
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """Deep merge two dictionaries"""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        self._config = deep_merge(self._config, loaded_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., 'training.epochs')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration as a dataclass"""
        training = self._config.get('training', {})
        return TrainingConfig(
            epochs=training.get('epochs', 600),
            batch_size=training.get('batch_size', 16),
            image_size=training.get('image_size', 640),
            device=training.get('device', 'cuda'),
            early_stopping_patience=training.get('early_stopping', {}).get('patience', 20),
            early_stopping_monitor=training.get('early_stopping', {}).get('monitor', 'val_mAP50'),
            learning_rate_schedule_type=training.get('learning_rate_schedule', {}).get('type', 'cosine_annealing'),
            lr0=training.get('learning_rate_schedule', {}).get('lr0', 0.01),
            lrf=training.get('learning_rate_schedule', {}).get('lrf', 0.01),
            optimizer_type=training.get('optimizer', {}).get('type', 'SGD'),
            momentum=training.get('optimizer', {}).get('momentum', 0.937),
            weight_decay=training.get('optimizer', {}).get('weight_decay', 0.0005),
            label_smoothing=training.get('regularization', {}).get('label_smoothing', 0.1),
            warmup_epochs=training.get('warmup_epochs', 3),
            warmup_momentum=training.get('warmup_momentum', 0.8),
            warmup_bias_lr=training.get('warmup_bias_lr', 0.1)
        )
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as a dataclass"""
        model = self._config.get('model', {})
        return ModelConfig(
            name=model.get('name', 'ECL-YOLOv11'),
            num_classes=model.get('num_classes', 7),
            classes=model.get('classes', ['car', 'person', 'bus', 'bicycle', 'truck', 'train', 'motorcycle']),
            input_channels=model.get('input_channels', 3),
            reg_max=model.get('reg_max', 16)
        )
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration as a dataclass"""
        data = self._config.get('data', {})
        return DataConfig(
            format=data.get('format', 'yolo'),
            train_split=data.get('train_split', 0.8),
            val_split=data.get('val_split', 0.1),
            test_split=data.get('test_split', 0.1),
            data_root=data.get('data_root', './data'),
            augmentation_weather=data.get('augmentation', {}).get('weather', ['fog', 'rain', 'snow'])
        )
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration as a dataclass"""
        eval_config = self._config.get('evaluation', {})
        inference = self._config.get('inference', {})
        return EvaluationConfig(
            metrics=eval_config.get('metrics', ['mAP@50', 'mAP@50-95', 'Precision', 'Recall', 'FPS']),
            map50_threshold=eval_config.get('iou_thresholds', {}).get('map50', 0.50),
            confidence_threshold=inference.get('confidence_threshold', 0.25),
            iou_threshold=inference.get('iou_threshold', 0.45)
        )
    
    def get_inference_config(self) -> InferenceConfig:
        """Get inference configuration as a dataclass"""
        inference = self._config.get('inference', {})
        return InferenceConfig(
            confidence_threshold=inference.get('confidence_threshold', 0.25),
            iou_threshold=inference.get('iou_threshold', 0.45),
            max_detections=inference.get('max_detections', 300),
            input_size=inference.get('input_size', 640),
            half_precision=inference.get('half_precision', False),
            batch_size=inference.get('batch_size', 1)
        )
    
    def get_device(self, force_device: Optional[str] = None) -> torch.device:
        """
        Get the torch device for computation.
        
        Args:
            force_device: Optional device override ('cuda' or 'cpu')
            
        Returns:
            torch.device: The selected device
        """
        if ConfigManager._device is not None and force_device is None:
            return ConfigManager._device
        
        if force_device:
            device_str = force_device
        else:
            device_str = self._config.get('training', {}).get('device', 'cuda')
        
        # Check CUDA availability
        if device_str == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device_str = 'cpu'
        
        # Create device
        device = torch.device(device_str)
        
        # Cache the device
        if force_device is None:
            ConfigManager._device = device
        
        return device
    
    def get_class_to_idx(self) -> Dict[str, int]:
        """Get class name to index mapping"""
        classes = self._config.get('model', {}).get('classes', 
            ['car', 'person', 'bus', 'bicycle', 'truck', 'train', 'motorcycle'])
        return {cls: idx for idx, cls in enumerate(classes)}
    
    def get_idx_to_class(self) -> Dict[int, str]:
        """Get index to class name mapping"""
        classes = self._config.get('model', {}).get('classes',
            ['car', 'person', 'bus', 'bicycle', 'truck', 'train', 'motorcycle'])
        return {idx: cls for idx, cls in enumerate(classes)}
    
    def get_num_classes(self) -> int:
        """Get the number of classes"""
        return self._config.get('model', {}).get('num_classes', 7)
    
    def get_classes(self) -> List[str]:
        """Get the list of class names"""
        return self._config.get('model', {}).get('classes',
            ['car', 'person', 'bus', 'bicycle', 'truck', 'train', 'motorcycle'])
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss function weights"""
        return self._config.get('loss', {'box': 7.5, 'cls': 0.5, 'dfl': 1.5})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self._config.get('logging', {
            'log_dir': './runs',
            'save_dir': './weights',
            'log_interval': 10,
            'save_interval': 50,
            'tensorboard': True,
            'verbose': True
        })
    
    def get_ce_module_config(self) -> Dict[str, Any]:
        """Get CE module configuration"""
        return self._config.get('model', {}).get('ce_module', {
            'enabled': True,
            'sobel_kernel_size': 3,
            'use_residual': True,
            'activation': 'SiLU'
        })
    
    def get_aenet_config(self) -> Dict[str, Any]:
        """Get AENet module configuration"""
        return self._config.get('model', {}).get('aenet', {
            'enabled': True,
            'pyramid_channels': [256, 512, 1024],
            'rcm_stages': 2,
            'use_dif': True,
            'use_fbm': True
        })
    
    def get_ldhead_config(self) -> Dict[str, Any]:
        """Get LDHead module configuration"""
        return self._config.get('model', {}).get('ldhead', {
            'enabled': True,
            'shared_conv': True,
            'use_groupnorm': True,
            'groupnorm_groups': 32,
            'depthwise_separable': True
        })
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get data paths configuration"""
        data = self._config.get('data', {})
        return {
            'data_root': data.get('data_root', './data'),
            'train_images': data.get('train_images', './data/images/train'),
            'val_images': data.get('val_images', './data/images/val'),
            'test_images': data.get('test_images', './data/images/test'),
            'train_labels': data.get('train_labels', './data/labels/train'),
            'val_labels': data.get('val_labels', './data/labels/val'),
            'test_labels': data.get('test_labels', './data/labels/test')
        }
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            key: Dot-notation key (e.g., 'training.batch_size')
            value: New value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def validate(self) -> List[str]:
        """
        Validate the configuration values.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Training validation
        training = self._config.get('training', {})
        epochs = training.get('epochs', 600)
        if not 1 <= epochs <= 10000:
            errors.append(f"Invalid epochs: {epochs} (should be 1-10000)")
        
        batch_size = training.get('batch_size', 16)
        if not 1 <= batch_size <= 512:
            errors.append(f"Invalid batch_size: {batch_size} (should be 1-512)")
        
        image_size = training.get('image_size', 640)
        if not 128 <= image_size <= 2048:
            errors.append(f"Invalid image_size: {image_size} (should be 128-2048)")
        
        # Model validation
        model = self._config.get('model', {})
        num_classes = model.get('num_classes', 7)
        if not 1 <= num_classes <= 1000:
            errors.append(f"Invalid num_classes: {num_classes} (should be 1-1000)")
        
        classes = model.get('classes', [])
        if len(classes) != num_classes:
            errors.append(f"Class list length ({len(classes)}) doesn't match num_classes ({num_classes})")
        
        # Inference validation
        inference = self._config.get('inference', {})
        conf_thresh = inference.get('confidence_threshold', 0.25)
        if not 0.0 <= conf_thresh <= 1.0:
            errors.append(f"Invalid confidence_threshold: {conf_thresh} (should be 0.0-1.0)")
        
        iou_thresh = inference.get('iou_threshold', 0.45)
        if not 0.0 <= iou_thresh <= 1.0:
            errors.append(f"Invalid iou_threshold: {iou_thresh} (should be 0.0-1.0)")
        
        return errors
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get the full configuration dictionary (copy)"""
        import copy
        return copy.deepcopy(self._config)
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the entire configuration dictionary.
        
        Args:
            config: New configuration dictionary
        """
        self._config = config
    
    def reset(self) -> None:
        """Reset configuration to defaults"""
        self._config = DEFAULT_CONFIG.copy()
        ConfigManager._device = None


# =============================================================================
# Module-Level Functions (Convenience Accessors)
# =============================================================================

# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager: The singleton configuration manager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        ConfigManager: The configured manager
    """
    manager = get_config_manager()
    if config_path:
        manager.load_from_yaml(config_path)
    return manager


def get_training_config() -> TrainingConfig:
    """Get training configuration"""
    return get_config_manager().get_training_config()


def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return get_config_manager().get_model_config()


def get_data_config() -> DataConfig:
    """Get data configuration"""
    return get_config_manager().get_data_config()


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation configuration"""
    return get_config_manager().get_evaluation_config()


def get_inference_config() -> InferenceConfig:
    """Get inference configuration"""
    return get_config_manager().get_inference_config()


def get_device(force_device: Optional[str] = None) -> torch.device:
    """Get computation device"""
    return get_config_manager().get_device(force_device)


def get_num_classes() -> int:
    """Get number of classes"""
    return get_config_manager().get_num_classes()


def get_classes() -> List[str]:
    """Get class names"""
    return get_config_manager().get_classes()


def get_class_to_idx() -> Dict[str, int]:
    """Get class to index mapping"""
    return get_config_manager().get_class_to_idx()


def get_idx_to_class() -> Dict[int, str]:
    """Get index to class mapping"""
    return get_config_manager().get_idx_to_class()


def get_loss_weights() -> Dict[str, float]:
    """Get loss function weights"""
    return get_config_manager().get_loss_weights()


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value by key"""
    return get_config_manager().get(key, default)


# =============================================================================
# Main entry point for standalone execution
# =============================================================================

if __name__ == "__main__":
    # Test the configuration module
    print("Testing ECL-YOLOv11 Configuration Module")
    print("=" * 50)
    
    # Try to load config from default locations
    config_manager = load_config()
    
    # Validate
    errors = config_manager.validate()
    if errors:
        print("\nConfiguration Validation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration validated successfully!")
    
    # Print some configuration values
    print("\nConfiguration Summary:")
    print(f"  Model: {get_model_config().name}")
    print(f"  Num Classes: {get_num_classes()}")
    print(f"  Classes: {get_classes()}")
    print(f"  Epochs: {get_training_config().epochs}")
    print(f"  Batch Size: {get_training_config().batch_size}")
    print(f"  Image Size: {get_training_config().image_size}")
    print(f"  Device: {get_device()}")
    print(f"  Learning Rate: {get_training_config().lr0}")
    print(f"  Label Smoothing: {get_training_config().label_smoothing}")
    print(f"  Confidence Threshold: {get_inference_config().confidence_threshold}")
    print(f"  IoU Threshold: {get_inference_config().iou_threshold}")
    
    print("\nLoss Weights:")
    loss_weights = get_loss_weights()
    for key, value in loss_weights.items():
        print(f"  {key}: {value}")
    
    print("\nCE Module Config:")
    ce_config = get_config_manager().get_ce_module_config()
    for key, value in ce_config.items():
        print(f"  {key}: {value}")
    
    print("\nAENet Config:")
    aenet_config = get_config_manager().get_aenet_config()
    for key, value in aenet_config.items():
        print(f"  {key}: {value}")
    
    print("\nLDHead Config:")
    ldhead_config = get_config_manager().get_ldhead_config()
    for key, value in ldhead_config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Configuration module test completed!")
