"""
ECL-YOLOv11 Main Entry Point

This file serves as the main entry point for the ECL-YOLOv11 object detection framework.
It orchestrates data preparation, model initialization, training, and evaluation.

Based on the paper: "Robust Object Detection in Adverse Weather Conditions: 
ECL-YOLOv11 for Automotive Vision Systems"

Author: ECL-YOLOv11 Reproduction Team
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from utils.config import (
    ConfigManager,
    get_config_manager,
    load_config,
    get_training_config,
    get_model_config,
    get_data_config,
    get_evaluation_config,
    get_device
)
from model.ecl_yolo import ECLYOLOv11, create_ecl_yolov11
from model.modules.ce_module import CEModule, CEBackbone
from model.modules.aenet import AENet
from model.modules.ldhead import LDHead
from data.weather_augmentation import WeatherAugmentation
from data.dataset import YOLODataset, create_dataloader
from trainer.trainer import Trainer, create_trainer
from evaluation.evaluator import Evaluator, create_evaluator, measure_fps


# =============================================================================
# Default Configuration Paths
# =============================================================================

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# =============================================================================
# Utility Functions
# =============================================================================

def setup_environment(config: Dict) -> torch.device:
    """
    Setup the training environment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device: Selected compute device
    """
    # Get hardware info
    hardware = config.get('hardware', {})
    software = config.get('software', {})
    
    print("="*70)
    print("ECL-YOLOv11: Robust Object Detection in Adverse Weather Conditions")
    print("="*70)
    print(f"\nHardware Configuration:")
    print(f"  CPU: {hardware.get('cpu', 'Unknown')}")
    print(f"  GPU: {hardware.get('gpu', 'Unknown')}")
    print(f"  GPU Memory: {hardware.get('gpu_memory_gb', 12)} GB")
    print(f"  OS: {hardware.get('os', 'Unknown')}")
    
    print(f"\nSoftware Configuration:")
    print(f"  Python: {software.get('python', '3.10')}")
    print(f"  PyTorch: {software.get('pytorch', '2.5.0')}")
    print(f"  CUDA: {software.get('cuda', '12.1')}")
    print(f"  Framework: {software.get('framework', 'Ultralytics 8.3.9')}")
    print()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    return device


def create_experiment_directories(config: Dict) -> Dict[str, Path]:
    """
    Create directories for experiment outputs.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of directory paths
    """
    logging_config = config.get('logging', {})
    
    log_dir = Path(logging_config.get('log_dir', './runs'))
    save_dir = Path(logging_config.get('save_dir', './weights'))
    
    # Create timestamp-based experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = log_dir / f"experiment_{timestamp}"
    
    # Create all directories
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    return {
        'experiment': exp_dir,
        'log': log_dir,
        'save': save_dir,
        'checkpoints': exp_dir / "checkpoints",
        'logs': exp_dir / "logs",
        'visualizations': exp_dir / "visualizations",
        'results': exp_dir / "results"
    }


def prepare_dataset(config: Dict, device: torch.device) -> Tuple[Any, Any, Any]:
    """
    Prepare the dataset for training.
    
    Since the original dataset is not publicly available, this function:
    1. Creates a synthetic dataset using weather augmentation on public data
    2. Falls back to using a simple synthetic dataset if no public data available
    
    Args:
        config: Configuration dictionary
        device: Compute device
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print("\n" + "="*70)
    print("Preparing Dataset")
    print("="*70)
    
    data_config = get_data_config()
    training_config = get_training_config()
    
    # Get data paths
    data_paths = config.get('data', {})
    data_root = data_paths.get('data_root', './data')
    
    print(f"\nData Configuration:")
    print(f"  Format: {data_config.format}")
    print(f"  Train/Val/Test Split: {data_config.train_split}/{data_config.val_split}/{data_config.test_split}")
    print(f"  Augmentation Weather: {data_config.augmentation_weather}")
    
    # Try to load existing dataset, or create a simple one for testing
    try:
        # Check if dataset exists
        train_images = Path(data_root) / "train" / "images"
        
        if train_images.exists() and len(list(train_images.glob("*.jpg"))) > 0:
            print(f"\nLoading dataset from: {data_root}")
            
            train_dataset = YOLODataset(
                root_dir=data_root,
                split='train',
                image_size=training_config.image_size,
                augment=True,
                weather_types=data_config.augmentation_weather,
                use_weather_prob=0.5,
                config=config
            )
            
            val_dataset = YOLODataset(
                root_dir=data_root,
                split='val',
                image_size=training_config.image_size,
                augment=False,
                config=config
            )
            
            test_dataset = YOLODataset(
                root_dir=data_root,
                split='test',
                image_size=training_config.image_size,
                augment=False,
                config=config
            )
            
            print(f"\nDataset loaded successfully:")
            print(f"  Train: {len(train_dataset)} images")
            print(f"  Val: {len(val_dataset)} images")
            print(f"  Test: {len(test_dataset)} images")
            
            return train_dataset, val_dataset, test_dataset
    except Exception as e:
        print(f"Warning: Could not load dataset from {data_root}: {e}")
    
    # Create synthetic dataset for demonstration
    print("\nCreating synthetic dataset for demonstration...")
    print("Note: In practice, use the original dataset or create similar weather-degraded data")
    
    # Create a simple synthetic dataset
    train_dataset = _create_synthetic_dataset(
        config=config,
        split='train',
        image_size=training_config.image_size,
        num_samples=100
    )
    
    val_dataset = _create_synthetic_dataset(
        config=config,
        split='val',
        image_size=training_config.image_size,
        num_samples=20
    )
    
    test_dataset = _create_synthetic_dataset(
        config=config,
        split='test',
        image_size=training_config.image_size,
        num_samples=20
    )
    
    print(f"\nSynthetic dataset created:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    return train_dataset, val_dataset, test_dataset


def _create_synthetic_dataset(
    config: Dict,
    split: str,
    image_size: int,
    num_samples: int
) -> Any:
    """
    Create a synthetic dataset for testing.
    
    Args:
        config: Configuration dictionary
        split: Dataset split name
        image_size: Image size
        num_samples: Number of samples
        
    Returns:
        Dataset object
    """
    import tempfile
    import shutil
    import cv2
    import numpy as np
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create split directories
        split_dir = temp_dir / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Generate sample images and labels
        np.random.seed(42 if split == 'train' else 123)
        
        for i in range(num_samples):
            # Create random image
            img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Add some structure (simulate objects)
            num_objects = np.random.randint(1, 5)
            
            bboxes = []
            for _ in range(num_objects):
                x1 = np.random.randint(50, 500)
                y1 = np.random.randint(50, 350)
                w = np.random.randint(40, 150)
                h = np.random.randint(40, 150)
                x2 = min(x1 + w, 630)
                y2 = min(y1 + h, 470)
                
                # Draw rectangle
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / 640
                y_center = ((y1 + y2) / 2) / 480
                width = (x2 - x1) / 640
                height = (y2 - y1) / 480
                
                # Random class
                class_id = np.random.randint(0, 7)
                bboxes.append([class_id, x_center, y_center, width, height])
            
            # Save image
            img_path = images_dir / f"img_{i:04d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Save label
            label_path = labels_dir / f"img_{i:04d}.txt"
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
        
        # Create dataset from temp directory
        dataset = YOLODataset(
            root_dir=temp_dir,
            split=split,
            image_size=image_size,
            augment=(split == 'train'),
            weather_types=['fog', 'rain', 'snow'] if split == 'train' else [],
            use_weather_prob=0.5 if split == 'train' else 0.0,
            config=config
        )
        
        return dataset
        
    finally:
        # Note: temp_dir will be cleaned up when process ends
        pass


def initialize_model(config: Dict, device: torch.device) -> nn.Module:
    """
    Initialize the ECL-YOLOv11 model.
    
    Args:
        config: Configuration dictionary
        device: Compute device
        
    Returns:
        Initialized model
    """
    print("\n" + "="*70)
    print("Initializing ECL-YOLOv11 Model")
    print("="*70)
    
    model_config = get_model_config()
    
    # Get model parameters
    num_classes = model_config.num_classes
    reg_max = model_config.reg_max
    channels = config.get('model', {}).get('aenet', {}).get('pyramid_channels', [256, 512, 1024])
    
    print(f"\nModel Configuration:")
    print(f"  Model Name: {model_config.name}")
    print(f"  Number of Classes: {num_classes}")
    print(f"  Classes: {', '.join(model_config.classes)}")
    print(f"  Reg Max: {reg_max}")
    print(f"  Pyramid Channels: {channels}")
    
    # Get CE module config
    ce_config = config.get('model', {}).get('ce_module', {})
    print(f"\nCE Module Configuration:")
    print(f"  Enabled: {ce_config.get('enabled', True)}")
    print(f"  Sobel Kernel Size: {ce_config.get('sobel_kernel_size', 3)}")
    print(f"  Use Residual: {ce_config.get('use_residual', True)}")
    print(f"  Activation: {ce_config.get('activation', 'SiLU')}")
    
    # Get AENet config
    aenet_config = config.get('model', {}).get('aenet', {})
    print(f"\nAENet Configuration:")
    print(f"  Enabled: {aenet_config.get('enabled', True)}")
    print(f"  RCM Stages: {aenet_config.get('rcm_stages', 2)}")
    print(f"  Use DIF: {aenet_config.get('use_dif', True)}")
    print(f"  Use FBM: {aenet_config.get('use_fbm', True)}")
    
    # Get LDHead config
    ldhead_config = config.get('model', {}).get('ldhead', {})
    print(f"\nLDHead Configuration:")
    print(f"  Enabled: {ldhead_config.get('enabled', True)}")
    print(f"  Shared Conv: {ldhead_config.get('shared_conv', True)}")
    print(f"  Use GroupNorm: {ldhead_config.get('use_groupnorm', True)}")
    print(f"  Depthwise Separable: {ldhead_config.get('depthwise_separable', True)}")
    
    # Create model
    model = ECLYOLOv11(
        num_classes=num_classes,
        reg_max=reg_max,
        channels=channels,
        use_ce_in_backbone=ce_config.get('enabled', True),
        use_aenet_in_neck=aenet_config.get('enabled', True),
        use_ldhead_in_head=ldhead_config.get('enabled', True),
        config=config,
        device=device
    )
    
    model = model.to(device)
    
    # Print model summary
    summary = model.summary()
    print(f"\nModel Summary:")
    print(f"  Total Parameters: {summary['total_parameters']:,}")
    print(f"  Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"  Backbone Parameters: {summary['backbone_parameters']:,}")
    print(f"  Neck Parameters: {summary['neck_parameters']:,}")
    print(f"  Head Parameters: {summary['head_parameters']:,}")
    
    # Compare with paper
    paper_params = 3001194
    diff = summary['total_parameters'] - paper_params
    print(f"\n  Paper Parameters: {paper_params:,}")
    print(f"  Difference: {diff:,} ({diff/paper_params*100:.2f}%)")
    
    return model


def train_model(
    model: nn.Module,
    train_dataset: Any,
    val_dataset: Any,
    config: Dict,
    device: torch.device,
    dirs: Dict[str, Path]
) -> Tuple[nn.Module, Dict]:
    """
    Train the model.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary
        device: Compute device
        dirs: Experiment directories
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    training_config = get_training_config()
    
    # Create data loaders
    train_loader = create_dataloader(
        root_dir=train_dataset.root_dir if hasattr(train_dataset, 'root_dir') else './data',
        split='train',
        batch_size=training_config.batch_size,
        image_size=training_config.image_size,
        augment=True,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
        config=config
    )
    
    val_loader = create_dataloader(
        root_dir=val_dataset.root_dir if hasattr(val_dataset, 'root_dir') else './data',
        split='val',
        batch_size=training_config.batch_size,
        image_size=training_config.image_size,
        augment=False,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
        config=config
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Batch Size: {training_config.batch_size}")
    print(f"  Image Size: {training_config.image_size}")
    print(f"  Optimizer: {training_config.optimizer_type}")
    print(f"  Learning Rate: {training_config.lr0}")
    print(f"  Weight Decay: {training_config.weight_decay}")
    print(f"  Momentum: {training_config.momentum}")
    print(f"  Label Smoothing: {training_config.label_smoothing}")
    print(f"  Early Stopping Patience: {training_config.early_stopping_patience}")
    print(f"  Learning Rate Schedule: {training_config.learning_rate_schedule_type}")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    
    # Save training history
    trainer.save_history(dirs['results'] / 'training_history.json')
    
    # Get best model
    best_model = trainer.get_model()
    
    return best_model, history


def evaluate_model(
    model: nn.Module,
    test_dataset: Any,
    config: Dict,
    device: torch.device,
    dirs: Dict[str, Path]
) -> Dict:
    """
    Evaluate the model on the test set.
    
    Args:
        model: Model to evaluate
        test_dataset: Test dataset
        config: Configuration dictionary
        device: Compute device
        dirs: Experiment directories
        
    Returns:
        Evaluation metrics dictionary
    """
    print("\n" + "="*70)
    print("Evaluating Model")
    print("="*70)
    
    training_config = get_training_config()
    model_config = get_model_config()
    
    # Create test data loader
    test_loader = create_dataloader(
        root_dir=test_dataset.root_dir if hasattr(test_dataset, 'root_dir') else './data',
        split='test',
        batch_size=1,  # Use batch size 1 for accurate FPS measurement
        image_size=training_config.image_size,
        augment=False,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
        config=config
    )
    
    # Create evaluator
    evaluator = create_evaluator(
        model=model,
        config=config,
        device=device
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluator.evaluate(
        data_loader=test_loader,
        confidence_threshold=0.25,
        compute_fps=True,
        verbose=True
    )
    
    # Save results
    evaluator.save_metrics(metrics, dirs['results'] / 'evaluation_metrics.json')
    
    # Print summary
    print("\n" + "="*70)
    print("Final Evaluation Results")
    print("="*70)
    print(f"  mAP@50:     {metrics.mAP50:.4f}")
    print(f"  mAP@50-95:  {metrics.mAP50_95:.4f}")
    print(f"  Precision:  {metrics.Precision:.4f}")
    print(f"  Recall:     {metrics.Recall:.4f}")
    print(f"  FPS:        {metrics.FPS:.2f}")
    print("="*70)
    
    return metrics.to_dict()


def run_ablation_study(
    config: Dict,
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Any,
    device: torch.device,
    dirs: Dict[str, Path]
) -> None:
    """
    Run ablation study with different module combinations.
    
    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        device: Compute device
        dirs: Experiment directories
    """
    print("\n" + "="*70)
    print("Running Ablation Study")
    print("="*70)
    
    ablation_results = {}
    
    # Module combinations to test
    module_combinations = [
        {'name': 'Baseline (YOLOv11)', 'ce': False, 'aenet': False, 'ldhead': False},
        {'name': '+ CE', 'ce': True, 'aenet': False, 'ldhead': False},
        {'name': '+ AENet', 'ce': False, 'aenet': True, 'ldhead': False},
        {'name': '+ LDHead', 'ce': False, 'aenet': False, 'ldhead': True},
        {'name': '+ CE + AENet', 'ce': True, 'aenet': True, 'ldhead': False},
        {'name': '+ AENet + LDHead', 'ce': False, 'aenet': True, 'ldhead': True},
        {'name': '+ CE + LDHead', 'ce': True, 'aenet': False, 'ldhead': True},
        {'name': 'ECL-YOLOv11 (Full)', 'ce': True, 'aenet': True, 'ldhead': True},
    ]
    
    for combo in module_combinations:
        print(f"\n{'='*50}")
        print(f"Testing: {combo['name']}")
        print(f"{'='*50}")
        
        # Create model with specific modules
        model = ECLYOLOv11(
            num_classes=7,
            reg_max=16,
            use_ce_in_backbone=combo['ce'],
            use_aenet_in_neck=combo['aenet'],
            use_ldhead_in_head=combo['ldhead'],
            config=config,
            device=device
        ).to(device)
        
        # Train for a few epochs (abbreviated for speed)
        trainer = Trainer(
            model=model,
            train_loader=create_dataloader('./data', 'train', batch_size=8, image_size=640),
            val_loader=create_dataloader('./data', 'val', batch_size=8, image_size=640),
            config=config,
            device=device
        )
        
        # Quick training
        trainer.epochs = 5  # Abbreviated for ablation
        history = trainer.train()
        
        # Evaluate
        evaluator = create_evaluator(model=model, config=config, device=device)
        
        # Quick FPS test
        fps = measure_fps(model, input_size=(640, 640), num_warmup=5, num_iterations=20, device=device)
        
        # Record results
        ablation_results[combo['name']] = {
            'parameters': model.get_num_parameters(),
            'fps': fps,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0.0
        }
        
        print(f"  Parameters: {model.get_num_parameters():,}")
        print(f"  FPS: {fps:.2f}")
    
    # Save ablation results
    with open(dirs['results'] / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print("\nAblation study complete. Results saved.")


def compare_with_baseline(results: Dict, baseline: Dict) -> None:
    """
    Compare results with baseline from paper.
    
    Args:
        results: Current model results
        baseline: Baseline metrics from paper
    """
    print("\n" + "="*70)
    print("Comparison with Paper Baseline (YOLOv11)")
    print("="*70)
    
    print(f"\n{'Metric':<20} {'Baseline':>12} {'ECL-YOLOv11':>15} {'Improvement':>15}")
    print("-"*70)
    
    metrics_to_compare = [
        ('mAP@50', 'mAP50', True),
        ('mAP@50-95', 'mAP50_95', True),
        ('Precision', 'Precision', True),
        ('Recall', 'Recall', True),
        ('FPS', 'FPS', False)  # Lower is not better
    ]
    
    for display_name, key, higher_is_better in metrics_to_compare:
        base_val = baseline.get(key, 0)
        curr_val = results.get(key, 0)
        
        if higher_is_better:
            diff = curr_val - base_val
            diff_pct = (diff / base_val * 100) if base_val != 0 else 0
            change_str = f"+{diff:.4f} ({diff_pct:+.2f}%)" if diff >= 0 else f"{diff:.4f} ({diff_pct:.2f}%)"
        else:
            # For FPS, compare differently
            diff = curr_val - base_val
            change_str = f"{diff:.2f} ({diff/base_val*100:.1f}%)" if base_val != 0 else f"{curr_val:.2f}"
        
        print(f"{display_name:<20} {base_val:>12.4f} {curr_val:>15.4f} {change_str:>15}")
    
    print("="*70)


# =============================================================================
# Main Entry Point
# =============================================================================

def main(
    config_path: Optional[Union[str, Path]] = None,
    run_evaluation: bool = True,
    run_ablation: bool = False,
    test_mode: bool = False
) -> Dict:
    """
    Main entry point for ECL-YOLOv11.
    
    Args:
        config_path: Path to configuration file
        run_evaluation: Whether to run evaluation
        run_ablation: Whether to run ablation study
        test_mode: If True, use minimal configuration for testing
        
    Returns:
        Dictionary with experiment results
    """
    # Load configuration
    config_manager = load_config(config_path or DEFAULT_CONFIG_PATH)
    config = config_manager.get_config_dict()
    
    # Override config for test mode
    if test_mode:
        # Modify training config for quick testing
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 2
        config['training']['early_stopping']['patience'] = 1
        print("Test mode enabled: Using minimal configuration")
    
    # Setup environment
    device = setup_environment(config)
    
    # Create experiment directories
    dirs = create_experiment_directories(config)
    
    # Save configuration
    with open(dirs['experiment'] / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nExperiment directory: {dirs['experiment']}")
    print(f"Results will be saved to: {dirs['results']}")
    
    # Prepare dataset
    train_dataset, val_dataset, test_dataset = prepare_dataset(config, device)
    
    # Initialize model
    model = initialize_model(config, device)
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        dirs=dirs
    )
    
    # Evaluate model
    eval_results = {}
    if run_evaluation:
        eval_results = evaluate_model(
            model=trained_model,
            test_dataset=test_dataset,
            config=config,
            device=device,
            dirs=dirs
        )
        
        # Compare with baseline
        baseline_results = {
            'mAP50': 0.614,
            'mAP50_95': 0.397,
            'Precision': 0.688,
            'Recall': 0.553,
            'FPS': 406.5
        }
        compare_with_baseline(eval_results, baseline_results)
    
    # Run ablation study
    if run_ablation:
        run_ablation_study(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
            dirs=dirs
        )
    
    # Save final results
    final_results = {
        'evaluation': eval_results,
        'training_history': history,
        'experiment_dir': str(dirs['experiment'])
    }
    
    with open(dirs['results'] / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print(f"Results saved to: {dirs['results']}")
    print("="*70)
    
    return final_results


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='ECL-YOLOv11: Robust Object Detection in Adverse Weather Conditions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--no-evaluation',
        action='store_true',
        help='Skip evaluation after training'
    )
    
    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run ablation study'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: use minimal configuration for quick testing'
    )
    
    return parser.parse_args()


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Run main
    results = main(
        config_path=args.config,
        run_evaluation=not args.no_evaluation,
        run_ablation=args.ablation,
        test_mode=args.test
    )
    
    print("\nExecution completed successfully!")
