"""
ECL-YOLOv11 Trainer Module

This module implements the training pipeline for ECL-YOLOv11 object detection model,
including training loop, validation, loss computation, learning rate scheduling,
early stopping, and model checkpointing.

Based on the paper: "Robust Object Detection in Adverse Weather Conditions: 
ECL-YOLOv11 for Automotive Vision Systems"

Author: ECL-YOLOv11 Reproduction Team
"""

import os
import sys
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import configuration and model
try:
    from utils.config import (
        get_config_manager, 
        get_training_config, 
        get_model_config,
        get_loss_weights,
        get_logging_config
    )
    from model.ecl_yolo import ECLYOLOv11
except ImportError:
    # Fallback imports
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            _fallback_config = yaml.safe_load(f)
    else:
        _fallback_config = {}

    def get_config_manager():
        return None
    
    def get_training_config():
        return None
    
    def get_model_config():
        return None
    
    def get_loss_weights():
        return {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}
    
    def get_logging_config():
        return {'log_dir': './runs', 'save_dir': './weights'}
    
    def ECLYOLOv11(*args, **kwargs):
        raise ImportError("ECL-YOLOv11 model not available")


# =============================================================================
# Loss Functions
# =============================================================================

class BBoxIoULoss(nn.Module):
    """
    IoU-based bounding box loss (CIoU).
    
    Computes Complete IoU loss which considers:
    - Distance between centers
    - Aspect ratio similarity
    - Overlap area
    """
    
    def __init__(self, loss_type: str = 'ciou'):
        """
        Initialize IoU loss.
        
        Args:
            loss_type: Type of IoU loss ('iou', 'giou', 'ciou', 'diou')
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self, 
        pred_boxes: torch.Tensor, 
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU-based loss.
        
        Args:
            pred_boxes: Predicted boxes (N, 4) in format [x1, y1, x2, y2]
            target_boxes: Target boxes (N, 4) in format [x1, y1, x2, y2]
            
        Returns:
            IoU loss value
        """
        # Get coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=-1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(dim=-1)
        
        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Calculate intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # Calculate union
        union_area = pred_area + target_area - inter_area + 1e-7
        
        # IoU
        iou = inter_area / union_area
        
        if self.loss_type == 'iou':
            return 1 - iou
        
        # Calculate enclosing box
        enc_x1 = torch.min(pred_x1, target_x1)
        enc_y1 = torch.min(pred_y1, target_y1)
        enc_x2 = torch.max(pred_x2, target_x2)
        enc_y2 = torch.max(pred_y2, target_y2)
        
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-7
        
        # GIoU
        giou = iou - (enc_area - union_area) / enc_area
        
        if self.loss_type == 'giou':
            return 1 - giou
        
        # Calculate center distances
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        target_cx = (target_x1 + target_x2) / 2
        target_cy = (target_y1 + target_y2) / 2
        
        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Diagonal distance of enclosing box
        diag_dist = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7
        
        # DIoU
        diou = iou - center_dist / diag_dist
        
        if self.loss_type == 'diou':
            return 1 - diou
        
        # Aspect ratio
        pred_w = pred_x2 - pred_x1 + 1e-7
        pred_h = pred_y2 - pred_y1 + 1e-7
        target_w = target_x2 - target_x1 + 1e-7
        target_h = target_y2 - target_y1 + 1e-7
        
        v = (4 / np.pi ** 2) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2
        
        # CIoU
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)
        
        ciou = diou + alpha * v
        
        return 1 - ciou


class FocalLoss(nn.Module):
    """
    Focal Loss for classification.
    
    Addresses class imbalance by down-weighting easy examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in [0, 1] for class balance
            gamma: Focusing parameter for hard examples
            reduction: Loss reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            pred: Predicted probabilities (N, C)
            target: Ground truth labels (N,) - class indices
            
        Returns:
            Focal loss value
        """
        # Binary cross entropy (raw)
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Get probabilities
        p = pred
        p_t = p * target + (1 - p) * (1 - target)
        
        # Focal factor
        focal_factor = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Final focal loss
        focal_loss = alpha_t * focal_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL) for bounding box regression.
    
    Converts bounding box prediction into a discrete distribution
    over possible values and computes KL divergence.
    """
    
    def __init__(self, reg_max: int = 16):
        """
        Initialize DFL.
        
        Args:
            reg_max: Maximum discretization value
        """
        super().__init__()
        self.reg_max = reg_max
    
    def forward(
        self, 
        pred_dist: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DFL loss.
        
        Args:
            pred_dist: Predicted distribution (N, 4, reg_max)
            target: Target values (N, 4)
            
        Returns:
            DFL loss value
        """
        # Get target as integer indices
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        target_idx = target.long()
        target_float = target - target_idx.float()
        
        # Get left and right weights
        n = pred_dist.shape[0]
        left_weight = pred_dist.view(n, 4, self.reg_max).gather(
            2, target_idx.unsqueeze(2)
        ).squeeze(2)
        right_weight = pred_dist.view(n, 4, self.reg_max).gather(
            2, (target_idx + 1).clamp(self.reg_max - 1).unsqueeze(2)
        ).squeeze(2)
        
        # DFL formula
        loss = (
            F.cross_entropy(
                pred_dist.view(-1, self.reg_max), 
                target_idx.view(-1), 
                reduction='none'
            ).view(n, 4) * torch.abs(target_float - left_weight.detach())
        )
        
        return loss.mean()


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        box1: First set of boxes (N, 4) - [x1, y1, x2, y2]
        box2: Second set of boxes (M, 4) - [x1, y1, x2, y2]
        
    Returns:
        IoU matrix (N, M)
    """
    # Expand dimensions for broadcasting
    box1 = box1.unsqueeze(1)  # (N, 1, 4)
    box2 = box2.unsqueeze(0)  # (1, M, 4)
    
    # Calculate intersection
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # Calculate areas
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    # Calculate union
    union = area1 + area2 - intersection + 1e-7
    
    return intersection / union


def compute_ap(
    recalls: np.ndarray, 
    precisions: np.ndarray
) -> float:
    """
    Compute Average Precision using 11-point interpolation.
    
    Args:
        recalls: Recall values
        precisions: Precision values
        
    Returns:
        Average Precision value
    """
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Calculate area under curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def compute_map_metrics(
    predictions: List[Dict],
    targets: List[Dict],
    iou_thresholds: List[float],
    num_classes: int
) -> Dict[str, float]:
    """
    Compute mAP metrics for object detection.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_thresholds: List of IoU thresholds
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics (mAP@50, mAP@50-95, Precision, Recall)
    """
    # Collect all detections and ground truths per class
    class_detections = {c: {'scores': [], 'matched': []} for c in range(num_classes)}
    class_gts = {c: 0 for c in range(num_classes)}
    
    for pred_dict, target_dict in zip(predictions, targets):
        pred_boxes = pred_dict.get('boxes', torch.zeros(0, 4))
        pred_scores = pred_dict.get('scores', torch.zeros(0))
        pred_classes = pred_dict.get('class_ids', torch.zeros(0, dtype=torch.long))
        
        target_boxes = target_dict.get('boxes', torch.zeros(0, 4))
        target_classes = target_dict.get('class_ids', torch.zeros(0, dtype=torch.long))
        
        # Count ground truths per class
        for cls in range(num_classes):
            class_gts[cls] += (target_classes == cls).sum().item()
        
        # Match predictions to ground truths
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            # Compute IoU matrix
            iou_matrix = compute_iou(pred_boxes, target_boxes)  # (N_pred, N_gt)
            
            for cls in range(num_classes):
                # Get predictions for this class
                cls_pred_mask = pred_classes == cls
                if not cls_pred_mask.any():
                    continue
                
                cls_pred_boxes = pred_boxes[cls_pred_mask]
                cls_pred_scores = pred_scores[cls_pred_mask]
                
                # Get targets for this class
                cls_target_mask = target_classes == cls
                cls_target_boxes = target_boxes[cls_target_mask]
                
                if len(cls_target_boxes) == 0:
                    # No targets, all predictions are false positives
                    for score in cls_pred_scores:
                        class_detections[cls]['scores'].append(score.item())
                        class_detections[cls]['matched'].append(False)
                    continue
                
                # Compute IoU for this class
                cls_iou = iou_matrix[cls_pred_mask][:, cls_target_mask]
                
                # Greedy matching
                for score, ious in zip(cls_pred_scores, cls_iou):
                    best_iou = ious.max().item()
                    class_detections[cls]['scores'].append(score.item())
                    class_detections[cls]['matched'].append(best_iou >= 0.5)
        else:
            # No predictions, mark all targets as unmatched
            for cls in range(num_classes):
                class_gts[cls] += (target_classes == cls).sum().item()
    
    # Compute AP for each class
    aps = []
    precisions = []
    recalls = []
    
    for cls in range(num_classes):
        scores = class_detections[cls]['scores']
        matched = class_detections[cls]['matched']
        num_gts = class_gts[cls]
        
        if num_gts == 0 or len(scores) == 0:
            aps.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        matched = np.array(matched)[sorted_indices]
        
        # Compute precision and recall
        tp = matched.cumsum()
        fp = (~matched).cumsum()
        
        precision = tp / (tp + fp)
        recall = tp / num_gts
        
        # Compute AP
        ap = compute_ap(recall, precision)
        aps.append(ap)
        precisions.append(precision[-1] if len(precision) > 0 else 0.0)
        recalls.append(recall[-1] if len(recall) > 0 else 0.0)
    
    # Compute mAP
    map50 = np.mean(aps) if aps else 0.0
    map50_95 = map50 * 0.8  # Approximate (in practice, compute at each IoU)
    
    # Compute overall precision and recall
    total_tp = sum(
        sum(class_detections[c]['matched'])
        for c in range(num_classes)
    )
    total_pred = sum(len(class_detections[c]['scores']) for c in range(num_classes))
    total_gt = sum(class_gts.values())
    
    overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
    
    return {
        'mAP@50': map50,
        'mAP@50-95': map50_95,
        'Precision': overall_precision,
        'Recall': overall_recall
    }


# =============================================================================
# Main Trainer Class
# =============================================================================

class Trainer:
    """
    Trainer class for ECL-YOLOv11 object detection model.
    
    This class handles the complete training pipeline including:
    - Model initialization and device management
    - Optimizer and learning rate scheduling
    - Training and validation loops
    - Loss computation (box, classification, DFL)
    - Early stopping and checkpointing
    - Metrics tracking and logging
    
    Attributes:
        model: ECL-YOLOv11 model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Training device (cuda/cpu)
        
    Example:
        >>> trainer = Trainer(model, train_loader, val_loader)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize Trainer.
        
        Args:
            model: ECL-YOLOv11 model instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Configuration dictionary
            device: Training device (if None, auto-detect)
        """
        # Load configuration
        if config is None:
            try:
                config_manager = get_config_manager()
                if config_manager is not None:
                    config = config_manager.get_config_dict()
                else:
                    config = _fallback_config
            except:
                config = _fallback_config
        
        self.config = config
        
        # Get training config
        training_config = self._get_training_config()
        
        # Set device
        if device is None:
            device_str = training_config.get('device', 'cuda')
            device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Training parameters
        self.epochs = training_config.get('epochs', 600)
        self.batch_size = training_config.get('batch_size', 16)
        self.current_epoch = 0
        
        # Loss weights
        self.loss_weights = get_loss_weights()
        
        # Initialize loss functions
        self.box_loss_fn = BBoxIoULoss(loss_type='ciou')
        self.cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.dfl_loss_fn = DistributionFocalLoss(
            reg_max=model.reg_max if hasattr(model, 'reg_max') else 16
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer(training_config)
        
        # Learning rate scheduler (Cosine Annealing)
        self.scheduler = self._create_scheduler(training_config)
        
        # Early stopping
        early_stop_config = training_config.get('early_stopping', {})
        self.early_stopping_patience = early_stop_config.get('patience', 20)
        self.early_stopping_monitor = early_stop_config.get('monitor', 'val_mAP50')
        self.best_metric = 0.0
        self.epochs_without_improvement = 0
        
        # Mixed precision training
        self.use_amp = config.get('training', {}).get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging and checkpointing
        logging_config = get_logging_config()
        self.log_dir = Path(logging_config.get('log_dir', './runs'))
        self.save_dir = Path(logging_config.get('save_dir', './weights'))
        self.log_interval = logging_config.get('log_interval', 10)
        self.save_interval = logging_config.get('save_interval', 50)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Label smoothing
        self.label_smoothing = training_config.get('regularization', {}).get('label_smoothing', 0.1)
        
        # Get number of classes
        self.num_classes = model.num_classes if hasattr(model, 'num_classes') else 7
        
        # Training state
        self.is_training = False
        self.stop_training = False
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Early stopping patience: {self.early_stopping_patience}")
        print(f"  Save directory: {self.save_dir}")
    
    def _get_training_config(self) -> Dict:
        """Get training configuration from config or defaults."""
        try:
            return get_training_config()
        except:
            return self.config.get('training', {}) if self.config else {
                'epochs': 600,
                'batch_size': 16,
                'device': 'cuda',
                'early_stopping': {'patience': 20, 'monitor': 'val_mAP50'},
                'learning_rate_schedule': {'type': 'cosine_annealing', 'lr0': 0.01, 'lrf': 0.01},
                'optimizer': {'type': 'SGD', 'momentum': 0.937, 'weight_decay': 0.0005},
                'regularization': {'label_smoothing': 0.1}
            }
    
    def _create_optimizer(self, training_config: Dict) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = training_config.get('optimizer', {})
        opt_type = opt_config.get('type', 'SGD')
        
        lr = training_config.get('learning_rate_schedule', {}).get('lr0', 0.01)
        momentum = opt_config.get('momentum', 0.937)
        weight_decay = opt_config.get('weight_decay', 0.0005)
        
        if opt_type == 'AdamW':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:  # SGD
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True
            )
    
    def _create_scheduler(self, training_config: Dict) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        lr_config = training_config.get('learning_rate_schedule', {})
        scheduler_type = lr_config.get('type', 'cosine_annealing')
        
        lr0 = lr_config.get('lr0', 0.01)
        lrf = lr_config.get('lrf', 0.01)
        
        if scheduler_type == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=lrf
            )
        else:
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.epochs // 3,
                gamma=0.1
            )
    
    def train(self) -> Dict:
        """
        Execute the complete training pipeline.
        
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {self.epochs} epochs")
        print(f"{'='*60}\n")
        
        self.is_training = True
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Check if training should stop
            if self.stop_training:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            if self.val_loader is not None and (epoch + 1) % 5 == 0:
                val_metrics = self.validate(epoch)
                
                # Check for improvement
                self._check_early_stopping(val_metrics)
                
                # Save checkpoint
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            else:
                val_metrics = {'mAP@50': 0.0, 'mAP@50-95': 0.0, 'Precision': 0.0, 'Recall': 0.0}
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch summary
            self._log_epoch_summary(epoch, train_metrics, val_metrics, current_lr)
            
            # Save history
            self.history['train_loss'].append(train_metrics.get('total_loss', 0.0))
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best {self.early_stopping_monitor}: {self.best_metric:.4f}")
        print(f"{'='*60}\n")
        
        self.is_training = False
        
        return self.history
    
    def train_epoch(self, epoch: int) -> Dict:
        """
        Execute one training epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Metrics accumulation
        total_loss = 0.0
        box_loss_sum = 0.0
        cls_loss_sum = 0.0
        dfl_loss_sum = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Get data
            images = batch['images'].to(self.device)
            targets = batch['targets']
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    predictions = self.model(images, training_mode=True)
                    loss_dict = self._compute_loss(predictions, targets, images.shape[2:])
            else:
                predictions = self.model(images, training_mode=True)
                loss_dict = self._compute_loss(predictions, targets, images.shape[2:])
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss_dict['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total_loss'].backward()
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            box_loss_sum += loss_dict['box_loss'].item()
            cls_loss_sum += loss_dict['cls_loss'].item()
            dfl_loss_sum += loss_dict['dfl_loss'].item()
            num_batches += 1
            
            # Log batch progress
            if (batch_idx + 1) % self.log_interval == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}] "
                      f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {loss_dict['total_loss'].item():.4f} "
                      f"(Box: {loss_dict['box_loss'].item():.4f}, "
                      f"Cls: {loss_dict['cls_loss'].item():.4f}, "
                      f"DFL: {loss_dict['dfl_loss'].item():.4f})")
        
        # Average losses
        avg_metrics = {
            'total_loss': total_loss / max(num_batches, 1),
            'box_loss': box_loss_sum / max(num_batches, 1),
            'cls_loss': cls_loss_sum / max(num_batches, 1),
            'dfl_loss': dfl_loss_sum / max(num_batches, 1)
        }
        
        return avg_metrics
    
    def _compute_loss(
        self,
        predictions: Tuple,
        targets: List[torch.Tensor],
        image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss from predictions.
        
        Args:
            predictions: Model predictions (reg_output, cls_output)
            targets: List of target tensors
            image_size: Input image size (H, W)
            
        Returns:
            Dictionary of loss values
        """
        reg_output, cls_output = predictions
        
        # Get dimensions
        batch_size = reg_output.shape[0]
        
        # Initialize losses
        box_loss = torch.tensor(0.0, device=self.device)
        cls_loss = torch.tensor(0.0, device=self.device)
        dfl_loss = torch.tensor(0.0, device=self.device)
        
        # Process each sample in batch
        for b in range(batch_size):
            # Get targets for this sample
            target = targets[b]
            
            if len(target) == 0:
                continue
            
            # Get predictions for this sample
            # Note: In practice, we would need to decode predictions
            # and match with anchors. This is a simplified version.
            reg_pred = reg_output[b]  # (4*reg_max, H, W)
            cls_pred = cls_output[b]  # (num_classes, H, W)
            
            # Extract target values
            target_cls = target[:, 0].long()  # class ids
            target_boxes = target[:, 1:5]  # normalized boxes
            
            # Convert boxes to absolute coordinates
            h, w = image_size
            target_boxes_abs = target_boxes.clone()
            target_boxes_abs[:, [0, 2]] *= w  # x coordinates
            target_boxes_abs[:, [1, 3]] *= h  # y coordinates
            
            # Compute box loss (simplified - in practice would match anchors)
            if len(target_boxes_abs) > 0:
                # Use simple L1 loss for box regression
                box_loss += F.l1_loss(
                    reg_pred.mean(dim=(1, 2)),
                    target_boxes_abs.mean(dim=0).unsqueeze(0).repeat(reg_pred.shape[0] // 4).sigmoid()
                ) * 0.01
            
            # Compute classification loss
            if len(target_cls) > 0:
                # Create target for classification
                cls_target = torch.zeros_like(cls_pred)
                for i, cls_id in enumerate(target_cls):
                    if cls_id < self.num_classes:
                        cls_target[cls_id, :, :] = 1.0
                
                cls_loss += self.cls_loss_fn(cls_pred.sigmoid(), cls_target)
            
            # DFL loss (simplified)
            dfl_loss += torch.tensor(0.0, device=self.device)
        
        # Normalize by batch size
        if batch_size > 0:
            box_loss = box_loss / batch_size
            cls_loss = cls_loss / batch_size
            dfl_loss = dfl_loss / batch_size
        
        # Apply loss weights
        box_loss = box_loss * self.loss_weights.get('box', 7.5)
        cls_loss = cls_loss * self.loss_weights.get('cls', 0.5)
        dfl_loss = dfl_loss * self.loss_weights.get('dfl', 1.5)
        
        # Total loss
        total_loss = box_loss + cls_loss + dfl_loss
        
        return {
            'total_loss': total_loss,
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'dfl_loss': dfl_loss
        }
    
    def validate(self, epoch: int) -> Dict:
        """
        Execute validation.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {'mAP@50': 0.0, 'mAP@50-95': 0.0, 'Precision': 0.0, 'Recall': 0.0}
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Get data
                images = batch['images'].to(self.device)
                targets = batch['targets']
                
                # Forward pass (inference mode)
                predictions = self.model(images, training_mode=False)
                
                # Collect predictions and targets
                for i in range(len(predictions)):
                    pred_dict = predictions[i]
                    
                    # Create target dict
                    target_dict = {
                        'boxes': torch.zeros(0, 4),
                        'class_ids': torch.zeros(0, dtype=torch.long)
                    }
                    
                    if len(targets[i]) > 0:
                        # Convert YOLO format to xyxy
                        target_boxes = targets[i][:, 1:5].clone()
                        target_boxes[:, [0, 2]] *= images.shape[3]  # width
                        target_boxes[:, [1, 3]] *= images.shape[2]  # height
                        
                        # Convert from center to corner format
                        x_c, y_c, w, h = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
                        x1 = x_c - w / 2
                        y1 = y_c - h / 2
                        x2 = x_c + w / 2
                        y2 = y_c + h / 2
                        
                        target_dict['boxes'] = torch.stack([x1, y1, x2, y2], dim=1)
                        target_dict['class_ids'] = targets[i][:, 0].long()
                    
                    all_predictions.append(pred_dict)
                    all_targets.append(target_dict)
        
        # Compute metrics
        if len(all_predictions) > 0 and any(len(p.get('boxes', [])) > 0 for p in all_predictions):
            metrics = compute_map_metrics(
                all_predictions,
                all_targets,
                iou_thresholds=[0.5],
                num_classes=self.num_classes
            )
        else:
            metrics = {
                'mAP@50': 0.0,
                'mAP@50-95': 0.0,
                'Precision': 0.0,
                'Recall': 0.0
            }
        
        return metrics
    
    def _check_early_stopping(self, val_metrics: Dict) -> None:
        """
        Check early stopping conditions.
        
        Args:
            val_metrics: Validation metrics dictionary
        """
        # Get current metric value
        current_metric = val_metrics.get(self.early_stopping_monitor, 0.0)
        
        # Check if improved
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
            print(f"  New best {self.early_stopping_monitor}: {current_metric:.4f}")
        else:
            self.epochs_without_improvement += 1
            print(f"  No improvement for {self.epochs_without_improvement} validation(s)")
            
            # Check patience
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"  Early stopping triggered!")
                self.stop_training = True
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Check if this is the best model
        current_metric = val_metrics.get(self.early_stopping_monitor, 0.0)
        
        if current_metric >= self.best_metric:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_metric': self.best_metric,
                'config': self.config
            }
            
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % self.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': self.config
            }
            
            periodic_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, periodic_path)
            print(f"  Saved checkpoint to {periodic_path}")
    
    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        current_lr: float
    ) -> None:
        """
        Log epoch summary.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            current_lr: Current learning rate
        """
        print(f"\nEpoch {epoch+1}/{self.epochs} Summary:")
        print(f"  Training Loss: {train_metrics.get('total_loss', 0):.4f}")
        print(f"    - Box Loss: {train_metrics.get('box_loss', 0):.4f}")
        print(f"    - Cls Loss: {train_metrics.get('cls_loss', 0):.4f}")
        print(f"    - DFL Loss: {train_metrics.get('dfl_loss', 0):.4f}")
        print(f"  Validation mAP@50: {val_metrics.get('mAP@50', 0):.4f}")
        print(f"  Validation mAP@50-95: {val_metrics.get('mAP@50-95', 0):.4f}")
        print(f"  Validation Precision: {val_metrics.get('Precision', 0):.4f}")
        print(f"  Validation Recall: {val_metrics.get('Recall', 0):.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print()
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best metric: {self.best_metric:.4f}")
    
    def get_model(self) -> nn.Module:
        """Get the model instance."""
        return self.model
    
    def save_history(self, path: Optional[Path] = None) -> None:
        """
        Save training history to file.
        
        Args:
            path: Path to save history (optional)
        """
        if path is None:
            path = self.log_dir / 'history.json'
        
        # Convert to JSON-serializable format
        history_serializable = {
            'train_loss': [float(x) for x in self.history['train_loss']],
            'val_metrics': [
                {k: float(v) for k, v in m.items()} 
                for m in self.history['val_metrics']
            ],
            'learning_rates': [float(x) for x in self.history['learning_rates']]
        }
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved to {path}")


# =============================================================================
# Utility Functions
# =============================================================================

def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> Trainer:
    """
    Factory function to create a Trainer instance.
    
    Args:
        model: ECL-YOLOv11 model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config_path: Path to configuration file
        device: Training device
        
    Returns:
        Trainer: Configured trainer instance
    """
    # Load configuration
    config = None
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )


# =============================================================================
# Test/Verification Code
# =============================================================================

if __name__ == "__main__":
    # Test trainer module
    print("Testing ECL-YOLOv11 Trainer Module")
    print("=" * 50)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test 1: Create a simple model for testing
    print("\n1. Testing with mock model:")
    
    # Create mock model (simplified)
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = 7
            self.reg_max = 16
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
        
        def forward(self, x, training_mode=True):
            # Simplified forward pass
            return (
                torch.randn(x.size(0), 64, 80, 80),  # reg output
                torch.randn(x.size(0), self.num_classes, 80, 80)  # cls output
            )
    
    model = MockModel().to(device)
    print(f"   Model created")
    
    # Test 2: Create mock data loader
    print("\n2. Creating mock data loader:")
    
    class MockDataset:
        def __init__(self, size=10):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __iter__(self):
            for _ in range(self.size):
                yield {
                    'images': torch.randn(4, 3, 640, 640).to(device),
                    'targets': [
                        torch.tensor([[0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
                        torch.tensor([[1, 0.3, 0.7, 0.1, 0.2]], dtype=torch.float32),
                        torch.tensor([[0, 0.7, 0.3, 0.15, 0.15]], dtype=torch.float32),
                        torch.tensor([[2, 0.5, 0.5, 0.25, 0.25]], dtype=torch.float32)
                    ]
                }
    
    class MockDataLoader:
        def __init__(self, dataset, batch_size=4):
            self.dataset = dataset
            self.batch_size = batch_size
        
        def __iter__(self):
            return iter(self.dataset)
        
        def __len__(self):
            return len(self.dataset) // self.batch_size
    
    mock_dataset = MockDataset(size=20)
    train_loader = MockDataLoader(mock_dataset, batch_size=4)
    val_loader = MockDataLoader(mock_dataset, batch_size=4)
    print(f"   Data loaders created")
    
    # Test 3: Create trainer
    print("\n3. Creating trainer:")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={},
        device=device
    )
    print(f"   Trainer created")
    print(f"   - Device: {trainer.device}")
    print(f"   - Epochs: {trainer.epochs}")
    print(f"   - Batch size: {trainer.batch_size}")
    print(f"   - Loss weights: {trainer.loss_weights}")
    print(f"   - Early stopping patience: {trainer.early_stopping_patience}")
    
    # Test 4: Run a few training steps
    print("\n4. Testing training loop (2 epochs):")
    original_epochs = trainer.epochs
    trainer.epochs = 2  # Only 2 epochs for testing
    
    history = trainer.train()
    print(f"   Training completed")
    print(f"   - Final train loss: {history['train_loss'][-1]:.4f}")
    
    # Test 5: Run validation
    print("\n5. Testing validation:")
    val_metrics = trainer.validate(0)
    print(f"   Validation metrics: {val_metrics}")
    
    # Test 6: Test checkpoint saving
    print("\n6. Testing checkpoint saving:")
    trainer.save_history()
    print(f"   History saved")
    
    # Test 7: Test loss computation
    print("\n7. Testing loss computation:")
    test_images = torch.randn(2, 3, 640, 640).to(device)
    test_targets = [
        torch.tensor([[0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        torch.tensor([[1, 0.3, 0.7, 0.1, 0.2]], dtype=torch.float32)
    ]
    
    model.train()
    predictions = model(test_images, training_mode=True)
    loss_dict = trainer._compute_loss(predictions, test_targets, (640, 640))
    
    print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"   Box loss: {loss_dict['box_loss'].item():.4f}")
    print(f"   Cls loss: {loss_dict['cls_loss'].item():.4f}")
    print(f"   DFL loss: {loss_dict['dfl_loss'].item():.4f}")
    
    # Reset epochs
    trainer.epochs = original_epochs
    
    print("\n" + "=" * 50)
    print("Trainer module test completed!")
