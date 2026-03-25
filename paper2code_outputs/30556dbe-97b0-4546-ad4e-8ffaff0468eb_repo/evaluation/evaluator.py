"""
ECL-YOLOv11 Evaluator Module

This module implements the evaluation pipeline for ECL-YOLOv11 object detection model,
including metrics computation (mAP@50, mAP@50-95, Precision, Recall), FPS measurement,
and comprehensive evaluation reports.

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
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import configuration
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import get_config_manager, get_evaluation_config, get_model_config
except ImportError:
    # Fallback configuration functions
    def get_config_manager():
        return None
    
    def get_evaluation_config():
        return None
    
    def get_model_config():
        return None


# =============================================================================
# Data Classes for Evaluation Results
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    Precision: float = 0.0
    Recall: float = 0.0
    FPS: float = 0.0
    class_aps: Dict[str, float] = field(default_factory=dict)
    class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    num_images: int = 0
    num_predictions: int = 0
    num_ground_truths: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mAP@50': self.mAP50,
            'mAP@50-95': self.mAP50_95,
            'Precision': self.Precision,
            'Recall': self.Recall,
            'FPS': self.FPS,
            'class_aps': self.class_aps,
            'class_metrics': self.class_metrics,
            'num_images': self.num_images,
            'num_predictions': self.num_predictions,
            'num_ground_truths': self.num_ground_truths
        }


@dataclass
class Prediction:
    """Single prediction container."""
    bbox: List[float]  # [x1, y1, x2, y2]
    score: float
    class_id: int
    image_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bbox': self.bbox,
            'score': self.score,
            'class_id': self.class_id,
            'image_id': self.image_id
        }


@dataclass
class GroundTruth:
    """Single ground truth container."""
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    image_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bbox': self.bbox,
            'class_id': self.class_id,
            'image_id': self.image_id
        }


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_EVALUATION_CONFIG = {
    'metrics': ['mAP@50', 'mAP@50-95', 'Precision', 'Recall', 'FPS'],
    'iou_thresholds': {
        'map50': 0.50,
        'map50_95_start': 0.50,
        'map50_95_end': 0.95,
        'steps': 10
    },
    'confidence_thresholds': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    'max_detections': 300,
    'nms_threshold': 0.45,
    'save_predictions': True,
    'visualize_results': True
}

DEFAULT_CLASS_NAMES = ['car', 'person', 'bus', 'bicycle', 'truck', 'train', 'motorcycle']


# =============================================================================
# IoU Computation Functions
# =============================================================================

def compute_iou_box(
    box1: Union[List[float], np.ndarray, torch.Tensor],
    box2: Union[List[float], np.ndarray, torch.Tensor]
) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Determine input type and convert to numpy
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
    if isinstance(box1, (list, tuple)):
        box1 = np.array(box1)
    if isinstance(box2, (list, tuple)):
        box2 = np.array(box2)
    
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    if union <= 0:
        return 0.0
    
    return intersection / union


def compute_iou_matrix(
    boxes1: Union[np.ndarray, torch.Tensor],
    boxes2: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: Array of boxes (N, 4) in format [x1, y1, x2, y2]
        boxes2: Array of boxes (M, 4) in format [x1, y1, x2, y2]
        
    Returns:
        IoU matrix of shape (N, M)
    """
    # Convert to numpy
    if isinstance(boxes1, torch.Tensor):
        boxes1 = boxes1.cpu().numpy()
    if isinstance(boxes2, torch.Tensor):
        boxes2 = boxes2.cpu().numpy()
    
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    
    n1 = len(boxes1)
    n2 = len(boxes2)
    
    # Initialize IoU matrix
    iou_matrix = np.zeros((n1, n2), dtype=np.float32)
    
    if n1 == 0 or n2 == 0:
        return iou_matrix
    
    # Compute areas
    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute IoU for each pair
    for i in range(n1):
        for j in range(n2):
            # Calculate intersection
            x1 = max(boxes1[i, 0], boxes2[j, 0])
            y1 = max(boxes1[i, 1], boxes2[j, 1])
            x2 = min(boxes1[i, 2], boxes2[j, 2])
            y2 = min(boxes1[i, 3], boxes2[j, 3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            union = areas1[i] + areas2[j] - intersection
            
            if union > 0:
                iou_matrix[i, j] = intersection / union
    
    return iou_matrix


# =============================================================================
# NMS (Non-Maximum Suppression)
# =============================================================================

def non_maximum_suppression(
    predictions: List[Prediction],
    iou_threshold: float = 0.45,
    score_threshold: float = 0.001
) -> List[Prediction]:
    """
    Apply Non-Maximum Suppression to predictions.
    
    Args:
        predictions: List of predictions
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score to keep
        
    Returns:
        Filtered list of predictions
    """
    if len(predictions) == 0:
        return []
    
    # Filter by score threshold
    predictions = [p for p in predictions if p.score >= score_threshold]
    
    if len(predictions) == 0:
        return []
    
    # Sort by score (descending)
    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)
    
    # Keep flags
    keep = [True] * len(predictions)
    
    # Apply NMS
    for i in range(len(predictions)):
        if not keep[i]:
            continue
        
        for j in range(i + 1, len(predictions)):
            if not keep[j]:
                continue
            
            # Skip if different class
            if predictions[i].class_id != predictions[j].class_id:
                continue
            
            # Compute IoU
            iou = compute_iou_box(predictions[i].bbox, predictions[j].bbox)
            
            # Suppress if IoU above threshold
            if iou >= iou_threshold:
                keep[j] = False
    
    # Return kept predictions
    return [predictions[i] for i in range(len(predictions)) if keep[i]]


# =============================================================================
# AP Calculation Functions
# =============================================================================

def compute_precision_recall(
    predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    iou_threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute precision and recall for a set of predictions and ground truths.
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (precision, recall)
    """
    if len(ground_truths) == 0:
        # No ground truths: if predictions exist, precision=0, else undefined
        precision = 0.0 if len(predictions) > 0 else 0.0
        recall = 0.0
        return precision, recall
    
    if len(predictions) == 0:
        # No predictions: precision undefined, recall = 0
        return 0.0, 0.0
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)
    
    # Track which ground truths have been matched
    gt_matched = [False] * len(ground_truths)
    
    # Count TP and FP
    tp = 0
    fp = 0
    
    for pred in predictions:
        # Find best matching ground truth
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            
            if pred.class_id != gt.class_id:
                continue
            
            # Compute IoU
            iou = compute_iou_box(pred.bbox, gt.bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if matched
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    # Compute precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(ground_truths) if len(ground_truths) > 0 else 0.0
    
    return precision, recall


def compute_ap_at_threshold(
    predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    iou_threshold: float = 0.5,
    confidence_thresholds: Optional[List[float]] = None
) -> float:
    """
    Compute Average Precision at a specific IoU threshold.
    
    Uses 101-point interpolation (COCO style).
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_threshold: IoU threshold
        confidence_thresholds: List of confidence thresholds to evaluate
        
    Returns:
        Average Precision value
    """
    if len(ground_truths) == 0:
        return 0.0
    
    if len(predictions) == 0:
        return 0.0
    
    # Use default confidence thresholds if not provided
    if confidence_thresholds is None:
        confidence_thresholds = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    # Sort predictions by score (descending)
    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)
    
    # Collect all scores and match status
    scores = []
    matched = []
    
    for thresh in confidence_thresholds:
        # Filter predictions by confidence threshold
        preds_above_thresh = [p for p in predictions if p.score >= thresh]
        
        if len(preds_above_thresh) == 0:
            continue
        
        # Track ground truth matching
        gt_matched = [False] * len(ground_truths)
        
        for pred in preds_above_thresh:
            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx]:
                    continue
                
                if pred.class_id != gt.class_id:
                    continue
                
                iou = compute_iou_box(pred.bbox, gt.bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Record match status
            is_match = best_iou >= iou_threshold and best_gt_idx >= 0
            scores.append(pred.score)
            matched.append(is_match)
            
            if is_match:
                gt_matched[best_gt_idx] = True
    
    if len(scores) == 0:
        return 0.0
    
    # Sort by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    scores = np.array(scores)[sorted_indices]
    matched = np.array(matched)[sorted_indices]
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(matched.astype(int))
    fp_cumsum = np.cumsum((~matched).astype(int))
    
    # Compute precision and recall
    total_gt = len(ground_truths)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / total_gt
    
    # Compute AP using 101-point interpolation
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        if np.sum(recall >= r) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= r])
        ap += p
    
    ap /= 101.0
    
    return ap


def compute_ap_per_class(
    predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    class_ids: List[int],
    iou_threshold: float = 0.5,
    confidence_thresholds: Optional[List[float]] = None
) -> Dict[int, float]:
    """
    Compute AP for each class separately.
    
    Args:
        predictions: List of all predictions
        ground_truths: List of all ground truths
        class_ids: List of class IDs to evaluate
        iou_threshold: IoU threshold
        confidence_thresholds: Confidence thresholds to evaluate
        
    Returns:
        Dictionary mapping class_id to AP value
    """
    ap_per_class = {}
    
    for class_id in class_ids:
        # Filter predictions and ground truths for this class
        class_predictions = [p for p in predictions if p.class_id == class_id]
        class_gts = [gt for gt in ground_truths if gt.class_id == class_id]
        
        # Compute AP for this class
        ap = compute_ap_at_threshold(
            class_predictions,
            class_gts,
            iou_threshold,
            confidence_thresholds
        )
        
        ap_per_class[class_id] = ap
    
    return ap_per_class


# =============================================================================
# FPS Measurement
# =============================================================================

def measure_fps(
    model: nn.Module,
    input_size: Tuple[int, int] = (640, 640),
    num_warmup: int = 20,
    num_iterations: int = 200,
    device: Optional[torch.device] = None,
    batch_size: int = 1
) -> float:
    """
    Measure inference FPS (frames per second) for the model.
    
    Args:
        model: Model to evaluate
        input_size: Input image size (width, height)
        num_warmup: Number of warmup iterations
        num_iterations: Number of timing iterations
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        FPS value
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size[1], input_size[0]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input, training_mode=False)
    
    # Synchronize for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timing loop
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input, training_mode=False)
    
    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate FPS
    total_time = end_time - start_time
    total_images = num_iterations * batch_size
    fps = total_images / total_time
    
    return fps


def measure_fps_simple(
    inference_fn: Callable,
    num_warmup: int = 20,
    num_iterations: int = 200
) -> float:
    """
    Measure FPS using a simple inference function.
    
    Args:
        inference_fn: Function that runs inference (no arguments)
        num_warmup: Number of warmup iterations
        num_iterations: Number of timing iterations
        
    Returns:
        FPS value
    """
    # Warmup
    for _ in range(num_warmup):
        inference_fn()
    
    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timing loop
    start_time = time.time()
    
    for _ in range(num_iterations):
        inference_fn()
    
    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate FPS
    total_time = end_time - start_time
    fps = num_iterations / total_time
    
    return fps


# =============================================================================
# Main Evaluator Class
# =============================================================================

class Evaluator:
    """
    Evaluator class for ECL-YOLOv11 object detection model.
    
    This class handles comprehensive evaluation including:
    - mAP@50 and mAP@50-95 computation
    - Precision and Recall calculation
    - Per-class AP computation
    - FPS measurement
    - Comprehensive evaluation reports
    
    Attributes:
        model: Model to evaluate
        class_names: List of class names
        config: Configuration dictionary
        device: Device for evaluation
        
    Example:
        >>> evaluator = Evaluator(model, class_names)
        >>> metrics = evaluator.evaluate(dataset)
        >>> print(f"mAP@50: {metrics.mAP50}")
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        class_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize Evaluator.
        
        Args:
            model: Model to evaluate (optional, can be set later)
            class_names: List of class names
            config: Configuration dictionary
            device: Device for evaluation
        """
        # Load configuration
        if config is None:
            try:
                config_manager = get_config_manager()
                if config_manager is not None:
                    config = config_manager.get_config_dict()
                else:
                    config = {}
            except:
                config = {}
        
        self.config = config
        
        # Get evaluation config
        eval_config = self._get_evaluation_config()
        
        # Set device
        if device is None:
            device_str = eval_config.get('device', 'cuda')
            device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Set model
        self.model = model
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        
        # Set class names
        if class_names is None:
            try:
                model_config = get_model_config()
                if model_config is not None:
                    class_names = model_config.classes
                else:
                    class_names = DEFAULT_CLASS_NAMES
            except:
                class_names = DEFAULT_CLASS_NAMES
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(class_names)}
        
        # Evaluation parameters
        self.iou_thresholds = eval_config.get('iou_thresholds', DEFAULT_EVALUATION_CONFIG['iou_thresholds'])
        self.confidence_thresholds = eval_config.get('confidence_thresholds', DEFAULT_EVALUATION_CONFIG['confidence_thresholds'])
        self.max_detections = eval_config.get('max_detections', 300)
        self.nms_threshold = eval_config.get('nms_threshold', 0.45)
        
        # Output directory
        self.output_dir = Path(eval_config.get('output_dir', './eval_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluator initialized:")
        print(f"  Device: {self.device}")
        print(f"  Classes: {self.num_classes} ({', '.join(self.class_names)})")
        print(f"  IoU thresholds: {self.iou_thresholds}")
        print(f"  Confidence thresholds: {len(self.confidence_thresholds)} points")
        print(f"  NMS threshold: {self.nms_threshold}")
    
    def _get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration from config or defaults."""
        try:
            eval_config = get_evaluation_config()
            if eval_config is not None:
                return eval_config
        except:
            pass
        
        # Return default config
        return DEFAULT_EVALUATION_CONFIG.copy()
    
    def set_model(self, model: nn.Module) -> None:
        """
        Set the model to evaluate.
        
        Args:
            model: Model to evaluate
        """
        self.model = model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        images: torch.Tensor,
        confidence_threshold: float = 0.25,
        apply_nms: bool = True
    ) -> List[List[Prediction]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: Batch of images (B, C, H, W)
            confidence_threshold: Confidence threshold for filtering
            apply_nms: Whether to apply NMS
            
        Returns:
            List of predictions per image
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            # Run inference
            predictions = self.model(images, training_mode=False)
        
        # Process predictions
        batch_predictions = []
        
        for i in range(len(predictions)):
            pred_dict = predictions[i]
            
            # Extract predictions
            boxes = pred_dict.get('boxes', torch.zeros(0, 4))
            scores = pred_dict.get('scores', torch.zeros(0))
            class_ids = pred_dict.get('class_ids', torch.zeros(0, dtype=torch.long))
            
            # Convert to Prediction objects
            img_predictions = []
            for j in range(len(boxes)):
                if scores[j] < confidence_threshold:
                    continue
                
                pred = Prediction(
                    bbox=boxes[j].cpu().tolist(),
                    score=scores[j].item(),
                    class_id=class_ids[j].item()
                )
                img_predictions.append(pred)
            
            # Apply NMS if requested
            if apply_nms and len(img_predictions) > 0:
                img_predictions = non_maximum_suppression(
                    img_predictions,
                    iou_threshold=self.nms_threshold,
                    score_threshold=confidence_threshold
                )
            
            batch_predictions.append(img_predictions)
        
        return batch_predictions
    
    def evaluate(
        self,
        data_loader: Any,
        confidence_threshold: float = 0.25,
        compute_fps: bool = True,
        verbose: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            confidence_threshold: Confidence threshold for evaluation
            compute_fps: Whether to compute FPS
            verbose: Whether to print progress
            
        Returns:
            EvaluationMetrics object with all metrics
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        if verbose:
            print(f"\nStarting evaluation...")
            print(f"  Confidence threshold: {confidence_threshold}")
            print(f"  NMS threshold: {self.nms_threshold}")
        
        # Collect all predictions and ground truths
        all_predictions: List[Prediction] = []
        all_ground_truths: List[GroundTruth] = []
        num_images = 0
        total_inference_time = 0.0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Get images and targets
                images = batch['images'].to(self.device)
                targets = batch['targets']
                image_ids = batch.get('image_ids', [f"image_{batch_idx}_{i}" for i in range(len(images))])
                
                # Run inference
                batch_predictions = self.predict(images, confidence_threshold, apply_nms=True)
                
                # Collect predictions and ground truths
                for i in range(len(images)):
                    img_pred = batch_predictions[i]
                    img_target = targets[i] if i < len(targets) else torch.zeros(0, 5)
                    
                    # Add predictions
                    for pred in img_pred:
                        pred.image_id = image_ids[i]
                        all_predictions.append(pred)
                    
                    # Add ground truths
                    if len(img_target) > 0:
                        for gt in img_target:
                            class_id = int(gt[0].item()) if isinstance(gt, torch.Tensor) else int(gt[0])
                            # Convert from YOLO format to xyxy
                            x_center = gt[1].item() if isinstance(gt, torch.Tensor) else gt[1]
                            y_center = gt[2].item() if isinstance(gt, torch.Tensor) else gt[2]
                            width = gt[3].item() if isinstance(gt, torch.Tensor) else gt[3]
                            height = gt[4].item() if isinstance(gt, torch.Tensor) else gt[4]
                            
                            # Convert to absolute coordinates
                            h, w = images.shape[2], images.shape[3]
                            x_center_px = x_center * w
                            y_center_px = y_center * h
                            width_px = width * w
                            height_px = height * h
                            
                            # Calculate corners
                            x1 = x_center_px - width_px / 2
                            y1 = y_center_px - height_px / 2
                            x2 = x_center_px + width_px / 2
                            y2 = y_center_px + height_px / 2
                            
                            gt_obj = GroundTruth(
                                bbox=[x1, y1, x2, y2],
                                class_id=class_id,
                                image_id=image_ids[i]
                            )
                            all_ground_truths.append(gt_obj)
                
                num_images += len(images)
                
                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        if verbose:
            print(f"  Total images: {num_images}")
            print(f"  Total predictions: {len(all_predictions)}")
            print(f"  Total ground truths: {len(all_ground_truths)}")
        
        # Compute mAP@50
        if verbose:
            print(f"\nComputing mAP@50...")
        
        iou_threshold_50 = self.iou_thresholds.get('map50', 0.50)
        
        # Compute AP per class at IoU=0.50
        class_ids = list(range(self.num_classes))
        ap_per_class_50 = compute_ap_per_class(
            all_predictions,
            all_ground_truths,
            class_ids,
            iou_threshold_50,
            self.confidence_thresholds
        )
        
        # Compute overall AP@50
        mAP50 = np.mean(list(ap_per_class_50.values())) if ap_per_class_50 else 0.0
        
        # Compute mAP@50-95
        if verbose:
            print(f"Computing mAP@50-95...")
        
        map50_95_values = []
        iou_start = self.iou_thresholds.get('map50_95_start', 0.50)
        iou_end = self.iou_thresholds.get('map50_95_end', 0.95)
        steps = self.iou_thresholds.get('steps', 10)
        
        iou_thresholds_95 = np.linspace(iou_start, iou_end, steps)
        
        for iou_th in iou_thresholds_95:
            ap_per_class = compute_ap_per_class(
                all_predictions,
                all_ground_truths,
                class_ids,
                iou_th,
                self.confidence_thresholds
            )
            map50_95_values.append(np.mean(list(ap_per_class.values())))
        
        mAP50_95 = np.mean(map50_95_values)
        
        # Compute Precision and Recall at the given confidence threshold
        if verbose:
            print(f"Computing Precision and Recall...")
        
        precision, recall = compute_precision_recall(
            all_predictions,
            all_ground_truths,
            iou_threshold_50
        )
        
        # Compute FPS
        fps = 0.0
        if compute_fps:
            if verbose:
                print(f"Measuring FPS...")
            try:
                fps = measure_fps(
                    self.model,
                    input_size=(640, 640),
                    num_warmup=20,
                    num_iterations=100,
                    device=self.device
                )
            except Exception as e:
                print(f"  Warning: FPS measurement failed: {e}")
        
        # Create metrics
        class_aps = {
            self.idx_to_class[idx]: ap 
            for idx, ap in ap_per_class_50.items()
        }
        
        # Compute per-class precision and recall
        class_metrics = {}
        for class_id in class_ids:
            class_preds = [p for p in all_predictions if p.class_id == class_id]
            class_gts = [gt for gt in all_ground_truths if gt.class_id == class_id]
            
            if len(class_gts) > 0:
                p, r = compute_precision_recall(class_preds, class_gts, iou_threshold_50)
                class_metrics[self.idx_to_class[class_id]] = {
                    'precision': p,
                    'recall': r,
                    'ap': ap_per_class_50.get(class_id, 0.0)
                }
        
        metrics = EvaluationMetrics(
            mAP50=mAP50,
            mAP50_95=mAP50_95,
            Precision=precision,
            Recall=recall,
            FPS=fps,
            class_aps=class_aps,
            class_metrics=class_metrics,
            num_images=num_images,
            num_predictions=len(all_predictions),
            num_ground_truths=len(all_ground_truths)
        )
        
        if verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def evaluate_with_fps(
        self,
        data_loader: Any,
        num_fps_iterations: int = 100,
        confidence_threshold: float = 0.25,
        verbose: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate and measure FPS separately.
        
        Args:
            data_loader: DataLoader for evaluation
            num_fps_iterations: Number of iterations for FPS measurement
            confidence_threshold: Confidence threshold
            verbose: Whether to print progress
            
        Returns:
            EvaluationMetrics with FPS
        """
        # First evaluate without FPS
        metrics = self.evaluate(
            data_loader,
            confidence_threshold=confidence_threshold,
            compute_fps=False,
            verbose=verbose
        )
        
        # Then measure FPS separately
        if verbose:
            print(f"Measuring FPS...")
        
        fps = measure_fps(
            self.model,
            input_size=(640, 640),
            num_warmup=20,
            num_iterations=num_fps_iterations,
            device=self.device
        )
        
        metrics.FPS = fps
        
        if verbose:
            print(f"  FPS: {fps:.2f}")
        
        return metrics
    
    def _print_metrics(self, metrics: EvaluationMetrics) -> None:
        """Print evaluation metrics in a formatted way."""
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"mAP@50:        {metrics.mAP50:.4f}")
        print(f"mAP@50-95:     {metrics.mAP50_95:.4f}")
        print(f"Precision:     {metrics.Precision:.4f}")
        print(f"Recall:        {metrics.Recall:.4f}")
        print(f"FPS:           {metrics.FPS:.2f}")
        print(f"\nPer-class AP@50:")
        for class_name, ap in metrics.class_aps.items():
            print(f"  {class_name:12s}: {ap:.4f}")
        print(f"\nStatistics:")
        print(f"  Images:           {metrics.num_images}")
        print(f"  Predictions:      {metrics.num_predictions}")
        print(f"  Ground Truths:    {metrics.num_ground_truths}")
        print(f"{'='*60}\n")
    
    def save_metrics(
        self,
        metrics: EvaluationMetrics,
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics: EvaluationMetrics to save
            output_path: Output file path (optional)
        """
        if output_path is None:
            output_path = self.output_dir / 'evaluation_metrics.json'
        else:
            output_path = Path(output_path)
        
        # Convert to dict and save
        metrics_dict = metrics.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"Metrics saved to {output_path}")
    
    def compare_with_baseline(
        self,
        baseline_metrics: EvaluationMetrics,
        current_metrics: EvaluationMetrics,
        model_names: Tuple[str, str] = ("Baseline", "ECL-YOLOv11")
    ) -> Dict[str, Any]:
        """
        Compare current metrics with baseline metrics.
        
        Args:
            baseline_metrics: Baseline model metrics
            current_metrics: Current model metrics
            model_names: Tuple of (baseline_name, current_name)
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\n{'='*70}")
        print(f"Comparison: {model_names[0]} vs {model_names[1]}")
        print(f"{'='*70}")
        
        # Metrics to compare
        metrics_names = ['mAP50', 'mAP50_95', 'Precision', 'Recall', 'FPS']
        
        print(f"{'Metric':<20} {model_names[0]:>15} {model_names[1]:>15} {'Change':>15}")
        print(f"{'-'*70}")
        
        comparison = {}
        
        for metric_name in metrics_names:
            baseline_val = getattr(baseline_metrics, metric_name, 0.0)
            current_val = getattr(current_metrics, metric_name, 0.0)
            change = current_val - baseline_val
            change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0.0
            
            change_str = f"+{change:.4f} ({change_pct:+.2f}%)" if change >= 0 else f"{change:.4f} ({change_pct:.2f}%)"
            
            print(f"{metric_name:<20} {baseline_val:>15.4f} {current_val:>15.4f} {change_str:>15}")
            
            comparison[metric_name] = {
                'baseline': baseline_val,
                'current': current_val,
                'change': change,
                'change_percent': change_pct
            }
        
        # Per-class comparison
        print(f"\nPer-class AP@50 comparison:")
        print(f"{'Class':<20} {model_names[0]:>15} {model_names[1]:>15} {'Change':>15}")
        print(f"{'-'*70}")
        
        class_comparison = {}
        
        for class_name in current_metrics.class_aps.keys():
            baseline_ap = baseline_metrics.class_aps.get(class_name, 0.0)
            current_ap = current_metrics.class_aps.get(class_name, 0.0)
            change = current_ap - baseline_ap
            change_pct = (change / baseline_ap * 100) if baseline_ap != 0 else 0.0
            
            change_str = f"+{change:.4f} ({change_pct:+.2f}%)" if change >= 0 else f"{change:.4f} ({change_pct:.2f}%)"
            
            print(f"{class_name:<20} {baseline_ap:>15.4f} {current_ap:>15.4f} {change_str:>15}")
            
            class_comparison[class_name] = {
                'baseline': baseline_ap,
                'current': current_ap,
                'change': change,
                'change_percent': change_pct
            }
        
        print(f"{'='*70}\n")
        
        comparison['class_comparison'] = class_comparison
        
        return comparison


# =============================================================================
# Utility Functions
# =============================================================================

def create_evaluator(
    model: nn.Module,
    config_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> Evaluator:
    """
    Factory function to create an Evaluator instance.
    
    Args:
        model: Model to evaluate
        config_path: Path to configuration file
        device: Device for evaluation
        
    Returns:
        Evaluator: Configured evaluator instance
    """
    # Load configuration
    config = None
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
    
    # Get class names from config
    class_names = None
    if config:
        model_config = config.get('model', {})
        class_names = model_config.get('classes', None)
    
    return Evaluator(
        model=model,
        class_names=class_names,
        config=config,
        device=device
    )


def load_predictions_from_file(prediction_path: Path) -> List[Prediction]:
    """
    Load predictions from a JSON file.
    
    Args:
        prediction_path: Path to prediction file
        
    Returns:
        List of Prediction objects
    """
    with open(prediction_path, 'r') as f:
        pred_dicts = json.load(f)
    
    predictions = []
    for pd in pred_dicts:
        pred = Prediction(
            bbox=pd['bbox'],
            score=pd['score'],
            class_id=pd['class_id'],
            image_id=pd.get('image_id', '')
        )
        predictions.append(pred)
    
    return predictions


def load_ground_truths_from_file(ground_truth_path: Path) -> List[GroundTruth]:
    """
    Load ground truths from a JSON file.
    
    Args:
        ground_truth_path: Path to ground truth file
        
    Returns:
        List of GroundTruth objects
    """
    with open(ground_truth_path, 'r') as f:
        gt_dicts = json.load(f)
    
    ground_truths = []
    for gd in gt_dicts:
        gt = GroundTruth(
            bbox=gd['bbox'],
            class_id=gd['class_id'],
            image_id=gd.get('image_id', '')
        )
        ground_truths.append(gt)
    
    return ground_truths


# =============================================================================
# Test/Verification Code
# =============================================================================

if __name__ == "__main__":
    # Test evaluator module
    print("Testing ECL-YOLOv11 Evaluator Module")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test 1: Test IoU computation
    print("\n1. Testing IoU computation:")
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    iou = compute_iou_box(box1, box2)
    print(f"   Box1: {box1}")
    print(f"   Box2: {box2}")
    print(f"   IoU: {iou:.4f}")
    
    # Test IoU matrix
    boxes1 = np.array([[0, 0, 100, 100], [200, 200, 300, 300]])
    boxes2 = np.array([[50, 50, 150, 150], [250, 250, 350, 350]])
    iou_mat = compute_iou_matrix(boxes1, boxes2)
    print(f"   IoU matrix:\n{iou_mat}")
    
    # Test 2: Test NMS
    print("\n2. Testing NMS:")
    predictions = [
        Prediction([10, 10, 50, 50], 0.9, 0),
        Prediction([15, 15, 55, 55], 0.8, 0),
        Prediction([100, 100, 200, 200], 0.7, 1),
        Prediction([300, 300, 400, 400], 0.6, 0),
    ]
    nms_predictions = non_maximum_suppression(predictions, iou_threshold=0.5)
    print(f"   Input predictions: {len(predictions)}")
    print(f"   Output predictions: {len(nms_predictions)}")
    
    # Test 3: Test precision/recall computation
    print("\n3. Testing precision/recall computation:")
    predictions = [
        Prediction([10, 10, 50, 50], 0.9, 0),
        Prediction([15, 15, 55, 55], 0.8, 0),
        Prediction([100, 100, 200, 200], 0.7, 1),
    ]
    ground_truths = [
        GroundTruth([10, 10, 50, 50], 0),
        GroundTruth([100, 100, 200, 200], 1),
        GroundTruth([300, 300, 400, 400], 0),  # Not matched
    ]
    precision, recall = compute_precision_recall(predictions, ground_truths, iou_threshold=0.5)
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    
    # Test 4: Test AP computation
    print("\n4. Testing AP computation:")
    ap = compute_ap_at_threshold(predictions, ground_truths, iou_threshold=0.5)
    print(f"   AP@50: {ap:.4f}")
    
    # Test 5: Test Evaluator initialization
    print("\n5. Testing Evaluator initialization:")
    
    # Create mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_classes = 7
            self.reg_max = 16
        
        def forward(self, x, training_mode=True):
            if training_mode:
                return (
                    torch.randn(x.size(0), 64, 80, 80),
                    torch.randn(x.size(0), self.num_classes, 80, 80)
                )
            else:
                # Return empty predictions
                results = []
                for _ in range(x.size(0)):
                    results.append({
                        'boxes': torch.zeros(0, 4),
                        'scores': torch.zeros(0),
                        'class_ids': torch.zeros(0, dtype=torch.long)
                    })
                return results
    
    model = MockModel()
    evaluator = Evaluator(model=model, class_names=DEFAULT_CLASS_NAMES)
    print(f"   Evaluator created successfully")
    print(f"   Classes: {evaluator.class_names}")
    print(f"   Num classes: {evaluator.num_classes}")
    
    # Test 6: Test FPS measurement
    print("\n6. Testing FPS measurement:")
    try:
        fps = measure_fps(model, input_size=(640, 640), num_warmup=5, num_iterations=10, device=device)
        print(f"   FPS: {fps:.2f}")
    except Exception as e:
        print(f"   FPS measurement: {e}")
    
    # Test 7: Test metrics structure
    print("\n7. Testing EvaluationMetrics structure:")
    metrics = EvaluationMetrics(
        mAP50=0.627,
        mAP50_95=0.405,
        Precision=0.731,
        Recall=0.556,
        FPS=237.5,
        class_aps={'car': 0.855, 'person': 0.734, 'bus': 0.462},
        class_metrics={'car': {'precision': 0.8, 'recall': 0.7, 'ap': 0.855}},
        num_images=100,
        num_predictions=500,
        num_ground_truths=450
    )
    print(f"   mAP@50: {metrics.mAP50}")
    print(f"   mAP@50-95: {metrics.mAP50_95}")
    print(f"   Precision: {metrics.Precision}")
    print(f"   Recall: {metrics.Recall}")
    print(f"   FPS: {metrics.FPS}")
    print(f"   Class APs: {metrics.class_aps}")
    print(f"   to_dict: {metrics.to_dict()}")
    
    # Test 8: Test save/load predictions
    print("\n8. Testing prediction I/O:")
    test_preds = [
        Prediction([10, 10, 50, 50], 0.9, 0, "img1"),
        Prediction([100, 100, 200, 200], 0.7, 1, "img2"),
    ]
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([p.to_dict() for p in test_preds], f)
        temp_path = f.name
    
    loaded_preds = load_predictions_from_file(Path(temp_path))
    print(f"   Saved predictions: {len(test_preds)}")
    print(f"   Loaded predictions: {len(loaded_preds)}")
    print(f"   First prediction: {loaded_preds[0].to_dict()}")
    
    os.remove(temp_path)
    
    print("\n" + "=" * 50)
    print("Evaluator module test completed!")
