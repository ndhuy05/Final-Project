"""
ECL-YOLOv11: Complete Model Implementation

This module implements the complete ECL-YOLOv11 object detection architecture by integrating:
1. CE (Convolutional Edge-Enhancement) modules in the backbone
2. AENet (Context-Guided Multi-Scale Fusion Network) in the neck
3. LDHead (Lightweight Shared Convolutional Detection Head)

Based on the paper: "Robust Object Detection in Adverse Weather Conditions: 
ECL-YOLOv11 for Automotive Vision Systems"

Author: ECL-YOLOv11 Reproduction Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import custom modules
try:
    from model.modules.ce_module import CEModule, CEBackbone
    from model.modules.aenet import AENet
    from model.modules.ldhead import LDHead
except ImportError:
    # Fallback imports if modules not available
    from .modules.ce_module import CEModule, CEBackbone
    from .modules.aenet import AENet
    from .modules.ldhead import LDHead

# Try to import configuration
try:
    from utils.config import get_config_manager, load_config
except ImportError:
    # Fallback configuration
    def get_config_manager():
        return None
    
    def load_config(path=None):
        return None


# =============================================================================
# Complete ECL-YOLOv11 Model
# =============================================================================

class ECLYOLOv11(nn.Module):
    """
    ECL-YOLOv11: Edge-enhanced, Context-guided, and Lightweight YOLOv11.
    
    This is the complete object detection model that integrates three key modules:
    1. CE (Convolutional Edge-Enhancement) module in the backbone for edge preservation
    2. AENet (Context-Guided Multi-Scale Fusion Network) in the neck for semantic fusion
    3. LDHead (Lightweight Shared Convolutional Detection Head) in the head for efficient inference
    
    Architecture:
        Input Image -> Backbone (with CE) -> Neck (AENet) -> Head (LDHead) -> Predictions
    
    Attributes:
        num_classes (int): Number of object classes
        reg_max (int): Maximum value for DFL regression discretization
        channels (List[int]): Channel dimensions for multi-scale features [P3, P4, P5]
        
    Input:
        x (torch.Tensor): Input image tensor of shape (B, 3, H, W)
        
    Output:
        Training: Tuple of (reg_outputs, cls_outputs) - raw predictions for loss computation
        Inference: Post-processed detections with boxes, scores, and class IDs
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        reg_max: int = 16,
        channels: Optional[List[int]] = None,
        image_size: int = 640,
        use_ce_in_backbone: bool = True,
        use_aenet_in_neck: bool = True,
        use_ldhead_in_head: bool = True,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ECL-YOLOv11 model.
        
        Args:
            num_classes: Number of object classes (default: 7 for paper's dataset)
            reg_max: Maximum value for distribution focal loss (default: 16)
            channels: List of channel dimensions for P3, P4, P5 features
            image_size: Input image size (default: 640)
            use_ce_in_backbone: Whether to use CE modules in backbone
            use_aenet_in_neck: Whether to use AENet in neck
            use_ldhead_in_head: Whether to use LDHead in head
            config: Optional configuration dictionary
            device: Optional torch device
        """
        super().__init__()
        
        # Try to load configuration
        if config is None:
            try:
                config_manager = get_config_manager()
                if config_manager is not None:
                    config = config_manager.get_config_dict()
            except:
                config = {}
        
        # Apply configuration defaults
        self.num_classes = config.get('model', {}).get('num_classes', num_classes) if isinstance(config, dict) else num_classes
        self.reg_max = config.get('model', {}).get('reg_max', reg_max) if isinstance(config, dict) else reg_max
        self.image_size = config.get('training', {}).get('image_size', image_size) if isinstance(config, dict) else image_size
        
        # Default channel dimensions
        if channels is None:
            channels = config.get('model', {}).get('aenet', {}).get('pyramid_channels', [256, 512, 1024]) if isinstance(config, dict) else [256, 512, 1024]
        self.channels = channels
        
        # Module flags
        self.use_ce_in_backbone = use_ce_in_backbone
        self.use_aenet_in_neck = use_aenet_in_neck
        self.use_ldhead_in_head = use_ldhead_in_head
        
        # Device
        self._device = device
        
        # Get CE module configuration
        ce_config = None
        if config:
            ce_config = config.get('model', {}).get('ce_module', {})
        
        # Get AENet configuration
        aenet_config = None
        if config:
            aenet_config = config.get('model', {}).get('aenet', {})
        
        # Get LDHead configuration
        ldhead_config = None
        if config:
            ldhead_config = config.get('model', {}).get('ldhead', {})
        
        # =====================================================================
        # Build Backbone with CE modules
        # =====================================================================
        if use_ce_in_backbone:
            self.backbone = CEBackbone(
                in_channels=3,
                channels=tuple(channels),
                num_classes=self.num_classes,
                use_ce_at_stages=(True, True, False),  # CE at P3 and P4, not P5
                config=ce_config
            )
        else:
            # Standard YOLOv11-style backbone without CE
            self.backbone = self._build_standard_backbone()
        
        # =====================================================================
        # Build Neck with AENet
        # =====================================================================
        if use_aenet_in_neck:
            self.neck = AENet(
                in_channels_list=channels,
                out_channels=channels[0],  # Use P3 channel count as output
                rcm_stages=aenet_config.get('rcm_stages', 2) if aenet_config else 2,
                use_dif=aenet_config.get('use_dif', True) if aenet_config else True,
                use_fbm=aenet_config.get('use_fbm', True) if aenet_config else True,
                config=aenet_config
            )
        else:
            # Standard YOLOv11-style neck (simplified FPN)
            self.neck = self._build_standard_neck()
        
        # =====================================================================
        # Build Detection Head with LDHead
        # =====================================================================
        if use_ldhead_in_head:
            # For LDHead, we need to adjust channels to match AENet output
            # AENet outputs all scales with the same channel count (channels[0])
            head_channels = [channels[0], channels[0], channels[0]]  # All scales same channels
            self.head = LDHead(
                num_classes=self.num_classes,
                in_channels_list=head_channels,
                reg_max=self.reg_max,
                groupnorm_groups=ldhead_config.get('groupnorm_groups', 32) if ldhead_config else 32,
                use_groupnorm=ldhead_config.get('use_groupnorm', True) if ldhead_config else True,
                depthwise_separable=ldhead_config.get('depthwise_separable', True) if ldhead_config else True,
                config=ldhead_config
            )
        else:
            # Standard detection head
            self.head = self._build_standard_head()
        
        # Initialize weights
        self._init_weights()
    
    def _build_standard_backbone(self) -> nn.Module:
        """
        Build a standard YOLOv11-style backbone without CE modules.
        
        Returns:
            nn.Module: Standard backbone network
        """
        channels = self.channels
        
        class StandardBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                # Stem
                self.stem = nn.Sequential(
                    nn.Conv2d(3, channels[0] // 4, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels[0] // 4),
                    nn.SiLU(inplace=True)
                )
                
                # Stage 1 -> P3
                self.stage1 = nn.Sequential(
                    nn.Conv2d(channels[0] // 4, channels[0] // 2, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels[0] // 2),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels[0] // 2, channels[0], 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(channels[0]),
                    nn.SiLU(inplace=True)
                )
                
                # Stage 2 -> P4
                self.stage2 = nn.Sequential(
                    nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels[1]),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels[1], channels[1], 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(channels[1]),
                    nn.SiLU(inplace=True)
                )
                
                # Stage 3 -> P5
                self.stage3 = nn.Sequential(
                    nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels[2]),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(channels[2], channels[2], 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(channels[2]),
                    nn.SiLU(inplace=True)
                )
            
            def forward(self, x):
                x = self.stem(x)
                p3 = self.stage1(x)
                p4 = self.stage2(p3)
                p5 = self.stage3(p4)
                return p3, p4, p5
        
        return StandardBackbone()
    
    def _build_standard_neck(self) -> nn.Module:
        """
        Build a standard FPN-style neck without AENet.
        
        Returns:
            nn.Module: Standard neck network
        """
        channels = self.channels
        
        class StandardNeck(nn.Module):
            def __init__(self):
                super().__init__()
                # Top-down pathway
                self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
                
                # Lateral convolutions
                self.lateral_conv0 = nn.Conv2d(channels[2], channels[1], 1, bias=False)
                self.lateral_conv1 = nn.Conv2d(channels[1], channels[0], 1, bias=False)
                
                # Output convolutions
                self.c3_1 = nn.Conv2d(channels[1] + channels[1], channels[1], 3, padding=1, bias=False)
                self.c3_2 = nn.Conv2d(channels[0] + channels[0], channels[0], 3, padding=1, bias=False)
            
            def forward(self, features):
                p3, p4, p5 = features
                
                # Top-down pathway
                p5_up = self.upsample(p5)
                p4_out = self.c3_1(torch.cat([self.lateral_conv0(p5_up), p4], dim=1))
                
                p4_up = self.upsample(p4_out)
                p3_out = self.c3_2(torch.cat([self.lateral_conv1(p4_up), p3], dim=1))
                
                return [p3_out, p4_out, p5]
        
        return StandardNeck()
    
    def _build_standard_head(self) -> nn.Module:
        """
        Build a standard YOLOv11 detection head without LDHead.
        
        Returns:
            nn.Module: Standard detection head
        """
        channels = self.channels
        num_classes = self.num_classes
        reg_max = self.reg_max
        
        class StandardHead(nn.Module):
            def __init__(self):
                super().__init__()
                # Detection convolutions for each scale
                self.det_convs = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(ch, 2 * ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(2 * ch),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(2 * ch, 2 * ch, 3, padding=1, bias=False),
                        nn.BatchNorm2d(2 * ch),
                        nn.SiLU(inplace=True)
                    )
                    for ch in channels
                ])
                
                # Regression and classification heads
                self.reg_heads = nn.ModuleList([
                    nn.Conv2d(ch, 4 * reg_max, 1) for ch in channels
                ])
                self.cls_heads = nn.ModuleList([
                    nn.Conv2d(ch, num_classes, 1) for ch in channels
                ])
            
            def forward(self, features):
                outputs = []
                for i, feat in enumerate(features):
                    x = self.det_convs[i](feat)
                    reg_out = self.reg_heads[i](x)
                    cls_out = torch.sigmoid(self.cls_heads[i](x))
                    outputs.append((reg_out, cls_out))
                return outputs
        
        return StandardHead()
    
    def _init_weights(self) -> None:
        """
        Initialize model weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor,
        training_mode: bool = True
    ) -> Union[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]], 
                List[Dict[str, torch.Tensor]]]:
        """
        Forward pass through ECL-YOLOv11.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            training_mode: If True, return raw predictions for training;
                          If False, return post-processed detections for inference
            
        Returns:
            Training mode: Tuple of (reg_outputs, cls_outputs)
                - reg_outputs: List of regression tensors for each scale
                - cls_outputs: List of classification tensors for each scale
            Inference mode: List of detection dictionaries with 'boxes', 'scores', 'class_ids'
        """
        # =====================================================================
        # Step 1: Backbone feature extraction
        # =====================================================================
        features = self.backbone(x)  # Returns (p3, p4, p5)
        
        # =====================================================================
        # Step 2: AENet multi-scale fusion
        # =====================================================================
        fused_features = self.neck(features)  # Returns [p3_fused, p4_fused, p5_fused]
        
        # =====================================================================
        # Step 3: Detection head
        # =====================================================================
        if self.use_ldhead_in_head:
            reg_output, cls_output = self.head(fused_features)
            
            # For training: return raw predictions
            if training_mode:
                # Reshape outputs to match expected format
                # reg_output: (B, 4*reg_max, H, W)
                # cls_output: (B, num_classes, H, W)
                return reg_output, cls_output
            else:
                # For inference: apply post-processing
                return self._post_process(reg_output, cls_output, x.shape[2:])
        else:
            # Standard head
            outputs = self.head(fused_features)
            
            if training_mode:
                reg_outputs = [o[0] for o in outputs]
                cls_outputs = [o[1] for o in outputs]
                return reg_outputs, cls_outputs
            else:
                # Combine outputs from all scales
                all_reg = torch.cat([o[0] for o in outputs], dim=1)
                all_cls = torch.cat([o[1] for o in outputs], dim=1)
                return self._post_process(all_reg, all_cls, x.shape[2:])
    
    def _post_process(
        self,
        reg_output: torch.Tensor,
        cls_output: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process raw predictions to obtain final detections.
        
        This is a simplified post-processing. For production use, implement
        proper NMS with confidence thresholding.
        
        Args:
            reg_output: Regression outputs from head
            cls_output: Classification outputs from head
            original_size: Original image size (H, W)
            
        Returns:
            List of detection dictionaries with 'boxes', 'scores', 'class_ids'
        """
        # Get batch size
        batch_size = reg_output.shape[0]
        
        # Parse dimensions
        num_classes = self.num_classes
        reg_max = self.reg_max
        
        # Reshape regression output to (B, 4, reg_max, H, W)
        reg_output = reg_output.view(batch_size, 4, reg_max, -1)
        
        # Apply softmax to get probability distribution over DFL
        reg_output = F.softmax(reg_output, dim=2)
        
        # Compute bounding box coordinates from distribution
        # This is a simplified version - full DFL would compute proper offsets
        # For now, use simple decoding
        h, w = reg_output.shape[3], reg_output.shape[4]
        
        # Generate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=reg_output.device),
            torch.arange(w, device=reg_output.device),
            indexing='ij'
        )
        
        # Extract predictions for each image in batch
        detections = []
        
        for b in range(batch_size):
            # Get class predictions
            cls_scores = cls_output[b]  # (num_classes, H, W)
            
            # Get maximum class score and class ID for each position
            max_scores, pred_classes = cls_scores.max(dim=0)  # (H, W)
            
            # Flatten for easier processing
            max_scores = max_scores.flatten()
            pred_classes = pred_classes.flatten()
            
            # Apply confidence threshold
            conf_threshold = 0.25
            mask = max_scores > conf_threshold
            
            if mask.sum() == 0:
                # No detections above threshold
                detections.append({
                    'boxes': torch.empty((0, 4), device=reg_output.device),
                    'scores': torch.empty(0, device=reg_output.device),
                    'class_ids': torch.empty(0, dtype=torch.long, device=reg_output.device)
                })
                continue
            
            # Filter by confidence
            filtered_scores = max_scores[mask]
            filtered_classes = pred_classes[mask]
            
            # Get regression predictions for filtered positions
            # This is simplified - proper implementation would decode DFL
            reg_pred = reg_output[b]  # (4, reg_max, H, W)
            
            # For simplicity, use center of feature map as box center
            # and a fixed scale
            grid_flat = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)[mask]
            
            # Create boxes (normalized coordinates)
            # Convert grid positions to normalized coordinates
            stride_h = original_size[0] / h
            stride_w = original_size[1] / w
            
            cx = (grid_flat[:, 0].float() + 0.5) * stride_w / original_size[1]
            cy = (grid_flat[:, 1].float() + 0.5) * stride_h / original_size[0]
            
            # Fixed box size (simplified)
            box_size = 0.1
            x1 = (cx - box_size / 2).clamp(0, 1)
            y1 = (cy - box_size / 2).clamp(0, 1)
            x2 = (cx + box_size / 2).clamp(0, 1)
            y2 = (cy + box_size / 2).clamp(0, 1)
            
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            
            # Apply NMS (simplified)
            if boxes.shape[0] > 0:
                # Use PyTorch's NMS
                keep_indices = self._nms(boxes, filtered_scores, iou_threshold=0.45)
                boxes = boxes[keep_indices]
                filtered_scores = filtered_scores[keep_indices]
                filtered_classes = filtered_classes[keep_indices]
            
            detections.append({
                'boxes': boxes,
                'scores': filtered_scores,
                'class_ids': filtered_classes
            })
        
        return detections
    
    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float = 0.45
    ) -> torch.Tensor:
        """
        Apply Non-Maximum Suppression.
        
        Args:
            boxes: Bounding boxes (N, 4) in normalized coordinates
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Indices of boxes to keep
        """
        if boxes.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # Convert to xyxy format if not already
        # boxes should be in format [x1, y1, x2, y2]
        
        # Sort by scores
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep = []
        while sorted_indices.shape[0] > 0:
            # Get the box with highest score
            current_idx = sorted_indices[0]
            keep.append(current_idx)
            
            if sorted_indices.shape[0] == 1:
                break
            
            # Compute IoU with remaining boxes
            current_box = boxes[current_idx].unsqueeze(0)
            remaining_boxes = boxes[sorted_indices[1:]]
            
            # Calculate intersection coordinates
            x1 = torch.max(current_box[:, 0], remaining_boxes[:, 0])
            y1 = torch.max(current_box[:, 1], remaining_boxes[:, 1])
            x2 = torch.min(current_box[:, 2], remaining_boxes[:, 2])
            y2 = torch.min(current_box[:, 3], remaining_boxes[:, 3])
            
            # Calculate intersection area
            intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            
            # Calculate union area
            current_area = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
            remaining_area = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = current_area + remaining_area - intersection
            
            # Calculate IoU
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU below threshold
            sorted_indices = sorted_indices[1:][iou < iou_threshold]
        
        return torch.stack(keep) if len(keep) > 0 else torch.empty(0, dtype=torch.long, device=boxes.device)
    
    def get_num_parameters(self) -> int:
        """
        Get the total number of parameters in the model.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self) -> torch.device:
        """
        Get the device of the model.
        
        Returns:
            torch.device: Device where model parameters are located
        """
        return next(self.parameters()).device
    
    def to(self, device: Union[str, torch.device]) -> 'ECLYOLOv11':
        """
        Move the model to the specified device.
        
        Args:
            device: Target device ('cuda', 'cpu', or torch.device)
            
        Returns:
            self: Model on the target device
        """
        self._device = torch.device(device) if isinstance(device, str) else device
        return super().to(device)
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model architecture.
        
        Returns:
            Dictionary containing model summary information
        """
        total_params = self.get_num_parameters()
        trainable_params = self.get_num_trainable_parameters()
        
        # Count parameters by component
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        return {
            'model_name': 'ECL-YOLOv11',
            'num_classes': self.num_classes,
            'reg_max': self.reg_max,
            'channels': self.channels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'neck_parameters': neck_params,
            'head_parameters': head_params,
            'device': str(self.get_device()),
            'use_ce': self.use_ce_in_backbone,
            'use_aenet': self.use_aenet_in_neck,
            'use_ldhead': self.use_ldhead_in_head
        }
    
    def load_pretrained(self, path: Union[str, Path], strict: bool = False) -> None:
        """
        Load pretrained weights from a checkpoint.
        
        Args:
            path: Path to the checkpoint file
            strict: Whether to strictly enforce key matching
        """
        path = Path(path)
        if not path.exists():
            print(f"Warning: Checkpoint file not found at {path}")
            return
        
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Try to load with strict=False to allow for minor differences
        try:
            self.load_state_dict(state_dict, strict=strict)
            print(f"Successfully loaded pretrained weights from {path}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights strictly: {e}")
            # Try loading with strict=False
            self.load_state_dict(state_dict, strict=False)
            print("Loaded weights with strict=False (some mismatches ignored)")
    
    def freeze_backbone(self) -> None:
        """
        Freeze the backbone parameters for transfer learning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def freeze_neck(self) -> None:
        """
        Freeze the neck parameters for transfer learning.
        """
        for param in self.neck.parameters():
            param.requires_grad = False
    
    def freeze_head(self) -> None:
        """
        Freeze the head parameters for transfer learning.
        """
        for param in self.head.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self) -> None:
        """
        Unfreeze all parameters for fine-tuning.
        """
        for param in self.parameters():
            param.requires_grad = True


# =============================================================================
# Factory Functions
# =============================================================================

def create_ecl_yolov11(
    num_classes: int = 7,
    reg_max: int = 16,
    channels: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    pretrained: Optional[Union[str, Path]] = None
) -> ECLYOLOv11:
    """
    Factory function to create ECL-YOLOv11 model.
    
    Args:
        num_classes: Number of object classes
        reg_max: Maximum value for DFL regression
        channels: Channel dimensions for multi-scale features
        config: Configuration dictionary
        device: Target device
        pretrained: Path to pretrained weights (optional)
        
    Returns:
        ECLYOLOv11: Configured model instance
    """
    model = ECLYOLOv11(
        num_classes=num_classes,
        reg_max=reg_max,
        channels=channels,
        config=config,
        device=device
    )
    
    if pretrained:
        model.load_pretrained(pretrained)
    
    return model


def create_ecl_yolov11_from_config(
    config_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> ECLYOLOv11:
    """
    Create ECL-YOLOv11 model from configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        device: Target device
        
    Returns:
        ECLYOLOv11: Configured model instance
    """
    # Load configuration
    config = {}
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get model configuration
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    return create_ecl_yolov11(
        num_classes=model_config.get('num_classes', 7),
        reg_max=model_config.get('reg_max', 16),
        channels=model_config.get('aenet', {}).get('pyramid_channels', [256, 512, 1024]),
        config=config,
        device=device
    )


# =============================================================================
# Test/Verification Code
# =============================================================================

if __name__ == "__main__":
    # Test ECL-YOLOv11 implementation
    print("Testing ECL-YOLOv11 Implementation")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test 1: Create model with default configuration
    print("\n1. Testing model creation:")
    model = ECLYOLOv11(
        num_classes=7,
        reg_max=16,
        channels=[256, 512, 1024]
    ).to(device)
    
    print(f"   Model created successfully")
    print(f"   Total parameters: {model.get_num_parameters():,}")
    print(f"   Trainable parameters: {model.get_num_trainable_parameters():,}")
    
    # Print model summary
    summary = model.summary()
    print(f"\n   Model Summary:")
    print(f"   - Model name: {summary['model_name']}")
    print(f"   - Num classes: {summary['num_classes']}")
    print(f"   - Channels: {summary['channels']}")
    print(f"   - Backbone params: {summary['backbone_parameters']:,}")
    print(f"   - Neck params: {summary['neck_parameters']:,}")
    print(f"   - Head params: {summary['head_parameters']:,}")
    
    # Test 2: Forward pass in training mode
    print("\n2. Testing forward pass (training mode):")
    test_input = torch.randn(2, 3, 640, 640).to(device)
    
    reg_out, cls_out = model(test_input, training_mode=True)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Regression output shape: {reg_out.shape}")
    print(f"   Classification output shape: {cls_out.shape}")
    
    # Test 3: Forward pass in inference mode
    print("\n3. Testing forward pass (inference mode):")
    detections = model(test_input, training_mode=False)
    print(f"   Number of detections: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"   Batch {i}: {det['boxes'].shape[0]} boxes, scores range: [{det['scores'].min():.4f}, {det['scores'].max():.4f}]")
    
    # Test 4: Gradient flow
    print("\n4. Testing gradient flow:")
    test_input_grad = torch.randn(1, 3, 640, 640, requires_grad=True).to(device)
    model_grad = ECLYOLOv11(num_classes=7).to(device)
    
    reg_out, cls_out = model_grad(test_input_grad, training_mode=True)
    loss = reg_out.sum() + cls_out.sum()
    loss.backward()
    
    has_grad = test_input_grad.grad is not None
    print(f"   Gradient flow works: {has_grad}")
    if has_grad:
        print(f"   Gradient shape: {test_input_grad.grad.shape}")
    
    # Test 5: Parameter count comparison with paper
    print("\n5. Comparing with paper results:")
    paper_params = 3001194
    actual_params = model.get_num_parameters()
    diff = actual_params - paper_params
    print(f"   Paper parameters: {paper_params:,}")
    print(f"   Actual parameters: {actual_params:,}")
    print(f"   Difference: {diff:,} ({diff/paper_params*100:.2f}%)")
    
    # Test 6: Test with different configurations
    print("\n6. Testing different configurations:")
    
    # Without CE
    model_no_ce = ECLYOLOv11(
        num_classes=7,
        use_ce_in_backbone=False
    ).to(device)
    print(f"   Without CE: {model_no_ce.get_num_parameters():,} params")
    
    # Without AENet
    model_no_aenet = ECLYOLOv11(
        num_classes=7,
        use_aenet_in_neck=False
    ).to(device)
    print(f"   Without AENet: {model_no_aenet.get_num_parameters():,} params")
    
    # Without LDHead
    model_no_ldhead = ECLYOLOv11(
        num_classes=7,
        use_ldhead_in_head=False
    ).to(device)
    print(f"   Without LDHead: {model_no_ldhead.get_num_parameters():,} params")
    
    # Test 7: Test different input sizes
    print("\n7. Testing different input sizes:")
    for size in [320, 416, 640, 1280]:
        try:
            test_input = torch.randn(1, 3, size, size).to(device)
            model_test = ECLYOLOv11(num_classes=7).to(device)
            reg_out, cls_out = model_test(test_input, training_mode=True)
            print(f"   Input size {size}x{size}: OK (output shapes: {reg_out.shape}, {cls_out.shape})")
        except Exception as e:
            print(f"   Input size {size}x{size}: FAILED - {e}")
    
    # Test 8: Device transfer
    print("\n8. Testing device transfer:")
    model_cpu = ECLYOLOv11(num_classes=7)
    model_cpu.to(device)
    print(f"   Model on: {model_cpu.get_device()}")
    
    model_cpu2 = model_cpu.to('cpu')
    print(f"   After to('cpu'): {model_cpu2.get_device()}")
    
    # Test 9: Freezing layers
    print("\n9. Testing layer freezing:")
    model_freeze = ECLYOLOv11(num_classes=7).to(device)
    model_freeze.freeze_backbone()
    print(f"   After freeze_backbone: {model_freeze.get_num_trainable_parameters():,} trainable params")
    
    model_freeze2 = ECLYOLOv11(num_classes=7).to(device)
    model_freeze2.freeze_neck()
    print(f"   After freeze_neck: {model_freeze2.get_num_trainable_parameters():,} trainable params")
    
    model_freeze3 = ECLYOLOv11(num_classes=7).to(device)
    model_freeze3.freeze_head()
    print(f"   After freeze_head: {model_freeze3.get_num_trainable_parameters():,} trainable params")
    
    model_freeze4 = ECLYOLOv11(num_classes=7).to(device)
    model_freeze4.freeze_backbone()
    model_freeze4.freeze_neck()
    print(f"   After freeze_backbone+neck: {model_freeze4.get_num_trainable_parameters():,} trainable params")
    
    print("\n" + "=" * 50)
    print("ECL-YOLOv11 test completed!")
