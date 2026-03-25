"""
ECL-YOLOv11 LDHead (Lightweight Shared Convolutional Detection Head) Module

This module implements the LDHead as described in the paper:
"Robust Object Detection in Adverse Weather Conditions: ECL-YOLOv11 for Automotive Vision Systems"

LDHead replaces the original YOLOv11 detection head with three key innovations:
1. Cross-scale parameter sharing through a single shared convolution
2. GroupNorm for stable normalization under varying batch sizes
3. Depthwise separable convolutions for computational efficiency

Based on paper equations (12-13):
- y_i = Concat(Scale(W_reg * F_i), σ(W_cls * F_i))          [Equation 12]
- b̂_i = dist2bbox(Γ_I(y_i,reg), anchors)                    [Equation 13]

Author: ECL-YOLOv11 Reproduction Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import configuration, with fallback to defaults
try:
    from utils.config import get_config_manager, get_ldhead_config
except ImportError:
    # Fallback configuration
    def get_config_manager():
        return None
    
    def get_ldhead_config() -> Dict[str, Any]:
        return {
            'enabled': True,
            'shared_conv': True,
            'use_groupnorm': True,
            'groupnorm_groups': 32,
            'depthwise_separable': True
        }


# =============================================================================
# LDHead Class - Lightweight Shared Convolution Detection Head
# =============================================================================

class LDHead(nn.Module):
    """
    Lightweight Shared Convolutional Detection Head (LDHead).
    
    This module implements an efficient detection head with cross-scale parameter sharing.
    It processes multi-scale features from the neck and produces regression and
    classification outputs for object detection.
    
    The key innovations are:
    1. Cross-scale parameter sharing: A single shared convolution processes
       concatenated multi-scale features, dramatically reducing parameters
    2. GroupNorm: Replaces BatchNorm for stable training under varying
       batch sizes and adverse weather conditions
    3. Depthwise separable convolution: Reduces computational cost
    
    Based on paper equations (12-13):
    y_i = Concat(Scale(W_reg * F_i), σ(W_cls * F_i))
    b̂_i = dist2bbox(Γ_I(y_i,reg), anchors)
    
    Attributes:
        num_classes (int): Number of detection classes
        in_channels_list (List[int]): List of input channels for each scale [P3, P4, P5]
        reg_max (int): Maximum value for DFL regression discretization
        
    Input:
        features (List[torch.Tensor]): List of multi-scale features from neck [P3, P4, P5]
        
    Output:
        Tuple[torch.Tensor, torch.Tensor]: (regression_outputs, classification_outputs)
            - regression_outputs: (B, 4*reg_max, H, W) - bounding box predictions
            - classification_outputs: (B, num_classes, H, W) - class probabilities
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        in_channels_list: List[int] = [256, 512, 1024],
        reg_max: int = 16,
        groupnorm_groups: int = 32,
        use_groupnorm: bool = True,
        depthwise_separable: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LDHead.
        
        Args:
            num_classes: Number of detection classes
            in_channels_list: List of input channels for each scale [P3, P4, P5]
            reg_max: Maximum value for DFL regression discretization
            groupnorm_groups: Number of groups for GroupNorm
            use_groupnorm: Whether to use GroupNorm (instead of BatchNorm)
            depthwise_separable: Whether to use depthwise separable convolution
            config: Optional configuration dictionary
        """
        super().__init__()
        
        # Get configuration if not provided
        if config is None:
            try:
                config = get_ldhead_config()
                if callable(config):
                    config = config()
            except:
                config = {
                    'enabled': True,
                    'shared_conv': True,
                    'use_groupnorm': True,
                    'groupnorm_groups': 32,
                    'depthwise_separable': True
                }
        
        # Apply configuration
        if config:
            groupnorm_groups = config.get('groupnorm_groups', groupnorm_groups)
            use_groupnorm = config.get('use_groupnorm', use_groupnorm)
            depthwise_separable = config.get('depthwise_separable', depthwise_separable)
        
        self.num_classes = num_classes
        self.in_channels_list = in_channels_list
        self.reg_max = reg_max
        self.groupnorm_groups = groupnorm_groups
        self.use_groupnorm = use_groupnorm
        self.depthwise_separable = depthwise_separable
        
        # Total input channels after concatenation
        self.total_in_channels = sum(in_channels_list)
        
        # Ensure groupnorm_groups is valid (must divide channels evenly)
        self.groupnorm_groups = min(groupnorm_groups, self.total_in_channels)
        if self.total_in_channels % self.groupnorm_groups != 0:
            self.groupnorm_groups = self.total_in_channels
        
        # =====================================================================
        # Per-scale input processing layers
        # Each scale gets: 1x1 conv + GroupNorm
        # =====================================================================
        self.input_convs = nn.ModuleList()
        self.input_norms = nn.ModuleList()
        
        for in_channels in in_channels_list:
            # 1x1 convolution for channel adjustment
            self.input_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 1, bias=False),
                )
            )
            
            # GroupNorm for stable normalization
            if use_groupnorm:
                self.input_norms.append(
                    nn.GroupNorm(self.groupnorm_groups, in_channels)
                )
            else:
                # Fallback to BatchNorm2d (not recommended but included)
                self.input_norms.append(
                    nn.BatchNorm2d(in_channels)
                )
        
        # =====================================================================
        # Shared Depthwise Separable Convolution
        # Single 3x3 convolution shared across all scales
        # =====================================================================
        if depthwise_separable:
            # Depthwise separable: groups = in_channels for each spatial position
            # This applies the same filter across all channels but processes
            # each channel separately (depthwise), then combines with pointwise conv
            self.shared_conv = nn.Sequential(
                # Depthwise part
                nn.Conv2d(
                    self.total_in_channels,
                    self.total_in_channels,
                    kernel_size=3,
                    padding=1,
                    groups=self.total_in_channels,  # Depthwise
                    bias=False
                ),
                nn.BatchNorm2d(self.total_in_channels),
                nn.SiLU(inplace=True),
                # Pointwise part to combine channels
                nn.Conv2d(
                    self.total_in_channels,
                    self.total_in_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(self.total_in_channels),
                nn.SiLU(inplace=True)
            )
        else:
            # Standard 3x3 convolution (not recommended for efficiency)
            self.shared_conv = nn.Sequential(
                nn.Conv2d(
                    self.total_in_channels,
                    self.total_in_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(self.total_in_channels),
                nn.SiLU(inplace=True)
            )
        
        # =====================================================================
        # Output Branches
        # =====================================================================
        
        # Regression branch: outputs 4 * reg_max channels per position
        # These represent the bounding box offsets as discrete probability distribution
        # Will be decoded using DFL (Distribution Focal Loss)
        self.reg_conv = nn.Conv2d(
            self.total_in_channels,
            4 * reg_max,
            kernel_size=1,
            bias=False
        )
        
        # Classification branch: outputs num_classes channels per position
        # Apply sigmoid activation for multi-label classification
        self.cls_conv = nn.Conv2d(
            self.total_in_channels,
            num_classes,
            kernel_size=1,
            bias=False
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize convolutional weights using Kaiming initialization."""
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
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LDHead.
        
        Args:
            features: List of multi-scale features [P3, P4, P5] from AENet
                     P3: (B, C0, H0, W0) - highest resolution
                     P4: (B, C1, H1, W1) - medium resolution
                     P5: (B, C2, H2, W2) - lowest resolution
                     where H0 > H1 > H2 and W0 > W1 > W2
            
        Returns:
            Tuple of (regression_outputs, classification_outputs):
            - regression_outputs: (B, 4*reg_max, H_total, W_total)
              Concatenated regression predictions across all scales
            - classification_outputs: (B, num_classes, H_total, W_total)
              Concatenated classification predictions across all scales
        """
        assert len(features) == len(self.in_channels_list), \
            f"Expected {len(self.in_channels_list)} input features, got {len(features)}"
        
        # =====================================================================
        # Step 1: Per-scale processing
        # Process each scale with 1x1 conv + GroupNorm + SiLU
        # =====================================================================
        processed_features = []
        
        for i, feat in enumerate(features):
            # Apply 1x1 convolution
            x = self.input_convs[i](feat)
            
            # Apply normalization (GroupNorm or BatchNorm)
            x = self.input_norms[i](x)
            
            # Apply SiLU activation
            x = F.silu(x)
            
            processed_features.append(x)
        
        # =====================================================================
        # Step 2: Concatenate all processed features
        # Shape: (B, total_channels, H_max, W_max) where H_max = max(H_i)
        # =====================================================================
        # Find the maximum spatial dimensions
        max_h = max(f.shape[2] for f in processed_features)
        max_w = max(f.shape[3] for f in processed_features)
        
        # Upsample smaller features to match the largest
        upsampled_features = []
        for f in processed_features:
            if f.shape[2] != max_h or f.shape[3] != max_w:
                f_upsampled = F.interpolate(
                    f,
                    size=(max_h, max_w),
                    mode='bilinear',
                    align_corners=False
                )
                upsampled_features.append(f_upsampled)
            else:
                upsampled_features.append(f)
        
        # Concatenate along channel dimension
        concat_features = torch.cat(upsampled_features, dim=1)
        
        # =====================================================================
        # Step 3: Shared Depthwise Separable Convolution
        # Single convolution shared across all scales
        # =====================================================================
        shared_out = self.shared_conv(concat_features)
        
        # =====================================================================
        # Step 4: Output branches
        # =====================================================================
        
        # Regression branch (no activation - raw logits for DFL)
        reg_output = self.reg_conv(shared_out)
        
        # Classification branch (sigmoid activation for multi-label)
        cls_output = self.cls_conv(shared_out)
        cls_output = torch.sigmoid(cls_output)
        
        return reg_output, cls_output
    
    def get_output_shape(
        self,
        input_shapes: List[Tuple[int, int, int, int]]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Calculate output shapes for given input shapes.
        
        Args:
            input_shapes: List of (B, C, H, W) tuples for each scale
            
        Returns:
            Tuple of (reg_output_shape, cls_output_shape)
        """
        max_h = max(s[2] for s in input_shapes)
        max_w = max(s[3] for s in input_shapes)
        
        reg_shape = (input_shapes[0][0], 4 * self.reg_max, max_h, max_w)
        cls_shape = (input_shapes[0][0], self.num_classes, max_h, max_w)
        
        return reg_shape, cls_shape


# =============================================================================
# Utility Functions
# =============================================================================

def create_ldhead(
    num_classes: int = 7,
    in_channels_list: List[int] = [256, 512, 1024],
    reg_max: int = 16,
    config: Optional[Dict[str, Any]] = None
) -> LDHead:
    """
    Factory function to create LDHead.
    
    Args:
        num_classes: Number of detection classes
        in_channels_list: List of input channels for each scale
        reg_max: Maximum value for DFL regression discretization
        config: Optional configuration dictionary
        
    Returns:
        LDHead: Configured LDHead module
    """
    return LDHead(
        num_classes=num_classes,
        in_channels_list=in_channels_list,
        reg_max=reg_max,
        config=config
    )


# =============================================================================
# Test/Verification Code
# =============================================================================

if __name__ == "__main__":
    # Test LDHead implementation
    print("Testing LDHead Implementation")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration from config
    num_classes = 7
    reg_max = 16
    in_channels_list = [256, 512, 1024]
    
    # Test 1: Basic LDHead
    print("\n1. Testing Basic LDHead:")
    ldhead = LDHead(
        num_classes=num_classes,
        in_channels_list=in_channels_list,
        reg_max=reg_max
    ).to(device)
    
    # Create test inputs (P3, P4, P5 from AENet)
    p3 = torch.randn(2, 256, 80, 80).to(device)
    p4 = torch.randn(2, 512, 40, 40).to(device)
    p5 = torch.randn(2, 1024, 20, 20).to(device)
    
    reg_out, cls_out = ldhead([p3, p4, p5])
    print(f"   Input P3 shape: {p3.shape}")
    print(f"   Input P4 shape: {p4.shape}")
    print(f"   Input P5 shape: {p5.shape}")
    print(f"   Regression output shape: {reg_out.shape}")
    print(f"   Classification output shape: {cls_out.shape}")
    print(f"   Expected reg shape: (2, {4*reg_max}, 80, 80)")
    print(f"   Expected cls shape: (2, {num_classes}, 80, 80)")
    print(f"   Parameters: {sum(p.numel() for p in ldhead.parameters()):,}")
    
    # Verify shapes
    assert reg_out.shape == (2, 4 * reg_max, 80, 80), \
        f"Expected reg shape (2, {4*reg_max}, 80, 80), got {reg_out.shape}"
    assert cls_out.shape == (2, num_classes, 80, 80), \
        f"Expected cls shape (2, {num_classes}, 80, 80), got {cls_out.shape}"
    print("   Shapes verified ✓")
    
    # Test 2: Output value ranges
    print("\n2. Testing output value ranges:")
    print(f"   Regression output range: [{reg_out.min():.4f}, {reg_out.max():.4f}]")
    print(f"   Classification output range: [{cls_out.min():.4f}, {cls_out.max():.4f}]")
    
    # Verify classification is in [0, 1] range (after sigmoid)
    assert cls_out.min() >= 0.0, "Classification values below 0!"
    assert cls_out.max() <= 1.0, "Classification values above 1!"
    print("   Classification in [0,1] range ✓")
    
    # Test 3: Gradient flow
    print("\n3. Testing gradient flow:")
    test_features = [
        torch.randn(1, 256, 80, 80, requires_grad=True).to(device),
        torch.randn(1, 512, 40, 40, requires_grad=True).to(device),
        torch.randn(1, 1024, 20, 20, requires_grad=True).to(device)
    ]
    
    ldhead_grad = LDHead(
        num_classes=num_classes,
        in_channels_list=in_channels_list,
        reg_max=reg_max
    ).to(device)
    
    reg_out, cls_out = ldhead_grad(test_features)
    loss = reg_out.sum() + cls_out.sum()
    loss.backward()
    
    has_grad = all(feat.grad is not None for feat in test_features)
    print(f"   Gradient flow works: {has_grad}")
    if has_grad:
        print(f"   Grad shapes: {[feat.grad.shape for feat in test_features]}")
    
    # Test 4: Different configurations
    print("\n4. Testing different configurations:")
    
    # Without GroupNorm
    ldhead_no_gn = LDHead(
        num_classes=num_classes,
        in_channels_list=in_channels_list,
        reg_max=reg_max,
        use_groupnorm=False
    ).to(device)
    print(f"   Without GroupNorm - Parameters: {sum(p.numel() for p in ldhead_no_gn.parameters()):,}")
    
    # Without depthwise separable
    ldhead_no_dw = LDHead(
        num_classes=num_classes,
        in_channels_list=in_channels_list,
        reg_max=reg_max,
        depthwise_separable=False
    ).to(device)
    print(f"   Without Depthwise - Parameters: {sum(p.numel() for p in ldhead_no_dw.parameters()):,}")
    
    # Test 5: Different input channels
    print("\n5. Testing different input channels:")
    test_configs = [
        [128, 256, 512],
        [64, 128, 256],
        [256, 256, 256],  # Same channels
    ]
    
    for in_ch_list in test_configs:
        try:
            ldhead_test = LDHead(
                num_classes=num_classes,
                in_channels_list=in_ch_list,
                reg_max=reg_max
            ).to(device)
            
            test_inputs = [
                torch.randn(1, in_ch_list[0], 80, 80).to(device),
                torch.randn(1, in_ch_list[1], 40, 40).to(device),
                torch.randn(1, in_ch_list[2], 20, 20).to(device)
            ]
            
            reg_out, cls_out = ldhead_test(test_inputs)
            print(f"   In channels {in_ch_list} -> Parameters: {sum(p.numel() for p in ldhead_test.parameters()):,}")
        except Exception as e:
            print(f"   In channels {in_ch_list} -> Error: {e}")
    
    # Test 6: Different num_classes
    print("\n6. Testing different num_classes:")
    for nc in [1, 2, 5, 10, 20]:
        try:
            ldhead_nc = LDHead(
                num_classes=nc,
                in_channels_list=in_channels_list,
                reg_max=reg_max
            ).to(device)
            
            test_inputs = [
                torch.randn(1, 256, 80, 80).to(device),
                torch.randn(1, 512, 40, 40).to(device),
                torch.randn(1, 1024, 20, 20).to(device)
            ]
            
            reg_out, cls_out = ldhead_nc(test_inputs)
            print(f"   num_classes={nc} -> cls_out shape: {cls_out.shape}, Parameters: {sum(p.numel() for p in ldhead_nc.parameters()):,}")
        except Exception as e:
            print(f"   num_classes={nc} -> Error: {e}")
    
    # Test 7: Output shape calculation
    print("\n7. Testing output shape calculation:")
    input_shapes = [(2, 256, 80, 80), (2, 512, 40, 40), (2, 1024, 20, 20)]
    reg_shape, cls_shape = ldhead.get_output_shape(input_shapes)
    print(f"   Input shapes: {input_shapes}")
    print(f"   Reg output shape: {reg_shape}")
    print(f"   Cls output shape: {cls_shape}")
    
    # Compare with actual forward pass
    actual_reg, actual_cls = ldhead([p3, p4, p5])
    assert actual_reg.shape == reg_shape, f"Shape mismatch: {actual_reg.shape} != {reg_shape}"
    assert actual_cls.shape == cls_shape, f"Shape mismatch: {actual_cls.shape} != {cls_shape}"
    print("   Shapes match ✓")
    
    print("\n" + "=" * 50)
    print("LDHead test completed!")
