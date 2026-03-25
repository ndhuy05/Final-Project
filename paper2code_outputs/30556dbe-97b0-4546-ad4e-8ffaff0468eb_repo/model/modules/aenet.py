"""
ECL-YOLOv11 AENet (Context-Guided Multi-Scale Fusion Network) Module

This module implements the AENet as described in the paper:
"Robust Object Detection in Adverse Weather Conditions: ECL-YOLOv11 for Automotive Vision Systems"

AENet replaces the original YOLOv11 Neck and comprises four sub-modules:
1. PCE (Pyramid Context Extraction): Unifies spatial dimensions and extracts cross-scale context
2. RCM (Rectangular Calibration Module): Captures elongated structures using strip convolutions
3. DIF (Down-to-Up Information Flow): Transfers semantic guidance from high-level to low-level features
4. FBM (Feedback Block Module): Provides detail feedback from high-resolution to low-resolution features

Based on paper equations (8)-(11):
- RCM: Output = σ(Excite(AdaptiveAvgPool(x))) × dwconv_hw(x)  [Equation 8]
- PCE: x' = RCM(PyramidPoolAggPCE(P3, P4, P5))                 [Equation 9]
- DIF: Out_DIF = x1 + Conv(Interp(x2))                         [Equation 10]
- FBM: Out_FBM = Conv(x_l) × σ(Interp(Conv(x_h)))             [Equation 11]

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
    from utils.config import get_config_manager, get_aenet_config
except ImportError:
    # Fallback configuration
    def get_config_manager():
        return None
    
    def get_aenet_config() -> Dict[str, Any]:
        return {
            'enabled': True,
            'pyramid_channels': [256, 512, 1024],
            'rcm_stages': 2,
            'use_dif': True,
            'use_fbm': True
        }


# =============================================================================
# RCM (Rectangular Calibration Module)
# =============================================================================

class RCMModule(nn.Module):
    """
    Rectangular Calibration Module (RCM).
    
    This module captures elongated structures using strip convolutions along
    horizontal and vertical directions. It's particularly effective for detecting
    regular-shaped objects like vehicles, lanes, and pedestrians.
    
    Based on paper Equation 8:
    Output = σ(Excite(AdaptiveAvgPool(x))) × dwconv_hw(x)
    
    where Excite refers to strip convolution operations.
    
    Attributes:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        reduction (int): Channel reduction ratio for excitation
        
    Input:
        x (torch.Tensor): Input feature map of shape (B, C, H, W)
        
    Output:
        torch.Tensor: Rectangular-calibrated feature map of shape (B, C_out, H, W)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: Optional[int] = None,
        reduction: int = 4
    ):
        """
        Initialize RCM Module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (default: same as in_channels)
            reduction: Channel reduction ratio for excitation
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.reduction = reduction
        
        # Channel reduction for excitation
        reduced_channels = max(in_channels // reduction, 8)
        
        # =====================================================================
        # Horizontal Strip Convolution (1×3 kernel)
        # Captures horizontal elongated structures
        # =====================================================================
        self.horiz_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                reduced_channels, 
                kernel_size=(1, 3), 
                padding=(0, 1),
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True)
        )
        
        # =====================================================================
        # Vertical Strip Convolution (3×1 kernel)
        # Captures vertical elongated structures
        # =====================================================================
        self.vert_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                reduced_channels, 
                kernel_size=(3, 1), 
                padding=(1, 0),
                bias=False
            ),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True)
        )
        
        # =====================================================================
        # Excitation fusion
        # =====================================================================
        self.excite_fusion = nn.Sequential(
            nn.Conv2d(reduced_channels * 2, in_channels, 1, bias=False),
            nn.Sigmoid()  # Activation for attention weights
        )
        
        # =====================================================================
        # Depthwise Separable Convolution
        # =====================================================================
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                self.out_channels, 
                kernel_size=3, 
                padding=1,
                groups=in_channels,  # Depthwise
                bias=False
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.SiLU(inplace=True)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RCM module.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Rectangular-calibrated feature tensor
        """
        # =====================================================================
        # Apply horizontal and vertical strip convolutions
        # =====================================================================
        h_feat = self.horiz_conv(x)
        v_feat = self.vert_conv(x)
        
        # =====================================================================
        # Concatenate and generate excitation weights
        # =====================================================================
        concat_feat = torch.cat([h_feat, v_feat], dim=1)
        excite_weights = self.excite_fusion(concat_feat)
        
        # =====================================================================
        # Apply depthwise separable convolution
        # =====================================================================
        dw_out = self.dwconv(x)
        
        # =====================================================================
        # Apply attention-weighted multiplication
        # Output = dw_out × excite_weights
        # =====================================================================
        output = dw_out * excite_weights
        
        return output


# =============================================================================
# PCE (Pyramid Context Extraction) Module
# =============================================================================

class PCEModule(nn.Module):
    """
    Pyramid Context Extraction (PCE) Module.
    
    This module unifies spatial dimensions of multi-scale features and extracts
    cross-scale context through pooling and RCM processing.
    
    Based on paper Equation 9:
    x' = RCM(PyramidPoolAggPCE(P3, P4, P5))
    
    Attributes:
        in_channels_list (List[int]): List of input channel dimensions [P3, P4, P5]
        out_channels (int): Output channel dimension for each scale
        rcm_stages (int): Number of RCM processing stages
        
    Input:
        features_list (List[torch.Tensor]): List of multi-scale features [P3, P4, P5]
        
    Output:
        List[torch.Tensor]: Context-enhanced features [P3', P4', P5']
    """
    
    def __init__(
        self,
        in_channels_list: List[int] = [256, 512, 1024],
        out_channels: int = 256,
        rcm_stages: int = 2
    ):
        """
        Initialize PCE Module.
        
        Args:
            in_channels_list: List of input channels for each scale
            out_channels: Output channels (will be used for projection)
            rcm_stages: Number of RCM stages to apply
        """
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.rcm_stages = rcm_stages
        
        # Total input channels for concatenation
        total_in_channels = sum(in_channels_list)
        
        # =====================================================================
        # Adaptive pooling to unify spatial dimensions
        # Each feature will be pooled to 1×1
        # =====================================================================
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # =====================================================================
        # RCM processing for context extraction
        # =====================================================================
        self.rcm_layers = nn.ModuleList()
        for i in range(rcm_stages):
            if i == 0:
                self.rcm_layers.append(
                    RCMModule(total_in_channels, total_in_channels)
                )
            else:
                self.rcm_layers.append(
                    RCMModule(total_in_channels, total_in_channels)
                )
        
        # =====================================================================
        # Channel projection for each scale
        # =====================================================================
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
            for in_ch in in_channels_list
        ])
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
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
    
    def forward(self, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through PCE module.
        
        Args:
            features_list: List of multi-scale features [P3, P4, P5]
                         Shapes: (B, C1, H1, W1), (B, C2, H2, W2), (B, C3, H3, W3)
            
        Returns:
            List of context-enhanced features
        """
        # =====================================================================
        # Pool each scale to unify spatial dimensions
        # =====================================================================
        pooled_list = [self.pool(feat) for feat in features_list]
        
        # =====================================================================
        # Concatenate pooled features along channel dimension
        # =====================================================================
        concat_feat = torch.cat(pooled_list, dim=1)  # (B, C1+C2+C3, 1, 1)
        
        # =====================================================================
        # Apply RCM for context enhancement
        # =====================================================================
        for rcm_layer in self.rcm_layers:
            concat_feat = rcm_layer(concat_feat)
        
        # =====================================================================
        # Project each scale to output channels
        # Note: We broadcast the processed context back to each scale
        # =====================================================================
        output_list = []
        for i, (feat, proj) in enumerate(zip(features_list, self.scale_projs)):
            # Use the same processed context for all scales
            # The context is 1×1, so it broadcasts across spatial dimensions
            context = proj(concat_feat)  # (B, out_channels, 1, 1)
            
            # Broadcast multiply with original features
            # This applies the context as attention weights
            h, w = feat.shape[2], feat.shape[3]
            context_broadcast = F.interpolate(
                context, 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Element-wise multiplication to apply context
            enhanced = feat * context_broadcast
            output_list.append(enhanced)
        
        return output_list


# =============================================================================
# DIF (Down-to-Up Information Flow) Module
# =============================================================================

class DIFModule(nn.Module):
    """
    Down-to-Up Information Flow (DIF) Module.
    
    This module transfers semantic guidance from high-level features to 
    low-level features through upsampling and fusion.
    
    Based on paper Equation 10:
    Out_DIF = x_low + Conv(Interp(x_high))
    
    where:
    - x_low: Lower-level features (higher resolution)
    - x_high: Higher-level features (lower resolution)
    - Interp: Upsampling operation
    - Conv: 1×1 convolution for channel alignment
    
    Attributes:
        low_channels (int): Number of channels in low-level features
        high_channels (int): Number of channels in high-level features
        
    Input:
        x_low (torch.Tensor): Lower-level features (e.g., P3)
        x_high (torch.Tensor): Higher-level features (e.g., P4/P5)
        
    Output:
        torch.Tensor: Fused features with semantic guidance
    """
    
    def __init__(
        self, 
        low_channels: int, 
        high_channels: int,
        rcm_stages: int = 1
    ):
        """
        Initialize DIF Module.
        
        Args:
            low_channels: Number of channels in low-level features
            high_channels: Number of channels in high-level features
            rcm_stages: Number of RCM stages for enhancement
        """
        super().__init__()
        
        self.low_channels = low_channels
        self.high_channels = high_channels
        
        # =====================================================================
        # RCM for enhancing low-level features
        # =====================================================================
        self.rcm_low = RCMModule(low_channels, low_channels)
        
        # =====================================================================
        # Channel adjustment for upsampled high-level features
        # =====================================================================
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, 1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.SiLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
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
    
    def forward(
        self, 
        x_low: torch.Tensor, 
        x_high: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through DIF module.
        
        Args:
            x_low: Lower-level features (e.g., P3), shape (B, C_low, H_l, W_l)
            x_high: Higher-level features (e.g., P4/P5), shape (B, C_high, H_h, W_h)
            
        Returns:
            Fused features with semantic guidance, shape (B, C_low, H_l, W_l)
        """
        # =====================================================================
        # Enhance low-level features with RCM
        # =====================================================================
        low_enhanced = self.rcm_low(x_low)
        
        # =====================================================================
        # Upsample high-level features to match low-level resolution
        # =====================================================================
        upsampled_high = F.interpolate(
            x_high,
            size=x_low.shape[2:],
            mode='nearest'  # Use nearest for structure preservation
        )
        
        # =====================================================================
        # Channel adjustment for upsampled features
        # =====================================================================
        high_adjusted = self.conv_high(upsampled_high)
        
        # =====================================================================
        # Fuse: Out = low_enhanced + high_adjusted
        # =====================================================================
        output = low_enhanced + high_adjusted
        
        return output


# =============================================================================
# FBM (Feedback Block Module)
# =============================================================================

class FBMModule(nn.Module):
    """
    Feedback Block Module (FBM).
    
    This module provides detail feedback from high-resolution features to
    low-resolution features through gated attention.
    
    Based on paper Equation 11:
    Out_FBM = Conv(x_l) × σ(Interp(Conv(x_h)))
    
    Attributes:
        low_channels (int): Number of channels in low-resolution features
        high_channels (int): Number of channels in high-resolution features
        
    Input:
        x_low (torch.Tensor): Low-resolution features (e.g., P5)
        x_high (torch.Tensor): High-resolution features (e.g., P3)
        
    Output:
        torch.Tensor: Detail-compensated features
    """
    
    def __init__(
        self, 
        low_channels: int, 
        high_channels: int
    ):
        """
        Initialize FBM Module.
        
        Args:
            low_channels: Number of channels in low-resolution features
            high_channels: Number of channels in high-resolution features
        """
        super().__init__()
        
        self.low_channels = low_channels
        self.high_channels = high_channels
        
        # =====================================================================
        # Convolution for high-resolution features (to generate gate)
        # =====================================================================
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, 1, bias=False),
            nn.Sigmoid()  # Generate attention weights in [0, 1]
        )
        
        # =====================================================================
        # Convolution for low-resolution features
        # =====================================================================
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, 1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.SiLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
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
    
    def forward(
        self, 
        x_low: torch.Tensor, 
        x_high: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through FBM module.
        
        Args:
            x_low: Low-resolution features (e.g., P5), shape (B, C_l, H_l, W_l)
            x_high: High-resolution features (e.g., P3), shape (B, C_h, H_h, W_h)
            
        Returns:
            Detail-compensated features, shape (B, C_l, H_l, W_l)
        """
        # =====================================================================
        # Generate gated attention weights from high-resolution features
        # =====================================================================
        high_feat = self.conv_high(x_high)  # (B, C_l, H_h, W_h)
        
        # =====================================================================
        # Upsample gate to match low-resolution spatial dimensions
        # =====================================================================
        upsampled_gate = F.interpolate(
            high_feat,
            size=x_low.shape[2:],
            mode='nearest'  # Use nearest for binary-like gating
        )
        
        # =====================================================================
        # Process low-resolution features
        # =====================================================================
        low_feat = self.conv_low(x_low)
        
        # =====================================================================
        # Apply gated multiplication
        # Out = low_feat × gate
        # =====================================================================
        output = low_feat * upsampled_gate
        
        return output


# =============================================================================
# C3K2 Block (Cross-Stage Partial Connection from YOLOv11)
# =============================================================================

class C3K2Block(nn.Module):
    """
    C3K2 Block from YOLOv11.
    
    This block provides final processing of fused multi-scale features
    before passing to the detection head.
    
    Attributes:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_blocks (int): Number of bottleneck blocks
        
    Input:
        x (torch.Tensor): Input features
        
    Output:
        torch.Tensor: Processed features
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 0.5
    ):
        """
        Initialize C3K2 Block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_blocks: Number of bottleneck blocks
            expansion: Expansion ratio for hidden channels
        """
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        # =====================================================================
        # Main path
        # =====================================================================
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        )
        
        # Bottleneck blocks
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            for _ in range(num_blocks)
        ])
        
        # =====================================================================
        # Output convolution
        # =====================================================================
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self.activation = nn.SiLU(inplace=True)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C3K2 block.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        # Apply bottleneck blocks
        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1)
            x1 = self.activation(x1)
        
        # Concatenate and process
        concat = torch.cat([x1, x2], dim=1)
        output = self.conv_out(concat)
        
        return output


# =============================================================================
# AENet (Complete Network)
# =============================================================================

class AENet(nn.Module):
    """
    Context-Guided Multi-Scale Fusion Network (AENet).
    
    AENet replaces the original YOLOv11 Neck and provides enhanced multi-scale
    feature fusion for robust object detection in adverse weather conditions.
    
    The network integrates:
    - PCE: Pyramid Context Extraction for cross-scale context
    - DIF: Down-to-Up Information Flow for semantic guidance
    - FBM: Feedback Block Module for detail compensation
    - C3K2: Final feature refinement
    
    Based on Figure 7 in the paper.
    
    Attributes:
        in_channels_list (List[int]): Input channels [P3, P4, P5]
        out_channels (int): Output channels for each scale
        rcm_stages (int): Number of RCM processing stages
        use_dif (bool): Whether to use DIF modules
        use_fbm (bool): Whether to use FBM modules
        
    Input:
        features_list (List[torch.Tensor]): List of multi-scale features [P3, P4, P5]
        
    Output:
        List[torch.Tensor]: Fused multi-scale features for detection head
    """
    
    def __init__(
        self,
        in_channels_list: List[int] = [256, 512, 1024],
        out_channels: int = 256,
        rcm_stages: int = 2,
        use_dif: bool = True,
        use_fbm: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize AENet.
        
        Args:
            in_channels_list: List of input channel dimensions for each scale
            out_channels: Output channels for each scale
            rcm_stages: Number of RCM stages in PCE
            use_dif: Whether to use DIF modules
            use_fbm: Whether to use FBM modules
            config: Optional configuration dictionary
        """
        super().__init__()
        
        # Get configuration if not provided
        if config is None:
            try:
                config = get_aenet_config()
                if callable(config):
                    config = config()
            except:
                config = {
                    'enabled': True,
                    'pyramid_channels': [256, 512, 1024],
                    'rcm_stages': 2,
                    'use_dif': True,
                    'use_fbm': True
                }
        
        # Apply configuration
        if config:
            in_channels_list = config.get('pyramid_channels', in_channels_list)
            rcm_stages = config.get('rcm_stages', rcm_stages)
            use_dif = config.get('use_dif', use_dif)
            use_fbm = config.get('use_fbm', use_fbm)
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.use_dif = use_dif
        self.use_fbm = use_fbm
        
        # =====================================================================
        # PCE Module for initial context extraction
        # =====================================================================
        self.pce = PCEModule(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            rcm_stages=rcm_stages
        )
        
        # =====================================================================
        # Channel projection layers to align input channels to output channels
        # =====================================================================
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
            for in_ch in in_channels_list
        ])
        
        # =====================================================================
        # DIF Modules for each scale pair
        # DIF: P5->P4, P4->P3 (semantic guidance from high to low)
        # =====================================================================
        if use_dif:
            # DIF from P5 to P4
            self.dif_54 = DIFModule(
                low_channels=out_channels,
                high_channels=out_channels,
                rcm_stages=1
            )
            
            # DIF from P4 to P3
            self.dif_43 = DIFModule(
                low_channels=out_channels,
                high_channels=out_channels,
                rcm_stages=1
            )
        
        # =====================================================================
        # FBM Modules for detail feedback
        # FBM: P3->P5, P4->P5 (detail compensation from high to low)
        # =====================================================================
        if use_fbm:
            # FBM from P3 to P5 (P5 receives P3 details)
            self.fbm_35 = FBMModule(
                low_channels=out_channels,
                high_channels=out_channels
            )
            
            # FBM from P4 to P5 (P5 receives P4 details)
            self.fbm_45 = FBMModule(
                low_channels=out_channels,
                high_channels=out_channels
            )
        
        # =====================================================================
        # C3K2 blocks for final processing
        # =====================================================================
        self.c3k2_p3 = C3K2Block(out_channels, out_channels, num_blocks=1)
        self.c3k2_p4 = C3K2Block(out_channels, out_channels, num_blocks=1)
        self.c3k2_p5 = C3K2Block(out_channels, out_channels, num_blocks=1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
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
    
    def forward(self, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through AENet.
        
        Args:
            features_list: List of multi-scale features from backbone [P3, P4, P5]
                          Shapes: (B, 256, H/8, W/8), (B, 512, H/16, W/16), (B, 1024, H/32, W/32)
            
        Returns:
            List of fused features [P3_out, P4_out, P5_out] for detection head
        """
        assert len(features_list) == 3, "AENet expects exactly 3 input features (P3, P4, P5)"
        
        p3, p4, p5 = features_list
        
        # =====================================================================
        # Step 1: Project input channels to output channels
        # =====================================================================
        p3_proj = self.input_projs[0](p3)
        p4_proj = self.input_projs[1](p4)
        p5_proj = self.input_projs[2](p5)
        
        # =====================================================================
        # Step 2: PCE - Pyramid Context Extraction
        # =====================================================================
        p3_pce, p4_pce, p5_pce = self.pce([p3_proj, p4_proj, p5_proj])
        
        # =====================================================================
        # Step 3: DIF - Down-to-Up Information Flow
        # Semantic guidance from high-level to low-level features
        # =====================================================================
        if self.use_dif:
            # DIF: P5 -> P4 (P4 receives semantic guidance from P5)
            p4_dif = self.dif_54(p4_pce, p5_pce)
            
            # DIF: P4 -> P3 (P3 receives semantic guidance from P4)
            p3_dif = self.dif_43(p3_pce, p4_dif)
        else:
            p4_dif = p4_pce
            p3_dif = p3_pce
        
        # =====================================================================
        # Step 4: FBM - Feedback Block Module
        # Detail feedback from high-resolution to low-resolution
        # =====================================================================
        if self.use_fbm:
            # FBM: P3 -> P5 (P5 receives details from P3)
            p5_fbm_1 = self.fbm_35(p5_pce, p3_dif)
            
            # FBM: P4 -> P5 (P5 receives details from P4)
            p5_fbm_2 = self.fbm_45(p5_pce, p4_dif)
            
            # Combine FBM outputs
            p5_fused = p5_fbm_1 + p5_fbm_2
        else:
            p5_fused = p5_pce
        
        # =====================================================================
        # Step 5: C3K2 - Final feature refinement
        # =====================================================================
        p3_out = self.c3k2_p3(p3_dif)
        p4_out = self.c3k2_p4(p4_dif)
        p5_out = self.c3k2_p5(p5_fused)
        
        return [p3_out, p4_out, p5_out]


# =============================================================================
# Utility Functions
# =============================================================================

def create_aenet(
    in_channels_list: List[int] = [256, 512, 1024],
    out_channels: int = 256,
    config: Optional[Dict[str, Any]] = None
) -> AENet:
    """
    Factory function to create AENet.
    
    Args:
        in_channels_list: List of input channel dimensions
        out_channels: Output channels for each scale
        config: Optional configuration dictionary
        
    Returns:
        AENet: Configured AENet module
    """
    return AENet(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        config=config
    )


# =============================================================================
# Test/Verification Code
# =============================================================================

if __name__ == "__main__":
    # Test AENet implementation
    print("Testing AENet Implementation")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test RCM Module
    print("\n1. Testing RCMModule:")
    rcm = RCMModule(in_channels=256, out_channels=256).to(device)
    test_input = torch.randn(2, 256, 80, 80).to(device)
    output = rcm(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in rcm.parameters()):,}")
    
    # Test PCE Module
    print("\n2. Testing PCEModule:")
    pce = PCEModule(
        in_channels_list=[256, 512, 1024],
        out_channels=256,
        rcm_stages=2
    ).to(device)
    
    p3_in = torch.randn(2, 256, 80, 80).to(device)
    p4_in = torch.randn(2, 512, 40, 40).to(device)
    p5_in = torch.randn(2, 1024, 20, 20).to(device)
    
    p3_out, p4_out, p5_out = pce([p3_in, p4_in, p5_in])
    print(f"   Input shapes: {p3_in.shape}, {p4_in.shape}, {p5_in.shape}")
    print(f"   Output shapes: {p3_out.shape}, {p4_out.shape}, {p5_out.shape}")
    
    # Test DIF Module
    print("\n3. Testing DIFModule:")
    dif = DIFModule(low_channels=256, high_channels=256).to(device)
    low_feat = torch.randn(2, 256, 80, 80).to(device)
    high_feat = torch.randn(2, 256, 40, 40).to(device)
    output = dif(low_feat, high_feat)
    print(f"   Low input shape: {low_feat.shape}")
    print(f"   High input shape: {high_feat.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test FBM Module
    print("\n4. Testing FBMModule:")
    fbm = FBMModule(low_channels=256, high_channels=256).to(device)
    low_feat = torch.randn(2, 256, 20, 20).to(device)
    high_feat = torch.randn(2, 256, 80, 80).to(device)
    output = fbm(low_feat, high_feat)
    print(f"   Low input shape: {low_feat.shape}")
    print(f"   High input shape: {high_feat.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test Complete AENet
    print("\n5. Testing Complete AENet:")
    aenet = AENet(
        in_channels_list=[256, 512, 1024],
        out_channels=256,
        rcm_stages=2,
        use_dif=True,
        use_fbm=True
    ).to(device)
    
    p3 = torch.randn(2, 256, 80, 80).to(device)
    p4 = torch.randn(2, 512, 40, 40).to(device)
    p5 = torch.randn(2, 1024, 20, 20).to(device)
    
    p3_out, p4_out, p5_out = aenet([p3, p4, p5])
    print(f"   Input shapes: {p3.shape}, {p4.shape}, {p5.shape}")
    print(f"   Output shapes: {p3_out.shape}, {p4_out.shape}, {p5_out.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in aenet.parameters()):,}")
    
    # Test gradient flow
    print("\n6. Testing gradient flow:")
    aenet = AENet(
        in_channels_list=[256, 512, 1024],
        out_channels=256
    ).to(device)
    
    test_input = [
        torch.randn(1, 256, 80, 80, requires_grad=True).to(device),
        torch.randn(1, 512, 40, 40, requires_grad=True).to(device),
        torch.randn(1, 1024, 20, 20, requires_grad=True).to(device)
    ]
    
    outputs = aenet(test_input)
    loss = sum(o.sum() for o in outputs)
    loss.backward()
    
    has_grad = all(inp.grad is not None for inp in test_input)
    print(f"   Gradient flow works: {has_grad}")
    print(f"   Grad shapes: {[inp.grad.shape for inp in test_input if inp.grad is not None]}")
    
    # Test with different configurations
    print("\n7. Testing with different configurations:")
    
    # Without DIF
    aenet_no_dif = AENet(
        in_channels_list=[256, 512, 1024],
        out_channels=256,
        use_dif=False,
        use_fbm=True
    ).to(device)
    print(f"   Without DIF - Parameters: {sum(p.numel() for p in aenet_no_dif.parameters()):,}")
    
    # Without FBM
    aenet_no_fbm = AENet(
        in_channels_list=[256, 512, 1024],
        out_channels=256,
        use_dif=True,
        use_fbm=False
    ).to(device)
    print(f"   Without FBM - Parameters: {sum(p.numel() for p in aenet_no_fbm.parameters()):,}")
    
    # Without both
    aenet_base = AENet(
        in_channels_list=[256, 512, 1024],
        out_channels=256,
        use_dif=False,
        use_fbm=False
    ).to(device)
    print(f"   Without DIF+FBM - Parameters: {sum(p.numel() for p in aenet_base.parameters()):,}")
    
    print("\n" + "=" * 50)
    print("AENet test completed!")
