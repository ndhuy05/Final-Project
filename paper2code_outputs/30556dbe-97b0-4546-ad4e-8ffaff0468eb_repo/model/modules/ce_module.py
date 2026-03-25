"""
ECL-YOLOv11 CE (Convolutional Edge-enhancement) Module

This module implements the Edge-Enhancement Convolution (CE) module as described in the paper:
"Robust Object Detection in Adverse Weather Conditions: ECL-YOLOv11 for Automotive Vision Systems"

The CE module is designed to preserve edge and contour information in images by combining
fixed Sobel operator-based edge extraction with learnable convolutional features.

Architecture:
- Sobel edge extraction branch (fixed kernels, non-trainable)
- Learnable convolutional branch (3×3 conv with SiLU activation)
- Feature fusion via concatenation and 1×1 convolution
- Residual connection for gradient flow
- Output projection

Author: ECL-YOLOv11 Reproduction Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import configuration, with fallback to defaults
try:
    from utils.config import get_config_manager, get_ce_module_config
except ImportError:
    # Fallback configuration functions
    def get_config_manager():
        return None
    
    def get_ce_module_config():
        return {
            'enabled': True,
            'sobel_kernel_size': 3,
            'use_residual': True,
            'activation': 'SiLU'
        }


# =============================================================================
# Sobel Operator Definition
# =============================================================================

# Sobel kernels for edge extraction (as defined in paper Equation 1)
# K_x: Horizontal gradient kernel
SOBEL_X = torch.tensor([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=torch.float32)

# K_y: Vertical gradient kernel
SOBEL_Y = torch.tensor([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
], dtype=torch.float32)


# =============================================================================
# SobelConv Class - Fixed Edge Extraction
# =============================================================================

class SobelConv(nn.Module):
    """
    Sobel edge extraction module with fixed (non-trainable) kernels.
    
    This module applies horizontal and vertical Sobel operators to extract
    gradient features that emphasize edges and contours in the input.
    
    Based on paper Equation 2: x_sobel = Conv(x; K_x) + Conv(x; K_y)
    
    Attributes:
        in_channels (int): Number of input channels
        
    Input:
        x (torch.Tensor): Input feature map of shape (B, C, H, W)
        
    Output:
        torch.Tensor: Edge features of shape (B, C, H, W)
    """
    
    def __init__(self, in_channels: int):
        """
        Initialize SobelConv module.
        
        Args:
            in_channels: Number of input channels
        """
        super().__init__()
        
        self.in_channels = in_channels
        
        # Register Sobel kernels as buffers (non-trainable, but move with device)
        # Shape: (C, 1, 3, 3) for grouped convolution
        sobel_x = SOBEL_X.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        sobel_y = SOBEL_Y.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Sobel edge extraction.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Edge features of shape (B, C, H, W)
        """
        # Apply horizontal Sobel filter with grouped convolution
        edge_x = F.conv2d(
            x, 
            self.sobel_x, 
            padding=1, 
            groups=self.in_channels
        )
        
        # Apply vertical Sobel filter with grouped convolution
        edge_y = F.conv2d(
            x, 
            self.sobel_y, 
            padding=1, 
            groups=self.in_channels
        )
        
        # Sum horizontal and vertical responses (Equation 2)
        edge_features = edge_x + edge_y
        
        return edge_features


# =============================================================================
# CEModule Class - Main Edge Enhancement Module
# =============================================================================

class CEModule(nn.Module):
    """
    Convolutional Edge-Enhancement (CE) Module.
    
    This module explicitly preserves edge and contour information by fusing:
    1. Fixed Sobel-based edge features (gradient extraction)
    2. Learnable convolutional features (semantic extraction)
    
    The fusion maintains both edge detail and high-level semantics,
    which is particularly important for object detection in adverse weather
    conditions where edge information is degraded.
    
    Based on paper equations (2)-(7):
    - x_sobel = Conv(x; K_x) + Conv(x; K_y)  [Equation 2]
    - x_conv = sigma(W * x + b)               [Equation 3]
    - x_cat = [x_sobel, x_conv]             [Equation 4]
    - x_f = phi(W_1 * x_cat)              [Equation 5]
    - x_r = x_f + x                       [Equation 6]
    - y = W_2 * x_r                       [Equation 7]
    
    Attributes:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        activation (str): Activation function name ('SiLU' or 'ReLU')
        
    Input:
        x (torch.Tensor): Input feature map of shape (B, C_in, H, W)
        
    Output:
        torch.Tensor: Enhanced feature map of shape (B, C_out, H, W)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: Optional[int] = None,
        activation: str = 'SiLU',
        use_residual: bool = True,
        config: Optional[dict] = None
    ):
        """
        Initialize CE Module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (default: same as in_channels)
            activation: Activation function ('SiLU' or 'ReLU')
            use_residual: Whether to use residual connection
            config: Optional configuration dictionary
        """
        super().__init__()
        
        # Get configuration if not provided
        if config is None:
            config = get_ce_module_config()
            if config is None:
                config = get_ce_module_config() if callable(get_ce_module_config) else {}
        
        # Apply configuration
        if config:
            activation = config.get('activation', activation)
            use_residual = config.get('use_residual', use_residual)
        
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.activation = activation
        self.use_residual = use_residual
        
        # =====================================================================
        # Branch 1: Sobel Edge Extraction (fixed, non-trainable)
        # =====================================================================
        self.sobel_conv = SobelConv(in_channels)
        
        # =====================================================================
        # Branch 2: Learnable Convolutional Branch (Equation 3)
        # =====================================================================
        self.conv_branch = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size=3, 
                padding=1,
                bias=False
            ),
            self._get_activation(activation)
        )
        
        # =====================================================================
        # Fusion: Concatenation + 1×1 Convolution (Equations 4-5)
        # After concatenation: 2 * in_channels
        # Output after fusion: in_channels
        # =====================================================================
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                2 * in_channels, 
                in_channels, 
                kernel_size=1,
                bias=False
            ),
            self._get_activation(activation)
        )
        
        # =====================================================================
        # Output Projection (Equation 7)
        # =====================================================================
        # If output channels differ from input, project; otherwise keep identity
        if self.out_channels != in_channels:
            self.output_conv = nn.Conv2d(
                in_channels, 
                self.out_channels, 
                kernel_size=1,
                bias=False
            )
        else:
            # Identity mapping - no parameters needed
            self.output_conv = nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """
        Get activation function module.
        
        Args:
            activation: Activation name ('SiLU', 'ReLU', 'LeakyReLU', or 'GELU')
            
        Returns:
            nn.Module: Activation module
        """
        activation = activation.lower()
        
        if activation == 'silu' or activation == 'swish':
            return nn.SiLU(inplace=True)
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        else:
            # Default to SiLU
            return nn.SiLU(inplace=True)
    
    def _init_weights(self) -> None:
        """
        Initialize convolutional weights using Kaiming initialization.
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CE module.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            
        Returns:
            Enhanced feature tensor of shape (B, C_out, H, W)
        """
        # =====================================================================
        # Step 1: Sobel Edge Extraction (Equation 2)
        # =====================================================================
        edge_features = self.sobel_conv(x)
        
        # =====================================================================
        # Step 2: Learnable Convolutional Features (Equation 3)
        # =====================================================================
        conv_features = self.conv_branch(x)
        
        # =====================================================================
        # Step 3: Concatenate Features (Equation 4)
        # =====================================================================
        concat_features = torch.cat([edge_features, conv_features], dim=1)
        
        # =====================================================================
        # Step 4: Fusion via 1×1 Convolution (Equation 5)
        # =====================================================================
        fused_features = self.fusion_conv(concat_features)
        
        # =====================================================================
        # Step 5: Residual Connection (Equation 6)
        # =====================================================================
        if self.use_residual:
            # Project input if channels differ
            if self.in_channels != self.out_channels:
                x_projected = nn.functional.conv2d(
                    x, 
                    torch.eye(
                        self.out_channels, 
                        self.in_channels, 
                        device=x.device, 
                        dtype=x.dtype
                    ).unsqueeze(-1).unsqueeze(-1),
                    bias=None,
                    groups=1
                ) if self.in_channels == self.out_channels else x
                # Simpler approach: just add directly if same channels
                if self.in_channels == self.out_channels:
                    residual_features = fused_features + x
                else:
                    # Need to handle channel mismatch
                    residual_features = fused_features + x[:, :self.in_channels, :, :]
            else:
                residual_features = fused_features + x
        else:
            residual_features = fused_features
        
        # =====================================================================
        # Step 6: Output Projection (Equation 7)
        # =====================================================================
        output = self.output_conv(residual_features)
        
        return output


# =============================================================================
# CE-Enhanced Backbone (CEB) - Integration with YOLOv11 Backbone
# =============================================================================

class CEBackbone(nn.Module):
    """
    CE-Enhanced Backbone (CEB) for YOLOv11.
    
    This class integrates CE modules into the YOLOv11 backbone to enhance
    edge feature extraction throughout the network.
    
    The CE modules can be inserted at various positions in the backbone
    to progressively enhance edge information.
    
    Attributes:
        num_classes (int): Number of output classes
        channels (list): List of channel dimensions at each stage
        
    Input:
        x (torch.Tensor): Input image tensor
        
    Output:
        List[torch.Tensor]: Multi-scale feature maps [P3, P4, P5]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int, int, int] = (256, 512, 1024),
        num_classes: int = 7,
        use_ce_at_stages: Tuple[bool, bool, bool] = (True, True, False),
        config: Optional[dict] = None
    ):
        """
        Initialize CE-Enhanced Backbone.
        
        Args:
            in_channels: Number of input image channels
            channels: Tuple of (P3, P4, P5) channel dimensions
            num_classes: Number of object classes
            use_ce_at_stages: Which stages to apply CE module
            config: Optional configuration dictionary
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.use_ce_at_stages = use_ce_at_stages
        
        # Get configuration
        if config is None:
            config = get_ce_module_config() if callable(get_ce_module_config) else {}
        
        # =====================================================================
        # Backbone layers (simplified YOLOv11-style architecture)
        # Stage 0: Initial convolution
        # =====================================================================
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0] // 4, 3, stride=2, padding=1),
            nn.SiLU()
        )
        
        # =====================================================================
        # Stage 1: P3 generation (with optional CE)
        # =====================================================================
        self.stage1_conv = nn.Sequential(
            nn.Conv2d(channels[0] // 4, channels[0] // 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels[0] // 2, channels[0], 3, stride=1, padding=1),
            nn.SiLU()
        )
        
        if use_ce_at_stages[0]:
            self.ce_stage1 = CEModule(
                channels[0], 
                channels[0],
                config=config
            )
        else:
            self.ce_stage1 = nn.Identity()
        
        # =====================================================================
        # Stage 2: P4 generation (with optional CE)
        # =====================================================================
        self.stage2_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels[1], channels[1], 3, stride=1, padding=1),
            nn.SiLU()
        )
        
        if use_ce_at_stages[1]:
            self.ce_stage2 = CEModule(
                channels[1],
                channels[1],
                config=config
            )
        else:
            self.ce_stage2 = nn.Identity()
        
        # =====================================================================
        # Stage 3: P5 generation (no CE by default)
        # =====================================================================
        self.stage3_conv = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels[2], channels[2], 3, stride=1, padding=1),
            nn.SiLU()
        )
        
        if use_ce_at_stages[2]:
            self.ce_stage3 = CEModule(
                channels[2],
                channels[2],
                config=config
            )
        else:
            self.ce_stage3 = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through CE-enhanced backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of feature maps (P3, P4, P5)
        """
        # Stem
        x = self.stem(x)  # H/2 × W/2
        
        # Stage 1 -> P3
        x = self.stage1_conv(x)
        p3 = self.ce_stage1(x)  # H/4 × W/4
        
        # Stage 2 -> P4
        x = self.stage2_conv(p3)
        p4 = self.ce_stage2(x)  # H/8 × W/8
        
        # Stage 3 -> P5
        x = self.stage3_conv(p4)
        p5 = self.ce_stage3(x)  # H/16 × W/16
        
        return p3, p4, p5


# =============================================================================
# Utility Functions
# =============================================================================

def create_ce_module(
    in_channels: int, 
    out_channels: Optional[int] = None,
    config: Optional[dict] = None
) -> CEModule:
    """
    Factory function to create a CE module.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        config: Optional configuration dictionary
        
    Returns:
        CEModule: Configured CE module
    """
    return CEModule(
        in_channels=in_channels,
        out_channels=out_channels,
        config=config
    )


# =============================================================================
# Test/Verification Code
# =============================================================================

if __name__ == "__main__":
    # Test the CE module
    print("Testing CE Module Implementation")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test SobelConv
    print("\n1. Testing SobelConv:")
    test_input = torch.randn(2, 64, 128, 128).to(device)
    sobel = SobelConv(64).to(device)
    output = sobel(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Sobel kernels are buffers: {sobel.sobel_x.requires_grad}")
    
    # Test CEModule
    print("\n2. Testing CEModule:")
    ce = CEModule(in_channels=64, out_channels=64).to(device)
    output = ce(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Number of parameters: {sum(p.numel() for p in ce.parameters())}")
    
    # Test with different channels
    print("\n3. Testing CEModule with channel mismatch:")
    ce_mismatch = CEModule(in_channels=64, out_channels=128).to(device)
    output = ce_mismatch(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test CEBackbone
    print("\n4. Testing CEBackbone:")
    backbone = CEBackbone(
        in_channels=3,
        channels=(256, 512, 1024),
        num_classes=7
    ).to(device)
    
    test_image = torch.randn(2, 3, 640, 640).to(device)
    p3, p4, p5 = backbone(test_image)
    print(f"   Input shape: {test_image.shape}")
    print(f"   P3 shape: {p3.shape}")
    print(f"   P4 shape: {p4.shape}")
    print(f"   P5 shape: {p5.shape}")
    
    # Test gradient flow
    print("\n5. Testing gradient flow:")
    test_input = torch.randn(2, 64, 128, 128, requires_grad=True).to(device)
    ce = CEModule(in_channels=64).to(device)
    output = ce(test_input)
    loss = output.sum()
    loss.backward()
    has_grad = test_input.grad is not None
    print(f"   Gradient flow works: {has_grad}")
    print(f"   Gradient shape: {test_input.grad.shape if has_grad else 'N/A'}")
    
    # Print activation functions
    print("\n6. Supported activation functions:")
    activations = ['SiLU', 'ReLU', 'LeakyReLU', 'GELU']
    for act_name in activations:
        try:
            ce_test = CEModule(in_channels=64, activation=act_name)
            print(f"   {act_name}: OK")
        except Exception as e:
            print(f"   {act_name}: Failed - {e}")
    
    print("\n" + "=" * 50)
    print("CE Module test completed!")
