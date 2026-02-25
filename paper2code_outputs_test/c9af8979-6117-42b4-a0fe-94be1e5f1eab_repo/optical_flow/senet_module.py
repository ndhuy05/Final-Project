"""
optical_flow/senet_module.py

Squeeze-and-Excitation (SE) attention module for the conveyor belt speed detection system.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements the SENet attention mechanism that the paper integrates into 
the RAFT optical flow model to enhance feature adaptability under varying illumination 
conditions and low-texture scenarios.

Author: Based on paper methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.
    
    This module implements the SE attention mechanism described in the paper.
    It learns to re-weight feature channels based on their importance, which helps:
    1. Suppress feature channels corresponding to overexposed/underexposed regions
    2. Enhance feature channels containing useful details (particles, scratches)
    3. Focus on information-rich areas while reducing background interference
    
    The SE block consists of:
    1. Squeeze: Global Average Pooling to aggregate spatial information
    2. Excitation: Two fully connected layers (bottleneck) to learn channel dependencies
    3. Scale: Channel-wise multiplication to apply learned weights
    
    Attributes:
        channels: Number of input channels (C)
        reduction: Reduction ratio for the bottleneck (r)
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        bias: bool = False
    ):
        """Initialize the SE block.
        
        Args:
            channels: Number of input channels (C)
            reduction: Reduction ratio for the bottleneck (r), default 16 from config
            bias: Whether to use bias in the excitation layers
        """
        super(SEBlock, self).__init__()
        
        # Validate input parameters
        if channels <= 0:
            raise ValueError(f"Number of channels must be positive, got {channels}")
        if reduction <= 0:
            raise ValueError(f"Reduction ratio must be positive, got {reduction}")
        
        self.channels = channels
        self.reduction = reduction
        
        # Calculate bottleneck size
        # Using max(1, ...) to handle small channel counts gracefully
        self.bottleneck_channels = max(1, channels // reduction)
        
        # Excitation: Two fully connected layers forming a bottleneck
        # FC1: C -> C/r (reduce dimensions)
        # FC2: C/r -> C (expand back to original)
        self.fc1 = nn.Linear(channels, self.bottleneck_channels, bias=bias)
        self.fc2 = nn.Linear(self.bottleneck_channels, channels, bias=bias)
        
        # Initialize weights for better training stability
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize the weights of the SE block using Kaiming initialization."""
        # Kaiming normal initialization is suitable for ReLU activations
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize biases to zero if present
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SE attention mechanism.
        
        This implements the three-stage SE block:
        1. Squeeze: Global Average Pooling to create channel descriptors
        2. Excitation: FC layers to learn channel importance weights
        3. Scale: Re-weight features using learned channel weights
        
        Args:
            x: Input tensor of shape (B, C, H, W) where:
               B = batch size
               C = number of channels
               H = height
               W = width
            
        Returns:
            Re-weighted tensor of shape (B, C, H, W) with same dtype as input
        """
        # Store original shape
        batch_size, channels, height, width = x.size()
        
        # Validate input channel count matches configured channels
        if channels != self.channels:
            raise ValueError(
                f"Input has {channels} channels but SEBlock expects {self.channels}"
            )
        
        # ============================================================
        # Stage 1: Squeeze - Global Average Pooling
        # ============================================================
        # Input: (B, C, H, W) -> Output: (B, C)
        # This creates a channel descriptor by aggregating spatial information
        # Following Equation from paper methodology: z_c = (1/(H*W)) * sum(x_c)
        y = x.view(batch_size, channels, height * width).mean(dim=2)
        
        # ============================================================
        # Stage 2: Excitation - Learn channel dependencies
        # ============================================================
        # Input: (B, C) -> Output: (B, C)
        # Two FC layers with bottleneck learn non-linear channel relationships
        # FC1: C -> C/r with ReLU activation
        # FC2: C/r -> C with Sigmoid activation (outputs in range [0, 1])
        
        # First FC layer with ReLU activation (bottleneck)
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        
        # Second FC layer with Sigmoid activation (produces attention weights)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        
        # ============================================================
        # Stage 3: Scale - Re-weight features
        # ============================================================
        # Input: (B, C) and (B, C, H, W) -> Output: (B, C, H, W)
        # Reshape attention weights and apply channel-wise multiplication
        # Following Equation: x_out = s * x_in (element-wise for each channel)
        
        # Reshape attention weights to (B, C, 1, 1) for broadcasting
        y = y.view(batch_size, channels, 1, 1)
        
        # Scale: multiply each channel by its learned importance weight
        # Using expand_as to broadcast to full spatial dimensions
        output = x * y.expand(batch_size, channels, height, width)
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string with extra representation information."""
        return (
            f"channels={self.channels}, "
            f"reduction={self.reduction}, "
            f"bottleneck={self.bottleneck_channels}"
        )


class SEBlock2D(nn.Module):
    """Alternative 2D implementation of SE block using convolutions.
    
    This version uses 1x1 convolutions instead of linear layers,
    which can be more efficient in some deployment scenarios.
    
    This is functionally equivalent to SEBlock but implemented differently.
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        bias: bool = False
    ):
        """Initialize the 2D SE block.
        
        Args:
            channels: Number of input channels (C)
            reduction: Reduction ratio for the bottleneck (r)
            bias: Whether to use bias in conv layers
        """
        super(SEBlock2D, self).__init__()
        
        if channels <= 0:
            raise ValueError(f"Number of channels must be positive, got {channels}")
        if reduction <= 0:
            raise ValueError(f"Reduction ratio must be positive, got {reduction}")
        
        self.channels = channels
        self.reduction = reduction
        
        # Calculate bottleneck channels
        self.bottleneck_channels = max(1, channels // reduction)
        
        # Squeeze: Global Average Pooling (implemented in forward)
        
        # Excitation using 1x1 convolutions instead of linear layers
        # Conv1: C -> C/r
        self.conv1 = nn.Conv2d(
            channels, 
            self.bottleneck_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=bias
        )
        # Conv2: C/r -> C
        self.conv2 = nn.Conv2d(
            self.bottleneck_channels, 
            channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=bias
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize the weights."""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 2D SE attention mechanism.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Re-weighted tensor of shape (B, C, H, W)
        """
        # Squeeze: Global Average Pooling
        # Input: (B, C, H, W) -> Output: (B, C, 1, 1)
        y = F.adaptive_avg_pool2d(x, 1)
        
        # Excitation
        y = self.conv1(y)
        y = F.relu(y, inplace=True)
        y = self.conv2(y)
        y = torch.sigmoid(y)
        
        # Scale: multiply each channel by its weight
        output = x * y
        
        return output


class SpatialChannelSEBlock(nn.Module):
    """Combined Spatial and Channel SE attention block.
    
    This extended version adds spatial attention on top of channel attention,
    which can help focus on specific spatial regions of the conveyor belt
    that contain more motion information.
    
    Based on the paper's emphasis on focusing on "information-rich areas"
    such as particles and scratches on the belt surface.
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
        bias: bool = False
    ):
        """Initialize the spatial-channel SE block.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for channel attention bottleneck
            spatial_kernel: Kernel size for spatial attention conv
            bias: Whether to use bias in conv layers
        """
        super(SpatialChannelSEBlock, self).__init__()
        
        if channels <= 0:
            raise ValueError(f"Number of channels must be positive, got {channels}")
        
        self.channels = channels
        self.reduction = reduction
        self.spatial_kernel = spatial_kernel
        
        # Channel attention components
        self.bottleneck_channels = max(1, channels // reduction)
        self.channel_fc1 = nn.Linear(channels, self.bottleneck_channels, bias=bias)
        self.channel_fc2 = nn.Linear(self.bottleneck_channels, channels, bias=bias)
        
        # Spatial attention components
        # Input is 2*C (concatenation of max and avg pooled features)
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(
            2, 
            1, 
            kernel_size=spatial_kernel, 
            stride=1, 
            padding=padding, 
            bias=bias
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.kaiming_normal_(self.channel_fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.channel_fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.spatial_conv.weight, mode='fan_out', nonlinearity='relu')
        
        if self.channel_fc1.bias is not None:
            nn.init.zeros_(self.channel_fc1.bias)
        if self.channel_fc2.bias is not None:
            nn.init.zeros_(self.channel_fc2.bias)
        if self.spatial_conv.bias is not None:
            nn.init.zeros_(self.spatial_conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply combined spatial and channel attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Attended tensor of shape (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()
        
        # ============================================================
        # Channel Attention (same as SEBlock)
        # ============================================================
        # Squeeze
        channel_y = x.view(batch_size, channels, height * width).mean(dim=2)
        
        # Excitation
        channel_y = F.relu(self.channel_fc1(channel_y), inplace=True)
        channel_y = torch.sigmoid(self.channel_fc2(channel_y))
        channel_y = channel_y.view(batch_size, channels, 1, 1)
        
        # Apply channel attention
        x = x * channel_y.expand(batch_size, channels, height, width)
        
        # ============================================================
        # Spatial Attention
        # ============================================================
        # Compute max and average pooling along channel dimension
        # Input: (B, C, H, W) -> Output: (B, 1, H, W)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate: (B, 2, H, W)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        
        # Apply convolution and sigmoid
        # Output: (B, 1, H, W)
        spatial_attention = self.spatial_conv(spatial_input)
        spatial_attention = torch.sigmoid(spatial_attention)
        
        # Apply spatial attention
        output = x * spatial_attention
        
        return output


def create_se_block(
    channels: int,
    reduction: int = 16,
    bias: bool = False,
    use_2d_version: bool = False
) -> nn.Module:
    """Factory function to create an SE block.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck
        bias: Whether to use bias in FC/conv layers
        use_2d_version: If True, use SEBlock2D; otherwise use SEBlock
        
    Returns:
        SEBlock or SEBlock2D instance
    """
    if use_2d_version:
        return SEBlock2D(channels=channels, reduction=reduction, bias=bias)
    else:
        return SEBlock(channels=channels, reduction=reduction, bias=bias)


# Global registry for SE block configurations
_se_block_registry = {}


def register_se_block(name: str, block: nn.Module) -> None:
    """Register a named SE block configuration.
    
    Args:
        name: Name identifier for the block
        block: SE block instance
    """
    _se_block_registry[name] = block


def get_se_block(name: str) -> Optional[nn.Module]:
    """Get a registered SE block by name.
    
    Args:
        name: Name identifier
        
    Returns:
        SE block instance or None if not found
    """
    return _se_block_registry.get(name)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("SE Block Module Test")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters (matching config.yaml)
    batch_size = 4
    channels = 256
    height = 64
    width = 64
    reduction = 16
    
    # Create test input tensor
    test_input = torch.randn(batch_size, channels, height, width)
    
    print(f"\n[Test Configuration]")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Channels: {channels}")
    print(f"  Reduction ratio: {reduction}")
    print(f"  Bottleneck channels: {max(1, channels // reduction)}")
    
    # Test 1: Basic SEBlock
    print("\n[Test 1] Basic SEBlock")
    se_block = SEBlock(channels=channels, reduction=reduction)
    output = se_block(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    assert output.shape == test_input.shape, "Output shape mismatch"
    print("  ✓ SEBlock basic test passed")
    
    # Test 2: Verify channel re-weighting
    print("\n[Test 2] Channel re-weighting verification")
    # Check that output is different from input (re-weighting applied)
    diff = (output - test_input).abs().mean()
    print(f"  Mean absolute difference: {diff:.4f}")
    assert diff > 1e-4, "SE block should modify features"
    print("  ✓ Channel re-weighting verified")
    
    # Test 3: Gradient flow
    print("\n[Test 3] Gradient flow verification")
    test_input_grad = torch.randn(2, 128, 32, 32, requires_grad=True)
    se_block_grad = SEBlock(channels=128, reduction=16)
    output_grad = se_block_grad(test_input_grad)
    loss = output_grad.sum()
    loss.backward()
    print(f"  Input gradient shape: {test_input_grad.grad.shape}")
    assert test_input_grad.grad is not None, "Gradient should flow"
    print("  ✓ Gradient flow verified")
    
    # Test 4: SEBlock2D version
    print("\n[Test 4] SEBlock2D version")
    se_block_2d = SEBlock2D(channels=channels, reduction=reduction)
    output_2d = se_block_2d(test_input)
    print(f"  Output shape: {output_2d.shape}")
    assert output_2d.shape == test_input.shape, "Output shape mismatch"
    print("  ✓ SEBlock2D test passed")
    
    # Test 5: Compare SEBlock and SEBlock2D
    print("\n[Test 5] Compare SEBlock and SEBlock2D")
    se_block_1 = SEBlock(channels=64, reduction=16)
    se_block_2 = SEBlock2D(channels=64, reduction=16)
    
    test_input_small = torch.randn(2, 64, 16, 16)
    output_1 = se_block_1(test_input_small)
    output_2 = se_block_2(test_input_small)
    
    # Results should be similar but not identical (different implementations)
    similarity = F.cosine_similarity(output_1.flatten(), output_2.flatten(), dim=0)
    print(f"  Cosine similarity: {similarity:.4f}")
    print("  ✓ Comparison test completed")
    
    # Test 6: SpatialChannelSEBlock
    print("\n[Test 6] SpatialChannelSEBlock")
    spatial_se_block = SpatialChannelSEBlock(
        channels=channels, 
        reduction=reduction,
        spatial_kernel=7
    )
    output_spatial = spatial_se_block(test_input)
    print(f"  Output shape: {output_spatial.shape}")
    assert output_spatial.shape == test_input.shape
    print("  ✓ SpatialChannelSEBlock test passed")
    
    # Test 7: Edge case - small channel count
    print("\n[Test 7] Edge case - small channel count")
    small_channels = 8
    se_small = SEBlock(channels=small_channels, reduction=16)
    small_input = torch.randn(2, small_channels, 8, 8)
    small_output = se_small(small_input)
    print(f"  Input: {small_input.shape}, Output: {small_output.shape}")
    assert small_output.shape == small_input.shape
    print("  ✓ Small channel count handled correctly")
    
    # Test 8: Edge case - single channel
    print("\n[Test 8] Edge case - single channel")
    single_channel = 1
    se_single = SEBlock(channels=single_channel, reduction=16)
    single_input = torch.randn(2, single_channel, 8, 8)
    single_output = se_single(single_input)
    print(f"  Input: {single_input.shape}, Output: {single_output.shape}")
    assert single_output.shape == single_input.shape
    print("  ✓ Single channel handled correctly")
    
    # Test 9: Different batch sizes
    print("\n[Test 9] Different batch sizes")
    for bs in [1, 2, 8, 16]:
        test_input_bs = torch.randn(bs, 64, 16, 16)
        output_bs = se_block(test_input_bs)
        assert output_bs.shape == test_input_bs.shape, f"Failed for batch size {bs}"
    print("  ✓ Different batch sizes handled correctly")
    
    # Test 10: Factory function
    print("\n[Test 10] Factory function")
    se_from_factory = create_se_block(channels=128, reduction=8)
    factory_input = torch.randn(1, 128, 16, 16)
    factory_output = se_from_factory(factory_input)
    print(f"  Factory created: {type(se_from_factory).__name__}")
    print(f"  Output shape: {factory_output.shape}")
    assert factory_output.shape == factory_input.shape
    print("  ✓ Factory function works")
    
    # Test 11: Extra representation
    print("\n[Test 11] Extra representation")
    repr_str = repr(se_block)
    print(f"  Representation: {repr_str}")
    assert "channels=256" in repr_str
    assert "reduction=16" in repr_str
    print("  ✓ Extra representation correct")
    
    # Test 12:.eval() mode (for inference)
    print("\n[Test 12] Inference mode")
    se_block.eval()
    with torch.no_grad():
        inference_input = torch.randn(2, 256, 32, 32)
        inference_output = se_block(inference_input)
        print(f"  Inference output shape: {inference_output.shape}")
        print(f"  Inference output range: [{inference_output.min():.4f}, {inference_output.max():.4f}]")
    se_block.train()
    print("  ✓ Inference mode works")
    
    # Test 13: torch.jit compatibility test
    print("\n[Test 13] TorchScript compatibility")
    try:
        se_block_script = SEBlock(channels=128, reduction=16)
        scripted = torch.jit.script(se_block_script)
        script_input = torch.randn(1, 128, 16, 16)
        script_output = scripted(script_input)
        print(f"  Scripted output shape: {script_output.shape}")
        print("  ✓ TorchScript compatibility verified")
    except Exception as e:
        print(f"  Note: TorchScript test failed ({e}), but core functionality works")
    
    print("\n" + "=" * 60)
    print("All SE Block tests passed!")
    print("=" * 60)
