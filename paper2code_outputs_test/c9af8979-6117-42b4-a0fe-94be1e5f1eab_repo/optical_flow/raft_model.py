"""
optical_flow/raft_model.py

RAFT-SEnet optical flow model for conveyor belt speed detection.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements the RAFT (Recurrent All-Pairs Field Transforms) architecture
with integrated SENet attention mechanism for enhanced feature extraction under
varying illumination conditions and low-texture scenarios.

Author: Based on paper methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import math

# Try to import config and SENet module
try:
    from config import Config, get_config, OpticalFlowConfig
    from optical_flow.senet_module import SEBlock, create_se_block
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    OpticalFlowConfig = None
    SEBlock = None
    create_se_block = None


class ConvEncoder(nn.Module):
    """Convolutional encoder for feature extraction with optional SENet integration.
    
    This encoder extracts hierarchical features from input images using a series
    of convolutional layers with batch normalization and ReLU activation.
    Optionally integrates SENet attention for enhanced feature re-weighting.
    
    Based on the paper's RAFT-SEnet architecture for motion feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 6,
        use_senet: bool = True,
        senet_reduction: int = 16
    ):
        """Initialize the convolutional encoder.
        
        Args:
            in_channels: Number of input image channels (3 for RGB)
            base_channels: Base number of channels for the first layer
            num_levels: Number of feature pyramid levels
            use_senet: Whether to use SENet attention modules
            senet_reduction: Reduction ratio for SENet bottleneck
        """
        super(ConvEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.use_senet = use_senet
        self.senet_reduction = senet_reduction
        
        # Initial convolution to expand channels
        self.conv0 = nn.Conv2d(in_channels, base_channels, 7, 2, 3)
        self.bn0 = nn.BatchNorm2d(base_channels)
        
        # First residual block
        self.conv1 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(base_channels)
        
        # Additional downsampling layers to build feature pyramid
        self.downsample_layers = nn.ModuleList()
        self.senet_layers = nn.ModuleList()
        
        for i in range(num_levels - 2):
            in_ch = base_channels * (2 ** min(i, 2))
            out_ch = base_channels * (2 ** min(i + 1, 2))
            
            layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.downsample_layers.append(layer)
            
            # Add SENet after first two downsampling layers (for better illumination adaptation)
            if use_senet and i < 2:
                self.senet_layers.append(create_se_block(out_ch, reduction=senet_reduction))
            else:
                self.senet_layers.append(None)
        
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from input image.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            List of feature tensors at different scales (coarse to fine)
        """
        features = []
        
        # Initial convolution
        x = self.relu(self.bn0(self.conv0(x)))
        
        # First residual block
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + identity
        features.append(x)
        
        # Downsampling with optional SENet attention
        for idx, layer in enumerate(self.downsample_layers):
            x = layer(x)
            
            # Apply SENet attention for illumination adaptation
            if self.use_senet and self.senet_layers[idx] is not None:
                x = self.senet_layers[idx](x)
            
            features.append(x)
        
        return features


class ContextEncoder(nn.Module):
    """Context encoder for extracting contextual features from reference frame.
    
    This encoder processes the reference frame to generate contextual information
    that is used by the update operator to refine the optical flow estimate.
    Based on Section 3.1 of the paper.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        output_channels: int = 128
    ):
        """Initialize the context encoder.
        
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels
            output_channels: Output channel count (matches GRU hidden size)
        """
        super(ContextEncoder, self).__init__()
        
        self.conv0 = nn.Conv2d(in_channels, base_channels, 7, 2, 3)
        self.bn0 = nn.BatchNorm2d(base_channels)
        
        self.conv1 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(base_channels)
        
        # Downsample to reduce resolution
        self.downsample1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(base_channels * 2)
        
        self.conv4 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(base_channels * 2)
        
        self.downsample2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.conv5 = nn.Conv2d(base_channels * 4, output_channels, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(output_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract contextual features from reference frame.
        
        Args:
            x: Reference frame tensor [B, 3, H, W]
            
        Returns:
            Contextual features [B, output_channels, H/8, W/8]
        """
        x = self.relu(self.bn0(self.conv0(x)))
        
        # First block
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + identity
        
        # Downsample and second block
        x = self.downsample1(x)
        identity = x
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x + identity
        
        # Final downsample
        x = self.downsample2(x)
        
        # Output projection
        x = self.relu(self.bn5(self.conv5(x)))
        
        return x


class CorrelationVolume(nn.Module):
    """4D correlation volume builder with multi-scale pyramid.
    
    Builds correlation volumes from feature similarities using a 4-level
    feature pyramid as described in Section 3.1 of the paper.
    """
    
    def __init__(
        self,
        feature_channels: int = 64,
        num_levels: int = 4
    ):
        """Initialize correlation volume builder.
        
        Args:
            feature_channels: Number of channels in input features
            num_levels: Number of pyramid levels
        """
        super(CorrelationVolume, self).__init__()
        
        self.feature_channels = feature_channels
        self.num_levels = num_levels
        
        # 1x1 conv to reduce feature channels for correlation computation
        self.proj = nn.Conv2d(feature_channels, feature_channels, 1)
    
    def forward(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor
    ) -> List[torch.Tensor]:
        """Build correlation volume pyramid from feature maps.
        
        Args:
            fmap1: Feature map from first frame [B, C, H, W]
            fmap2: Feature map from second frame [B, C, H, W]
            
        Returns:
            List of correlation volumes at different scales
        """
        # Reduce feature dimensions
        fmap1 = self.proj(fmap1)
        fmap2 = self.proj(fmap2)
        
        batch, channels, height, width = fmap1.shape
        
        # Reshape for correlation computation
        fmap1 = fmap1.view(batch, channels, height * width)
        fmap2 = fmap2.view(batch, channels, height * width)
        
        # Compute correlation matrix using matrix multiplication
        # [B, H*W, H*W] correlation matrix
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        
        # Normalize correlations
        corr = corr / math.sqrt(channels)
        
        # Reshape to 4D correlation volume [B, H, W, H, W]
        corr = corr.view(batch, height, width, height, width)
        
        correlation_volumes = []
        
        # Build multi-scale pyramid
        for level in range(self.num_levels):
            if level == 0:
                correlation_volumes.append(corr)
            else:
                # Downsample correlation volume by pooling
                scale = 2 ** level
                h, w = height // scale, width // scale
                
                # Use adaptive pooling to get consistent spatial dimensions
                corr_reshaped = corr.view(batch, height, width, height, width)
                
                # Reshape for efficient downsampling
                corr_reshaped = corr_reshaped.permute(0, 3, 4, 1, 2).contiguous()
                corr_reshaped = corr_reshaped.view(batch * height * width, 1, height, width)
                
                # Downsample
                corr_down = F.adaptive_avg_pool2d(corr_reshaped, (h, w))
                corr_down = corr_down.view(batch, height, width, h, w)
                
                correlation_volumes.append(corr_down)
        
        return correlation_volumes


class CorrelationSampler(nn.Module):
    """Sample from correlation volume at positions specified by flow field."""
    
    def __init__(self):
        """Initialize correlation sampler."""
        super(CorrelationSampler, self).__init__()
    
    def forward(
        self,
        correlation: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """Sample correlation features at flow positions.
        
        Args:
            correlation: Correlation volume [B, H, W, H, W]
            flow: Flow field [B, 2, H, W] in normalized coordinates
            
        Returns:
            Sampled correlation features [B, H, W, 1]
        """
        batch, _, height, width = flow.shape
        
        # Create meshgrid for sampling
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=flow.device, dtype=torch.float32),
            torch.arange(width, device=flow.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Normalize coordinates to [-1, 1]
        x_normalized = 2.0 * x_grid / (width - 1) - 1.0
        y_normalized = 2.0 * y_grid / (height - 1) - 1.0
        
        # Add flow displacement
        flow_u = flow[:, 0, :, :]  # Horizontal component
        flow_v = flow[:, 1, :, :]  # Vertical component
        
        # Scale flow to match correlation volume dimensions
        # Flow is in image coordinates, need to scale to correlation coordinates
        x_offset = flow_u / width * 2.0
        y_offset = flow_v / height * 2.0
        
        # Compute sampling coordinates
        x_coords = x_normalized.unsqueeze(0).unsqueeze(0) + x_offset.unsqueeze(1)
        y_coords = y_normalized.unsqueeze(0).unsqueeze(0) + y_offset.unsqueeze(1)
        
        # Stack coordinates [B, 2, H, W]
        sampling_grid = torch.stack([x_coords, y_coords], dim=1)
        
        # Reshape correlation for sampling
        # [B, H, W, H, W] -> [B, H*W, H, W]
        corr_reshaped = correlation.view(batch, height, width, height * width)
        corr_reshaped = corr_reshaped.permute(0, 3, 1, 2).contiguous()
        
        # Sample using grid_sample
        corr_sampled = F.grid_sample(
            corr_reshaped,
            sampling_grid.permute(0, 2, 3, 1),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Reshape output
        corr_sampled = corr_sampled.squeeze(-1).permute(0, 2, 3, 1)
        
        return corr_sampled


class UpdateOperator(nn.Module):
    """GRU-based update operator for iterative flow refinement.
    
    This module updates the flow estimate using correlation features,
    contextual features, and the current flow state.
    Based on Section 3.1 of the paper (Updater module).
    """
    
    def __init__(
        self,
        hidden_channels: int = 128,
        correlation_channels: int = 64,
        context_channels: int = 128
    ):
        """Initialize the update operator.
        
        Args:
            hidden_channels: Hidden state channels (GRU hidden size)
            correlation_channels: Correlation feature channels
            context_channels: Context feature channels
        """
        super(UpdateOperator, self).__init__()
        
        self.hidden_channels = hidden_channels
        
        # Input channels: correlation + context + hidden state
        input_channels = 1 + context_channels + hidden_channels
        
        # GRU for recurrent updates
        self.gru = nn.GRUCell(input_channels, hidden_channels)
        
        # Convolution to process GRU output
        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        # Head to predict flow residual
        self.flow_head = nn.Conv2d(hidden_channels, 2, 3, 1, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        corr_features: torch.Tensor,
        context_features: torch.Tensor,
        current_flow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update flow estimate using correlation and context features.
        
        Args:
            hidden_state: Previous hidden state [B, hidden_channels, H, W]
            corr_features: Correlation features [B, H, W, 1]
            context_features: Contextual features [B, context_channels, H, W]
            current_flow: Current flow estimate [B, 2, H, W]
            
        Returns:
            Tuple of (new_hidden_state, flow_residual)
        """
        batch, _, height, width = hidden_state.shape
        
        # Prepare correlation features
        corr = corr_features.permute(0, 3, 1, 2)  # [B, 1, H, W]
        
        # Prepare context features (already [B, C, H, W])
        
        # Prepare flow (for skip connection)
        flow = current_flow
        
        # Concatenate all inputs
        x = torch.cat([corr, context_features, hidden_state], dim=1)
        
        # Reshape for GRU: [B, C, H, W] -> [B*H*W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch * height * width, -1)
        
        hidden_flat = hidden_state.permute(0, 2, 3, 1).contiguous()
        hidden_flat = hidden_flat.view(batch * height * width, -1)
        
        # Update hidden state
        new_hidden_flat = self.gru(x, hidden_flat)
        new_hidden = new_hidden_flat.view(batch, height, width, self.hidden_channels)
        new_hidden = new_hidden.permute(0, 3, 1, 2)
        
        # Process updated hidden state
        x = self.relu(self.bn1(self.conv1(new_hidden)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Predict flow residual
        flow_delta = self.flow_head(x)
        
        return new_hidden, flow_delta


class RAFTSEnetModel(nn.Module):
    """RAFT optical flow model with integrated SENet attention.
    
    This is the main model class implementing the RAFT-SEnet architecture
    as described in the paper for conveyor belt speed detection.
    
    Architecture components:
    - Feature Encoder (Siamese CNN with shared weights)
    - Context Encoder
    - 4D Correlation Volume Builder
    - SENet attention modules
    - GRU-based Update Operator
    
    The model performs iterative flow refinement to estimate dense optical flow.
    """
    
    def __init__(
        self,
        config: Optional[OpticalFlowConfig] = None,
        base_channels: int = 64,
        hidden_channels: int = 128,
        num_iterations: int = 12,
        correlation_levels: int = 4
    ):
        """Initialize the RAFT-SEnet model.
        
        Args:
            config: Optical flow configuration (from config.yaml)
            base_channels: Base number of channels for feature extraction
            hidden_channels: Hidden channels for GRU (default matches paper)
            num_iterations: Number of update iterations
            correlation_levels: Number of correlation pyramid levels
        """
        super(RAFTSEnetModel, self).__init__()
        
        # Use config if provided
        if config is not None:
            self.input_height = config.input_height
            self.input_width = config.input_width
            self.use_senet = config.use_senet
            self.senet_reduction = config.senet_reduction
        else:
            self.input_height = 448
            self.input_width = 256
            self.use_senet = True
            self.senet_reduction = 16
        
        self.base_channels = base_channels
        self.hidden_channels = hidden_channels
        self.num_iterations = num_iterations
        self.correlation_levels = correlation_levels
        
        # Feature encoder (Siamese network - shared weights for both frames)
        self.feature_encoder = ConvEncoder(
            in_channels=3,
            base_channels=base_channels,
            num_levels=6,
            use_senet=self.use_senet,
            senet_reduction=self.senet_reduction
        )
        
        # Context encoder (processes reference frame only)
        self.context_encoder = ContextEncoder(
            in_channels=3,
            base_channels=base_channels,
            output_channels=hidden_channels
        )
        
        # Correlation volume builder
        self.correlation_volume = CorrelationVolume(
            feature_channels=base_channels * 4,
            num_levels=correlation_levels
        )
        
        # Correlation sampler
        self.correlation_sampler = CorrelationSampler()
        
        # Update operator (GRU-based)
        self.update_operator = UpdateOperator(
            hidden_channels=hidden_channels,
            correlation_channels=1,
            context_channels=hidden_channels
        )
        
        # Flow head for initialization
        self.flow_head = nn.Conv2d(base_channels * 4, 2, 3, 1, 1)
        
        # Initialize weights
        self._init_weights()
        
        # Store intermediate flows for training
        self.intermediate_flows: List[torch.Tensor] = []
    
    def _init_weights(self):
        """Initialize all model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRUCell):
                nn.init.xavier_uniform_(m.weight_ih)
                nn.init.xavier_uniform_(m.weight_hh)
                nn.init.zeros_(m.bias_ih)
                nn.init.zeros_(m.bias_hh)
    
    def initialize_flow(self, features: torch.Tensor) -> torch.Tensor:
        """Initialize the flow field to zero.
        
        Args:
            features: Feature tensor to determine output dimensions
            
        Returns:
            Initial zero flow field [B, 2, H, W]
        """
        batch, _, height, width = features.shape
        return torch.zeros(batch, 2, height, width, device=features.device)
    
    def upsample_flow(
        self,
        flow: torch.Tensor,
        target_height: int,
        target_width: int
    ) -> torch.Tensor:
        """Upsample flow field to higher resolution using bilinear interpolation.
        
        Args:
            flow: Flow field [B, 2, H, W]
            target_height: Target height
            target_width: Target width
            
        Returns:
            Upsampled flow field [B, 2, target_height, target_width]
        """
        batch, _, height, width = flow.shape
        
        # Create upsampled flow
        flow_upsampled = F.interpolate(
            flow,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=True
        )
        
        # Scale flow values to match new resolution
        scale_y = target_height / height
        scale_x = target_width / width
        flow_upsampled[:, 0] *= scale_x
        flow_upsampled[:, 1] *= scale_y
        
        return flow_upsampled
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_iterations: Optional[int] = None,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """Forward pass through RAFT-SEnet model.
        
        Args:
            frame1: First frame [B, 3, H, W]
            frame2: Second frame [B, 3, H, W]
            num_iterations: Number of update iterations (uses default if None)
            return_intermediate: Whether to return intermediate flow estimates
            
        Returns:
            Dictionary containing:
                - 'flow': Final optical flow [B, 2, H, W]
                - 'intermediate_flows': List of intermediate flows (if return_intermediate=True)
        """
        if num_iterations is None:
            num_iterations = self.num_iterations
        
        # Reset intermediate flows
        if return_intermediate:
            self.intermediate_flows = []
        
        # Extract features from both frames using shared-weight encoder
        features1 = self.feature_encoder(frame1)
        features2 = self.feature_encoder(frame2)
        
        # Use the finest features for correlation
        fmap1 = features1[-1]  # Coarsest level
        fmap2 = features2[-1]
        
        # Extract context features from reference frame
        context = self.context_encoder(frame1)
        
        # Build correlation volume pyramid
        correlation_pyramid = self.correlation_volume(fmap1, fmap2)
        
        # Initialize flow to zero
        flow = self.initialize_flow(fmap1)
        
        # Initialize hidden state
        batch, _, height, width = fmap1.shape
        hidden_state = torch.zeros(
            batch, self.hidden_channels, height, width,
            device=fmap1.device
        )
        
        # Iterative flow refinement
        for iteration in range(num_iterations):
            # Sample correlation features at current flow positions
            corr_features = self.correlation_sampler(correlation_pyramid[0], flow)
            
            # Update hidden state and predict flow residual
            hidden_state, flow_delta = self.update_operator(
                hidden_state,
                corr_features,
                context,
                flow
            )
            
            # Add residual to current flow estimate
            flow = flow + flow_delta
            
            # Store intermediate flow
            if return_intermediate:
                self.intermediate_flows.append(flow)
        
        # Upsample flow to original input resolution
        _, _, orig_height, orig_width = frame1.shape
        flow_final = self.upsample_flow(flow, orig_height, orig_width)
        
        # Prepare output
        output = {'flow': flow_final}
        
        if return_intermediate:
            # Upsample intermediate flows
            intermediate_upsampled = []
            for intermediate_flow in self.intermediate_flows:
                intermediate_upsampled.append(
                    self.upsample_flow(intermediate_flow, orig_height, orig_width)
                )
            output['intermediate_flows'] = intermediate_upsampled
        
        return output
    
    def get_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_iterations: Optional[int] = None
    ) -> torch.Tensor:
        """Simplified interface for getting optical flow.
        
        This is a convenience method for inference.
        
        Args:
            frame1: First frame [B, 3, H, W]
            frame2: Second frame [B, 3, H, W]
            num_iterations: Number of update iterations
            
        Returns:
            Optical flow field [B, 2, H, W]
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(frame1, frame2, num_iterations=num_iterations)
        return result['flow']


def create_raft_senet_model(
    config: Optional[OpticalFlowConfig] = None,
    pretrained_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> RAFTSEnetModel:
    """Factory function to create a RAFT-SEnet model.
    
    Args:
        config: Optical flow configuration
        pretrained_path: Path to pretrained weights (optional)
        device: Device to load model on
        
    Returns:
        Initialized RAFTSEnetModel
    """
    # Get config if not provided
    if config is None and CONFIG_AVAILABLE:
        try:
            config = get_config().model.optical_flow
        except:
            pass
    
    # Create model
    model = RAFTSEnetModel(config=config)
    
    # Load pretrained weights if provided
    if pretrained_path is not None:
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    # Move to device
    model = model.to(device)
    
    return model


class RAFTWrapper(nn.Module):
    """Wrapper class for simplified RAFT-SEnet integration.
    
    This wrapper provides a simple interface compatible with the
    broader speed detection pipeline.
    """
    
    def __init__(
        self,
        config: Optional[OpticalFlowConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize RAFT wrapper.
        
        Args:
            config: Optical flow configuration
            device: Device to run on
        """
        super(RAFTWrapper, self).__init__()
        
        self.device = device
        self.model = create_raft_senet_model(config=config, device=device)
        
        # Set to evaluation mode
        self.model.eval()
    
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Estimate optical flow between two frames.
        
        Args:
            frame1: First frame tensor [B, 3, H, W]
            frame2: Second frame tensor [B, 3, H, W]
            
        Returns:
            Optical flow field [B, 2, H, W]
        """
        # Move to device if needed
        if frame1.device != self.device:
            frame1 = frame1.to(self.device)
        if frame2.device != self.device:
            frame2 = frame2.to(self.device)
        
        # Get flow
        flow = self.model.get_flow(frame1, frame2)
        
        return flow
    
    def estimate_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        num_iterations: int = 12
    ) -> torch.Tensor:
        """Estimate optical flow with custom iteration count.
        
        Args:
            frame1: First frame
            frame2: Second frame
            num_iterations: Number of refinement iterations
            
        Returns:
            Optical flow field
        """
        with torch.no_grad():
            result = self.model.forward(frame1, frame2, num_iterations=num_iterations)
        return result['flow']


# Global model registry
_models: Dict[str, RAFTSEnetModel] = {}


def get_model(
    name: str = "default",
    config: Optional[OpticalFlowConfig] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> RAFTSEnetModel:
    """Get or create a RAFT-SEnet model.
    
    Args:
        name: Model identifier
        config: Configuration
        device: Device
        
    Returns:
        RAFTSEnetModel instance
    """
    global _models
    
    if name in _models:
        return _models[name]
    
    model = create_raft_senet_model(config=config, device=device)
    _models[name] = model
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("RAFT-SEnet Model Test")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Test configuration
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            opt_config = config.model.optical_flow
            print(f"Config - Input size: {opt_config.input_size}")
            print(f"Config - Use SENet: {opt_config.use_senet}")
            print(f"Config - SENet reduction: {opt_config.senet_reduction}")
        except:
            print("Using default configuration")
            opt_config = None
    else:
        opt_config = None
    
    # Create model
    print("\n[Test 1] Creating model...")
    model = RAFTSEnetModel(config=opt_config)
    model = model.to(device)
    print(f"  Model created: {type(model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test input
    print("\n[Test 2] Testing forward pass...")
    batch_size = 2
    height, width = 448, 256
    
    # Create dummy inputs (normalized RGB)
    frame1 = torch.randn(batch_size, 3, height, width).to(device)
    frame2 = torch.randn(batch_size, 3, height, width).to(device)
    
    print(f"  Input shape: {frame1.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        result = model(frame1, frame2, num_iterations=4)
    
    flow = result['flow']
    print(f"  Output flow shape: {flow.shape}")
    print(f"  Flow range: [{flow.min():.4f}, {flow.max():.4f}]")
    assert flow.shape == (batch_size, 2, height, width)
    print("  ✓ Forward pass successful")
    
    # Test with intermediate flows
    print("\n[Test 3] Testing intermediate flow outputs...")
    model.train()
    result_train = model(
        frame1, frame2, 
        num_iterations=4, 
        return_intermediate=True
    )
    
    intermediate = result_train['intermediate_flows']
    print(f"  Number of intermediate flows: {len(intermediate)}")
    for i, flow_i in enumerate(intermediate):
        print(f"    Iteration {i+1}: {flow_i.shape}")
    print("  ✓ Intermediate flows working")
    
    # Test gradient flow
    print("\n[Test 4] Testing gradient flow...")
    frame1_grad = torch.randn(1, 3, 224, 128, requires_grad=True, device=device)
    frame2_grad = torch.randn(1, 3, 224, 128, device=device)
    
    model_train = RAFTSEnetModel(config=opt_config).to(device)
    output = model_train(frame1_grad, frame2_grad, num_iterations=2)
    loss = output['flow'].sum()
    loss.backward()
    
    print(f"  Input gradient shape: {frame1_grad.grad.shape}")
    assert frame1_grad.grad is not None
    print("  ✓ Gradient flow verified")
    
    # Test different batch sizes
    print("\n[Test 5] Testing different batch sizes...")
    for bs in [1, 2, 4]:
        test_input1 = torch.randn(bs, 3, 224, 128).to(device)
        test_input2 = torch.randn(bs, 3, 224, 128).to(device)
        
        with torch.no_grad():
            test_output = model.get_flow(test_input1, test_input2)
        
        assert test_output.shape == (bs, 2, 224, 128)
        print(f"  Batch size {bs}: {test_output.shape}")
    print("  ✓ Different batch sizes handled")
    
    # Test different resolutions
    print("\n[Test 6] Testing different resolutions...")
    resolutions = [(224, 128), (448, 256), (384, 512)]
    for h, w in resolutions:
        test_input1 = torch.randn(1, 3, h, w).to(device)
        test_input2 = torch.randn(1, 3, h, w).to(device)
        
        with torch.no_grad():
            test_output = model.get_flow(test_input1, test_input2)
        
        assert test_output.shape == (1, 2, h, w)
        print(f"  Resolution {h}x{w}: {test_output.shape}")
    print("  ✓ Different resolutions handled")
    
    # Test wrapper class
    print("\n[Test 7] Testing RAFTWrapper...")
    wrapper = RAFTWrapper(config=opt_config, device=device)
    test_input1 = torch.randn(1, 3, 224, 128).to(device)
    test_input2 = torch.randn(1, 3, 224, 128).to(device)
    
    with torch.no_grad():
        wrapper_output = wrapper(test_input1, test_input2)
    
    print(f"  Wrapper output shape: {wrapper_output.shape}")
    print("  ✓ Wrapper class working")
    
    # Test factory function
    print("\n[Test 8] Testing factory function...")
    factory_model = create_raft_senet_model(config=opt_config, device=device)
    print(f"  Factory model type: {type(factory_model).__name__}")
    print("  ✓ Factory function working")
    
    # Test get_model registry
    print("\n[Test 9] Testing model registry...")
    registry_model = get_model("test_model", config=opt_config, device=device)
    print(f"  Registry model type: {type(registry_model).__name__}")
    print("  ✓ Model registry working")
    
    # Test with different number of iterations
    print("\n[Test 10] Testing different iteration counts...")
    for num_iters in [1, 4, 8, 12, 16]:
        test_input1 = torch.randn(1, 3, 224, 128).to(device)
        test_input2 = torch.randn(1, 3, 224, 128).to(device)
        
        with torch.no_grad():
            output = model.estimate_flow(test_input1, test_input2, num_iterations=num_iters)
        
        print(f"  Iterations={num_iters}: output shape {output.shape}")
    print("  ✓ Different iteration counts working")
    
    # Performance test
    print("\n[Test 11] Performance test...")
    import time
    
    test_input1 = torch.randn(1, 3, 448, 256).to(device)
    test_input2 = torch.randn(1, 3, 448, 256).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.get_flow(test_input1, test_input2)
    
    # Timed runs
    num_runs = 10
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model.get_flow(test_input1, test_input2)
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs * 1000
    
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Estimated FPS: {1000/avg_time:.1f}")
    print("  ✓ Performance test complete")
    
    print("\n" + "=" * 60)
    print("All RAFT-SEnet model tests passed!")
    print("=" * 60)
