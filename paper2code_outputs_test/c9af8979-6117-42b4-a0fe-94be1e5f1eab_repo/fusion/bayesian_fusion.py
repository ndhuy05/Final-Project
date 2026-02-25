"""
fusion/bayesian_fusion.py

Bayesian decision fusion module for conveyor belt speed detection.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements the Bayesian algorithm described in Section 3.3 of the paper
for dynamically combining speed estimates from optical flow and feature matching
methods based on image quality metrics (brightness and contrast).

Mathematical foundations from the paper:
- Raw weight calculation: w = aL + bC (Equation 8)
- Weight normalization: ω = w_raw / (w_optical_raw + w_feature_raw) (Equations 9-10)
- Final fusion: V_fusion = ω_optical × V_optical + ω_feature × V_feature (Equation in Section 3.3)

Author: Based on paper methodology
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass, field
import warnings

# Try to import configuration
try:
    from config import Config, get_config, FusionConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    get_config = None
    FusionConfig = None

# Try to import related modules
try:
    from fusion.image_quality import ImageQualityAnalyzer, ImageQualityMetrics
    IMAGE_QUALITY_AVAILABLE = True
except ImportError:
    IMAGE_QUALITY_AVAILABLE = False
    ImageQualityAnalyzer = None
    ImageQualityMetrics = None


@dataclass
class FusionResult:
    """Result container for Bayesian fusion output.
    
    This class stores the final fused speed estimate along with all
    relevant metadata from the fusion process.
    
    Based on Section 3.3 of the paper, the fusion combines estimates from
    optical flow and feature matching using Bayesian weight allocation.
    
    Attributes:
        fused_speed: Final fused speed in meters per second
        weight_optical: Normalized weight for optical flow method (0-1)
        weight_feature: Normalized weight for feature matching method (0-1)
        confidence_optical: Confidence score for optical flow estimate (0-1)
        confidence_feature: Confidence score for feature matching estimate (0-1)
        quality_metrics: Image quality metrics used in fusion (brightness, contrast)
        is_valid: Whether the fusion was successful
        error_message: Error message if fusion failed
    """
    fused_speed: float = 0.0
    weight_optical: float = 0.5
    weight_feature: float = 0.5
    confidence_optical: float = 0.0
    confidence_feature: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    is_valid: bool = False
    error_message: str = ""
    
    def __post_init__(self):
        """Validate and process results after initialization."""
        # Clip weights to [0, 1]
        self.weight_optical = max(0.0, min(1.0, float(self.weight_optical)))
        self.weight_feature = max(0.0, min(1.0, float(self.weight_feature)))
        
        # Clip confidences to [0, 1]
        self.confidence_optical = max(0.0, min(1.0, float(self.confidence_optical)))
        self.confidence_feature = max(0.0, min(1.0, float(self.confidence_feature)))
        
        # Ensure speed is non-negative
        self.fused_speed = max(0.0, float(self.fused_speed))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            'fused_speed': float(self.fused_speed),
            'weight_optical': float(self.weight_optical),
            'weight_feature': float(self.weight_feature),
            'confidence_optical': float(self.confidence_optical),
            'confidence_feature': float(self.confidence_feature),
            'quality_metrics': self.quality_metrics.copy() if self.quality_metrics else {},
            'is_valid': bool(self.is_valid),
            'error_message': str(self.error_message)
        }
    
    def get_method_speeds(
        self, 
        speed_optical: float, 
        speed_feature: float
    ) -> Dict[str, float]:
        """Get individual method speed estimates.
        
        Args:
            speed_optical: Speed from optical flow
            speed_feature: Speed from feature matching
            
        Returns:
            Dictionary with both estimates and weighted contribution
        """
        return {
            'optical_raw': speed_optical,
            'optical_contribution': self.weight_optical * speed_optical,
            'feature_raw': speed_feature,
            'feature_contribution': self.weight_feature * speed_feature
        }


class BayesianFusion:
    """Bayesian decision fusion for combining optical flow and feature matching speeds.
    
    This class implements the Bayesian weight allocation algorithm described in
    Section 3.3 of the paper. It dynamically adjusts weights based on:
    1. Image brightness (L) - affects feature matching reliability
    2. Image contrast (C) - affects both optical flow and feature matching
    3. Method-specific confidence metrics
    
    The fusion formula is:
        V_fusion = ω_optical × V_optical + ω_feature × V_feature
    
    where weights are normalized based on image quality and method confidence.
    
    Attributes:
        weight_a: Parameter for brightness weight (from config, default 0.5)
        weight_b: Parameter for contrast weight (from config, default 0.5)
        min_confidence_threshold: Minimum confidence to consider method valid
        enable_adaptive_fusion: Whether to use enhanced confidence-based fusion
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
        min_confidence_threshold: float = 0.1,
        enable_adaptive_fusion: bool = True,
        flow_confidence_weight: float = 0.5,
        feature_confidence_weight: float = 0.5
    ):
        """Initialize the Bayesian fusion module.
        
        Args:
            config: Configuration object. If None, uses provided parameter values.
            weight_a: Weight parameter a for brightness (Equation 8 from paper)
            weight_b: Weight parameter b for contrast (Equation 8 from paper)
            min_confidence_threshold: Minimum confidence threshold for valid estimates
            enable_adaptive_fusion: Whether to use enhanced confidence-based fusion
            flow_confidence_weight: Weight for flow consistency in confidence calculation
            feature_confidence_weight: Weight for inlier ratio in confidence calculation
        """
        # Get configuration if available
        if config is not None and CONFIG_AVAILABLE:
            try:
                fusion_config = config.fusion
                self.weight_a = fusion_config.weight_a
                self.weight_b = fusion_config.weight_b
            except:
                self.weight_a = weight_a
                self.weight_b = weight_b
        else:
            self.weight_a = weight_a
            self.weight_b = weight_b
        
        # Additional parameters
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_adaptive_fusion = enable_adaptive_fusion
        self.flow_confidence_weight = flow_confidence_weight
        self.feature_confidence_weight = feature_confidence_weight
        
        # Initialize image quality analyzer if available
        if IMAGE_QUALITY_AVAILABLE:
            self.quality_analyzer = ImageQualityAnalyzer()
        else:
            self.quality_analyzer = None
        
        # Statistics tracking
        self._stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'failed_low_confidence': 0,
            'failed_invalid_input': 0,
            'avg_weight_optical': 0.0,
            'avg_weight_feature': 0.0
        }
    
    @classmethod
    def from_config(
        cls,
        config: Optional[Config] = None
    ) -> 'BayesianFusion':
        """Create fusion module from configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            BayesianFusion instance with configured parameters
        """
        return cls(config=config)
    
    def _validate_inputs(
        self,
        speed_optical: float,
        speed_feature: float
    ) -> Tuple[bool, str]:
        """Validate input speed values.
        
        Args:
            speed_optical: Speed from optical flow method
            speed_feature: Speed from feature matching method
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for NaN or inf
        if np.isnan(speed_optical) or np.isinf(speed_optical):
            return False, f"Invalid optical flow speed: {speed_optical}"
        
        if np.isnan(speed_feature) or np.isinf(speed_feature):
            return False, f"Invalid feature matching speed: {speed_feature}"
        
        # Check for negative speeds
        if speed_optical < 0:
            return False, f"Negative optical flow speed: {speed_optical}"
        
        if speed_feature < 0:
            return False, f"Negative feature matching speed: {speed_feature}"
        
        return True, ""
    
    def calculate_raw_weights(
        self,
        brightness: float,
        contrast: float
    ) -> Tuple[float, float]:
        """Calculate raw weights from image quality metrics.
        
        Based on Equation 8 from the paper:
            w = aL + bC
        
        Args:
            brightness: Image brightness (L), range [0, 255]
            contrast: Image contrast (C), range [0, 255]
            
        Returns:
            Tuple of (w_optical_raw, w_feature_raw)
        """
        # Normalize brightness to [0, 1] range for better numerical stability
        brightness_normalized = brightness / 255.0
        
        # Normalize contrast to [0, 1] range (typical contrast range is 0-128)
        contrast_normalized = min(contrast / 128.0, 1.0)
        
        # Calculate raw weights using the formula from the paper
        w_optical_raw = self.weight_a * brightness_normalized + self.weight_b * contrast_normalized
        w_feature_raw = self.weight_a * brightness_normalized + self.weight_b * contrast_normalized
        
        return float(w_optical_raw), float(w_feature_raw)
    
    def calculate_flow_confidence(
        self,
        flow_magnitude: float,
        brightness: float,
        contrast: float,
        num_valid_pixels: int = 1
    ) -> float:
        """Calculate confidence score for optical flow estimation.
        
        The confidence is based on:
        1. Flow consistency - how uniform are the flow vectors
        2. Brightness suitability - penalize extreme brightness
        3. Contrast suitability - penalize low contrast
        
        Args:
            flow_magnitude: Average flow magnitude in pixels
            brightness: Image brightness (0-255)
            contrast: Image contrast (0-255)
            num_valid_pixels: Number of valid flow pixels
            
        Returns:
            Confidence score in [0, 1]
        """
        # Flow consistency factor (based on magnitude stability)
        # Higher flow magnitude with more valid pixels = higher confidence
        flow_factor = min(flow_magnitude / 50.0, 1.0)  # Normalize assuming max ~50px
        
        # Pixel coverage factor
        coverage_factor = min(num_valid_pixels / 1000.0, 1.0)  # Normalize assuming ~1000 pixels
        
        # Brightness suitability: penalize extreme brightness
        # Optimal brightness is around 127 (mid-gray)
        brightness_diff = abs(brightness - 127.5) / 127.5
        brightness_factor = max(0.0, 1.0 - brightness_diff)
        
        # Contrast suitability: penalize low contrast
        # Higher contrast is better for feature tracking
        contrast_factor = min(contrast / 50.0, 1.0)  # Normalize assuming max useful contrast ~50
        
        # Combine factors with weights
        confidence = (
            0.4 * flow_factor +
            0.3 * coverage_factor +
            0.15 * brightness_factor +
            0.15 * contrast_factor
        )
        
        return float(max(0.0, min(1.0, confidence)))
    def calculate_feature_confidence(
        self,
        num_inliers: int,
        total_matches: int,
        brightness: float,
        contrast: float,
        avg_match_distance: float = 10.0
    ) -> float:
        """Calculate confidence score for feature matching estimation.
        
        The confidence is based on:
        1. Inlier ratio - proportion of inlier matches after RANSAC
        2. Match quality - average descriptor distance
        3. Contrast suitability - penalize low contrast
        
        Args:
            num_inliers: Number of inlier matches after RANSAC
            total_matches: Total number of matches before filtering
            brightness: Image brightness (0-255)
            contrast: Image contrast (0-255)
            avg_match_distance: Average Hamming distance of matches
            
        Returns:
            Confidence score in [0, 1]
        """
        # Inlier ratio factor
        if total_matches > 0:
            inlier_ratio = num_inliers / total_matches
        else:
            inlier_ratio = 0.0
        
        # Match quality factor (lower distance = better quality)
        # Typical good matches have distance < 30 for binary descriptors
        match_quality = max(0.0, 1.0 - avg_match_distance / 50.0)
        
        # Brightness suitability
        brightness_diff = abs(brightness - 127.5) / 127.5
        brightness_factor = max(0.0, 1.0 - brightness_diff)
        
        # Contrast suitability (higher contrast = better feature detection)
        contrast_factor = min(contrast / 50.0, 1.0)
        
        # Combine factors with weights
        confidence = (
            0.5 * inlier_ratio +
            0.2 * match_quality +
            0.15 * brightness_factor +
            0.15 * contrast_factor
        )
        
        return float(max(0.0, min(1.0, confidence)))
    
    def _calculate_adaptive_weights(
        self,
        w_optical_raw: float,
        w_feature_raw: float,
        confidence_optical: float,
        confidence_feature: float
    ) -> Tuple[float, float]:
        """Calculate adaptive weights incorporating method confidence.
        
        This enhanced method multiplies raw weights by confidence scores
        to produce more intelligent weight allocation.
        
        Args:
            w_optical_raw: Raw weight for optical flow
            w_feature_raw: Raw weight for feature matching
            confidence_optical: Confidence in optical flow estimate
            confidence_feature: Confidence in feature matching estimate
            
        Returns:
            Tuple of (w_optical_adjusted, w_feature_adjusted)
        """
        # Adjust raw weights by method confidence
        w_optical_adj = w_optical_raw * (0.5 + 0.5 * confidence_optical)
        w_feature_adj = w_feature_raw * (0.5 + 0.5 * confidence_feature)
        
        return float(w_optical_adj), float(w_feature_adj)
    
    def _normalize_weights(
        self,
        w_optical: float,
        w_feature: float
    ) -> Tuple[float, float]:
        """Normalize weights to ensure they sum to 1.
        
        Based on Equations 9-10 from the paper:
            ω_optical = w_optical_raw / (w_optical_raw + w_feature_raw)
            ω_feature = w_feature_raw / (w_optical_raw + w_feature_raw)
        
        Args:
            w_optical: Raw weight for optical flow
            w_feature: Raw weight for feature matching
            
        Returns:
            Tuple of (ω_optical, ω_feature)
        """
        total = w_optical + w_feature
        
        # Handle edge case where both weights are zero
        if total < 1e-10:
            return 0.5, 0.5
        
        # Normalize
        omega_optical = w_optical / total
        omega_feature = w_feature / total
        
        return float(omega_optical), float(omega_feature)
    
    def _handle_extreme_asymmetry(
        self,
        omega_optical: float,
        omega_feature: float,
        confidence_optical: float,
        confidence_feature: float
    ) -> Tuple[float, float]:
        """Handle cases where one method has significantly higher confidence.
        
        If one method has much higher confidence than the other (>70% difference),
        assign more weight to the higher confidence method.
        
        Args:
            omega_optical: Current normalized weight for optical flow
            omega_feature: Current normalized weight for feature matching
            confidence_optical: Optical flow confidence
            confidence_feature: Feature matching confidence
            
        Returns:
            Tuple of adjusted (ω_optical, ω_feature)
        """
        confidence_diff = confidence_optical - confidence_feature
        
        # If one method has much higher confidence
        if abs(confidence_diff) > 0.4:
            if confidence_diff > 0:
                # Optical flow has higher confidence - favor it more
                omega_optical = min(0.95, omega_optical + 0.2)
                omega_feature = 1.0 - omega_optical
            else:
                # Feature matching has higher confidence - favor it more
                omega_feature = min(0.95, omega_feature + 0.2)
                omega_optical = 1.0 - omega_feature
        
        return float(omega_optical), float(omega_feature)
    
    def _fuse_speeds(
        self,
        speed_optical: float,
        speed_feature: float,
        omega_optical: float,
        omega_feature: float,
        confidence_optical: float,
        confidence_feature: float
    ) -> float:
        """Fuse speed estimates using weighted combination.
        
        Based on the paper's fusion equation:
            V_fusion = ω_optical × V_optical + ω_feature × V_feature
        
        Args:
            speed_optical: Speed estimate from optical flow (m/s)
            speed_feature: Speed estimate from feature matching (m/s)
            omega_optical: Normalized weight for optical flow
            omega_feature: Normalized weight for feature matching
            confidence_optical: Optical flow confidence
            confidence_feature: Feature matching confidence
            
        Returns:
            Fused speed in m/s
        """
        # Handle case where both estimates are zero
        if speed_optical <= 0 and speed_feature <= 0:
            return 0.0
        
        # Handle case where only one estimate is valid
        if speed_optical <= 0:
            return speed_feature
        if speed_feature <= 0:
            return speed_optical
        
        # Check for large discrepancy between estimates
        # If they differ by >50%, weight towards more confident method
        avg_speed = (speed_optical + speed_feature) / 2
        if avg_speed > 0:
            relative_diff = abs(speed_optical - speed_feature) / avg_speed
            
            if relative_diff > 0.5:
                # Large discrepancy - use confidence-weighted combination
                # More weight to the more consistent estimate
                if confidence_optical > confidence_feature:
                    # Favor optical flow more
                    adjusted_omega = 0.7
                else:
                    # Favor feature matching more
                    adjusted_omega = 0.3
                
                fused = adjusted_omega * speed_optical + (1 - adjusted_omega) * speed_feature
            else:
                # Normal case - use calculated weights
                fused = omega_optical * speed_optical + omega_feature * speed_feature
        else:
            # Average is zero, use equal weights
            fused = omega_optical * speed_optical + omega_feature * speed_feature
        
        return float(fused)
    
    def _get_image_quality_metrics(
        self,
        image: np.ndarray
    ) -> Tuple[float, float]:
        """Extract brightness and contrast from image.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (brightness, contrast)
        """
        if self.quality_analyzer is not None:
            try:
                quality_result = self.quality_analyzer.analyze(image)
                return float(quality_result.brightness), float(quality_result.contrast)
            except Exception as e:
                warnings.warn(f"Image quality analysis failed: {e}")
        
        # Fallback: simple calculation using numpy
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # BGR to grayscale
                    gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
                else:
                    gray = image.squeeze()
            else:
                gray = image
            
            # Calculate brightness (mean)
            brightness = np.mean(gray)
            
            # Calculate contrast (std)
            contrast = np.std(gray)
            
            return float(brightness), float(contrast)
        except Exception:
            return 127.5, 50.0  # Default values
    
    def fuse(
        self,
        image: np.ndarray,
        speed_optical: float,
        speed_feature: float,
        optical_flow_result: Optional[Dict[str, Any]] = None,
        feature_match_result: Optional[Dict[str, Any]] = None
    ) -> FusionResult:
        """Fuse speed estimates from optical flow and feature matching.
        
        This is the main method of the Bayesian fusion module. It:
        1. Extracts image quality metrics (brightness, contrast)
        2. Calculates raw weights from image quality
        3. Calculates method-specific confidence scores
        4. Adjusts weights based on confidence (if adaptive fusion enabled)
        5. Normalizes weights
        6. Fuses speed estimates
        
        Args:
            image: Input image frame (used for quality assessment)
            speed_optical: Speed estimate from optical flow method (m/s)
            speed_feature: Speed estimate from feature matching method (m/s)
            optical_flow_result: Optional result dict with additional flow metrics
            feature_match_result: Optional result dict with additional match metrics
            
        Returns:
            FusionResult containing fused speed and metadata
        """
        # Update statistics
        self._stats['total_fusions'] += 1
        
        # Validate inputs
        is_valid, error_msg = self._validate_inputs(speed_optical, speed_feature)
        if not is_valid:
            self._stats['failed_fusions'] += 1
            self._stats['failed_invalid_input'] += 1
            return FusionResult(
                is_valid=False,
                error_message=error_msg
            )
        
        try:
            # Step 1: Extract image quality metrics
            brightness, contrast = self._get_image_quality_metrics(image)
            quality_metrics = {
                'brightness': brightness,
                'contrast': contrast,
                'brightness_normalized': brightness / 255.0,
                'contrast_normalized': min(contrast / 128.0, 1.0)
            }
            
            # Step 2: Calculate raw weights from image quality
            w_optical_raw, w_feature_raw = self.calculate_raw_weights(
                brightness, contrast
            )
            
            # Step 3: Calculate method-specific confidence scores
            # Default confidence values
            conf_optical = 0.5
            conf_feature = 0.5
            
            if self.enable_adaptive_fusion:
                # Extract additional metrics if available
                flow_magnitude = 0.0
                num_valid_pixels = 1
                num_inliers = 0
                total_matches = 0
                avg_match_distance = 10.0
                
                if optical_flow_result is not None:
                    flow_magnitude = optical_flow_result.get('avg_flow_magnitude', 0.0)
                    num_valid_pixels = optical_flow_result.get('valid_pixel_ratio', 1.0)
                
                if feature_match_result is not None:
                    num_inliers = feature_match_result.get('num_inliers', 0)
                    total_matches = feature_match_result.get('num_matches', 1)
                
                conf_optical = self.calculate_flow_confidence(
                    flow_magnitude=flow_magnitude,
                    brightness=brightness,
                    contrast=contrast,
                    num_valid_pixels=num_valid_pixels
                )
                
                conf_feature = self.calculate_feature_confidence(
                    num_inliers=num_inliers,
                    total_matches=total_matches,
                    brightness=brightness,
                    contrast=contrast,
                    avg_match_distance=avg_match_distance
                )
            
            # Step 4: Calculate adaptive weights
            if self.enable_adaptive_fusion:
                w_optical_adj, w_feature_adj = self._calculate_adaptive_weights(
                    w_optical_raw,
                    w_feature_raw,
                    conf_optical,
                    conf_feature
                )
            else:
                w_optical_adj = w_optical_raw
                w_feature_adj = w_feature_raw
            
            # Step 5: Normalize weights
            omega_optical, omega_feature = self._normalize_weights(
                w_optical_adj, w_feature_adj
            )
            
            # Step 6: Handle extreme asymmetry
            if self.enable_adaptive_fusion:
                omega_optical, omega_feature = self._handle_extreme_asymmetry(
                    omega_optical,
                    omega_feature,
                    conf_optical,
                    conf_feature
                )
            
            # Step 7: Fuse speed estimates
            fused_speed = self._fuse_speeds(
                speed_optical,
                speed_feature,
                omega_optical,
                omega_feature,
                conf_optical,
                conf_feature
            )
            
            # Check confidence thresholds
            if conf_optical < self.min_confidence_threshold and conf_feature < self.min_confidence_threshold:
                self._stats['failed_low_confidence'] += 1
                # Still return result but mark as low confidence
            
            # Update statistics
            self._stats['successful_fusions'] += 1
            n = self._stats['successful_fusions']
            self._stats['avg_weight_optical'] = (
                (self._stats['avg_weight_optical'] * (n - 1) + omega_optical) / n
            )
            self._stats['avg_weight_feature'] = (
                (self._stats['avg_weight_feature'] * (n - 1) + omega_feature) / n
            )
            
            # Create result
            return FusionResult(
                fused_speed=fused_speed,
                weight_optical=omega_optical,
                weight_feature=omega_feature,
                confidence_optical=conf_optical,
                confidence_feature=conf_feature,
                quality_metrics=quality_metrics,
                is_valid=True,
                error_message=""
            )
            
        except Exception as e:
            self._stats['failed_fusions'] += 1
            return FusionResult(
                is_valid=False,
                error_message=f"Fusion failed: {str(e)}"
            )
    
    def fuse_with_results(
        self,
        image: np.ndarray,
        optical_result: Optional[Any] = None,
        feature_result: Optional[Any] = None
    ) -> FusionResult:
        """Fuse using result objects instead of raw speed values.
        
        This method accepts full result objects from the optical flow
        and feature matching pipelines and extracts necessary metrics.
        
        Args:
            image: Input image
            optical_result: OpticalFlowResult or similar object
            feature_result: FeatureMatchResult or similar object
            
        Returns:
            FusionResult with fused speed
        """
        # Extract speeds from result objects
        speed_optical = 0.0
        speed_feature = 0.0
        
        optical_dict = None
        feature_dict = None
        
        # Extract optical flow speed
        if optical_result is not None:
            if hasattr(optical_result, 'speed_m_per_sec'):
                speed_optical = optical_result.speed_m_per_sec
            elif hasattr(optical_result, 'speed_px_per_sec'):
                speed_optical = optical_result.speed_px_per_sec
            
            # Extract additional metrics
            if hasattr(optical_result, 'to_dict'):
                optical_dict = optical_result.to_dict()
            elif isinstance(optical_result, dict):
                optical_dict = optical_result
                
            if hasattr(optical_result, 'avg_flow_magnitude'):
                if optical_dict is None:
                    optical_dict = {}
                optical_dict['avg_flow_magnitude'] = optical_result.avg_flow_magnitude
            if hasattr(optical_result, 'valid_pixel_ratio'):
                if optical_dict is None:
                    optical_dict = {}
                optical_dict['valid_pixel_ratio'] = optical_result.valid_pixel_ratio
        
        # Extract feature matching speed
        if feature_result is not None:
            if hasattr(feature_result, 'speed_m_per_sec'):
                speed_feature = feature_result.speed_m_per_sec
            elif hasattr(feature_result, 'speed_px_per_sec'):
                speed_feature = feature_result.speed_px_per_sec
            
            # Extract additional metrics
            if hasattr(feature_result, 'to_dict'):
                feature_dict = feature_result.to_dict()
            elif isinstance(feature_result, dict):
                feature_dict = feature_result
                
            if hasattr(feature_result, 'num_inliers'):
                if feature_dict is None:
                    feature_dict = {}
                feature_dict['num_inliers'] = feature_result.num_inliers
            if hasattr(feature_result, 'num_inliers'):
                if feature_dict is None:
                    feature_dict = {}
                feature_dict['num_matches'] = getattr(feature_result, 'num_inliers', 1)
        
        # Call main fuse method
        return self.fuse(
            image=image,
            speed_optical=speed_optical,
            speed_feature=speed_feature,
            optical_flow_result=optical_dict,
            feature_match_result=feature_dict
        )
    
    def set_weight_parameters(
        self,
        weight_a: Optional[float] = None,
        weight_b: Optional[float] = None
    ) -> None:
        """Update weight parameters for brightness and contrast.
        
        Args:
            weight_a: New brightness weight parameter
            weight_b: New contrast weight parameter
        """
        if weight_a is not None and 0 <= weight_a <= 1:
            self.weight_a = weight_a
        if weight_b is not None and 0 <= weight_b <= 1:
            self.weight_b = weight_b
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fusion statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self._stats.copy()
        
        # Add derived statistics
        if self._stats['total_fusions'] > 0:
            stats['success_rate'] = self._stats['successful_fusions'] / self._stats['total_fusions']
            stats['failure_rate'] = self._stats['failed_fusions'] / self._stats['total_fusions']
            stats['low_confidence_rate'] = self._stats['failed_low_confidence'] / self._stats['total_fusions']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['low_confidence_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset fusion statistics."""
        self._stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'failed_fusions': 0,
            'failed_low_confidence': 0,
            'failed_invalid_input': 0,
            'avg_weight_optical': 0.0,
            'avg_weight_feature': 0.0
        }
    
    def __repr__(self) -> str:
        """String representation of the fusion module."""
        return (
            f"BayesianFusion(\n"
            f"  weight_a={self.weight_a},\n"
            f"  weight_b={self.weight_b},\n"
            f"  min_confidence_threshold={self.min_confidence_threshold},\n"
            f"  enable_adaptive_fusion={self.enable_adaptive_fusion},\n"
            f"  flow_confidence_weight={self.flow_confidence_weight},\n"
            f"  feature_confidence_weight={self.feature_confidence_weight}\n"
            f")"
        )


def create_bayesian_fusion(
    config: Optional[Config] = None,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    enable_adaptive_fusion: bool = True
) -> BayesianFusion:
    """Factory function to create a BayesianFusion instance.
    
    Args:
        config: Configuration object (takes priority if provided)
        weight_a: Weight parameter for brightness (used if config is None)
        weight_b: Weight parameter for contrast (used if config is None)
        enable_adaptive_fusion: Whether to use enhanced confidence-based fusion
        
    Returns:
        BayesianFusion instance
    """
    return BayesianFusion(
        config=config,
        weight_a=weight_a,
        weight_b=weight_b,
        enable_adaptive_fusion=enable_adaptive_fusion
    )


# Global fusion module registry
_fusion_modules: Dict[str, BayesianFusion] = {}


def get_bayesian_fusion(
    name: str = "default",
    config: Optional[Config] = None,
    **kwargs
) -> BayesianFusion:
    """Get or create a BayesianFusion module by name.
    
    Args:
        name: Module identifier
        config: Configuration object
        **kwargs: Additional arguments for BayesianFusion
        
    Returns:
        BayesianFusion instance
    """
    global _fusion_modules
    
    if name in _fusion_modules:
        return _fusion_modules[name]
    
    fusion_module = create_bayesian_fusion(config=config, **kwargs)
    _fusion_modules[name] = fusion_module
    
    return fusion_module


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Bayesian Fusion Module Test")
    print("=" * 60)
    
    # Test 1: Create fusion module with defaults
    print("\n[Test 1] Creating fusion module with defaults...")
    fusion = BayesianFusion()
    print(f"  Created: {repr(fusion)}")
    print("  ✓ Fusion module created")
    
    # Test 2: Create fusion module from config
    print("\n[Test 2] Creating fusion module from configuration...")
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            fusion_config = BayesianFusion.from_config(config)
            print(f"  Created from config")
            print(f"  Weight a: {fusion_config.weight_a}")
            print(f"  Weight b: {fusion_config.weight_b}")
            print("  ✓ Config integration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (config not available)")
    
    # Test 3: Create synthetic test image
    print("\n[Test 3] Creating synthetic test image...")
    test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    print(f"  Test image shape: {test_image.shape}")
    print("  ✓ Test image created")
    
    # Test 4: Test raw weight calculation
    print("\n[Test 4] Testing raw weight calculation...")
    w_optical, w_feature = fusion.calculate_raw_weights(brightness=127.5, contrast=50.0)
    print(f"  L=127.5, C=50.0: w_optical={w_optical:.4f}, w_feature={w_feature:.4f}")
    assert abs(w_optical - w_feature) < 1e-6, "Raw weights should be equal"
    
    w_optical2, w_feature2 = fusion.calculate_raw_weights(brightness=255.0, contrast=100.0)
    print(f"  L=255, C=100: w_optical={w_optical2:.4f}, w_feature={w_feature2:.4f}")
    assert w_optical2 > w_optical, "Higher quality should give higher weight"
    print("  ✓ Raw weight calculation works")
    
    # Test 5: Test weight normalization
    print("\n[Test 5] Testing weight normalization...")
    omega_optical, omega_feature = fusion._normalize_weights(0.3, 0.3)
    print(f"  Equal weights (0.3, 0.3): omega_optical={omega_optical:.4f}, omega_feature={omega_feature:.4f}")
    assert abs(omega_optical - 0.5) < 1e-6 and abs(omega_feature - 0.5) < 1e-6
    
    omega_optical2, omega_feature2 = fusion._normalize_weights(0.8, 0.2)
    print(f"  Unequal weights (0.8, 0.2): omega_optical={omega_optical2:.4f}, omega_feature={omega_feature2:.4f}")
    assert abs(omega_optical2 - 0.8) < 1e-6 and abs(omega_feature2 - 0.2) < 1e-6
    print("  ✓ Normalization works")
    
    # Test 6: Test flow confidence calculation
    print("\n[Test 6] Testing flow confidence calculation...")
    conf1 = fusion.calculate_flow_confidence(flow_magnitude=5.0, brightness=127.5, contrast=50.0, num_valid_pixels=1000)
    print(f"  Normal conditions: confidence={conf1:.4f}")
    
    conf2 = fusion.calculate_flow_confidence(flow_magnitude=1.0, brightness=10.0, contrast=10.0, num_valid_pixels=100)
    print(f"  Poor conditions: confidence={conf2:.4f}")
    assert conf1 > conf2, "Better conditions should give higher confidence"
    print("  ✓ Flow confidence works")
    
    # Test 7: Test feature confidence calculation
    print("\n[Test 7] Testing feature confidence calculation...")
    conf3 = fusion.calculate_feature_confidence(num_inliers=50, total_matches=100, brightness=127.5, contrast=50.0)
    print(f"  Good matches: confidence={conf3:.4f}")
    
    conf4 = fusion.calculate_feature_confidence(num_inliers=5, total_matches=100, brightness=10.0, contrast=10.0)
    print(f"  Poor matches: confidence={conf4:.4f}")
    assert conf3 > conf4, "Better matches should give higher confidence"
    print("  ✓ Feature confidence works")
    
    # Test 8: Test main fuse method
    print("\n[Test 8] Testing main fuse method...")
    result = fusion.fuse(
        image=test_image,
        speed_optical=1.5,
        speed_feature=1.2
    )
    print(f"  Fused speed: {result.fused_speed:.4f} m/s")
    print(f"  Weights: optical={result.weight_optical:.4f}, feature={result.weight_feature:.4f}")
    print(f"  Confidence: optical={result.confidence_optical:.4f}, feature={result.confidence_feature:.4f}")
    print(f"  Is valid: {result.is_valid}")
    print("  ✓ Main fuse works")
    
    # Test 9: Test input validation
    print("\n[Test 9] Testing input validation...")
    result_invalid = fusion.fuse(
        image=test_image,
        speed_optical=-1.0,  # Invalid negative speed
        speed_feature=1.0
    )
    print(f"  Negative optical speed: valid={result_invalid.is_valid}, error={result_invalid.error_message}")
    assert not result_invalid.is_valid
    
    result_nan = fusion.fuse(
        image=test_image,
        speed_optical=float('nan'),
        speed_feature=1.0
    )
    print(f"  NaN optical speed: valid={result_nan.is_valid}")
    assert not result_nan.is_valid
    print("  ✓ Input validation works")
    
    # Test 10: Test extreme asymmetry handling
    print("\n[Test 10] Testing extreme asymmetry handling...")
    omega_opt, omega_feat = fusion._handle_extreme_asymmetry(
        0.5, 0.5, 0.9, 0.1  # High confidence difference
    )
    print(f"  High asymmetry (0.9 vs 0.1): omega_optical={omega_opt:.4f}, omega_feature={omega_feat:.4f}")
    assert omega_opt > 0.5, "Should favor high confidence method"
    print("  ✓ Extreme asymmetry handling works")
    
    # Test 11: Test speed fusion with large discrepancy
    print("\n[Test 11] Testing speed fusion with large discrepancy...")
    fused = fusion._fuse_speeds(
        speed_optical=2.0,
        speed_feature=1.0,
        omega_optical=0.5,
        omega_feature=0.5,
        confidence_optical=0.9,
        confidence_feature=0.1
    )
    print(f"  Large discrepancy (2.0 vs 1.0): fused={fused:.4f}")
    # Should weight more towards optical flow (higher confidence)
    assert 1.0 < fused < 2.0, "Fused speed should be between inputs"
    print("  ✓ Speed fusion handles discrepancy")
    
    # Test 12: Test statistics tracking
    print("\n[Test 12] Testing statistics tracking...")
    stats = fusion.get_statistics()
    print(f"  Statistics: {stats}")
    print(f"  Total fusions: {stats['total_fusions']}")
    print(f"  Successful: {stats['successful_fusions']}")
    fusion.reset_statistics()
    stats_after = fusion.get_statistics()
    print(f"  After reset: {stats_after['total_fusions']}")
    assert stats_after['total_fusions'] == 0
    print("  ✓ Statistics work")
    
    # Test 13: Test with different brightness/contrast
    print("\n[Test 13] Testing with different brightness/contrast...")
    # Bright image
    bright_image = np.full((100, 100, 3), 240, dtype=np.uint8)
    result_bright = fusion.fuse(bright_image, 1.5, 1.2)
    print(f"  Bright image: brightness={result_bright.quality_metrics.get('brightness', 'N/A')}, weights=({result_bright.weight_optical:.2f}, {result_bright.weight_feature:.2f})")
    
    # Dark image
    dark_image = np.full((100, 100, 3), 20, dtype=np.uint8)
    result_dark = fusion.fuse(dark_image, 1.5, 1.2)
    print(f"  Dark image: brightness={result_dark.quality_metrics.get('brightness', 'N/A')}, weights=({result_dark.weight_optical:.2f}, {result_dark.weight_feature:.2f})")
    
    # High contrast
    contrast_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    result_contrast = fusion.fuse(contrast_image, 1.5, 1.2)
    print(f"  High contrast: contrast={result_contrast.quality_metrics.get('contrast', 'N/A'):.2f}, weights=({result_contrast.weight_optical:.2f}, {result_contrast.weight_feature:.2f})")
    print("  ✓ Different conditions handled")
    
    # Test 14: Test factory function
    print("\n[Test 14] Testing factory function...")
    factory_fusion = create_bayesian_fusion()
    print(f"  Factory type: {type(factory_fusion).__name__}")
    print("  ✓ Factory function works")
    
    # Test 15: Test registry
    print("\n[Test 15] Testing registry...")
    reg_fusion = get_bayesian_fusion("test_reg")
    print(f"  Registry type: {type(reg_fusion).__name__}")
    print("  ✓ Registry works")
    
    # Test 16: Test to_dict conversion
    print("\n[Test 16] Testing result to_dict...")
    result = fusion.fuse(test_image, 1.5, 1.2)
    result_dict = result.to_dict()
    print(f"  Keys: {list(result_dict.keys())}")
    assert 'fused_speed' in result_dict
    assert 'weight_optical' in result_dict
    print("  ✓ to_dict works")
    
    # Test 17: Test weight parameter setting
    print("\n[Test 17] Testing weight parameter setting...")
    fusion.set_weight_parameters(weight_a=0.3, weight_b=0.7)
    print(f"  Set weight_a=0.3, weight_b=0.7")
    print(f"  Current weight_a: {fusion.weight_a}")
    print(f"  Current weight_b: {fusion.weight_b}")
    assert fusion.weight_a == 0.3
    assert fusion.weight_b == 0.7
    print("  ✓ Weight setting works")
    
    # Test 18: Test get_method_speeds
    print("\n[Test 18] Testing get_method_speeds...")
    method_speeds = result.get_method_speeds(1.5, 1.2)
    print(f"  Method speeds: {method_speeds}")
    assert 'optical_raw' in method_speeds
    assert 'feature_contribution' in method_speeds
    print("  ✓ get_method_speeds works")
    
    # Test 19: Test adaptive vs non-adaptive fusion
    print("\n[Test 19] Testing adaptive vs non-adaptive fusion...")
    fusion_non_adaptive = BayesianFusion(enable_adaptive_fusion=False)
    result_non_adaptive = fusion_non_adaptive.fuse(test_image, 1.5, 1.2)
    
    fusion_adaptive = BayesianFusion(enable_adaptive_fusion=True)
    result_adaptive = fusion_adaptive.fuse(test_image, 1.5, 1.2)
    
    print(f"  Non-adaptive: weights=({result_non_adaptive.weight_optical:.4f}, {result_non_adaptive.weight_feature:.4f})")
    print(f"  Adaptive: weights=({result_adaptive.weight_optical:.4f}, {result_adaptive.weight_feature:.4f})")
    print("  ✓ Adaptive comparison works")
    
    # Test 20: Test config integration
    print("\n[Test 20] Testing config integration...")
    if CONFIG_AVAILABLE:
        try:
            cfg = get_config()
            print(f"  Config available: Yes")
            print(f"  Fusion weight_a: {cfg.fusion.weight_a}")
            print(f"  Fusion weight_b: {cfg.fusion.weight_b}")
            print("  ✓ Config integration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (config not available)")
    
    print("\n" + "=" * 60)
    print("All Bayesian Fusion tests passed!")
    print("=" * 60)
