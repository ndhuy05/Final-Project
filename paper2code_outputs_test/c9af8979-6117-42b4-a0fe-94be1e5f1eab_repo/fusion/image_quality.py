"""
fusion/image_quality.py

Image quality assessment module for the conveyor belt speed detection system.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements image quality assessment for the Bayesian decision fusion
process. According to Section 3.3 of the paper, the weight calculation depends on
image brightness (L) and contrast (C) to dynamically allocate weights between
optical flow and feature matching methods.

Mathematical foundation from the paper:
- Brightness: L = (1/(W*H)) * sum(I(i,j))
- Contrast: C = sqrt((1/(W*H)) * sum((I(i,j) - L)^2))
- Weight: w = aL + bC (computed in bayesian_fusion.py)

Author: Based on paper methodology
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

# Try to import OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    # Will use numpy fallback

# Try to import configuration
try:
    from config import Config, get_config, FusionConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    get_config = None
    FusionConfig = None


@dataclass
class ImageQualityMetrics:
    """Container for image quality assessment results.
    
    This class stores the computed quality metrics that are used by the
    Bayesian fusion module for weight calculation.
    
    Attributes:
        brightness: Mean pixel intensity (L value from paper), range [0, 255]
        contrast: Standard deviation of pixel intensities (C value from paper), range [0, 255]
        histogram: Optional histogram of pixel intensities for detailed analysis
        is_valid: Whether the image was valid for analysis
        error_message: Error message if analysis failed
    """
    brightness: float = 0.0
    contrast: float = 0.0
    histogram: Optional[np.ndarray] = None
    is_valid: bool = False
    error_message: str = ""
    
    def __post_init__(self):
        """Validate and process results after initialization."""
        # Ensure values are within valid ranges
        self.brightness = max(0.0, min(255.0, float(self.brightness)))
        self.contrast = max(0.0, min(255.0, float(self.contrast)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization.
        
        Returns:
            Dictionary representation of metrics
        """
        return {
            'brightness': float(self.brightness),
            'contrast': float(self.contrast),
            'histogram': self.histogram.tolist() if self.histogram is not None else None,
            'is_valid': bool(self.is_valid),
            'error_message': str(self.error_message)
        }
    
    def get_quality_score(self) -> float:
        """Get a combined quality score for debugging/analysis.
        
        Returns:
            Quality score based on brightness and contrast balance
        """
        # Optimal brightness range is roughly 80-180 for good feature detection
        # Optimal contrast is > 20 for meaningful feature extraction
        brightness_score = 1.0 - abs(self.brightness - 127.5) / 127.5
        contrast_score = min(self.contrast / 50.0, 1.0)  # Higher contrast is better
        
        # Combined score (weighted average)
        return 0.4 * brightness_score + 0.6 * contrast_score
    
    def is_overexposed(self) -> bool:
        """Check if image is overexposed (too bright).
        
        Returns:
            True if brightness > 220
        """
        return self.brightness > 220.0
    
    def is_underexposed(self) -> bool:
        """Check if image is underexposed (too dark).
        
        Returns:
            True if brightness < 35
        """
        return self.brightness < 35.0
    
    def is_low_contrast(self) -> bool:
        """Check if image has low contrast (poor feature visibility).
        
        Returns:
            True if contrast < 15
        """
        return self.contrast < 15.0


class ImageQualityAnalyzer:
    """Image quality analyzer for Bayesian fusion weight calculation.
    
    This class computes image quality metrics (brightness and contrast)
    that are used by the Bayesian fusion module to dynamically allocate
    weights between optical flow and feature matching methods.
    
    Based on Section 3.3 of the paper where the weight calculation formula
    w = aL + bC uses brightness (L) and contrast (C) values.
    
    Attributes:
        roi: Optional region of interest as (x, y, width, height)
        enable_histogram: Whether to compute histogram for detailed analysis
        validate_image: Whether to validate input images
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        enable_histogram: bool = False,
        validate_image: bool = True
    ):
        """Initialize the image quality analyzer.
        
        Args:
            config: Configuration object (optional, for future use)
            roi: Region of interest as (x, y, width, height) for focused analysis
            enable_histogram: Whether to compute pixel intensity histogram
            validate_image: Whether to validate input images before processing
        """
        self.config = config
        self.roi = roi
        self.enable_histogram = enable_histogram
        self.validate_image = validate_image
        
        # Statistics tracking
        self._stats = {
            'total_images': 0,
            'valid_images': 0,
            'overexposed_count': 0,
            'underexposed_count': 0,
            'low_contrast_count': 0,
            'failed_count': 0,
            'avg_brightness': 0.0,
            'avg_contrast': 0.0
        }
    
    @classmethod
    def from_config(
        cls,
        config: Optional[Config] = None
    ) -> 'ImageQualityAnalyzer':
        """Create analyzer from configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            ImageQualityAnalyzer instance
        """
        return cls(config=config)
    
    def _validate_image(
        self,
        image: np.ndarray
    ) -> Tuple[bool, str]:
        """Validate input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None:
            return False, "Image is None"
        
        if not isinstance(image, np.ndarray):
            return False, f"Image is not a numpy array, got {type(image)}"
        
        if image.size == 0:
            return False, "Image is empty"
        
        # Check dimensions
        if len(image.shape) < 2:
            return False, f"Image has invalid dimensions: {image.shape}"
        
        if len(image.shape) == 2:
            # Grayscale image - OK
            pass
        elif len(image.shape) == 3:
            # Color image - check channel count
            if image.shape[2] not in [1, 3, 4]:
                return False, f"Invalid number of channels: {image.shape[2]}"
        else:
            return False, f"Invalid image dimensions: {image.shape}"
        
        return True, ""
    
    def _convert_to_grayscale(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Convert color image to grayscale.
        
        Uses weighted average (ITU-R BT.601) for conversion:
        Gray = 0.299*R + 0.587*G + 0.114*B
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            Grayscale image as 2D numpy array
        """
        # Already grayscale
        if len(image.shape) == 2:
            return image
        
        # Handle different channel counts
        if image.shape[2] == 1:
            # Single channel - just squeeze
            return image.squeeze()
        
        if image.shape[2] == 3:
            # BGR color image
            if HAS_CV2:
                # Use OpenCV for efficiency
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    return gray
                except Exception:
                    pass
            
            # Fallback: manual conversion
            # Assume RGB format for numpy array
            r = image[:, :, 0].astype(np.float32)
            g = image[:, :, 1].astype(np.float32)
            b = image[:, :, 2].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.astype(np.uint8)
        
        if image.shape[2] == 4:
            # BGRA or RGBA - take first 3 channels
            if HAS_CV2:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                    return gray
                except Exception:
                    pass
            
            # Fallback
            r = image[:, :, 0].astype(np.float32)
            g = image[:, :, 1].astype(np.float32)
            b = image[:, :, 2].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return gray.astype(np.uint8)
        
        # Should not reach here, but handle gracefully
        return image.squeeze()
    
    def _extract_roi(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract region of interest from image.
        
        Args:
            image: Input image
            roi: Region of interest as (x, y, width, height)
            
        Returns:
            Extracted ROI image (or full image if ROI is invalid)
        """
        if roi is None:
            return image
        
        x, y, w, h = roi
        height, width = image.shape[:2]
        
        # Clamp ROI to image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        # Ensure valid dimensions
        if w <= 0 or h <= 0:
            return image
        
        return image[y:y+h, x:x+w]
    
    def _calculate_brightness(
        self,
        gray_image: np.ndarray
    ) -> float:
        """Calculate image brightness (mean pixel intensity).
        
        Based on paper Equation 8: L = (1/(W*H)) * sum(I(i,j))
        This is simply the arithmetic mean of all pixel values.
        
        Args:
            gray_image: Grayscale image as 2D numpy array
            
        Returns:
            Brightness value in range [0, 255]
        """
        # Use numpy for efficient computation
        brightness = np.mean(gray_image)
        
        return float(brightness)
    
    def _calculate_contrast(
        self,
        gray_image: np.ndarray,
        brightness: Optional[float] = None
    ) -> float:
        """Calculate image contrast (standard deviation of pixel intensities).
        
        Based on paper Equation 8: C = sqrt((1/(W*H)) * sum((I(i,j) - L)^2))
        This is the population standard deviation.
        
        Args:
            gray_image: Grayscale image as 2D numpy array
            brightness: Pre-computed brightness value (optional, for efficiency)
            
        Returns:
            Contrast value in range [0, 255]
        """
        # Use pre-computed brightness if available
        if brightness is None:
            brightness = np.mean(gray_image)
        
        # Calculate standard deviation (population, not sample)
        contrast = np.std(gray_image)
        
        return float(contrast)
    
    def _calculate_histogram(
        self,
        gray_image: np.ndarray,
        bins: int = 256
    ) -> np.ndarray:
        """Calculate histogram of pixel intensities.
        
        Args:
            gray_image: Grayscale image
            bins: Number of histogram bins
            
        Returns:
            Histogram array of shape (bins,)
        """
        # Calculate histogram
        hist, _ = np.histogram(gray_image.flatten(), bins=bins, range=(0, 256))
        return hist.astype(np.int32)
    
    def analyze(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> ImageQualityMetrics:
        """Analyze image quality and return metrics.
        
        This is the main method of the analyzer. It computes brightness (L)
        and contrast (C) values that are used by the Bayesian fusion module
        for weight calculation.
        
        Args:
            image: Input image as numpy array (H×W×C or H×W)
            roi: Optional region of interest (x, y, width, height)
            
        Returns:
            ImageQualityMetrics containing brightness, contrast, and analysis results
        """
        # Use instance ROI if not provided
        if roi is None:
            roi = self.roi
        
        # Update statistics
        self._stats['total_images'] += 1
        
        # Validate image if enabled
        if self.validate_image:
            is_valid, error_msg = self._validate_image(image)
            if not is_valid:
                self._stats['failed_count'] += 1
                return ImageQualityMetrics(
                    is_valid=False,
                    error_message=error_msg
                )
        
        try:
            # Convert to grayscale if needed
            gray_image = self._convert_to_grayscale(image)
            
            # Extract ROI if specified
            if roi is not None:
                gray_image = self._extract_roi(gray_image, roi)
            
            # Calculate brightness (L value from paper)
            brightness = self._calculate_brightness(gray_image)
            
            # Calculate contrast (C value from paper)
            contrast = self._calculate_contrast(gray_image, brightness)
            
            # Calculate histogram if enabled
            histogram = None
            if self.enable_histogram:
                histogram = self._calculate_histogram(gray_image)
            
            # Create result
            result = ImageQualityMetrics(
                brightness=brightness,
                contrast=contrast,
                histogram=histogram,
                is_valid=True,
                error_message=""
            )
            
            # Update statistics
            self._stats['valid_images'] += 1
            self._stats['avg_brightness'] = (
                (self._stats['avg_brightness'] * (self._stats['valid_images'] - 1) + brightness)
                / self._stats['valid_images']
            )
            self._stats['avg_contrast'] = (
                (self._stats['avg_contrast'] * (self._stats['valid_images'] - 1) + contrast)
                / self._stats['valid_images']
            )
            
            if result.is_overexposed():
                self._stats['overexposed_count'] += 1
            if result.is_underexposed():
                self._stats['underexposed_count'] += 1
            if result.is_low_contrast():
                self._stats['low_contrast_count'] += 1
            
            return result
            
        except Exception as e:
            self._stats['failed_count'] += 1
            return ImageQualityMetrics(
                is_valid=False,
                error_message=f"Analysis failed: {str(e)}"
            )
    
    def analyze_batch(
        self,
        images: list,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> list:
        """Analyze multiple images.
        
        Args:
            images: List of images to analyze
            roi: Optional region of interest
            
        Returns:
            List of ImageQualityMetrics
        """
        results = []
        for image in images:
            result = self.analyze(image, roi)
            results.append(result)
        return results
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> None:
        """Set the region of interest for analysis.
        
        Args:
            roi: Region of interest as (x, y, width, height)
        """
        self.roi = roi
    
    def clear_roi(self) -> None:
        """Clear the region of interest (analyze entire image)."""
        self.roi = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self._stats.copy()
        
        # Add derived statistics
        if self._stats['total_images'] > 0:
            stats['success_rate'] = self._stats['valid_images'] / self._stats['total_images']
            stats['failure_rate'] = self._stats['failed_count'] / self._stats['total_images']
            stats['overexposed_rate'] = self._stats['overexposed_count'] / self._stats['total_images']
            stats['underexposed_rate'] = self._stats['underexposed_count'] / self._stats['total_images']
            stats['low_contrast_rate'] = self._stats['low_contrast_count'] / self._stats['total_images']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['overexposed_rate'] = 0.0
            stats['underexposed_rate'] = 0.0
            stats['low_contrast_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset analysis statistics."""
        self._stats = {
            'total_images': 0,
            'valid_images': 0,
            'overexposed_count': 0,
            'underexposed_count': 0,
            'low_contrast_count': 0,
            'failed_count': 0,
            'avg_brightness': 0.0,
            'avg_contrast': 0.0
        }
    
    def __repr__(self) -> str:
        """String representation of the analyzer."""
        return (
            f"ImageQualityAnalyzer(\n"
            f"  roi={self.roi},\n"
            f"  enable_histogram={self.enable_histogram},\n"
            f"  validate_image={self.validate_image}\n"
            f")"
        )


def create_image_quality_analyzer(
    config: Optional[Config] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    enable_histogram: bool = False
) -> ImageQualityAnalyzer:
    """Factory function to create an ImageQualityAnalyzer.
    
    Args:
        config: Configuration object (optional)
        roi: Region of interest (optional)
        enable_histogram: Whether to compute histogram (optional)
        
    Returns:
        ImageQualityAnalyzer instance
    """
    return ImageQualityAnalyzer(
        config=config,
        roi=roi,
        enable_histogram=enable_histogram
    )


# Global analyzer registry
_analyzers: Dict[str, ImageQualityAnalyzer] = {}


def get_image_quality_analyzer(
    name: str = "default",
    config: Optional[Config] = None,
    **kwargs
) -> ImageQualityAnalyzer:
    """Get or create an ImageQualityAnalyzer by name.
    
    Args:
        name: Analyzer identifier
        config: Configuration object
        **kwargs: Additional arguments for ImageQualityAnalyzer
        
    Returns:
        ImageQualityAnalyzer instance
    """
    global _analyzers
    
    if name in _analyzers:
        return _analyzers[name]
    
    analyzer = create_image_quality_analyzer(config=config, **kwargs)
    _analyzers[name] = analyzer
    
    return analyzer


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Image Quality Analyzer Module Test")
    print("=" * 60)
    
    # Test 1: Create analyzer with defaults
    print("\n[Test 1] Creating analyzer with defaults...")
    analyzer = ImageQualityAnalyzer()
    print(f"  Created: {repr(analyzer)}")
    print("  ✓ Analyzer created")
    
    # Test 2: Create analyzer from config
    print("\n[Test 2] Creating analyzer from configuration...")
    if CONFIG_AVAILABLE:
        try:
            config = get_config()
            analyzer_config = ImageQualityAnalyzer.from_config(config)
            print(f"  Created from config")
            print("  ✓ Config integration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (config not available)")
    
    # Test 3: Test with uniform gray image
    print("\n[Test 3] Testing with uniform gray image (128)...")
    gray_uniform = np.full((100, 100), 128, dtype=np.uint8)
    result = analyzer.analyze(gray_uniform)
    print(f"  Brightness: {result.brightness:.2f} (expected: 128)")
    print(f"  Contrast: {result.contrast:.2f} (expected: 0)")
    assert abs(result.brightness - 128) < 1.0, "Brightness mismatch"
    assert result.contrast < 1.0, "Contrast should be ~0 for uniform image"
    print("  ✓ Uniform gray works")
    
    # Test 4: Test with black image
    print("\n[Test 4] Testing with black image...")
    black_image = np.zeros((100, 100), dtype=np.uint8)
    result = analyzer.analyze(black_image)
    print(f"  Brightness: {result.brightness:.2f} (expected: 0)")
    print(f"  Contrast: {result.contrast:.2f} (expected: 0)")
    assert result.brightness < 1.0
    assert result.contrast < 1.0
    print("  ✓ Black image works")
    
    # Test 5: Test with white image
    print("\n[Test 5] Testing with white image...")
    white_image = np.full((100, 100), 255, dtype=np.uint8)
    result = analyzer.analyze(white_image)
    print(f"  Brightness: {result.brightness:.2f} (expected: 255)")
    print(f"  Contrast: {result.contrast:.2f} (expected: 0)")
    assert result.brightness > 254.0
    assert result.contrast < 1.0
    print("  ✓ White image works")
    
    # Test 6: Test with random noise (high contrast)
    print("\n[Test 6] Testing with random noise...")
    np.random.seed(42)
    noise_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    result = analyzer.analyze(noise_image)
    print(f"  Brightness: {result.brightness:.2f} (expected: ~127)")
    print(f"  Contrast: {result.contrast:.2f} (expected: ~73)")
    assert 100 < result.brightness < 150, "Brightness out of expected range"
    assert result.contrast > 50, "Contrast should be high for random noise"
    print("  ✓ Random noise works")
    
    # Test 7: Test with color image
    print("\n[Test 7] Testing with color image...")
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[:, :, 0] = 100  # Blue
    color_image[:, :, 1] = 150  # Green
    color_image[:, :, 2] = 200  # Red
    
    result = analyzer.analyze(color_image)
    print(f"  Brightness (BGR converted): {result.brightness:.2f}")
    # For BGR: 0.114*B + 0.587*G + 0.299*R = 0.114*100 + 0.587*150 + 0.299*200 = 11.4 + 88.05 + 59.8 = 159.25
    print(f"  Expected approx: {0.114*100 + 0.587*150 + 0.299*200:.2f}")
    assert 140 < result.brightness < 180, "Brightness conversion incorrect"
    print("  ✓ Color image works")
    
    # Test 8: Test with ROI
    print("\n[Test 8] Testing with ROI...")
    # Create image with distinct regions
    image_with_roi = np.zeros((200, 200), dtype=np.uint8)
    image_with_roi[50:150, 50:150] = 200  # Bright region
    
    analyzer.set_roi((50, 50, 100, 100))
    result = analyzer.analyze(image_with_roi)
    print(f"  Brightness with ROI: {result.brightness:.2f}")
    print(f"  Contrast with ROI: {result.contrast:.2f}")
    
    analyzer.clear_roi()
    result_no_roi = analyzer.analyze(image_with_roi)
    print(f"  Brightness without ROI: {result_no_roi.brightness:.2f}")
    
    assert result.brightness != result_no_roi.brightness, "ROI should affect result"
    print("  ✓ ROI works")
    
    # Test 9: Test quality metrics helpers
    print("\n[Test 9] Testing quality metrics helpers...")
    result = ImageQualityMetrics(brightness=250.0, contrast=10.0, is_valid=True)
    print(f"  Is overexposed (brightness=250): {result.is_overexposed()}")
    result2 = ImageQualityMetrics(brightness=20.0, contrast=10.0, is_valid=True)
    print(f"  Is underexposed (brightness=20): {result2.is_underexposed()}")
    result3 = ImageQualityMetrics(brightness=127.0, contrast=10.0, is_valid=True)
    print(f"  Is low contrast (contrast=10): {result3.is_low_contrast()}")
    print("  ✓ Quality helpers work")
    
    # Test 10: Test histogram calculation
    print("\n[Test 10] Testing histogram calculation...")
    analyzer_hist = ImageQualityAnalyzer(enable_histogram=True)
    result = analyzer_hist.analyze(noise_image)
    print(f"  Histogram shape: {result.histogram.shape if result.histogram is not None else 'None'}")
    print(f"  Histogram sum: {result.histogram.sum() if result.histogram is not None else 0}")
    assert result.histogram is not None
    assert result.histogram.sum() == 10000  # 100*100 pixels
    print("  ✓ Histogram works")
    
    # Test 11: Test validation
    print("\n[Test 11] Testing input validation...")
    result = analyzer.analyze(np.array([]))  # Empty array
    print(f"  Empty array: valid={result.is_valid}, error={result.error_message}")
    assert not result.is_valid
    
    result = analyzer.analyze(None)  # None
    print(f"  None: valid={result.is_valid}, error={result.error_message}")
    assert not result.is_valid
    print("  ✓ Validation works")
    
    # Test 12: Test statistics tracking
    print("\n[Test 12] Testing statistics tracking...")
    stats = analyzer.get_statistics()
    print(f"  Statistics keys: {list(stats.keys())}")
    print(f"  Total images: {stats['total_images']}")
    analyzer.reset_statistics()
    stats_after = analyzer.get_statistics()
    print(f"  After reset: {stats_after['total_images']}")
    assert stats_after['total_images'] == 0
    print("  ✓ Statistics work")
    
    # Test 13: Test batch processing
    print("\n[Test 13] Testing batch processing...")
    batch_images = [gray_uniform, black_image, white_image, noise_image]
    results = analyzer.analyze_batch(batch_images)
    print(f"  Batch size: {len(results)}")
    print(f"  First result brightness: {results[0].brightness:.2f}")
    assert len(results) == 4
    print("  ✓ Batch processing works")
    
    # Test 14: Test factory function
    print("\n[Test 14] Testing factory function...")
    factory_analyzer = create_image_quality_analyzer()
    print(f"  Factory type: {type(factory_analyzer).__name__}")
    print("  ✓ Factory function works")
    
    # Test 15: Test registry
    print("\n[Test 15] Testing registry...")
    reg_analyzer = get_image_quality_analyzer("test_reg")
    print(f"  Registry type: {type(reg_analyzer).__name__}")
    print("  ✓ Registry works")
    
    # Test 16: Test to_dict
    print("\n[Test 16] Testing to_dict...")
    result = analyzer.analyze(gray_uniform)
    result_dict = result.to_dict()
    print(f"  Keys: {list(result_dict.keys())}")
    assert 'brightness' in result_dict
    assert 'contrast' in result_dict
    print("  ✓ to_dict works")
    
    # Test 17: Test get_quality_score
    print("\n[Test 17] Testing get_quality_score...")
    result = ImageQualityMetrics(brightness=127.0, contrast=50.0, is_valid=True)
    score = result.get_quality_score()
    print(f"  Quality score for balanced image: {score:.4f}")
    assert 0 <= score <= 1
    print("  ✓ Quality score works")
    
    # Test 18: Test config integration
    print("\n[Test 18] Testing config integration...")
    if CONFIG_AVAILABLE:
        try:
            cfg = get_config()
            print(f"  Config available: Yes")
            print(f"  Fusion config weight_a: {cfg.fusion.weight_a}")
            print(f"  Fusion config weight_b: {cfg.fusion.weight_b}")
            print("  ✓ Config integration works")
        except Exception as e:
            print(f"  Note: {e}")
    else:
        print("  Skipping (config not available)")
    
    # Test 19: Test grayscale conversion edge cases
    print("\n[Test 19] Testing grayscale conversion edge cases...")
    # Single channel image
    single_channel = np.random.randint(0, 256, (50, 50, 1), dtype=np.uint8)
    result = analyzer.analyze(single_channel)
    print(f"  Single channel: brightness={result.brightness:.2f}")
    assert result.is_valid
    
    # 4 channel image (BGRA)
    four_channel = np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)
    result = analyzer.analyze(four_channel)
    print(f"  4 channel: brightness={result.brightness:.2f}")
    assert result.is_valid
    print("  ✓ Edge cases handled")
    
    # Test 20: Performance test
    print("\n[Test 20] Performance test...")
    import time
    
    # Large image
    large_image = np.random.randint(0, 256, (1080, 1920), dtype=np.uint8)
    
    start_time = time.time()
    for _ in range(100):
        result = analyzer.analyze(large_image)
    elapsed = time.time() - start_time
    
    print(f"  Processed 100 frames in {elapsed:.3f}s")
    print(f"  Average per frame: {elapsed*10:.2f}ms")
    print(f"  Estimated FPS: {100/elapsed:.1f}")
    print("  ✓ Performance test complete")
    
    print("\n" + "=" * 60)
    print("All Image Quality Analyzer tests passed!")
    print("=" * 60)
