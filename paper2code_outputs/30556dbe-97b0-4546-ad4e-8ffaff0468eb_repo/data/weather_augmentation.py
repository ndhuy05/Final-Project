"""
ECL-YOLOv11 Weather Augmentation Module

This module implements synthetic adverse weather effects (fog, rain, snow) 
for training data augmentation as described in the paper:
"Robust Object Detection in Adverse Weather Conditions: ECL-YOLOv11 for Automotive Vision Systems"

Since the original dataset is not publicly available, synthetic weather 
augmentation enables reproduction of the experimental conditions by applying
realistic weather effects to clear weather images.

Author: ECL-YOLOv11 Reproduction Team
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import configuration
try:
    from utils.config import get_config_manager
except ImportError:
    # Fallback configuration functions
    def get_config_manager():
        return None


# =============================================================================
# Default Weather Parameters
# =============================================================================

DEFAULT_WEATHER_PARAMS: Dict[str, Dict[str, Any]] = {
    "fog": {
        "density": 0.5,  # Fog density coefficient (0.0-1.0)
        "atmospheric_light": [220, 220, 220],  # RGB atmospheric light
        "min_beta": 0.2,  # Minimum fog density coefficient
        "max_beta": 2.0,  # Maximum fog density coefficient
        "use_depth_guidance": False,  # Whether to use depth-guided fog
    },
    "rain": {
        "intensity": 0.5,  # Rain intensity (0.0-1.0)
        "min_streaks": 20,  # Minimum number of rain streaks
        "max_streaks": 100,  # Maximum number of rain streaks
        "min_angle": -15,  # Minimum rain angle (degrees)
        "max_angle": 15,  # Maximum rain angle (degrees)
        "min_length": 10,  # Minimum streak length
        "max_length": 40,  # Maximum streak length
        "streak_color": [180, 180, 180],  # Color of rain streaks
        "add_haze": True,  # Whether to add atmospheric haze
        "haze_opacity": 0.3,  # Haze blending opacity
    },
    "snow": {
        "intensity": 0.5,  # Snow intensity (0.0-1.0)
        "min_particles": 100,  # Minimum number of particles
        "max_particles": 400,  # Maximum number of particles
        "min_size": 1,  # Minimum particle radius
        "max_size": 4,  # Maximum particle radius
        "particle_color": [255, 255, 255],  # Color of snow particles
        "add_blur": True,  # Whether to add motion blur
        "blur_kernel_size": 3,  # Gaussian blur kernel size
        "brightness_increase": 20,  # Brightness increase amount
    }
}


# =============================================================================
# Utility Functions
# =============================================================================

def generate_random_seed() -> int:
    """
    Generate a random seed for reproducibility.
    
    Returns:
        int: Random seed value
    """
    return random.randint(0, 2**31 - 1)


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate input image format.
    
    Args:
        image: Input image to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if not isinstance(image, np.ndarray):
        return False, "Image is not a numpy array"
    
    if len(image.shape) != 3:
        return False, f"Image should be 3D (H, W, C), got {len(image.shape)}D"
    
    if image.shape[2] != 3:
        return False, f"Image should have 3 channels, got {image.shape[2]}"
    
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        return False, f"Unsupported dtype: {image.dtype}"
    
    return True, ""


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in uint8 format.
    
    Args:
        image: Input image (any dtype)
        
    Returns:
        np.ndarray: Image in uint8 format
    """
    if image.dtype == np.uint8:
        return image
    
    # Convert float to uint8
    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        else:
            return image.astype(np.uint8)
    
    return image.astype(np.uint8)


def generate_depth_map(
    shape: Tuple[int, int],
    method: str = "random",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a pseudo-depth map for fog simulation.
    
    Args:
        shape: (H, W) of the depth map
        method: Generation method ("random", "gradient", "edge")
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Depth map in range [0, 1]
    """
    h, w = shape
    
    if seed is not None:
        np.random.seed(seed)
    
    if method == "random":
        # Random depth with smooth variation using Perlin-like noise
        depth = np.random.rand(h, w).astype(np.float32)
        
        # Apply Gaussian blur for smoother depth variation
        depth = cv2.GaussianBlur(depth, (15, 15), 0)
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
    elif method == "gradient":
        # Linear gradient from top to bottom
        y_coords = np.linspace(0, 1, h).reshape(-1, 1)
        x_coords = np.linspace(0, 1, w).reshape(1, -1)
        
        # Combine vertical gradient with slight horizontal variation
        depth = 0.7 * y_coords + 0.3 * x_coords
        
    elif method == "edge":
        # Edge-based depth (objects are closer)
        # Create random "object" regions
        depth = np.random.rand(h, w).astype(np.float32)
        depth = cv2.GaussianBlur(depth, (21, 21), 0)
        
        # Add random rectangular regions for foreground objects
        num_objects = random.randint(3, 8)
        for _ in range(num_objects):
            x1 = random.randint(0, w - 50)
            y1 = random.randint(0, h - 50)
            x2 = x1 + random.randint(30, 100)
            y2 = y1 + random.randint(30, 100)
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            # Closer objects have higher depth values (closer = 1)
            depth[y1:y2, x1:x2] = random.uniform(0.6, 1.0)
        
        depth = cv2.GaussianBlur(depth, (11, 11), 0)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
    else:
        # Default: uniform depth
        depth = np.ones(shape, dtype=np.float32) * 0.5
    
    return depth


# =============================================================================
# Main Weather Augmentation Class
# =============================================================================

class WeatherAugmentation:
    """
    Weather Augmentation class for creating synthetic adverse weather effects.
    
    This class implements three types of weather effects:
    1. Fog: Atmospheric scattering model with depth-guided transmission
    2. Rain: Rain streaks with atmospheric haze
    3. Snow: Falling snow particles with motion blur
    
    The effects are designed to degrade image quality while preserving
    object boundaries, which is important for the CE module to enhance.
    
    Attributes:
        fog_density (float): Fog density coefficient (0.0-1.0)
        rain_intensity (float): Rain intensity (0.0-1.0)
        snow_intensity (float): Snow intensity (0.0-1.0)
        seed (Optional[int]): Random seed for reproducibility
        weather_params (Dict): Custom weather parameters
        
    Example:
        >>> weather = WeatherAugmentation(fog_density=0.5, rain_intensity=0.3)
        >>> foggy_image = weather.add_fog(image)
        >>> rainy_image = weather.add_rain(image)
        >>> snowy_image = weather.add_snow(image)
    """
    
    def __init__(
        self,
        fog_density: float = 0.5,
        rain_intensity: float = 0.5,
        snow_intensity: float = 0.5,
        seed: Optional[int] = None,
        weather_params: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Weather Augmentation.
        
        Args:
            fog_density: Fog density (0.0 - 1.0), default: 0.5
            rain_intensity: Rain intensity (0.0 - 1.0), default: 0.5
            snow_intensity: Snow intensity (0.0 - 1.0), default: 0.5
            seed: Random seed for reproducibility (optional)
            weather_params: Custom weather parameters (optional)
            config: Configuration dictionary (optional)
        """
        # Try to load from config
        if config is None:
            try:
                config_manager = get_config_manager()
                if config_manager is not None:
                    config = config_manager.get_config_dict()
            except:
                config = {}
        
        # Load weather parameters from config if available
        if config:
            data_config = config.get('data', {})
            weather_params_config = data_config.get('weather_params', {})
            weather_params_config = data_config.get('augmentation', {}).get('weather', ['fog', 'rain', 'snow'])
            
            fog_density = weather_params_config.get('fog_density', fog_density)
            rain_intensity = weather_params_config.get('rain_intensity', rain_intensity)
            snow_intensity = weather_params_config.get('snow_intensity', snow_intensity)
        
        # Set parameters
        self.fog_density = max(0.0, min(1.0, fog_density))
        self.rain_intensity = max(0.0, min(1.0, rain_intensity))
        self.snow_intensity = max(0.0, min(1.0, snow_intensity))
        self.seed = seed
        
        # Set random seeds
        if self.seed is not None:
            self.set_seed(self.seed)
        
        # Use custom params or defaults
        self.weather_params = weather_params if weather_params is not None else DEFAULT_WEATHER_PARAMS.copy()
    
    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        cv2.setRNGSeed(seed)
    
    def add_fog(
        self,
        image: np.ndarray,
        density: Optional[float] = None,
        atmospheric_light: Optional[List[int]] = None,
        use_depth_guidance: Optional[bool] = None
    ) -> np.ndarray:
        """
        Apply fog effect to image using atmospheric scattering model.
        
        Based on Koschmieder's law and atmospheric scattering:
        I(x) = J(x) × t(x) + A × (1 - t(x))
        
        where:
        - J(x) = clear input image
        - A = atmospheric light (global illumination)
        - t(x) = transmission map = exp(-β × d(x))
        - β = fog density coefficient
        - d(x) = scene depth
        
        Args:
            image: Input image (H, W, 3) in BGR format
            density: Fog density override (0.0 - 1.0)
            atmospheric_light: RGB atmospheric light values
            use_depth_guidance: Whether to use depth-guided fog
            
        Returns:
            np.ndarray: Foggy image (H, W, 3) in BGR format
        """
        # Validate image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            raise ValueError(f"Invalid image: {error_msg}")
        
        # Ensure uint8 format
        image = ensure_uint8(image)
        
        # Get parameters
        density = density if density is not None else self.fog_density
        density = max(0.0, min(1.0, density))
        
        if atmospheric_light is None:
            atmospheric_light = self.weather_params["fog"].get(
                "atmospheric_light", 
                [220, 220, 220]
            )
        
        if use_depth_guidance is None:
            use_depth_guidance = self.weather_params["fog"].get(
                "use_depth_guidance", 
                False
            )
        
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Get atmospheric light (BGR order for OpenCV)
        A = np.array([
            atmospheric_light[2],  # B
            atmospheric_light[1],  # G
            atmospheric_light[0]   # R
        ], dtype=np.float32) / 255.0
        
        # Calculate fog density coefficient (beta)
        min_beta = self.weather_params["fog"].get("min_beta", 0.2)
        max_beta = self.weather_params["fog"].get("max_beta", 2.0)
        beta = min_beta + density * (max_beta - min_beta)
        
        # Generate or use depth map
        h, w = image.shape[:2]
        
        if use_depth_guidance:
            # Use depth-guided transmission
            depth = generate_depth_map((h, w), method="edge", seed=self.seed)
        else:
            # Use uniform fog (simpler but still effective)
            depth = np.ones((h, w), dtype=np.float32) * 0.5
        
        # Calculate transmission: t(x) = exp(-beta * depth(x))
        transmission = np.exp(-beta * depth)
        
        # Ensure transmission is 3-channel for broadcasting
        transmission = transmission[:, :, np.newaxis]
        
        # Apply atmospheric scattering formula
        # I = J * t + A * (1 - t)
        foggy = img_float * transmission + A * (1 - transmission)
        
        # Clip to valid range and convert back to uint8
        foggy = np.clip(foggy, 0.0, 1.0)
        foggy = (foggy * 255).astype(np.uint8)
        
        return foggy
    
    def add_rain(
        self,
        image: np.ndarray,
        intensity: Optional[float] = None,
        add_haze: Optional[bool] = None,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Apply rain effect to image with streaks and atmospheric haze.
        
        The rain effect includes:
        1. Rain streaks: directional blur patterns
        2. Atmospheric haze: reduced contrast and brightness
        3. Optional noise: rain-induced image noise
        
        Args:
            image: Input image (H, W, 3) in BGR format
            intensity: Rain intensity override (0.0 - 1.0)
            add_haze: Whether to add atmospheric haze
            add_noise: Whether to add rain-induced noise
            
        Returns:
            np.ndarray: Rainy image (H, W, 3) in BGR format
        """
        # Validate image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            raise ValueError(f"Invalid image: {error_msg}")
        
        # Ensure uint8 format
        image = ensure_uint8(image)
        
        # Get parameters
        intensity = intensity if intensity is not None else self.rain_intensity
        intensity = max(0.0, min(1.0, intensity))
        
        if add_haze is None:
            add_haze = self.weather_params["rain"].get("add_haze", True)
        
        # Get rain parameters
        min_streaks = self.weather_params["rain"].get("min_streaks", 20)
        max_streaks = self.weather_params["rain"].get("max_streaks", 100)
        min_angle = self.weather_params["rain"].get("min_angle", -15)
        max_angle = self.weather_params["rain"].get("max_angle", 15)
        min_length = self.weather_params["rain"].get("min_length", 10)
        max_length = self.weather_params["rain"].get("max_length", 40)
        streak_color = self.weather_params["rain"].get("streak_color", [180, 180, 180])
        
        # Create output image
        rainy = image.copy()
        h, w = image.shape[:2]
        
        # Calculate number of rain streaks based on intensity
        num_streaks = int(min_streaks + intensity * (max_streaks - min_streaks))
        
        # Generate rain streaks
        for _ in range(num_streaks):
            # Random position
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            # Random angle and length
            angle = random.uniform(min_angle, max_angle)
            length = random.randint(min_length, max_length)
            
            # Calculate end point
            end_x = int(x + length * np.sin(np.radians(angle)))
            end_y = int(y + length * np.cos(np.radians(angle)))
            
            # Clip to image bounds
            end_x = max(0, min(w - 1, end_x))
            end_y = max(0, min(h - 1, end_y))
            
            # Draw rain streak with gradient opacity
            cv2.line(rainy, (x, y), (end_x, end_y), streak_color, 1)
        
        # Add atmospheric haze (reduced contrast and brightness)
        if add_haze:
            haze_opacity = self.weather_params["rain"].get("haze_opacity", 0.3) * intensity
            
            # Convert to HSV for brightness adjustment
            rainy_hsv = cv2.cvtColor(rainy, cv2.COLOR_BGR2HSV)
            
            # Reduce V (brightness) channel
            rainy_hsv[:, :, 2] = rainy_hsv[:, :, 2].astype(np.int16)
            rainy_hsv[:, :, 2] = np.clip(rainy_hsv[:, :, 2] * (1 - haze_opacity * 0.5), 0, 255).astype(np.uint8)
            
            rainy = cv2.cvtColor(rainy_hsv, cv2.COLOR_HSV2BGR)
            
            # Add slight blue tint for rainy atmosphere
            rainy = cv2.addWeighted(rainy, 1, np.full_like(rainy, [200, 210, 230]), haze_opacity * 0.2, 0)
        
        # Add rain-induced noise
        if add_noise and intensity > 0.3:
            noise_intensity = (intensity - 0.3) * 15
            noise = np.random.normal(0, noise_intensity, rainy.shape).astype(np.int16)
            rainy = np.clip(rainy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return rainy
    
    def add_snow(
        self,
        image: np.ndarray,
        intensity: Optional[float] = None,
        add_blur: Optional[bool] = None,
        add_brightness: bool = True
    ) -> np.ndarray:
        """
        Apply snow effect to image with particles and motion blur.
        
        The snow effect includes:
        1. Snow particles: scattered white dots of varying sizes
        2. Motion blur: slight directional blur for falling snow
        3. Brightness increase: overall brightness boost
        
        Args:
            image: Input image (H, W, 3) in BGR format
            intensity: Snow intensity override (0.0 - 1.0)
            add_blur: Whether to add motion blur
            add_brightness: Whether to increase brightness
            
        Returns:
            np.ndarray: Snowy image (H, W, 3) in BGR format
        """
        # Validate image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            raise ValueError(f"Invalid image: {error_msg}")
        
        # Ensure uint8 format
        image = ensure_uint8(image)
        
        # Get parameters
        intensity = intensity if intensity is not None else self.snow_intensity
        intensity = max(0.0, min(1.0, intensity))
        
        if add_blur is None:
            add_blur = self.weather_params["snow"].get("add_blur", True)
        
        # Get snow parameters
        min_particles = self.weather_params["snow"].get("min_particles", 100)
        max_particles = self.weather_params["snow"].get("max_particles", 400)
        min_size = self.weather_params["snow"].get("min_size", 1)
        max_size = self.weather_params["snow"].get("max_size", 4)
        particle_color = self.weather_params["snow"].get("particle_color", [255, 255, 255])
        blur_kernel_size = self.weather_params["snow"].get("blur_kernel_size", 3)
        brightness_increase = self.weather_params["snow"].get("brightness_increase", 20)
        
        # Create output image
        snowy = image.copy()
        h, w = image.shape[:2]
        
        # Calculate number of snow particles based on intensity
        num_particles = int(min_particles + intensity * (max_particles - min_particles))
        
        # Generate snow particles
        for _ in range(num_particles):
            # Random position
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            # Random size
            size = random.randint(min_size, max_size)
            
            # Draw snow particle (circle)
            cv2.circle(snowy, (x, y), size, particle_color, -1)
        
        # Add motion blur for falling snow effect
        if add_blur and intensity > 0.3:
            # Use directional blur (slightly vertical)
            kernel_size = blur_kernel_size + int(intensity * 2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[int(kernel_size / 2), :] = 1.0 / kernel_size
            
            # Apply directional blur
            snowy = cv2.filter2D(snowy, -1, kernel)
        
        # Increase brightness for snowy scene
        if add_brightness:
            brightness = int(brightness_increase * intensity)
            snowy = cv2.convertScaleAbs(snowy, alpha=1.0, beta=brightness)
        
        return snowy
    
    def apply_weather(
        self,
        image: np.ndarray,
        weather_type: str = "fog",
        **kwargs
    ) -> np.ndarray:
        """
        Apply weather effect to image based on specified type.
        
        This is a unified interface for applying any weather effect.
        
        Args:
            image: Input image (H, W, 3) in BGR format
            weather_type: Type of weather ("fog", "rain", "snow", "none")
            **kwargs: Additional parameters:
                - density (float): For fog effect (0.0-1.0)
                - intensity (float): For rain/snow effect (0.0-1.0)
                - add_haze (bool): For rain effect
                - add_blur (bool): For snow effect
                
        Returns:
            np.ndarray: Image with weather effect applied
        """
        # Validate image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            raise ValueError(f"Invalid image: {error_msg}")
        
        # Normalize weather type
        weather_type = weather_type.lower().strip()
        
        # Apply appropriate weather effect
        if weather_type == "fog":
            return self.add_fog(
                image,
                density=kwargs.get("density", None),
                atmospheric_light=kwargs.get("atmospheric_light", None),
                use_depth_guidance=kwargs.get("use_depth_guidance", None)
            )
        
        elif weather_type == "rain":
            return self.add_rain(
                image,
                intensity=kwargs.get("intensity", None),
                add_haze=kwargs.get("add_haze", None),
                add_noise=kwargs.get("add_noise", True)
            )
        
        elif weather_type == "snow":
            return self.add_snow(
                image,
                intensity=kwargs.get("intensity", None),
                add_blur=kwargs.get("add_blur", None),
                add_brightness=kwargs.get("add_brightness", True)
            )
        
        elif weather_type == "none" or weather_type == "clear":
            # Return original image
            return image.copy()
        
        else:
            raise ValueError(f"Unknown weather type: {weather_type}. "
                           f"Supported types: 'fog', 'rain', 'snow', 'none'")
    
    def apply_random_weather(
        self,
        image: np.ndarray,
        weather_types: Optional[List[str]] = None,
        prob_fog: float = 0.33,
        prob_rain: float = 0.33,
        prob_snow: float = 0.34
    ) -> Tuple[np.ndarray, str]:
        """
        Apply a random weather effect to the image.
        
        Args:
            image: Input image (H, W, 3) in BGR format
            weather_types: List of weather types to choose from
            prob_fog: Probability of fog
            prob_rain: Probability of rain
            prob_snow: Probability of snow
            
        Returns:
            Tuple of (modified_image, weather_type)
        """
        if weather_types is None:
            weather_types = ["fog", "rain", "snow"]
        
        # Choose weather type
        weather_type = random.choice(weather_types)
        
        # Apply the selected weather effect
        result = self.apply_weather(image, weather_type)
        
        return result, weather_type
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get current weather parameters.
        
        Returns:
            Dictionary of current parameters
        """
        return {
            "fog_density": self.fog_density,
            "rain_intensity": self.rain_intensity,
            "snow_intensity": self.snow_intensity,
            "seed": self.seed
        }
    
    def set_parameters(
        self,
        fog_density: Optional[float] = None,
        rain_intensity: Optional[float] = None,
        snow_intensity: Optional[float] = None
    ) -> None:
        """
        Set weather parameters.
        
        Args:
            fog_density: New fog density (0.0-1.0)
            rain_intensity: New rain intensity (0.0-1.0)
            snow_intensity: New snow intensity (0.0-1.0)
        """
        if fog_density is not None:
            self.fog_density = max(0.0, min(1.0, fog_density))
        
        if rain_intensity is not None:
            self.rain_intensity = max(0.0, min(1.0, rain_intensity))
        
        if snow_intensity is not None:
            self.snow_intensity = max(0.0, min(1.0, snow_intensity))


# =============================================================================
# Functional Interface (for simpler usage)
# =============================================================================

def add_fog(
    image: np.ndarray,
    density: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Functional interface for adding fog to an image.
    
    Args:
        image: Input image (H, W, 3) in BGR format
        density: Fog density (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Foggy image
    """
    aug = WeatherAugmentation(fog_density=density, seed=seed)
    return aug.add_fog(image)


def add_rain(
    image: np.ndarray,
    intensity: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Functional interface for adding rain to an image.
    
    Args:
        image: Input image (H, W, 3) in BGR format
        intensity: Rain intensity (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Rainy image
    """
    aug = WeatherAugmentation(rain_intensity=intensity, seed=seed)
    return aug.add_rain(image)


def add_snow(
    image: np.ndarray,
    intensity: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Functional interface for adding snow to an image.
    
    Args:
        image: Input image (H, W, 3) in BGR format
        intensity: Snow intensity (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Snowy image
    """
    aug = WeatherAugmentation(snow_intensity=intensity, seed=seed)
    return aug.add_snow(image)


def apply_weather(
    image: np.ndarray,
    weather_type: str = "fog",
    intensity: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Functional interface for applying weather effects.
    
    Args:
        image: Input image (H, W, 3) in BGR format
        weather_type: Type of weather ("fog", "rain", "snow")
        intensity: Weather intensity (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Image with weather effect
    """
    if weather_type.lower() == "fog":
        return add_fog(image, density=intensity, seed=seed)
    elif weather_type.lower() == "rain":
        return add_rain(image, intensity=intensity, seed=seed)
    elif weather_type.lower() == "snow":
        return add_snow(image, intensity=intensity, seed=seed)
    else:
        return image.copy()


# =============================================================================
# Test/Verification Code
# =============================================================================

if __name__ == "__main__":
    # Test weather augmentation module
    print("Testing Weather Augmentation Module")
    print("=" * 50)
    
    # Create a test image (640x480 random noise)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some structure to the test image (simulate objects)
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(test_image, (300, 200), (400, 350), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(test_image, (500, 150), 30, (0, 0, 255), -1)  # Red circle
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Test image dtype: {test_image.dtype}")
    
    # Test 1: Create WeatherAugmentation instance
    print("\n1. Testing WeatherAugmentation initialization:")
    weather = WeatherAugmentation(
        fog_density=0.5,
        rain_intensity=0.5,
        snow_intensity=0.5,
        seed=42
    )
    print(f"   Instance created successfully")
    print(f"   Parameters: {weather.get_parameters()}")
    
    # Test 2: Add fog
    print("\n2. Testing add_fog:")
    foggy_image = weather.add_fog(test_image.copy())
    print(f"   Input shape: {test_image.shape}")
    print(f"   Output shape: {foggy_image.shape}")
    print(f"   Output dtype: {foggy_image.dtype}")
    print(f"   Mean pixel change: {np.abs(foggy_image.astype(float) - test_image.astype(float)).mean():.2f}")
    
    # Test 3: Add rain
    print("\n3. Testing add_rain:")
    rainy_image = weather.add_rain(test_image.copy())
    print(f"   Input shape: {test_image.shape}")
    print(f"   Output shape: {rainy_image.shape}")
    print(f"   Output dtype: {rainy_image.dtype}")
    print(f"   Mean pixel change: {np.abs(rainy_image.astype(float) - test_image.astype(float)).mean():.2f}")
    
    # Test 4: Add snow
    print("\n4. Testing add_snow:")
    snowy_image = weather.add_snow(test_image.copy())
    print(f"   Input shape: {test_image.shape}")
    print(f"   Output shape: {snowy_image.shape}")
    print(f"   Output dtype: {snowy_image.dtype}")
    print(f"   Mean pixel change: {np.abs(snowy_image.astype(float) - test_image.astype(float)).mean():.2f}")
    
    # Test 5: Apply different intensities
    print("\n5. Testing different intensities:")
    for density in [0.2, 0.5, 0.8]:
        fog_result = weather.add_fog(test_image.copy(), density=density)
        change = np.abs(fog_result.astype(float) - test_image.astype(float)).mean()
        print(f"   Fog density={density}: mean change={change:.2f}")
    
    # Test 6: Unified apply_weather interface
    print("\n6. Testing unified apply_weather interface:")
    for weather_type in ["fog", "rain", "snow", "none"]:
        result = weather.apply_weather(test_image.copy(), weather_type)
        change = np.abs(result.astype(float) - test_image.astype(float)).mean()
        print(f"   {weather_type}: mean change={change:.2f}")
    
    # Test 7: Random weather
    print("\n7. Testing random weather:")
    random.seed(123)
    np.random.seed(123)
    for i in range(5):
        result, wtype = weather.apply_random_weather(test_image.copy())
        change = np.abs(result.astype(float) - test_image.astype(float)).mean()
        print(f"   Random {i+1}: {wtype}, mean change={change:.2f}")
    
    # Test 8: Functional interfaces
    print("\n8. Testing functional interfaces:")
    fog_result = add_fog(test_image.copy(), density=0.5, seed=42)
    rain_result = add_rain(test_image.copy(), intensity=0.5, seed=42)
    snow_result = add_snow(test_image.copy(), intensity=0.5, seed=42)
    print(f"   add_fog: {fog_result.shape}")
    print(f"   add_rain: {rain_result.shape}")
    print(f"   add_snow: {snow_result.shape}")
    
    # Test 9: Test with different input image types
    print("\n9. Testing with different image types:")
    # Float32 image
    test_float = test_image.astype(np.float32) / 255.0
    result = weather.add_fog(test_float)
    print(f"   Float32 input: output shape={result.shape}, dtype={result.dtype}")
    
    # Test 10: Edge cases
    print("\n10. Testing edge cases:")
    # Zero intensity
    weather_zero = WeatherAugmentation(fog_density=0.0, rain_intensity=0.0, snow_intensity=0.0)
    result = weather_zero.add_fog(test_image.copy())
    change = np.abs(result.astype(float) - test_image.astype(float)).mean()
    print(f"   Zero fog density: mean change={change:.2f}")
    
    # Max intensity
    weather_max = WeatherAugmentation(fog_density=1.0, rain_intensity=1.0, snow_intensity=1.0)
    result = weather_max.add_fog(test_image.copy())
    change = np.abs(result.astype(float) - test_image.astype(float)).mean()
    print(f"   Max fog density: mean change={change:.2f}")
    
    print("\n" + "=" * 50)
    print("Weather augmentation test completed!")
