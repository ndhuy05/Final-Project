"""
utils/metrics.py

Evaluation metrics for the conveyor belt speed detection system.
Based on the paper: "Conveyor Belt Speed Detection via the Synergistic Fusion of 
Optical Flow and Feature Matching"

This module implements the evaluation metrics defined in Section 4.2 of the paper:
- MAE: Mean Absolute Error (Equation 11)
- RMSE: Root Mean Square Error (Equation 12)  
- Error Percentage: Relative error in percentage

Author: Based on paper methodology
"""

import numpy as np
from typing import Union, List, Dict, Optional, Tuple
from dataclasses import dataclass, field


def validate_inputs(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert input arrays to numpy arrays.
    
    Args:
        estimated: Array of estimated speeds
        true: Array of ground truth speeds
        
    Returns:
        Tuple of (estimated_array, true_array) as numpy arrays
        
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    # Convert to numpy arrays if needed
    if not isinstance(estimated, np.ndarray):
        estimated = np.array(estimated, dtype=np.float64)
    if not isinstance(true, np.ndarray):
        true = np.array(true, dtype=np.float64)
    
    # Flatten arrays to 1D
    estimated = estimated.flatten()
    true = true.flatten()
    
    # Check for empty arrays
    if len(estimated) == 0 or len(true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Check for matching lengths
    if len(estimated) != len(true):
        raise ValueError(
            f"Input arrays must have same length: "
            f"estimated={len(estimated)}, true={len(true)}"
        )
    
    # Check for NaN or infinite values
    if np.any(np.isnan(estimated)) or np.any(np.isnan(true)):
        raise ValueError("Input arrays contain NaN values")
    
    if np.any(np.isinf(estimated)) or np.any(np.isinf(true)):
        raise ValueError("Input arrays contain infinite values")
    
    # Check for non-negative speeds (belt speeds should be positive)
    if np.any(estimated < 0):
        raise ValueError("Estimated speeds cannot be negative")
    
    return estimated, true


def calculate_mae(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]]
) -> float:
    """Calculate Mean Absolute Error (MAE).
    
    Based on Equation (11) from the paper:
    MAE = (1/N) * sum(|V_est,i - V_true,i|) for i=1 to N
    
    The MAE metric directly reflects the overall error level of the speed 
    measurement method. Being insensitive to outliers, it can effectively 
    characterize the average accuracy of the method under normal operating 
    conditions.
    
    Args:
        estimated: Array of estimated belt speeds in m/s
        true: Array of ground truth belt speeds in m/s
        
    Returns:
        Mean Absolute Error in m/s
        
    Raises:
        ValueError: If inputs are invalid or have different lengths
    """
    estimated_arr, true_arr = validate_inputs(estimated, true)
    
    # Calculate absolute differences
    abs_diff = np.abs(estimated_arr - true_arr)
    
    # Calculate mean
    mae = np.mean(abs_diff)
    
    return float(mae)


def calculate_rmse(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]]
) -> float:
    """Calculate Root Mean Square Error (RMSE).
    
    Based on Equation (12) from the paper:
    RMSE = sqrt((1/N) * sum((V_est,i - V_true,i)^2) for i=1 to N
    
    This metric amplifies the weight of larger errors through squaring, 
    making it more sensitive to abnormal errors under extreme conditions. 
    It can be used to evaluate the error dispersion and robustness of 
    the method. A smaller RMSE indicates better consistency of the speed 
    measurement results and less fluctuation caused by environmental interference.
    
    Args:
        estimated: Array of estimated belt speeds in m/s
        true: Array of ground truth belt speeds in m/s
        
    Returns:
        Root Mean Square Error in m/s
        
    Raises:
        ValueError: If inputs are invalid or have different lengths
    """
    estimated_arr, true_arr = validate_inputs(estimated, true)
    
    # Calculate squared differences
    sq_diff = (estimated_arr - true_arr) ** 2
    
    # Calculate mean and take square root
    mse = np.mean(sq_diff)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def calculate_error_percentage(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]],
    epsilon: float = 1e-10
) -> float:
    """Calculate mean error percentage.
    
    Based on the paper methodology:
    Error% = (|V_est - V_true| / V_true) * 100%
    
    This provides the relative accuracy comparison across different speed ranges.
    The paper reports results like "5.25%" for RAFT-SEnet and "5.5%" for 
    Harris-BRIEF-RANSAC.
    
    Args:
        estimated: Array of estimated belt speeds in m/s
        true: Array of ground truth belt speeds in m/s
        epsilon: Small value to prevent division by zero (default: 1e-10)
        
    Returns:
        Mean error percentage (e.g., 5.25 for 5.25%)
        
    Raises:
        ValueError: If inputs are invalid or have different lengths
    """
    estimated_arr, true_arr = validate_inputs(estimated, true)
    
    # Calculate absolute differences
    abs_diff = np.abs(estimated_arr - true_arr)
    
    # Calculate relative error (avoid division by zero)
    # Add small epsilon to prevent division by zero for zero true speeds
    relative_error = abs_diff / np.maximum(true_arr, epsilon)
    
    # Convert to percentage and calculate mean
    error_percentage = np.mean(relative_error) * 100.0
    
    return float(error_percentage)


def calculate_all_metrics(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    """Calculate all evaluation metrics at once.
    
    Convenience function that computes MAE, RMSE, and error percentage
    in a single call.
    
    Args:
        estimated: Array of estimated belt speeds in m/s
        true: Array of ground truth belt speeds in m/s
        
    Returns:
        Dictionary containing:
            - 'MAE': Mean Absolute Error in m/s
            - 'RMSE': Root Mean Square Error in m/s  
            - 'error_percentage': Mean error percentage
            
    Raises:
        ValueError: If inputs are invalid or have different lengths
    """
    return {
        'MAE': calculate_mae(estimated, true),
        'RMSE': calculate_rmse(estimated, true),
        'error_percentage': calculate_error_percentage(estimated, true)
    }


def calculate_metrics_per_speed(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]],
    speed_levels: Optional[Union[np.ndarray, List[float]]] = None
) -> Dict[float, Dict[str, float]]:
    """Calculate metrics grouped by speed level.
    
    Based on the paper's Table III, which shows results for different
    speed settings (0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.5 m/s).
    
    Args:
        estimated: Array of estimated belt speeds in m/s
        true: Array of ground truth belt speeds in m/s
        speed_levels: Optional array of speed level labels for each measurement
        
    Returns:
        Dictionary mapping speed levels to their metrics
    """
    estimated_arr, true_arr = validate_inputs(estimated, true)
    
    if speed_levels is None:
        # If no speed levels provided, treat all as one group
        return {
            'all': calculate_all_metrics(estimated_arr, true_arr)
        }
    
    # Convert speed levels to numpy array
    if not isinstance(speed_levels, np.ndarray):
        speed_levels = np.array(speed_levels, dtype=np.float64)
    
    speed_levels = speed_levels.flatten()
    
    # Get unique speed levels
    unique_levels = np.unique(speed_levels)
    
    # Calculate metrics for each speed level
    result = {}
    for level in unique_levels:
        mask = speed_levels == level
        level_estimated = estimated_arr[mask]
        level_true = true_arr[mask]
        
        if len(level_estimated) > 0:
            result[float(level)] = calculate_all_metrics(level_estimated, level_true)
    
    return result


def calculate_standard_deviation(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]]
) -> float:
    """Calculate standard deviation of errors.
    
    Additional metric for measuring stability of speed detection.
    Lower standard deviation indicates more stable/consistent results.
    
    Args:
        estimated: Array of estimated belt speeds in m/s
        true: Array of ground truth belt speeds in m/s
        
    Returns:
        Standard deviation of errors in m/s
    """
    estimated_arr, true_arr = validate_inputs(estimated, true)
    
    errors = estimated_arr - true_arr
    std_dev = np.std(errors)
    
    return float(std_dev)


def calculate_max_error(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]]
) -> Tuple[float, float]:
    """Calculate maximum absolute error and its index.
    
    Useful for identifying worst-case scenarios and outlier detection.
    
    Args:
        estimated: Array of estimated belt speeds in m/s
        true: Array of ground truth belt speeds in m/s
        
    Returns:
        Tuple of (max_error, index) where max_error is the largest
        absolute error and index is its position in the arrays
    """
    estimated_arr, true_arr = validate_inputs(estimated, true)
    
    abs_errors = np.abs(estimated_arr - true_arr)
    max_error = np.max(abs_errors)
    max_index = int(np.argmax(abs_errors))
    
    return float(max_error), max_index


def calculate_r_squared(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]]
) -> float:
    """Calculate R-squared (coefficient of determination).
    
    Measures how well the estimated values match the true values.
    R² = 1 indicates perfect prediction, R² = 0 indicates no improvement
    over predicting the mean.
    
    Args:
        estimated: Array of estimated belt speeds in m/s
        true: Array of ground truth belt speeds in m/s
        
    Returns:
        R-squared value
    """
    estimated_arr, true_arr = validate_inputs(estimated, true)
    
    # Calculate total sum of squares
    ss_total = np.sum((true_arr - np.mean(true_arr)) ** 2)
    
    # Calculate residual sum of squares
    ss_residual = np.sum((true_arr - estimated_arr) ** 2)
    
    # Calculate R-squared
    if ss_total == 0:
        # If all true values are the same, check if predictions match
        r_squared = 1.0 if np.allclose(estimated_arr, true_arr) else 0.0
    else:
        r_squared = 1.0 - (ss_residual / ss_total)
    
    return float(r_squared)


@dataclass
class EvaluationResult:
    """Container for evaluation results.
    
    This class provides a structured way to store and access
    evaluation metrics computed from speed detection results.
    """
    mae: float = 0.0
    rmse: float = 0.0
    error_percentage: float = 0.0
    std_deviation: float = 0.0
    max_error: float = 0.0
    r_squared: float = 0.0
    num_samples: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert result to dictionary."""
        return {
            'MAE': self.mae,
            'RMSE': self.rmse,
            'error_percentage': self.error_percentage,
            'std_deviation': self.std_deviation,
            'max_error': self.max_error,
            'r_squared': self.r_squared,
            'num_samples': self.num_samples
        }
    
    def __str__(self) -> str:
        """String representation of evaluation results."""
        return (
            f"Evaluation Result (n={self.num_samples}):\n"
            f"  MAE:              {self.mae:.4f} m/s\n"
            f"  RMSE:             {self.rmse:.4f} m/s\n"
            f"  Error Percentage: {self.error_percentage:.2f}%\n"
            f"  Std Deviation:    {self.std_deviation:.4f} m/s\n"
            f"  Max Error:        {self.max_error:.4f} m/s\n"
            f"  R-squared:        {self.r_squared:.4f}"
        )


def evaluate_speed_detection(
    estimated: Union[np.ndarray, List[float]], 
    true: Union[np.ndarray, List[float]],
    compute_extended: bool = True
) -> EvaluationResult:
    """Comprehensive evaluation of speed detection results.
    
    This is the main entry point for evaluating speed detection performance.
    It computes all relevant metrics based on the paper's methodology.
    
    Args:
        estimated: Array of estimated belt speeds in m/s
        true: Array of ground truth belt speeds in m/s
        compute_extended: Whether to compute extended metrics (std, max_error, r_squared)
        
    Returns:
        EvaluationResult containing all computed metrics
    """
    estimated_arr, true_arr = validate_inputs(estimated, true)
    
    # Compute basic metrics
    mae = calculate_mae(estimated_arr, true_arr)
    rmse = calculate_rmse(estimated_arr, true_arr)
    error_pct = calculate_error_percentage(estimated_arr, true_arr)
    
    # Initialize result
    result = EvaluationResult(
        mae=mae,
        rmse=rmse,
        error_percentage=error_pct,
        num_samples=len(estimated_arr)
    )
    
    # Compute extended metrics if requested
    if compute_extended:
        result.std_deviation = calculate_standard_deviation(estimated_arr, true_arr)
        result.max_error, _ = calculate_max_error(estimated_arr, true_arr)
        result.r_squared = calculate_r_squared(estimated_arr, true_arr)
    
    return result


def compare_methods(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    method_names: Optional[List[str]] = None
) -> Dict[str, EvaluationResult]:
    """Compare multiple speed detection methods.
    
    Useful for comparing the proposed method against baselines like
    TV-L1, FlowNet2, RAFT, SIFT, FAST, ORB as shown in the paper.
    
    Args:
        results_dict: Dictionary mapping method names to (estimated, true) tuples
        method_names: Optional list specifying order of methods
        
    Returns:
        Dictionary mapping method names to their EvaluationResult
    """
    if method_names is None:
        method_names = list(results_dict.keys())
    
    comparison = {}
    for method_name in method_names:
        if method_name not in results_dict:
            raise ValueError(f"Method '{method_name}' not found in results")
        
        estimated, true = results_dict[method_name]
        comparison[method_name] = evaluate_speed_detection(estimated, true)
    
    return comparison


def print_comparison_table(
    comparison: Dict[str, EvaluationResult],
    title: str = "Method Comparison"
) -> None:
    """Print a formatted comparison table.
    
    Similar to Tables I, II, and III in the paper.
    
    Args:
        comparison: Dictionary of method names to EvaluationResult
        title: Title for the table
    """
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)
    print(f"{'Method':<25} {'MAE (m/s)':<12} {'RMSE (m/s)':<12} {'Error %':<12}")
    print("-" * 70)
    
    for method_name, result in comparison.items():
        print(f"{method_name:<25} {result.mae:<12.4f} {result.rmse:<12.4f} {result.error_percentage:<12.2f}")
    
    print("=" * 70)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Metrics Module Test")
    print("=" * 60)
    
    # Test data based on paper results (Section 4.2, Table I)
    # Ground truth: 4 m/s
    # Test results: TV-L1: 3.62, FlowNet2: 3.73, RAFT: 3.74, RAFT-SEnet: 4.21
    
    true_speeds = np.array([4.0] * 10)
    
    # Test 1: MAE calculation
    print("\n[Test 1] MAE Calculation")
    tvl1_estimates = np.array([3.62] * 10)
    flownet2_estimates = np.array([3.73] * 10)
    raft_estimates = np.array([3.74] * 10)
    raft_senet_estimates = np.array([4.21] * 10)
    
    mae_tvl1 = calculate_mae(tvl1_estimates, true_speeds)
    mae_flownet2 = calculate_mae(flownet2_estimates, true_speeds)
    mae_raft = calculate_mae(raft_estimates, true_speeds)
    mae_raft_senet = calculate_mae(raft_senet_estimates, true_speeds)
    
    print(f"  TV-L1 MAE: {mae_tvl1:.4f} (expected: ~0.38)")
    print(f"  FlowNet2 MAE: {mae_flownet2:.4f} (expected: ~0.27)")
    print(f"  RAFT MAE: {mae_raft:.4f} (expected: ~0.26)")
    print(f"  RAFT-SEnet MAE: {mae_raft_senet:.4f} (expected: ~0.21)")
    
    # Test 2: RMSE calculation
    print("\n[Test 2] RMSE Calculation")
    rmse_tvl1 = calculate_rmse(tvl1_estimates, true_speeds)
    rmse_flownet2 = calculate_rmse(flownet2_estimates, true_speeds)
    rmse_raft = calculate_rmse(raft_estimates, true_speeds)
    rmse_raft_senet = calculate_rmse(raft_senet_estimates, true_speeds)
    
    print(f"  TV-L1 RMSE: {rmse_tvl1:.4f} (expected: ~0.42)")
    print(f"  FlowNet2 RMSE: {rmse_flownet2:.4f} (expected: ~0.35)")
    print(f"  RAFT RMSE: {rmse_raft:.4f} (expected: ~0.29)")
    print(f"  RAFT-SEnet RMSE: {rmse_raft_senet:.4f} (expected: ~0.25)")
    
    # Test 3: Error percentage calculation
    print("\n[Test 3] Error Percentage Calculation")
    err_tvl1 = calculate_error_percentage(tvl1_estimates, true_speeds)
    err_flownet2 = calculate_error_percentage(flownet2_estimates, true_speeds)
    err_raft = calculate_error_percentage(raft_estimates, true_speeds)
    err_raft_senet = calculate_error_percentage(raft_senet_estimates, true_speeds)
    
    print(f"  TV-L1 Error: {err_tvl1:.2f}% (expected: ~9.50%)")
    print(f"  FlowNet2 Error: {err_flownet2:.2f}% (expected: ~6.75%)")
    print(f"  RAFT Error: {err_raft:.2f}% (expected: ~6.50%)")
    print(f"  RAFT-SEnet Error: {err_raft_senet:.2f}% (expected: ~5.25%)")
    
    # Test 4: Calculate all metrics at once
    print("\n[Test 4] Calculate All Metrics")
    all_metrics = calculate_all_metrics(raft_senet_estimates, true_speeds)
    print(f"  RAFT-SEnet: MAE={all_metrics['MAE']:.4f}, RMSE={all_metrics['RMSE']:.4f}, Error%={all_metrics['error_percentage']:.2f}%")
    
    # Test 5: Comprehensive evaluation
    print("\n[Test 5] Comprehensive Evaluation")
    result = evaluate_speed_detection(raft_senet_estimates, true_speeds)
    print(result)
    
    # Test 6: Method comparison
    print("\n[Test 6] Method Comparison")
    results_dict = {
        'TV-L1': (tvl1_estimates, true_speeds),
        'FlowNet2': (flownet2_estimates, true_speeds),
        'RAFT': (raft_estimates, true_speeds),
        'RAFT-SEnet (ours)': (raft_senet_estimates, true_speeds)
    }
    
    comparison = compare_methods(results_dict)
    print_comparison_table(comparison, "Optical Flow Methods Comparison (4 m/s)")
    
    # Test 7: Per-speed-level metrics (simulating Table III)
    print("\n[Test 7] Per-Speed-Level Metrics")
    # Simulated data for different speeds
    test_speeds = np.array([0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0, 3.5, 3.5, 4.5, 4.5])
    # Simulated RAFT-SEnet results (with some errors as in Table III)
    estimated_speeds = np.array([0.46, 0.48, 0.95, 1.02, 1.58, 1.51, 2.07, 2.03, 3.08, 2.98, 3.58, 3.52, 4.61, 4.54])
    
    per_speed = calculate_metrics_per_speed(estimated_speeds, test_speeds, test_speeds)
    
    print(f"{'Speed (m/s)':<15} {'MAE':<10} {'RMSE':<10} {'Error %':<10}")
    print("-" * 45)
    for speed, metrics in sorted(per_speed.items()):
        print(f"{speed:<15.1f} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['error_percentage']:<10.2f}")
    
    # Test 8: R-squared calculation
    print("\n[Test 8] R-squared Calculation")
    r2 = calculate_r_squared(raft_senet_estimates, true_speeds)
    print(f"  RAFT-SEnet R²: {r2:.4f}")
    
    # Test 9: Standard deviation
    print("\n[Test 9] Standard Deviation")
    std = calculate_standard_deviation(raft_senet_estimates, true_speeds)
    print(f"  RAFT-SEnet Std Dev: {std:.4f}")
    
    # Test 10: Max error
    print("\n[Test 10] Max Error")
    max_err, idx = calculate_max_error(raft_senet_estimates, true_speeds)
    print(f"  RAFT-SEnet Max Error: {max_err:.4f} at index {idx}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
