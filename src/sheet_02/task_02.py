import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from loguru import logger
import sys

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / 'data' / 'oversampling'
OUTPUT_PATH = BASE_DIR / 'plots'

# Configure Loguru
logger.remove()
logger.add(sys.stdout, level="INFO", filter=lambda record: record["level"].name == "INFO")

class DatasetAnalyzer:
    def __init__(self, real_data_path: Union[str, Path], synthetic_data_paths: List[Union[str, Path]],
                 output_path: Union[str, Path] = Path("../plots")):
        """
        Initialize analyzer with paths to real and synthetic datasets
        
        Args:
            real_data_path: Path to the real dataset CSV file
            synthetic_data_paths: List of paths to synthetic dataset CSV files
            output_path: Path to save output plots
            
        Raises:
            FileNotFoundError: If any dataset file is not found
            ValueError: If datasets are empty or have mismatching columns
        """
        self.real_data_path = Path(real_data_path)
        self.synthetic_data_paths = [Path(path) for path in synthetic_data_paths]
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and validate datasets
        self._load_and_validate_datasets()
        
        # Cache for intermediate results
        self._cache = {}
        
    def _load_and_validate_datasets(self) -> None:
        """
        Load and validate all datasets
        
        Raises:
            FileNotFoundError: If any dataset file is not found
            ValueError: If datasets are empty or have mismatching columns
        """
        # Validate paths exist
        if not self.real_data_path.exists():
            raise FileNotFoundError(f"Real dataset not found at {self.real_data_path}")
            
        for path in self.synthetic_data_paths:
            if not path.exists():
                raise FileNotFoundError(f"Synthetic dataset not found at {path}")
        
        # Load datasets with proper error handling
        try:
            self.real_data = pd.read_csv(self.real_data_path)
            self.synthetic_datasets = [pd.read_csv(path) for path in self.synthetic_data_paths]
            
            # Basic validation
            if self.real_data.empty:
                raise ValueError("Real dataset is empty")
            
            for i, df in enumerate(self.synthetic_datasets):
                if df.empty:
                    raise ValueError(f"Synthetic dataset {i} is empty")
                
            # Validate columns match
            real_cols = set(self.real_data.columns)
            for i, df in enumerate(self.synthetic_datasets):
                if set(df.columns) != real_cols:
                    raise ValueError(f"Columns in synthetic dataset {i} do not match real dataset")
                    
        except pd.errors.EmptyDataError:
            raise ValueError("One or more datasets are empty")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV files - please check file format")
            
    def _validate_column_data(self, column_name: str, values: np.ndarray) -> None:
        """
        Validate column data for analysis
        
        Args:
            column_name: Name of the column
            values: Array of values to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if len(values) == 0:
            raise ValueError(f"Column {column_name} contains no valid values")
            
        if np.all(np.isnan(values)):
            raise ValueError(f"Column {column_name} contains only NaN values")
            
        if np.any(np.isinf(values)):
            raise ValueError(f"Column {column_name} contains infinite values")
            
    def _calculate_cdf(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Cumulative Distribution Function
        
        Args:
            values: Array of values
            
        Returns:
            Tuple of sorted values and their cumulative probabilities
        """
        sorted_vals = np.sort(values)
        probs = np.linspace(0, 1, len(sorted_vals))
        return sorted_vals, probs
        
    def _calculate_ccdf(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Complementary Cumulative Distribution Function
        
        Args:
            values: Array of values
            
        Returns:
            Tuple of sorted values and their complementary cumulative probabilities
        """
        sorted_vals = np.sort(values)
        probs = 1 - np.linspace(0, 1, len(sorted_vals))
        return sorted_vals, probs

    def _minmax_stat(self, real: np.ndarray, synthetic_list: List[np.ndarray]) -> Dict:
        """
        Compute MinMax statistic with error margins, handling zero and negative values
        
        Args:
            real: Real dataset values
            synthetic_list: List of synthetic dataset values
            
        Returns:
            Dictionary containing violations and thresholds
        """
        min_real, max_real = np.min(real), np.max(real)
        range_real = max_real - min_real
        
        # Handle zero/negative values for range calculation
        if range_real == 0:
            # If all values are the same, use a small percentage of the absolute value
            range_real = abs(min_real) * 0.01 if min_real != 0 else 0.01
            
        min_err = min_real - (range_real / 2)
        max_err = max_real + (range_real / 2)
        
        violations = []
        violation_indices = []
        for synthetic in synthetic_list:
            outside_range = (synthetic < min_err) | (synthetic > max_err)
            violation_rate = float(np.sum(outside_range)) / len(synthetic)
            violations.append(violation_rate)
            violation_indices.append(np.where(outside_range)[0])
            
        return {
            'violations': violations,
            'violation_indices': violation_indices,
            'thresholds': (min_err, max_err)
        }
    
    def _gradient_stat(self, real: np.ndarray, synthetic_list: List[np.ndarray]) -> Dict:
        """
        Compute Gradient statistic with error margins
        
        Args:
            real: Real dataset values
            synthetic_list: List of synthetic dataset values
            
        Returns:
            Dictionary containing violations and thresholds
        """
        real_grad = np.diff(real)
        min_grad = np.min(real_grad)
        max_grad = np.max(real_grad)
        grad_range = max_grad - min_grad
        
        # Handle zero/negative gradients
        if grad_range == 0:
            grad_range = abs(min_grad) * 0.01 if min_grad != 0 else 0.01
            
        min_err = min_grad - (grad_range / 2)
        max_err = (max_grad + grad_range) / 2  # Corrected as per requirements
        
        violations = []
        violation_indices = []
        for synthetic in synthetic_list:
            synthetic_grad = np.diff(synthetic)
            outside_range = (synthetic_grad < min_err) | (synthetic_grad > max_err)
            violation_rate = float(np.sum(outside_range)) / len(synthetic_grad)
            violations.append(violation_rate)
            violation_indices.append(np.where(outside_range)[0])
            
        return {
            'violations': violations,
            'violation_indices': violation_indices,
            'thresholds': (min_err, max_err)
        }
    
    def _steadytime_stat(self, real: np.ndarray, synthetic_list: List[np.ndarray]) -> Optional[Dict]:
        """
        Compute Steadytime statistic for process values with <= 50 distinct values
        
        Args:
            real: Real dataset values
            synthetic_list: List of synthetic dataset values
            
        Returns:
            Dictionary containing violations and thresholds, or None if not applicable
        """
        def get_steady_durations(values: np.ndarray) -> List[int]:
            durations = []
            current_duration = 1
            
            for i in range(1, len(values)):
                if values[i] == values[i-1]:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_duration = 1
            durations.append(current_duration)
            return durations
        
        # Check if applicable (<=50 distinct values)
        if len(np.unique(real)) > 50:
            return None
            
        real_durations = get_steady_durations(real)
        max_real_duration = max(real_durations)
        
        # Handle edge case where max duration is 1
        if max_real_duration == 1:
            return None
            
        range_duration = max_real_duration
        min_err = max_real_duration - (range_duration / 2)
        max_err = max_real_duration + (range_duration / 2)
        
        violations = []
        violation_indices = []
        for synthetic in synthetic_list:
            if len(np.unique(synthetic)) > 50:
                violations.append(1.0)
                violation_indices.append([])
                continue
                
            synthetic_durations = get_steady_durations(synthetic)
            max_synthetic_duration = max(synthetic_durations)
            
            violation = 1.0 if (max_synthetic_duration < min_err or 
                              max_synthetic_duration > max_err) else 0.0
            violations.append(violation)
            
            # Find indices where violations occur
            violation_idx = []
            current_idx = 0
            for duration in synthetic_durations:
                if duration < min_err or duration > max_err:
                    violation_idx.extend(range(current_idx, current_idx + duration))
                current_idx += duration
            violation_indices.append(violation_idx)
            
        return {
            'violations': violations,
            'violation_indices': violation_indices,
            'thresholds': (min_err, max_err)
        }
    
    def _entropy_stat(self, real: np.ndarray, synthetic_list: List[np.ndarray]) -> Dict:
        """
        Compute Shannon entropy for all datasets
        
        Args:
            real: Real dataset values
            synthetic_list: List of synthetic dataset values
            
        Returns:
            Dictionary containing entropy values
        """
        def compute_entropy(values: np.ndarray) -> float:
            # Use Freedman-Diaconis rule for bin width
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(values) ** (1/3)) if iqr > 0 else 0.1
            bin_count = int(np.ceil((np.max(values) - np.min(values)) / bin_width))
            bin_count = max(min(bin_count, 100), 10)  # Limit bins between 10 and 100
            
            hist, _ = np.histogram(values, bins=bin_count, density=True)
            # Add small epsilon to avoid log(0)
            hist = hist + np.finfo(float).eps
            return float(entropy(hist))
        
        real_entropy = compute_entropy(real)
        synthetic_entropies = [compute_entropy(synthetic) for synthetic in synthetic_list]
        
        return {
            'real_entropy': real_entropy,
            'synthetic_entropies': synthetic_entropies
        }
    
    def analyze_column(self, column_name: str) -> Dict:
        """
        Analyze a single sensor/actuator column across all datasets
        
        Args:
            column_name: Name of column to analyze
            
        Returns:
            Dictionary containing analysis results for different metrics
            
        Raises:
            ValueError: If column data is invalid
        """
        if column_name not in self.real_data.columns:
            raise ValueError(f"Column {column_name} not found in datasets")
            
        # Get clean data
        real_values = self.real_data[column_name].dropna().values
        synthetic_values = [df[column_name].dropna().values for df in self.synthetic_datasets]
        
        # Validate data
        self._validate_column_data(column_name, real_values)
        for i, values in enumerate(synthetic_values):
            self._validate_column_data(f"{column_name} (synthetic {i+1})", values)
        
        # Compute statistics
        return {
            'minmax': self._minmax_stat(real_values, synthetic_values),
            'gradient': self._gradient_stat(real_values, synthetic_values),
            'steadytime': self._steadytime_stat(real_values, synthetic_values),
            'entropy': self._entropy_stat(real_values, synthetic_values)
        }
    
    def plot_combined_statistics(self, column_name: str, stats: Dict) -> None:
        """
        Create combined plot for all statistics
        
        Args:
            column_name: Name of the column being analyzed
            stats: Dictionary containing statistics
        """
        plt.style.use('ggplot')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Statistical Analysis for {column_name}', fontsize=16)
        
        # Plot Entropy CDF
        entropy_stats = stats['entropy']
        real_entropy = entropy_stats['real_entropy']
        synthetic_entropies = entropy_stats['synthetic_entropies']
        
        # Create CDF for entropy
        all_entropies = [real_entropy] + synthetic_entropies
        sorted_entropies = np.sort(all_entropies)
        probs = np.linspace(0, 1, len(sorted_entropies))
        
        ax1.plot(sorted_entropies, probs, 'k-', label='CDF')
        ax1.plot(real_entropy, 0.5, 'ro', label='Real Data')
        for i, entropy in enumerate(synthetic_entropies):
            ax1.plot(entropy, 0.5, 'bo', label=f'Synthetic {i+1}')
        ax1.set_xlabel('Entropy')
        ax1.set_ylabel('CDF')
        ax1.set_title('Entropy Distribution')
        ax1.legend()
        
        # Plot MinMax CCDF
        minmax_stats = stats['minmax']
        violations = minmax_stats['violations']
        ax2.plot([0, 1], [1, 0], 'k-', label='CCDF')
        for i, violation in enumerate(violations):
            ax2.plot(violation, 1-0.5, f'C{i}o', label=f'Synthetic {i+1}')
        ax2.set_xlabel('MinMax Violations')
        ax2.set_ylabel('CCDF')
        ax2.set_title('MinMax Violations Distribution')
        ax2.legend()

        # Plot Gradient CCDF
        gradient_stats = stats['gradient']
        violations = gradient_stats['violations']
        ax3.plot([0, 1], [1, 0], 'k-', label='CCDF')
        for i, violation in enumerate(violations):
            ax3.plot(violation, 1-0.5, f'C{i}o', label=f'Synthetic {i+1}')
        ax3.set_xlabel('Gradient Violations')
        ax3.set_ylabel('CCDF')
        ax3.set_title('Gradient Violations Distribution')
        ax3.legend()
        
        # Plot Steadytime CCDF if available
        steadytime_stats = stats['steadytime']
        if steadytime_stats is not None:
            violations = steadytime_stats['violations']
            ax4.plot([0, 1], [1, 0], 'k-', label='CCDF')
            for i, violation in enumerate(violations):
                ax4.plot(violation, 1-0.5, f'C{i}o', label=f'Synthetic {i+1}')
            ax4.set_xlabel('Steadytime Violations')
            ax4.set_ylabel('CCDF')
            ax4.set_title('Steadytime Violations Distribution')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Steadytime not applicable\n(>50 distinct values)',
                    ha='center', va='center')
            ax4.set_title('Steadytime Analysis')
        
        plt.tight_layout()
        plt.savefig(self.output_path / f'{column_name}_combined_stats.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self, column_name: str, stats: Dict) -> str:
        """
        Generate a text summary of the analysis results
        
        Args:
            column_name: Name of the column being analyzed
            stats: Dictionary containing statistics
            
        Returns:
            String containing the summary report
        """
        report = [f"Analysis Summary for {column_name}\n{'='*50}\n"]
        
        # Entropy Summary
        entropy_stats = stats['entropy']
        report.append("Entropy Analysis:")
        report.append(f"Real Data Entropy: {entropy_stats['real_entropy']:.4f}")
        for i, entropy in enumerate(entropy_stats['synthetic_entropies']):
            report.append(f"Synthetic {i+1} Entropy: {entropy:.4f}")
        report.append("")
        
        # MinMax Summary
        minmax_stats = stats['minmax']
        report.append("MinMax Analysis:")
        report.append(f"Thresholds: {minmax_stats['thresholds'][0]:.4f} to {minmax_stats['thresholds'][1]:.4f}")
        for i, violation in enumerate(minmax_stats['violations']):
            report.append(f"Synthetic {i+1} Violation Rate: {violation*100:.2f}%")
        report.append("")
        
        # Gradient Summary
        gradient_stats = stats['gradient']
        report.append("Gradient Analysis:")
        report.append(f"Thresholds: {gradient_stats['thresholds'][0]:.4f} to {gradient_stats['thresholds'][1]:.4f}")
        for i, violation in enumerate(gradient_stats['violations']):
            report.append(f"Synthetic {i+1} Violation Rate: {violation*100:.2f}%")
        report.append("")
        
        # Steadytime Summary
        steadytime_stats = stats['steadytime']
        if steadytime_stats is not None:
            report.append("Steadytime Analysis:")
            report.append(f"Thresholds: {steadytime_stats['thresholds'][0]:.4f} to {steadytime_stats['thresholds'][1]:.4f}")
            for i, violation in enumerate(steadytime_stats['violations']):
                report.append(f"Synthetic {i+1} Violation Rate: {violation*100:.2f}%")
        else:
            report.append("Steadytime Analysis: Not applicable (>50 distinct values)")
            
        return "\n".join(report)

    def analyze_all_columns(self) -> None:
        """
        Analyze all columns and generate plots and reports
        """
        logger.info("Starting analysis of all columns...")
        
        # Create report directory
        report_dir = self.output_path / 'reports'
        report_dir.mkdir(exist_ok=True)
        
        for column in self.real_data.columns:
            try:
                logger.info(f"Analyzing column: {column}")
                
                # Perform analysis
                stats = self.analyze_column(column)
                
                # Generate plots
                self.plot_combined_statistics(column, stats)
                
                # Generate and save report
                report = self.generate_summary_report(column, stats)
                with open(report_dir / f'{column}_report.txt', 'w') as f:
                    f.write(report)
                
                logger.info(f"Completed analysis of column: {column}")
                
            except Exception as e:
                logger.error(f"Error analyzing column {column}: {str(e)}")
                continue