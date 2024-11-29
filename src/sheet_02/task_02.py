import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

class DatasetAnalyzer:
    def __init__(self, real_data_path: Union[str, Path],
                 synthetic_data_paths: List[Union[str, Path]],
                 output_path: Union[str, Path]):
        self.real_data_path = Path(real_data_path)
        self.synthetic_data_paths = [Path(path) for path in synthetic_data_paths]
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.real_data = pd.read_csv(self.real_data_path)
        self.synthetic_datasets = [pd.read_csv(path) for path in self.synthetic_data_paths]

    def _compute_error_margins(self, values: np.ndarray) -> Tuple[float, float]:
        min_val, max_val = np.min(values), np.max(values)
        range_val = max_val - min_val
        if range_val == 0:
            range_val = abs(min_val) * 0.1 if min_val != 0 else 0.1
        min_err = min_val - (range_val / 2 )
        max_err = max_val + (range_val / 2)
        return min_err, max_err

    def _compute_statistics(self) -> Dict:
        minmax_violations = []
        gradient_violations = []
        steadytime_violations = []
        real_entropies = []
        synthetic_entropies = []

        for column in self.real_data.columns:
            real_values = self.real_data[column].dropna().values
            synthetic_values = [df[column].dropna().values for df in self.synthetic_datasets]

            # MinMax statistic
            min_err, max_err = self._compute_error_margins(real_values)
            column_minmax_violations = []
            for synthetic in synthetic_values:
                violations = np.sum((synthetic < min_err) | (synthetic > max_err))
                violation_rate = violations / len(synthetic)
                column_minmax_violations.append(float(violation_rate))
            minmax_violations.append(column_minmax_violations)

            # Gradient statistic
            real_gradients = np.diff(real_values)
            grad_min_err, grad_max_err = self._compute_error_margins(real_gradients)
            column_gradient_violations = []
            for synthetic in synthetic_values:
                violations = np.sum((np.diff(synthetic) < grad_min_err) | 
                                  (np.diff(synthetic) > grad_max_err))
                violation_rate = violations / len(np.diff(synthetic))
                column_gradient_violations.append(float(violation_rate))
            gradient_violations.append(column_gradient_violations)

            # Steadytime statistic
            if len(np.unique(real_values)) <= 50:
                def get_steady_durations(values):
                    durations = []
                    current_duration = 1
                    for i in range(1, len(values)):
                        if values[i] == values[i-1]:
                            current_duration += 1
                        else:
                            durations.append(current_duration)
                            current_duration = 1
                    durations.append(current_duration)
                    return np.array(durations)

                real_steady_durations = get_steady_durations(real_values)
                steady_min_err, steady_max_err = self._compute_error_margins(real_steady_durations)
                column_steadytime_violations = []
                for synthetic in synthetic_values:
                    if len(np.unique(synthetic)) > 50:
                        column_steadytime_violations.append(1.0)
                    else:
                        synth_durations = get_steady_durations(synthetic)
                        violations = np.sum((synth_durations < steady_min_err) | 
                                         (synth_durations > steady_max_err))
                        violation_rate = violations / len(synth_durations)
                        column_steadytime_violations.append(float(violation_rate))
            else:
                column_steadytime_violations = None
            steadytime_violations.append(column_steadytime_violations)

            # Entropy statistic
            def compute_entropy(values):
                hist, _ = np.histogram(values, bins=min(max(10, len(values)//10), 100), 
                                     density=True)
                hist = hist + np.finfo(float).eps
                return float(entropy(hist))

            real_entropies.append(compute_entropy(real_values))
            synthetic_entropy = [compute_entropy(synthetic) for synthetic in synthetic_values]
            synthetic_entropies.append(synthetic_entropy)

        return {
            'minmax_violations': minmax_violations,
            'gradient_violations': gradient_violations,
            'steadytime_violations': steadytime_violations,
            'real_entropies': real_entropies,
            'synthetic_entropies': synthetic_entropies
        }

    def plot_analysis(self, stats: Dict):
        plt.figure(figsize=(15, 10))
        
        # Entropy CDF
        plt.subplot(2, 2, 1)
        real_entropies = np.sort(stats['real_entropies'])
        plt.step(real_entropies, np.linspace(0, 1, len(real_entropies)), 
                'r-', label='Real', where='post')
        
        for i in range(len(stats['synthetic_entropies'][0])):
            synth_entropies = np.sort([e[i] for e in stats['synthetic_entropies']])
            plt.step(synth_entropies, np.linspace(0, 1, len(synth_entropies)), 
                    label=f'Synthetic {i+1}', where='post')
        
        plt.title('Entropy CDF')
        plt.xlabel('Entropy')
        plt.ylabel('Cumulative Probability')
        plt.legend()

        # Violation CCDFs
        violation_types = [
            ('MinMax Violations', stats['minmax_violations']),
            ('Gradient Violations', stats['gradient_violations']),
            ('Steadytime Violations', stats['steadytime_violations'])
        ]

        for i, (title, violations) in enumerate(violation_types, start=2):
            plt.subplot(2, 2, i)
            
            for j in range(len(stats['synthetic_entropies'][0])):
                valid_violations = [v[j] for v in violations if v is not None and v[j] is not None]
                if valid_violations:
                    sorted_violations = np.sort(valid_violations)
                    ccdf = 1 - np.linspace(0, 1, len(sorted_violations))
                    plt.plot(sorted_violations, ccdf, label=f'Synthetic {j+1}')
            
            plt.title(title)
            plt.xlabel('Violation Rate')
            plt.ylabel('Complementary CDF')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'dataset_analysis.png')
        plt.close()

    def analyze_datasets(self):
        stats = self._compute_statistics()
        self.plot_analysis(stats)