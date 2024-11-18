# ks_test.py
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Tuple, Dict
from .config import GANConfig
from .models import Generator
from sklearn.preprocessing import MinMaxScaler
from .data_utils import DataProcessor
import torch

class KSTestEvaluator:
    def __init__(self, config: GANConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = DataProcessor(config)

    def load_generator(self) -> Generator:
        model_path = self.config.model_save_path / 'generator.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        generator = Generator(self.config)
        generator.load_state_dict(checkpoint['model_state_dict'])
        generator.to(self.device)
        generator.eval()
        return generator

    def generate_synthetic_data(self, generator: Generator) -> np.ndarray:
        total_samples = self.config.samples_per_day * self.config.days_to_generate
        noise = torch.randn(total_samples, self.config.input_dim).to(self.device)
        
        with torch.no_grad():
            synthetic_data = generator(noise).detach().cpu().numpy()
        
        synthetic_data_reshaped = synthetic_data.reshape(total_samples, -1)
        
        if synthetic_data_reshaped.shape[1] == self.config.feature_dim:
            additional_column = np.random.normal(size=(total_samples, 1))
            synthetic_data_reshaped = np.hstack((synthetic_data_reshaped, additional_column))
            
        return synthetic_data_reshaped

    def evaluate(self) -> Dict:
        generator = self.load_generator()
        train_set_1, train_set_2 = self.data_processor.load_and_preprocess_data()
        
        synthetic_data = self.generate_synthetic_data(generator)
        synthetic_df = pd.DataFrame(synthetic_data, columns=train_set_1.columns)
        
        # Save synthetic data
        self.config.synthetic_data_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_df.to_csv(self.config.synthetic_data_path, index=False)
        
        normalized_sets = self._normalize_datasets(train_set_1, train_set_2, synthetic_df)
        return self._compute_ks_results(*normalized_sets)

    def _normalize_datasets(self, train_set_1, train_set_2, synthetic_df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scaler = MinMaxScaler()
        sample_size = self.config.samples_per_day * 3
        
        subset_1 = train_set_1.iloc[:sample_size]
        subset_2 = train_set_1.iloc[sample_size:sample_size*2]
        subset_3 = synthetic_df
        
        return (
            pd.DataFrame(scaler.fit_transform(subset_1), columns=subset_1.columns),
            pd.DataFrame(scaler.fit_transform(subset_2), columns=subset_2.columns),
            pd.DataFrame(scaler.fit_transform(subset_3), columns=subset_3.columns)
        )

    def _compute_ks_results(self, df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> Dict:
        results = {
            'real_real': self._compute_ks_statistics(df1, df2),
            'real_synthetic_1': self._compute_ks_statistics(df1, df3),
            'real_synthetic_2': self._compute_ks_statistics(df2, df3)
        }
        
        # for test_name, df in results.items():
        #     print(f"\nK-S Test Results ({test_name}):")
        #     print(f"Passing sensors: {df['passes'].sum()}")
        #     print(df)
            
        return results

    def _compute_ks_statistics(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        results = []
        for column in df1.columns:
            ks_stat, p_value = ks_2samp(df1[column], df2[column])
            results.append({
                'sensor': column,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'passes': ks_stat < 0.15 and p_value > 0.03
            })
        return pd.DataFrame(results)