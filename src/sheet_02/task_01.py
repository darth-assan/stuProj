import sys
import numpy as np
import pandas as pd
import argparse
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from loguru import logger
from typing import Tuple, Union, List

# Configure Loguru
logger.remove()
logger.add(sys.stdout, level="INFO", filter=lambda record: record["level"].name == "INFO")

class SyntheticDataGenerator:
    """A class to generate synthetic data using nearest neighbor interpolation."""
    
    def __init__(self, normalization_method: str = 'min_max'):
        """
        Initialize the SyntheticDataGenerator.
        
        Args:
            normalization_method: The method to use for data normalization ('min_max' or 'z_score')
        """
        self.normalization_method = normalization_method
        self.scaler = MinMaxScaler() if normalization_method == 'min_max' else StandardScaler()
        self.original_data = None
        self.normalized_data = None
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading dataset from: {filepath}")
        self.original_data = pd.read_csv(filepath)
        return self.original_data
    
    def normalize_data(self) -> np.ndarray:
        """
        Normalize the loaded dataset.
        
        Returns:
            Normalized data as numpy array
        """
        if self.original_data is None:
            raise ValueError("No data loaded. Please call load_dataset first.")
            
        logger.info(f"Normalizing data using '{self.normalization_method}' method")
        self.normalized_data = self.scaler.fit_transform(self.original_data)
        return self.normalized_data
    
    @staticmethod
    def get_decimal_precision(series: pd.Series) -> int:
        """
        Determine decimal precision for a pandas Series.
        
        Args:
            series: Input pandas Series
            
        Returns:
            Maximum decimal precision found in the series
        """
        precisions = series.astype(str).apply(lambda x: len(x.split('.')[-1]) if '.' in x else 0)
        return precisions.max()
    
    def inverse_normalize(self, synthetic_data: np.ndarray) -> pd.DataFrame:
        """
        Convert normalized synthetic data back to original scale.
        
        Args:
            synthetic_data: Normalized synthetic data
            
        Returns:
            DataFrame with synthetic data in original scale
        """
        logger.info("Inverse transforming the normalized data to original scale")
        inverse_transformed = self.scaler.inverse_transform(synthetic_data)
        synthetic_df = pd.DataFrame(inverse_transformed, columns=self.original_data.columns)
        
        # Preserve original precision
        for col in self.original_data.columns:
            precision = self.get_decimal_precision(self.original_data[col])
            synthetic_df[col] = synthetic_df[col].round(precision)
        
        return synthetic_df
    
    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute Manhattan distance between two arrays.
        
        Args:
            a: Single data point
            b: Dataset to compute distance from
            
        Returns:
            Array of distances
        """
        return np.sum(np.abs(a - b), axis=1)
    
    def find_k_nearest_neighbors(self, sample: np.ndarray, k: int) -> np.ndarray:
        """
        Find k nearest neighbors using Manhattan distance.
        
        Args:
            sample: Single data point
            k: Number of neighbors to find
            
        Returns:
            Array of k nearest neighbors
        """
        if self.normalized_data is None:
            raise ValueError("No normalized data available. Please normalize data first.")
            
        distances = self.manhattan_distance(sample, self.normalized_data)
        nearest_indices = np.argsort(distances)[:k]
        return self.normalized_data[nearest_indices]
    
    @staticmethod
    def generate_synthetic_sample(sample: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        """
        Generate synthetic sample through interpolation.
        
        Args:
            sample: Original sample
            neighbors: K nearest neighbors
            
        Returns:
            Generated synthetic sample
        """
        random_neighbor = neighbors[np.random.randint(0, len(neighbors))]
        diff = random_neighbor - sample
        return sample + np.random.random(sample.shape) * diff
    
    def generate_samples(self, sampling_percentage: float, k: int) -> pd.DataFrame:
        """
        Generate synthetic samples using custom oversampling.
        
        Args:
            sampling_percentage: Percentage of samples to generate (0-100)
            k: Number of nearest neighbors to consider
            
        Returns:
            DataFrame containing synthetic samples in original scale
        """
        if self.normalized_data is None:
            raise ValueError("No normalized data available. Please normalize data first.")
            
        sampling_strategy = sampling_percentage / 100
        num_samples = int(len(self.normalized_data) * sampling_strategy)
        logger.info(f"Generating {num_samples} synthetic samples ({sampling_percentage}%)")
        
        def create_synthetic_sample(i):
            sample = self.normalized_data[i]
            neighbors = self.find_k_nearest_neighbors(sample, k)
            return self.generate_synthetic_sample(sample, neighbors)
        
        logger.info("Generating synthetic samples in parallel")
        synthetic_samples = Parallel(n_jobs=-1)(
            delayed(create_synthetic_sample)(i) for i in range(num_samples)
        )
        synthetic_array = np.array(synthetic_samples)
        
        return self.inverse_normalize(synthetic_array)
    
    def save_synthetic_data(self, synthetic_data: pd.DataFrame, filename: str = "synthetic_data.csv"):
        """
        Save synthetic data to CSV.
        
        Args:
            synthetic_data: DataFrame of synthetic samples
            filename: Output filename
        """
        synthetic_data.to_csv(filename, index=False)
        logger.info(f"Synthetic data saved to {filename}")