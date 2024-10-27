import os
import re
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import logging

class DatasetProcessor:
    """A class to process HAI datasets and calculate Euclidean distances."""
    
    def __init__(self, base_dir: str, output_dir: str):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def _get_physical_columns(df: pd.DataFrame) -> List[str]:
        return [
            col for col in df.columns
            if not re.search(r'(time|timestamp|attack|Attack)', col, re.IGNORECASE)
        ]
    
    @staticmethod
    def _get_attack_columns(df: pd.DataFrame) -> List[str]:
        return [
            col for col in df.columns
            if re.search(r'attack', col, re.IGNORECASE)
        ]
    
    def filter_attack_data(self, df: pd.DataFrame) -> pd.DataFrame:
        attack_columns = self._get_attack_columns(df)
        if attack_columns:
            return df[(df[attack_columns] == 0).all(axis=1)].copy()
        return df.copy()
    
    def filter_non_numeric_and_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        physical_columns = self._get_physical_columns(df)
        df[physical_columns] = df[physical_columns].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=physical_columns).copy()
        
        Q1 = df[physical_columns].quantile(0.25)
        Q3 = df[physical_columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return df[~((df[physical_columns] < lower_bound) | 
                   (df[physical_columns] > upper_bound)).any(axis=1)]
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        physical_columns = self._get_physical_columns(df)
        scaler = MinMaxScaler()
        df[physical_columns] = scaler.fit_transform(df[physical_columns])
        return df
    
    def calculate_distances(self, df: pd.DataFrame) -> np.ndarray:
        physical_columns = self._get_physical_columns(df)
        data = df[physical_columns].values
        differences = np.diff(data, axis=0)
        return np.sqrt(np.sum(differences ** 2, axis=1))
    
    def process_file(self, file_path: Path) -> None:
        try:
            self.logger.info(f"Processing file: {file_path}")
            df = pd.read_csv(file_path)
            df = self.filter_attack_data(df)
            df = self.filter_non_numeric_and_outliers(df)
            df = self.normalize_data(df)
            distances = self.calculate_distances(df)
            
            dataset_version = file_path.parent.name
            output_dir = self.output_dir / dataset_version
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{file_path.name}_distances.csv"
            pd.DataFrame(distances, columns=["Euclidean_Distance"]).to_csv(
                output_file, index=False
            )
            
            self.logger.info(f"Saved distances to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
    
    def process_datasets(self) -> None:
        for file_path in self.base_dir.rglob("*"):
            if file_path.is_file() and re.search(r'(test|train)', file_path.name, re.IGNORECASE):
                self.process_file(file_path)
