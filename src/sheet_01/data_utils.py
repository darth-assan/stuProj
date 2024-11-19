import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
from .config import GANConfig
import os
import sys

class DataProcessor:
    def __init__(self, config: GANConfig):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_set_1 = pd.read_csv(self.config.train_set_1_path)
        train_set_2 = pd.read_csv(self.config.train_set_2_path)
        common_columns = train_set_1.columns.intersection(train_set_2.columns)
        return train_set_1[common_columns], train_set_2[common_columns]

    def normalize_data(self, data: pd.DataFrame, sample_size: Optional[int] = 100) -> torch.Tensor:
        data_subset = data[:sample_size].values if sample_size else data.values
        normalized_data = self.scaler.fit_transform(data_subset)
        return torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(1)

    def create_dataloader(self, data: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def check_directory_exists(directory: str):
        if not os.path.isdir(directory):
            error_message = f"Error: Directory '{directory}' does not exist."
            print(error_message, file=sys.stderr)
            sys.exit(1)

    def check_file_exists(file_path: str):
        if not os.path.isfile(file_path):
            error_message = f"Error: File '{file_path}' does not exist."
            print(error_message, file=sys.stderr)
            sys.exit(1)