from dataclasses import dataclass
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / 'data' / 'gan'
OUTPUT_PATH = BASE_DIR / 'results'

@dataclass
class GANConfig:
    # Model parameters
    feature_dim: int = 32
    input_dim: int = 100
    dropout_rate: float = 0.3
    
    # Training parameters
    batch_size: int = 16
    num_epochs: int = 2
    learning_rate_g: float = 0.0002
    learning_rate_d: float = 0.0001
    
    # Data generation parameters
    samples_per_day: int = 300
    days_to_generate: int = 3
    
    # Paths
    train_set_1_path: Path = DATA_PATH / 'train1_clean.csv'
    train_set_2_path: Path = DATA_PATH / 'train4_clean.csv'
    model_save_path: Path = BASE_DIR / 'saved_model'
    synthetic_data_path: Path = DATA_PATH / 'synthetic_data.csv'
    ks_summary_path: Path = OUTPUT_PATH / 'ks_summary.csv'