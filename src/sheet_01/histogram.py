from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / 'distances'
OUTPUT_PATH = BASE_DIR / 'plots'

class DistanceHistogramGenerator:
    def __init__(self):
        self.base_dirs = [
            DATA_PATH / 'hai-21.03',
            DATA_PATH / 'hai-22.04',
            DATA_PATH / 'haiend-23.05'
        ]
        self.train_distances = []
        self.test_distances = []

    def load_distances(self, directory: Path, file_type: str) -> list:
        """Load distances from CSV files, with validation checks."""
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}\n"
                "Please calculate distances first."
            )
        
        distance_files = list(directory.glob(f"*{file_type}*_distances.csv"))
        if not distance_files:
            raise FileNotFoundError(
                f"No {file_type} distance files found in: {directory}\n"
                "Please calculate distances first."
            )
            
        distances = []
        for file in distance_files:
            df = pd.read_csv(file)
            distances.extend(df['Distance'].tolist())
        return distances

    def load_all_distances(self):
        for dir_path in self.base_dirs:
            print(f"Loading {dir_path} distance dataset...")
            try:
                self.train_distances.extend(self.load_distances(dir_path, 'train'))
                self.test_distances.extend(self.load_distances(dir_path, 'test'))
            except FileNotFoundError as e:
                print(f"Warning: {str(e)}")
                continue

    def plot_histograms(self, save_path: Path = None):
        plt.figure(figsize=(14, 6))
        bins = np.linspace(0, 1, 50)

        # Calculate histograms to find max frequency
        train_hist, _ = np.histogram(self.train_distances, bins=bins)
        test_hist, _ = np.histogram(self.test_distances, bins=bins)
        max_freq = max(train_hist.max(), test_hist.max())

        # Train distances histogram
        if self.train_distances:
            plt.subplot(1, 2, 1)
            plt.hist(self.train_distances, bins=bins, color='blue', alpha=0.7, edgecolor='black')
            plt.xlabel("Euclidean Distance")
            plt.ylabel("Frequency")
            plt.title("Histogram of Euclidean Distances - Train Sets")
            plt.xlim(0, 1)
            plt.ylim(0, max_freq)

        # Test distances histogram
        if self.test_distances:
            plt.subplot(1, 2, 2)
            plt.hist(self.test_distances, bins=bins, color='green', alpha=0.7, edgecolor='black')
            plt.xlabel("Euclidean Distance")
            plt.ylabel("Frequency")
            plt.title("Histogram of Euclidean Distances - Test Sets")
            plt.xlim(0, 1) 
            plt.ylim(0, max_freq)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Histogram saved to {save_path}")
        else:
            plt.show()

    def generate(self, save_path: Path = None):
        """Generate histograms and save to file."""
        self.load_all_distances()
        if not self.train_distances and not self.test_distances:
            raise ValueError("No distance data found in any of the directories.")
            
        self.plot_histograms(save_path)