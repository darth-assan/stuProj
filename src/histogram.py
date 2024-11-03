import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define base directories for each dataset version
base_dirs = [
    '/Users/darth/Dev/stuProj/data/Processed/hai-21.03',
    '/Users/darth/Dev/stuProj/data/Processed/hai-22.04',
    '/Users/darth/Dev/stuProj/data/Processed/haiend-23.05'
]

# Function to load distance data from CSV files in a given directory
def load_distances(directory, file_type):
    distances = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith("_distances.csv") and file_type in file:
                file_path = os.path.join(directory, file)
                df = pd.read_csv(file_path)
                distances.extend(df['Distance'].tolist())
    return distances

# Load distances for all train and test datasets
def load_all_distances(base_dirs):
    train_distances = []
    test_distances = []
    for dir_path in base_dirs:
        print(f"Loading {dir_path} distance dataset...")
        train_distances.extend(load_distances(dir_path, 'train'))
        test_distances.extend(load_distances(dir_path, 'test'))
    return train_distances, test_distances

# Plot histograms for train and test distances using numpy and matplotlib
def plot_histograms(train_distances, test_distances, save_path=None):
    plt.figure(figsize=(14, 6))

    # Train distances histogram
    if train_distances:
        plt.subplot(1, 2, 1)
        plt.hist(train_distances, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Frequency")
        plt.title("Histogram of Euclidean Distances - Train Sets")
    else:
        print("No train distances found. Please check directories.")

    # Test distances histogram
    if test_distances:
        plt.subplot(1, 2, 2)
        plt.hist(test_distances, bins=50, color='green', alpha=0.7, edgecolor='black')
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Frequency")
        plt.title("Histogram of Euclidean Distances - Test Sets")
    else:
        print("No test distances found. Please check directories.")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()

# Main function to load distances and plot histograms
def main():
    try:
        train_distances, test_distances = load_all_distances(base_dirs)
        print(f"Generating histograms...")
        plot_histograms(train_distances, test_distances, save_path='/Users/darth/Dev/stuProj/data/Results/histogram.png')
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()