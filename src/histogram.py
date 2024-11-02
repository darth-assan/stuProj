import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory paths
DISTANCES_DIR = os.path.expanduser("G:/stuProj/data/distances")

def load_distances():
    # ... existing load_distances function remains the same ...
    train_distances = []
    test_distances = []

    for dataset_version in os.listdir(DISTANCES_DIR):
        version_path = os.path.join(DISTANCES_DIR, dataset_version)
        
        if not os.path.isdir(version_path):
            continue
            
        for file in os.listdir(version_path):
            file_path = os.path.join(version_path, file)
            try:
                if "train" in file.lower():
                    distances = pd.read_csv(file_path)["Euclidean_Distance"].tolist()
                    train_distances.extend(distances)
                    print(f"Loaded train data from: {file_path}")
                elif "test" in file.lower():
                    distances = pd.read_csv(file_path)["Euclidean_Distance"].tolist()
                    test_distances.extend(distances)
                    print(f"Loaded test data from: {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
    
    return train_distances, test_distances

def plot_histograms_matplotlib(train_distances, test_distances, num_bins=50):
    """
    Create and save histograms using matplotlib
    """
    # Set style to a built-in style
    plt.style.use('ggplot')  # Alternative options: 'classic', 'default', 'bmh', 'fivethirtyeight'
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot train data histogram
    if train_distances:
        ax1.hist(train_distances, bins=num_bins, color='blue', alpha=0.7)
        ax1.set_title('Train Set Euclidean Distance Histogram')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No train distances found', 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot test data histogram
    if test_distances:
        ax2.hist(test_distances, bins=num_bins, color='green', alpha=0.7)
        ax2.set_title('Test Set Euclidean Distance Histogram')
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No test distances found', 
                horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.expanduser("G:/stuProj/results")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'distance_histograms.png'), dpi=300, bbox_inches='tight')
    print(f"Histograms saved to: {os.path.join(output_dir, 'distance_histograms.png')}")
    
    # Display the plot
    plt.show()

def main():
    # Load data
    train_distances, test_distances = load_distances()
    
    # Create and save matplotlib visualizations
    plot_histograms_matplotlib(train_distances, test_distances)

if __name__ == "__main__":
    main()
