import argparse
from processor import DatasetProcessor
from histogram import load_distances, plot_histograms_matplotlib

def main():
    parser = argparse.ArgumentParser(description="Process datasets and generate histograms.")
    parser.add_argument('--all', action='store_true', help="Generate distances for all datasets.")
    parser.add_argument('--histogram', action='store_true', help="Generate histogram plots.")
    
    args = parser.parse_args()
    
    BASE_DIR = "G:/stuProj/data/original"
    OUTPUT_DIR = "G:/stuProj/data/distances"
    
    if args.all:
        processor = DatasetProcessor(BASE_DIR, OUTPUT_DIR)
        processor.process_datasets()
    
    if args.histogram:
        train_distances, test_distances = load_distances()
        plot_histograms_matplotlib(train_distances, test_distances)

if __name__ == "__main__":
    main()
