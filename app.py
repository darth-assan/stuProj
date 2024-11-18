import argparse
from pathlib import Path
from src.sheet_02.task_01 import SyntheticDataGenerator
from src.sheet_02.task_02 import DatasetAnalyzer
from src.sheet_01.processor import DataPreprocessor
from src.sheet_01.histogram import DistanceHistogramGenerator
from loguru import logger
import sys

# Configure Loguru
logger.remove()
logger.add(sys.stdout, level="INFO", filter=lambda record: record["level"].name == "INFO")

# Define paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data'
OUTPUT_PATH = BASE_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Industrial Process Data Analysis Toolbox
        ======================================
        This toolbox provides various functionalities for generating and analyzing ICS synthetically genrated data:
        1. Physical Readings Analysis (from Sheet 1)
        2. GAN-based Data Generation (from Sheet 1)
        3. Oversampling method for Synthetic Data Generation (from Sheet 2)
        4. Synthetic Data Analysis
        
        Use -s <keyword> to select the operation mode.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-s', '--start', type=str, required=True,
                       choices=['distance', 'gan', 'oversample', 'analyze', 'histogram'],
                       help="""Operation mode selection:
                       'distance': Analyze physical sensor readings (Sheet 1, Task 1)
                       'gan': Generate synthetic data using GAN (Sheet 1, Task 2)
                       'oversample': Generate synthetic data using oversampling
                       'analyze': Analyze synthetic datasets
                       'histogram': Generate histograms of distances(Sheet 1)""")

    # Additional arguments based on mode
    parser.add_argument('--dataset', type=str,
                       default=str(DATA_PATH / 'oversampling' / 'train_4_task_02.csv'),
                       help="Path to input dataset (CSV)")
    parser.add_argument('--output', type=str,
                       default=str(DATA_PATH / 'oversampling' / 'synthetic_data.csv'),
                       help="Path to output file/directory")
    
    # Oversampling specific arguments
    parser.add_argument('--percentage', type=float, default=100,
                       help="Percentage of samples to generate (for oversampling)")
    parser.add_argument('--k', type=int, choices=[2, 5], default=2,
                       help="Number of nearest neighbors (for oversampling)")
    parser.add_argument('--normalization', choices=['min_max', 'z_score'], 
                       default='min_max',
                       help="Normalization method (for oversampling)")
    
    # Analysis specific arguments
    parser.add_argument('--synthetic-data', type=str, nargs='+',
                       default=[str(DATA_PATH / 'oversampling' / 'k2_synthetic_data.csv'),
                              str(DATA_PATH / 'oversampling' / 'k5_synthetic_data.csv'),
                              str(DATA_PATH / 'oversampling' / 'synthetic_data-HU.csv')],
                       help="Paths to synthetic datasets (for analysis)")

    return parser.parse_args()

def main():
    args = parse_args()

    try:
        if args.start == 'distance':
            logger.info("Starting physical readings distance calculation...")
            base_dir = DATA_PATH / 'original'  
            output_dir = BASE_DIR / 'distances' 
            distance = DataPreprocessor(base_dir, output_dir)
            # Process each dataset version
            for version in ['hai-21.03', 'hai-22.04','haiend-23.05']:
                logger.info(f"Processing {version} dataset...")
                distance.process_directory(version)

        elif args.start == 'histogram':
            logger.info("Starting distance histogram generation...")
            generator = DistanceHistogramGenerator()
            output = OUTPUT_PATH / 'plots'
            # Create plots directory if it doesn't exist
            output.mkdir(parents=True, exist_ok=True)
            output_file = output / 'histogram.png'
            # Let exceptions propagate up
            generator.generate(output_file)

        elif args.start == 'gan':
            logger.info("Starting GAN-based data generation...")
            generator = GANGenerator()
            generator.generate_data(args.dataset, args.output)
            
        elif args.start == 'oversample':
            logger.info("Starting synthetic data generation using oversampling...")
            generator = SyntheticDataGenerator(args.normalization)
            generator.load_dataset(args.dataset)
            generator.normalize_data()
            synthetic_data = generator.generate_samples(args.percentage, args.k)
            generator.save_synthetic_data(synthetic_data, args.output)
            
        elif args.start == 'analyze':
            logger.info("Starting synthetic data analysis...")
            analyzer = DatasetAnalyzer(
                real_data_path=args.dataset,
                synthetic_data_paths=args.synthetic_data,
                output_path=args.output
            )
            analyzer.analyze_all_columns()
            
        logger.info(f"Operation '{args.start}' completed successfully")
        
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1) 

if __name__ == "__main__":
    main()