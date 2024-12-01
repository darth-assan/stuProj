import argparse
from pathlib import Path
from src.sheet_02.task_01 import SyntheticDataGenerator
from src.sheet_02.task_02 import DatasetAnalyzer
from src.sheet_01.processor import DataPreprocessor
from src.sheet_01.histogram import DistanceHistogramGenerator
from src.sheet_01.config import GANConfig
from src.sheet_01.train import GANTrainer
from src.sheet_01.ks_test import KSTestEvaluator
from src.sheet_01.data_utils import DataProcessor
from src.sheet_03.task_03 import PCA_algor
# import src.sheet_01.hparams as hp

from loguru import logger
import pandas as pd
import sys

# Configure Loguru
logger.remove()
logger.add(sys.stdout, level="INFO", filter=lambda record: record["level"].name == "INFO")

# Define paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'data'
OUTPUT_PATH = BASE_DIR

# Helper functions
check_file = DataProcessor.check_file_exists

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Industrial Process Data Analysis Toolbox
        ======================================
        This toolbox provides various functionalities for generating and analyzing ICS synthetically generated data:
        1. Physical Readings Analysis (from Sheet 1)
        2. GAN-based Data Generation (from Sheet 1)
        3. Oversampling method for Synthetic Data Generation (from Sheet 2)
        4. Synthetic Data Analysis
        5. Principal Component Analysis (from sheet 3)
        
        Use -s <keyword> to select the operation mode.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-s', '--start', type=str, required=True,
                        choices=['distance', 'oversample', 'analyze-1', 'analyze-2', 'histogram', 
                                 'gan-generate', 'train', 'hparams', 'pca'],
                        help="""Operation mode selection:
                        'distance': Analyze physical sensor readings (Sheet 1, Task 1)
                        'oversample': Generate synthetic data using oversampling
                        'analyze1': Analyze synthetic datasets (Sheet 1 - K-S Test)
                        'analyze2': Analyze synthetic datasets (Sheet 2 - CDF/CCDF)
                        'histogram': Generate histograms of distances(Sheet 1)
                        'gan-generate': Generate synthetic data using GAN (Sheet 1, Task 3)
                        'train': Train GAN model (Sheet 1, Task 3)
                        'hparams': Get hyperparameters for the GAN model
                        'pca': performing pca on the test and training data (sheet 3, Task 3)""")
    
    # General arguments
    parser.add_argument('-d', '--dataset', type=str,
                        default=None,
                        help="Path to input dataset (CSV)")
    parser.add_argument('-o', '--output', type=str,
                        default=None,
                        help="Path to output file/directory")

    # Oversampling-specific arguments
    oversample_group = parser.add_argument_group('Oversampling')
    oversample_group.add_argument('-p', '--percentage', type=float, default=100,
                                  help="Percentage of samples to generate (for oversampling)")
    oversample_group.add_argument('-k','--k', type=int, choices=[2, 5], default=2,
                                  help="Number of nearest neighbors (for oversampling)")
    oversample_group.add_argument('-n', '--normalization', choices=['min_max', 'z_score'], 
                                  default='min_max',
                                  help="Normalization method (for oversampling)")
    
    ####################################################################
    # PCA
    PCA_group = parser.add_argument_group('PCA')
    PCA_group.add_argument('-b', '--beta', type=float, choices=[0.998, 0.895, 0.879], default=0.998,
                                  help="the beta value of which the PCA trims to lower dimensions")
    ####################################################################


    # Analysis-specific arguments
    analysis_group = parser.add_argument_group('Analysis')
    analysis_group.add_argument('-sd', '--synthetic-data', type=str, nargs='+',
                                default=[str(DATA_PATH / 'oversampling' / 'k2_synthetic_data.csv'),
                                         str(DATA_PATH / 'oversampling' / 'k5_synthetic_data.csv'),
                                         str(DATA_PATH / 'oversampling' / 'synthetic_data-HU.csv')],
                                help="Paths to synthetic datasets (for analysis)")

    args = parser.parse_args()

    # Validate arguments based on the mode
    if args.start != 'oversample' and any(arg in sys.argv for arg in ['--k', '-k', '--percentage', '-p', '--normalization', '-n']):
        parser.error("--k, --percentage, and --normalization are only valid for 'oversample' mode")

    return args

def main():
    args = parse_args()
    config = GANConfig()

    try:
        if args.start == 'distance':
            logger.info("Starting physical readings distance calculation...")
            base_dir = DATA_PATH / 'original'  
            output_dir = BASE_DIR / 'distances' if not args.output else args.output
            distance = DataPreprocessor(base_dir, output_dir)
            # Process each dataset version
            for version in ['hai-21.03', 'hai-22.04','haiend-23.05']:
                logger.info(f"Processing {version} dataset...")
                distance.process_directory(version)

        elif args.start == 'histogram':
            logger.info("Starting distance histogram generation...")
            generator = DistanceHistogramGenerator()
            output = OUTPUT_PATH / 'plots'
            output.mkdir(parents=True, exist_ok=True)
            output_file = output / 'histogram.png'
            generator.generate(output_file)
            
        elif args.start == 'oversample':
            logger.info("Starting synthetic data generation using oversampling...")
            dataset = DATA_PATH / 'oversampling' / 'train_4_task_02.csv' if not args.dataset else args.dataset  
            check_file(dataset)
            output = DATA_PATH / 'oversampling' / 'synthetic_data_os.csv' if not args.output else args.output
            generator = SyntheticDataGenerator(args.normalization)
            generator.load_dataset(dataset)
            generator.normalize_data()
            synthetic_data = generator.generate_samples(args.percentage, args.k)
            generator.save_synthetic_data(synthetic_data, output)
            
        ##########################################################
        elif args.start == 'pca':
            logger.info("Starting principal component analysis with the folder...")
            dataset = DATA_PATH / 'original' / 'hai-21.03' if not args.dataset else args.dataset  
            ##check_file(dataset)  # not a file
            output = BASE_DIR/ 'results' / 'PCA'  if not args.output else args.output
            my_obj = PCA_algor(args.beta)
            my_obj.main_func(dataset, output, args.beta)
        ##########################################################
            
        elif args.start == 'analyze-2':
            logger.info("Starting synthetic data analysis...")
            dataset = DATA_PATH / 'oversampling' / 'train_4_task_02.csv' if not args.dataset else args.dataset
            check_file(dataset)
            output = OUTPUT_PATH / 'plots' if not args.output else args.output
            analyzer = DatasetAnalyzer(
                real_data_path = dataset,
                synthetic_data_paths = args.synthetic_data,
                output_path = output
            )
            analyzer.analyze_datasets()
            logger.info(f"Analysis completed. Results saved to {output}")   

        elif args.start == 'train':
            logger.info("Starting GAN model training...")
            trainer = GANTrainer(config)
            trainer.train()

        elif args.start == 'gan-generate':
            logger.info("Generating synthetic data using GAN...")
            evaluator = KSTestEvaluator(config)
            generator = evaluator.load_generator()
            synthetic_data = evaluator.generate_synthetic_data(generator)
            output_path = Path(config.synthetic_data_path) if not args.output else Path(args.output)
            pd.DataFrame(synthetic_data, columns=[f'feature_{i}' for i in range(synthetic_data.shape[1])]).to_csv(
                output_path, index=False)
            logger.info(f"Synthetic data saved to {output_path}")
        

        elif args.start == 'analyze-1':
            logger.info("Evaluating synthetic data using K-S...")
            evaluator = KSTestEvaluator(config)
            results = evaluator.evaluate()
            logger.info("Evaluation completed!")
            print(results)

            summary_path = Path(config.ks_summary_path)
            with open(summary_path, 'w') as f:
                f.write("K-S Test Results:\n" + '=' * 20 + '\n')
                for test_name, df in results.items():
                    passing = df['passes'].sum()
                    total = len(df)
                    f.write(f"K-S Test Results ({test_name}):\n")
                    f.write(f"Passing sensors: {passing}\n")
                    f.write(df.to_string(index=False) + "\n\n")
                
                f.write("Summary:\n")
                for test_name, df in results.items():
                    passing = df['passes'].sum()
                    total = len(df)
                    summary_line = f"{test_name}: {passing}/{total} sensors passed the KS test ({(passing/total*100):.1f}%)\n"
                    f.write(summary_line)

        elif args.start == 'hparams':
            logger.info("Starting hyperparameter tuning...")
            import src.sheet_01.hparams as hp
            #subprocess.run(["python", hp], check=True)
            
        logger.info(f"Operation '{args.start}' completed successfully")
        
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1) 

if __name__ == "__main__":
    main()