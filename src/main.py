import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Process datasets and generate histograms.")
    parser.add_argument('--all', action='store_true', help="Generate distances for all datasets.")
    parser.add_argument('--histogram', action='store_true', help="Generate histogram plots.")
    parser.add_argument('--train', action='store_true', help="Run the training script.")
    parser.add_argument('--evaluate', action='store_true', help="Run the ks-test script for calculating the Kolmogorov-Smirnov (K-S) statistic.")
    parser.add_argument('--hparams', action='store_true', help="Print hyperparameters.")
    
    args = parser.parse_args()

    if args.all:
        subprocess.run(["python", "processor.py"], check=True)

    if args.histogram:
        subprocess.run(["python", "histogram.py"], check=True)

    if args.train:
        subprocess.run(["python", "train.py"], check=True)

    if args.evaluate:
        subprocess.run(["python", "ks-test.py"], check=True)

    if args.hparams:
        subprocess.run(["python", "hparams.py"], check=True)

if __name__ == "__main__":
    main()
