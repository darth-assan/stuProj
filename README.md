# Sheet 01 - Group 5

## Task 01 

This project is designed to process HAI datasets, calculate Euclidean distances, and generate histograms for visual analysis. The project is organized into separate modules for processing datasets and generating histograms, providing a clean and maintainable codebase. Also includes the implementation of a GAN and evaluation of synthetic data.

## Project Structure

- `src/`
  - `main.py`: The main entry point for the project, handling command-line arguments.
  - `processor.py`: Contains the `DataProcessor` class for processing datasets and calculating distances.
  - `histogram.py`: Contains functions to load distances and generate histograms using matplotlib.
  - `models.py`: Contains the GAN model
  - `train.py`: Trains the model using the selected files from two distriburions from different versions of the Dataset.
  - `ks-test.py`: Computes the K-S statistic for the comparison of the various subsets
  - `hparams.py`: Hyperparameter tuning to get the best parameters to use for training the model.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- scipy
- torch

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib torch scipy
```


## Usage

The project can be run with different command-line options to either process datasets or generate histograms.

### Command-Line Options

- `--all`: Process all datasets and calculate Euclidean distances.
- `--histogram`: Generate histogram plots for the calculated distances.
- `--train`: Run the training script.
- `--evaluate`: First generate and save the Synthetic data, then Run the ks-test script for calculating the Kolmogorov-Smirnov (K-S) statistic.
- `--hparams`: Get the best parameters to use for training the model.

### Example Commands

1. **Process all datasets:**

   ```bash
   python src/main.py --all
   ```

   This command will process all datasets in the specified base directory, calculate Euclidean distances, and save them to the output directory.

2. **Generate histograms:**

   ```bash
   python src/main.py --histogram
   ```

   This command will load the calculated distances and generate histograms, saving the plots to the results directory.

## Configuration

- **Base Directory**: The directory containing the original datasets. Update the `BASE_DIR` variable in `main.py` to point to your dataset location.
- **Output Directory**: The directory where the calculated distances will be saved. Update the `OUTPUT_DIR` variable in `main.py` to your desired output location.

## Project Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. The directory structure for data should be processed as follows
- data
   - GAN/ _-> contains the two train dataset for training the GAN_
   - Original/ _-> The original dataset. can be obtained from ![[https://github.com/icsdataset/hai#hai-dataset]]_
   - Processed/ _-> Contains the preprocessed data and the distances of each dataset from the respective distributions_
   - Results/ _-> Contain results such as the histogram_

4. Run the project using the command-line options described above.
