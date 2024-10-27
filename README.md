# Sheet 01 - Group 5

## Task 01 

This project is designed to process HAI datasets, calculate Euclidean distances, and generate histograms for visual analysis. The project is organized into separate modules for processing datasets and generating histograms, providing a clean and maintainable codebase.

## Project Structure

- `src/`
  - `main.py`: The main entry point for the project, handling command-line arguments.
  - `processor.py`: Contains the `DatasetProcessor` class for processing datasets and calculating distances.
  - `histogram.py`: Contains functions to load distances and generate histograms using matplotlib.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```


## Usage

The project can be run with different command-line options to either process datasets or generate histograms.

### Command-Line Options

- `--all`: Process all datasets and calculate Euclidean distances.
- `--histogram`: Generate histogram plots for the calculated distances.

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

3. Update the `BASE_DIR` and `OUTPUT_DIR` variables in `main.py` to match your local setup.

4. Run the project using the command-line options described above.
