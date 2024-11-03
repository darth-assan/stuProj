import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from typing import List, Dict, Optional

class DataPreprocessor:
    def __init__(self, base_dir: str, output_dir: str):
        """
        Initialize the preprocessor with directory paths and dataset-specific configurations.
        
        Args:
            base_dir (str): Base directory containing all dataset versions
            output_dir (str): Base directory for saving processed files
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        # Main output directory is created if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset version specific configurations
        self.dataset_configs = {
            'hai-21.03': {
                'time_column': 'time',
                'attack_columns': ['attack', 'attack_P1', 'attack_P2', 'attack_P3']
            },
            'hai-22.04': {
                'time_column': 'timestamp',
                'attack_columns': ['Attack']
            },
            'haiend-23.05': {
                'time_column': None,
                'attack_columns': ['label'],
                'columns_to_include': [
                    "DM-FCV01-D", "DM-FCV01-Z", "DM-FCV02-D", "DM-FCV02-Z",
                    "DM-FCV03-D", "DM-FCV03-Z", "DM-FT01", "DM-FT01Z",
                    "DM-FT02", "DM-FT02Z", "DM-FT03", "DM-FT03Z",
                    "DM-HT01-D", "DM-LCV01-D", "DM-LCV01-MIS", "DM-LCV01-Z",
                    "DM-LIT01", "DM-PCV01-D", "DM-PCV01-Z", "DM-PCV02-D",
                    "DM-PCV02-Z", "DM-PIT01", "DM-PIT02", "DM-PP04-AO", 
                    "DM-PWIT-03", "DM-TIT01", "DM-TIT02", 
                    "DM-TWIT-03", "DM-TWIT-04", "DM-TWIT-05", 
                    "GATEOPEN"
                ]
            }
        }
        
        # Common columns to exclude for older versions
        self.common_exclude_columns = {
            'setpoints_thresholds': [
                'P1_B2004', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P3_LH01', 
                'P3_LL01', 'P2_RTR', 'P1_B4005', 'P1_B400B', 'P1_B4022', 
                'P2_AutoSD', 'P2_ManualSD', 'P4_ST_PS', 'P4_HT_PS'
            ],
            'boolean_controls': [
                'P1_PP01AD', 'P1_PP01AR', 'P1_PP01BD', 'P1_PP01BR', 'P1_PP02D',
                'P1_PP02R', 'P1_SOL01D', 'P1_SOL03D', 'P1_STSP', 'P2_OnOff',
                'P2_Emerg', 'P2_MASW', 'P2_AutoGo', 'P2_ManualGO', 'P2_ATSW_Lamp',
                'P2_MASW_Lamp', 'DQ03-LCV01-D', 'DM-ST-SP', 'DM-SW01-ST',
                'DM-SW02-SP', 'DM-SW03-EM'
            ],
            'preset_limits': [
                'P2_VTR01', 'P2_VTR02', 'P2_VTR03', 'P2_VTR04', 'P1_PIT01_HH',
                'P2_VT01', 'P4_LD'
            ],
            'control_outputs': [
                'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_LCV01D',
                'P1_LCV01Z', 'P1_PCV01D', 'P1_PCV01Z'
            ]
        }

    def get_version_output_dir(self, version: str) -> str:
        """
        Get the output directory for a specific version, creating it if needed.
        
        Args:
            version (str): Dataset version
            
        Returns:
            str: Path to version-specific output directory
        """
        version_output_dir = os.path.join(self.output_dir, version)
        os.makedirs(version_output_dir, exist_ok=True)
        return version_output_dir

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using MinMaxScaler."""
        scaler = MinMaxScaler()
        return pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
        )

    def calculate_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Euclidean distances between consecutive rows."""
        differences = np.diff(df.values, axis=0)
        distances = np.sqrt(np.sum(differences ** 2, axis=1))
        return pd.DataFrame({
            'Row_Pair': [f"{i}-{i+1}" for i in range(len(distances))],
            'Distance': distances
        })

    def drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with constant values."""
        constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        return df.drop(columns=constant_columns)

    def get_columns_to_exclude(self, version: str) -> List[str]:
        """Get list of columns to exclude based on dataset version."""
        if version in ['hai-21.03', 'hai-22.04']:
            time_col = [self.dataset_configs[version]['time_column']]
            return (time_col + 
                   self.common_exclude_columns['setpoints_thresholds'] +
                   self.common_exclude_columns['boolean_controls'] +
                   self.common_exclude_columns['preset_limits'] +
                   self.common_exclude_columns['control_outputs'])
        return []

    def process_file(self, file_path: str, version: str, 
                    label_file: Optional[str] = None) -> None:
        """
        Process a single file based on its version and type.
        
        Args:
            file_path (str): Path to the input file
            version (str): Dataset version (hai-21.03, hai-22.04, or haiend-23.05)
            label_file (str, optional): Path to label file for test datasets
        """
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Handle different dataset versions
        if version == 'haiend-23.05':
            # For haiend-23.05, use only available columns from specified columns_to_include
            columns_to_include = self.dataset_configs[version]['columns_to_include']
            available_columns = [col for col in columns_to_include if col in df.columns]
            
            if not available_columns:
                print(f"Warning: None of the specified columns found in {file_path}. Skipping.")
                return
            
            # Filter for available columns
            df_filtered = df[available_columns]
            
            # Append labels if it's a test file
            if label_file:
                labels = pd.read_csv(label_file)
                df_filtered = pd.concat([df_filtered, labels['label']], axis=1)
        else:
            # For older versions, exclude specified columns
            exclude_cols = self.get_columns_to_exclude(version)
            df_filtered = df.drop(columns=[col for col in exclude_cols if col in df.columns])

        # Filter out attack rows
        attack_cols = self.dataset_configs[version]['attack_columns']
        existing_attack_cols = [col for col in attack_cols if col in df_filtered.columns]
        if existing_attack_cols:
            data_without_attack = df_filtered[(df_filtered[existing_attack_cols] == 0).all(axis=1)]
        else:
            data_without_attack = df_filtered

        # Normalize and clean the data
        if not data_without_attack.empty:
            clean_data = self.normalize_data(
                data_without_attack.drop(columns=attack_cols, errors='ignore')
            )
            clean_data = self.drop_constant_columns(clean_data)
        else:
            print(f"Warning: No data left after filtering attack rows in {file_path}. Skipping.")
            return

        # Calculate distances
        distances = self.calculate_distances(clean_data)
        
        # Get version-specific output directory
        version_output_dir = self.get_version_output_dir(version)
        
        # Save processed files in version-specific directory
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        filtered_path = os.path.join(version_output_dir, f"{base_name}_clean.csv")
        distances_path = os.path.join(version_output_dir, f"{base_name}_distances.csv")
        
        df_filtered.to_csv(filtered_path, index=False)
        distances.to_csv(distances_path, index=False)
        
        print(f"Processed and saved: {base_name} in {version_output_dir}")


    def process_directory(self, version: str) -> None:
        """
        Process all files in a version-specific directory.
        
        Args:
            version (str): Dataset version to process
        """
        version_dir = os.path.join(self.base_dir, version)
        if not os.path.exists(version_dir):
            print(f"Directory not found: {version_dir}")
            return

        for filename in os.listdir(version_dir):
            if not filename.endswith('.csv'):
                continue

            file_path = os.path.join(version_dir, filename)
            
            # Handle test files for haiend-23.05
            if version == 'haiend-23.05':
                if 'test1' in filename:
                    label_file = os.path.join(version_dir, 'label-test1.csv')
                    self.process_file(file_path, version, label_file)
                elif 'test2' in filename:
                    label_file = os.path.join(version_dir, 'label-test2.csv')
                    self.process_file(file_path, version, label_file)
                elif 'train' in filename:
                    self.process_file(file_path, version)
            else:
                self.process_file(file_path, version)

# Example usage
if __name__ == "__main__":
    base_dir = "/Users/darth/Dev/stuProj/data/Original"  # Replace with actual base directory
    output_dir = "/Users/darth/Dev/stuProj/data/Processed"  # Replace with actual output directory
    
    preprocessor = DataPreprocessor(base_dir, output_dir)
    
    # Process each dataset version
    for version in ['hai-21.03', 'hai-22.04','haiend-23.05']:
        print(f"\nProcessing {version} dataset...")
        preprocessor.process_directory(version)