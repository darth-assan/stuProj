import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp
import torch
from models import Generator
import os

SAMPLES_PER_DAY = 300
DAYS_TO_GENERATE = 3
INPUT_DIM = 100
MODEL_SAVE_PATH = 'saved_models'

def load_generator(device, model_path='generator.pth'):
    """Load a trained generator model from disk."""
    full_path = os.path.join(MODEL_SAVE_PATH, model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No trained model found at {full_path}. Please train the model first.")
    
    checkpoint = torch.load(full_path, map_location=device)
    generator = Generator()
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.to(device)
    generator.eval()
    return generator

def generate_synthetic_data(generator, device):
    total_samples = SAMPLES_PER_DAY * DAYS_TO_GENERATE
    noise = torch.randn(total_samples, INPUT_DIM).to(device)
    
    with torch.no_grad():  # Disable gradient computation for inference
        synthetic_data = generator(noise).detach().cpu().numpy()
    
    synthetic_data_reshaped = synthetic_data.reshape(total_samples, -1)
    
    if synthetic_data_reshaped.shape[1] == 32:
        additional_column = np.random.normal(size=(total_samples, 1))
        synthetic_data_reshaped = np.hstack((synthetic_data_reshaped, additional_column))

    return synthetic_data_reshaped

def normalize_and_ks_test(train_set_1, train_set_2, synthetic_df):
    scaler = MinMaxScaler()
    subset_1 = train_set_1.iloc[:SAMPLES_PER_DAY * 3]
    subset_2 = train_set_1.iloc[SAMPLES_PER_DAY * 3:SAMPLES_PER_DAY * 6]
    subset_3 = synthetic_df

    # Saving synthetic data for future use
    if not os.path.exists('/Users/darth/Dev/stuProj/data/'):
        os.makedirs('/Users/darth/Dev/stuProj/data/')
    
    save_path = os.path.join('/Users/darth/Dev/stuProj/data/', 'synthetic_data.csv')
    with open (save_path, 'w'):
        pd.DataFrame(synthetic_df,columns=train_set_1.columns).to_csv(save_path, index=False)
    print(f"synthetic data {save_path}")
    
    normalized_subset_1 = pd.DataFrame(scaler.fit_transform(subset_1), columns=subset_1.columns)
    normalized_subset_2 = pd.DataFrame(scaler.fit_transform(subset_2), columns=subset_2.columns)
    normalized_subset_3 = pd.DataFrame(scaler.fit_transform(subset_3), columns=subset_3.columns)
    
    return normalized_subset_1, normalized_subset_2, normalized_subset_3

def compute_ks_statistics(df1, df2):
    ks_results = []
    for column in df1.columns:
        ks_stat, p_value = ks_2samp(df1[column], df2[column])
        ks_results.append({
            'sensor': column,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'passes': ks_stat < 0.15 and p_value > 0.03
        })
    return pd.DataFrame(ks_results)

def evaluate_synthetic_data(generator, train_set_1, train_set_2, device):
    synthetic_data = generate_synthetic_data(generator, device)
    synthetic_df = pd.DataFrame(synthetic_data, columns=train_set_1.columns)
    
    normalized_subset_1, normalized_subset_2, normalized_subset_3 = normalize_and_ks_test(
        train_set_1, train_set_2, synthetic_df)
    
    ks_results_real_real = compute_ks_statistics(normalized_subset_1, normalized_subset_2)
    ks_results_real_synthetic = compute_ks_statistics(normalized_subset_1, normalized_subset_3)
    ks_results_real_synthetic_2 = compute_ks_statistics(normalized_subset_2, normalized_subset_3)

    
    # Print results
    passes_real_real = ks_results_real_real['passes'].sum()
    passes_real_synthetic = ks_results_real_synthetic['passes'].sum()
    passes_real_synthetic_2 = ks_results_real_synthetic_2['passes'].sum()

    
    print("Number of sensors passing K-S test (real-real):", passes_real_real)
    print("Number of sensors passing K-S test (real-synthetic 1):", passes_real_synthetic)
    print("Number of sensors passing K-S test (real-synthetic 2):", passes_real_synthetic_2)
    print("\nDetailed K-S Test Results (Real-Real):\n", ks_results_real_real)
    print("\nDetailed K-S Test Results (Real-Synthetic 1):\n", ks_results_real_synthetic)
    print("\nDetailed K-S Test Results (Real-Synthetic 2):\n", ks_results_real_synthetic_2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # train a new model:
    # generator, train_set_1, train_set_2 = train_main()
    
    # load a previously trained model:
    from train import main as train_main, load_and_preprocess_data
    generator = load_generator(device)
    train_set_1, train_set_2 = load_and_preprocess_data()
    
    evaluate_synthetic_data(generator, train_set_1, train_set_2, device)