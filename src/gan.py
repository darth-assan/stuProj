import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ks_2samp
import numpy as np

# Constants
TRAIN_SET_1_PATH = '/Users/darth/Dev/stuProj/data/GAN/train1_clean.csv'
TRAIN_SET_2_PATH = '/Users/darth/Dev/stuProj/data/GAN/train4_clean.csv'
BATCH_SIZE = 128
INPUT_DIM = 100
FEATURE_DIM = 32
NUM_EPOCHS = 3
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0001
DROPOUT_RATE = 0.3
SAMPLES_PER_DAY = 300
DAYS_TO_GENERATE = 3

# Data Loading and Preprocessing
def load_and_preprocess_data():
    train_set_1 = pd.read_csv(TRAIN_SET_1_PATH)
    train_set_2 = pd.read_csv(TRAIN_SET_2_PATH)
    common_columns = train_set_1.columns.intersection(train_set_2.columns)
    train_set_1 = train_set_1[common_columns]
    train_set_2 = train_set_2[common_columns]
    return train_set_1, train_set_2

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = scaler.fit_transform(data.values)
    tensor_data = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(1)
    return tensor_data

# Model Definitions
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, FEATURE_DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(FEATURE_DIM, FEATURE_DIM * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv1d(FEATURE_DIM * 2, FEATURE_DIM * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(FEATURE_DIM * 4, FEATURE_DIM * 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(FEATURE_DIM * 8, FEATURE_DIM * 16, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(FEATURE_DIM * 16, FEATURE_DIM * 32, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(FEATURE_DIM * 32, 1, kernel_size=2, stride=1, padding=0)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(INPUT_DIM, FEATURE_DIM * 16),
            nn.ReLU(),
            nn.Linear(FEATURE_DIM * 16, FEATURE_DIM * 8),
            nn.ReLU(),
            nn.Linear(FEATURE_DIM * 8, FEATURE_DIM * 4),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(FEATURE_DIM * 4, FEATURE_DIM * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(FEATURE_DIM * 2, FEATURE_DIM, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(FEATURE_DIM, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), -1, 1)
        return self.deconv_layers(x)

# Training Setup
def train_gan(train_loader, discriminator, generator, criterion, optimizer_d, optimizer_g, device):
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(train_loader):
            real_data = batch[0].to(device)
            optimizer_d.zero_grad()
            real_labels = torch.full((real_data.size(0), 1), 0.9).to(device)
            fake_labels = torch.full((real_data.size(0), 1), 0.1).to(device)
            outputs = discriminator(real_data)
            d_loss_real = criterion(outputs, real_labels)
            noise = torch.randn(real_data.size(0), INPUT_DIM).to(device)
            fake_data = generator(noise)
            outputs = discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            optimizer_g.zero_grad()
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

# Synthetic Data Generation
def generate_synthetic_data(generator, device):
    total_samples = SAMPLES_PER_DAY * DAYS_TO_GENERATE
    noise = torch.randn(total_samples, INPUT_DIM).to(device)
    synthetic_data = generator(noise).detach().cpu().numpy()
    synthetic_data_reshaped = synthetic_data.reshape(total_samples, -1)
    if synthetic_data_reshaped.shape[1] == 32:
        additional_column = np.random.normal(size=(total_samples, 1))
        synthetic_data_reshaped = np.hstack((synthetic_data_reshaped, additional_column))
    return synthetic_data_reshaped

# Data Normalization and K-S Test
def normalize_and_ks_test(train_set_1, train_set_2, synthetic_df):
    scaler = MinMaxScaler()
    subset_1 = train_set_1.iloc[:SAMPLES_PER_DAY * 3]
    subset_2 = train_set_1.iloc[SAMPLES_PER_DAY * 3:SAMPLES_PER_DAY * 6]
    subset_3 = synthetic_df
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

def generate_and_evaluate_synthetic_data(generator, train_set_1, train_set_2, device):
    synthetic_data = generate_synthetic_data(generator, device)
    synthetic_df = pd.DataFrame(synthetic_data, columns=train_set_1.columns)
    normalized_subset_1, normalized_subset_2, normalized_subset_3 = normalize_and_ks_test(train_set_1, train_set_2, synthetic_df)
    ks_results_real_real = compute_ks_statistics(normalized_subset_1, normalized_subset_2)
    ks_results_real_synthetic = compute_ks_statistics(normalized_subset_1, normalized_subset_3)
    ks_results_real_synthetic_2 = compute_ks_statistics(normalized_subset_2, normalized_subset_3)
    return ks_results_real_real, ks_results_real_synthetic, ks_results_real_synthetic_2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set_1, train_set_2 = load_and_preprocess_data()
    tensor_data = normalize_data(train_set_2)
    dataset = TensorDataset(tensor_data)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    criterion = nn.BCELoss()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    train_gan(train_loader, discriminator, generator, criterion, optimizer_d, optimizer_g, device)
    ks_results_real_real, ks_results_real_synthetic, ks_results_real_synthetic_2 = generate_and_evaluate_synthetic_data(generator, train_set_1, train_set_2, device)
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
    main()