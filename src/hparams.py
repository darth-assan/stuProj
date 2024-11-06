import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import itertools

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter grid for tuning
param_grid = {
    "generator_lr": [0.0003, 0.0002, 0.0001],
    "discriminator_lr": [0.0001, 0.00005, 0.0002],
    "batch_size": [32, 64, 128],
    "noise_std_dev": [0.01, 0.02, 0.03],
    "num_epochs": [50, 100]
}

# Parameters
input_dim = 100               # Dimension of the input noise vector for the generator
feature_dim = 16              # Adjusted base feature size for convolutional layers

# Load data (replace with actual data loading and preprocessing steps)
# Sample data for tensor_data, replace this with actual data loading
tensor_data = torch.randn(1000, 49).float()  # Example: random data with 49 features

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(feature_dim * 2, feature_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(feature_dim * 4, feature_dim * 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(feature_dim * 8, feature_dim * 16, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(feature_dim * 16, feature_dim * 32, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(feature_dim * 32, 1, kernel_size=2, stride=1, padding=0)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 16),
            nn.ReLU(),
            nn.Linear(feature_dim * 16, feature_dim * 8),
            nn.ReLU(),
            nn.Linear(feature_dim * 8, feature_dim * 4),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(feature_dim * 4, feature_dim * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(feature_dim * 2, feature_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(feature_dim, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), -1, 1)
        return self.deconv_layers(x)

# Automated tuning
best_params = None
best_loss = float('inf')

param_combinations = list(itertools.product(
    param_grid["generator_lr"],
    param_grid["discriminator_lr"],
    param_grid["batch_size"],
    param_grid["noise_std_dev"],
    param_grid["num_epochs"]
))

# Iterate over each hyperparameter combination
for generator_lr, discriminator_lr, batch_size, noise_std_dev, num_epochs in param_combinations:
    print(f"Training with generator_lr={generator_lr}, discriminator_lr={discriminator_lr}, batch_size={batch_size}, noise_std_dev={noise_std_dev}, epochs={num_epochs}")
    
    # Initialize DataLoader for the current batch size
    train_loader = DataLoader(TensorDataset(tensor_data.unsqueeze(1)), batch_size=batch_size, shuffle=True)

    # Initialize models
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    
    # Loss and Optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for GAN
    optimizer_d = optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
    
    avg_d_loss = 0.0
    avg_g_loss = 0.0

    # Training the GAN with added noise to real data for regularization
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            real_data = batch[0].to(device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            real_data_noisy = real_data + torch.normal(0, noise_std_dev, real_data.shape).to(device)
            real_labels = torch.full((real_data.size(0), 1), 0.9).to(device)
            fake_labels = torch.full((real_data.size(0), 1), 0.1).to(device)

            # Discriminator on real data with noise
            outputs = discriminator(real_data_noisy)
            d_loss_real = criterion(outputs, real_labels)

            # Discriminator on fake data
            noise = torch.randn(real_data.size(0), input_dim).to(device)
            fake_data = generator(noise)
            outputs = discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            # Backprop and optimize discriminator
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            avg_d_loss += d_loss.item()

            # Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()
            avg_g_loss += g_loss.item()

    # Calculate average loss for this configuration
    avg_d_loss /= (num_epochs * len(train_loader))
    avg_g_loss /= (num_epochs * len(train_loader))
    total_loss = avg_d_loss + avg_g_loss

    # Update best parameters
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {
            "generator_lr": generator_lr,
            "discriminator_lr": discriminator_lr,
            "batch_size": batch_size,
            "noise_std_dev": noise_std_dev,
            "num_epochs": num_epochs
        }
    print(f"Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")

print(f"Best Parameters: {best_params}")