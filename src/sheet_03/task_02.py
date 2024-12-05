import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split, ParameterGrid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
from sklearn.preprocessing import MinMaxScaler


# Configure logger
logger.remove()
logger.add("autoencoder.log", rotation="500 MB", level="DEBUG")
logger.add(sys.stdout, level="INFO")

class ConfigurableAutoencoder(nn.Module):
    """
    A configurable autoencoder that supports different layer types and architectures.

    Args:
        input_dim (int): Dimension of input data
        compression_factor (int): Factor by which to compress the input
        num_hidden_layers (int): Number of hidden layers in encoder/decoder
        layer_type (str): Type of layer ('dense', 'conv1d', or 'lstm')
        activation (str): Activation function to use ('relu', 'tanh', or 'sigmoid')
    """
    def __init__(self, input_dim, compression_factor, num_hidden_layers, layer_type='dense', activation='relu'):
        super().__init__()
        logger.info(f"Initializing autoencoder with {layer_type} layers and {activation} activation")

        self.input_dim = input_dim
        self.compressed_dim = max(1, int(input_dim / compression_factor))
        self.layer_type = layer_type

        # Calculate layer sizes for encoder/decoder
        layer_sizes = np.linspace(input_dim, self.compressed_dim, num_hidden_layers + 1, dtype=int)

        # Map activation functions
        self.activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = self.activation_functions[activation]

        # Build network architecture
        self._build_network(layer_sizes)

        logger.debug(f"Network architecture - Input dim: {input_dim}, Compressed dim: {self.compressed_dim}")

    def _build_network(self, layer_sizes):
        """Build encoder and decoder networks based on specified layer type"""
        if self.layer_type == 'dense':
            self.encoder = self._build_dense_layers(layer_sizes, forward=True)
            self.decoder = self._build_dense_layers(layer_sizes, forward=False)
        elif self.layer_type == 'conv1d':
            self.encoder = self._build_conv1d_layers(layer_sizes, forward=True)
            self.decoder = self._build_conv1d_layers(layer_sizes, forward=False)
        elif self.layer_type == 'lstm':
            self.encoder = self._build_lstm_layers(layer_sizes, forward=True)
            self.decoder = self._build_lstm_layers(layer_sizes, forward=False)

    def _build_dense_layers(self, layer_sizes, forward=True):
        """Build fully connected layers for encoder/decoder"""
        layers = []
        sizes = list(layer_sizes) if forward else list(reversed(layer_sizes))

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(self.activation)

        return nn.Sequential(*layers)

    def _build_conv1d_layers(self, layer_sizes, forward=True):
        """Build 1D convolutional layers for encoder/decoder"""
        layers = []
        sizes = list(layer_sizes) if forward else list(reversed(layer_sizes))

        for i in range(len(sizes) - 1):
            if forward:
                layers.append(nn.Conv1d(1, 1, kernel_size=3, padding=1))
            else:
                layers.append(nn.ConvTranspose1d(1, 1, kernel_size=3, padding=1))
            layers.append(self.activation)

        return nn.Sequential(*layers)

    def _build_lstm_layers(self, layer_sizes, forward=True):
        """Build LSTM layers for encoder/decoder"""
        layers = nn.ModuleList()
        sizes = list(layer_sizes) if forward else list(reversed(layer_sizes))

        for i in range(len(sizes) - 1):
            layers.append(nn.LSTM(int(sizes[i]), int(sizes[i+1]), batch_first=True))

        return layers

    def forward(self, x):
        """Forward pass through the autoencoder"""
        if self.layer_type == 'dense':
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        elif self.layer_type == 'conv1d':
            x = x.unsqueeze(1)
            encoded = self.encoder(x).squeeze(1)
            decoded = self.decoder(encoded.unsqueeze(1)).squeeze(1)
        elif self.layer_type == 'lstm':
            for layer in self.encoder:
                x, _ = layer(x)
                x = self.activation(x)
            encoded = x
            for layer in self.decoder:
                x, _ = layer(x)
                x = self.activation(x)
            decoded = x

        return decoded, encoded

class ModelTrainer:
    """Handles model training and comprehensive hyperparameter tuning"""
    def __init__(self):
        self.hyperparameters = {
            'learning_rates': [0.0001, 0.001],  # Added more learning rates
            'batch_sizes': [64, 128],  # Added more batch sizes
            'epochs': [50, 100],  # Added more epoch options
            'layer_types': ['dense', 'conv1d', 'lstm'],
            'activations': ['relu', 'tanh', 'sigmoid'],
            'compression_factors': [2, 4],  # Added compression factor grid
            'num_hidden_layers': [2, 3]  # Added hidden layers grid
        }
        logger.info("Initialized ModelTrainer with expanded hyperparameters")

    def grid_search(self, X_train, X_test):
        """
        Perform exhaustive grid search over hyperparameters

        Args:
            X_train: Training data
            X_test: Test data

        Returns:
            Best model configurations and their performance
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_models = []

        # Create parameter grid
        param_grid = ParameterGrid({
            'learning_rate': self.hyperparameters['learning_rates'],
            'batch_size': self.hyperparameters['batch_sizes'],
            'epochs': self.hyperparameters['epochs'],
            'layer_type': self.hyperparameters['layer_types'],
            'activation': self.hyperparameters['activations'],
            'compression_factor': self.hyperparameters['compression_factors'],
            'num_hidden_layers': self.hyperparameters['num_hidden_layers']
        })

        logger.info(f"Starting grid search with {len(param_grid)} configurations")

        for params in param_grid:
            try:
                logger.info(f"Testing configuration: {params}")

                # Prepare data loader with current batch size
                train_loader = DataLoader(
                    torch.FloatTensor(X_train).unsqueeze(1),
                    batch_size=params['batch_size'],
                    shuffle=True
                )
                test_loader = DataLoader(
                    torch.FloatTensor(X_test).unsqueeze(1),
                    batch_size=params['batch_size']
                )

                # Create model with current hyperparameters
                model = ConfigurableAutoencoder(
                    input_dim=X_train.shape[1],
                    compression_factor=params['compression_factor'],
                    num_hidden_layers=params['num_hidden_layers'],
                    layer_type=params['layer_type'],
                    activation=params['activation']
                )

                # Train model
                best_model, best_loss = self.train_model(
                    model,
                    train_loader,
                    learning_rate=params['learning_rate'],
                    num_epochs=params['epochs']
                )

                # Store results
                best_models.append({
                    'params': params,
                    'best_model': best_model,
                    'best_loss': best_loss
                })

                logger.success(f"Configuration completed. Best loss: {best_loss:.6f}")

            except Exception as e:
                logger.error(f"Error in configuration {params}: {str(e)}")
                continue

        # Sort models by loss and return top configurations
        best_models.sort(key=lambda x: x['best_loss'])

        logger.info("Grid search completed. Top 3 model configurations:")
        for i, model_config in enumerate(best_models[:3], 1):
            logger.info(f"Rank {i}: Loss = {model_config['best_loss']:.6f}, Params = {model_config['params']}")

        return best_models

    def train_model(self, model, train_loader, learning_rate, num_epochs, patience=10):
        """
        Train the autoencoder model with early stopping and learning rate scheduling

        Args:
            model: The autoencoder model
            train_loader: DataLoader for training data
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            patience: Number of epochs to wait for improvement before stopping
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {device}")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience//2, verbose=True)

        best_loss = float('inf')
        best_model = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                batch = batch[0].to(device)
                optimizer.zero_grad()
                decoded, _ = model(batch)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.6f}")

            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = model.state_dict().copy()
                logger.info(f"New best model found with loss: {best_loss:.6f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        return best_model, best_loss

def extract_and_save_features(model, data_loader, save_path, scaler=None, original_columns=None):
    """Extract and save encoded features from the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Add this line to move model to device
    model.eval()
    features = []
    reconstructed = []

    logger.info(f"Extracting features to {save_path}")

    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0].to(device)
            decoded, encoded = model(batch)
            features.append(encoded.cpu().numpy())
            reconstructed.append(decoded.cpu().numpy())

    features = np.concatenate(features, axis=0)
    reconstructed = np.concatenate(reconstructed, axis=0)

    # Denormalize the reconstructed data if scaler is provided
    if scaler is not None:
        reconstructed = scaler.inverse_transform(reconstructed)

    # Convert to DataFrames and save as CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    features_df = pd.DataFrame(features)
    reconstructed_df = pd.DataFrame(reconstructed, columns=original_columns)

    csv_save_path = save_path.replace('.npy', '.csv')
    features_df.to_csv(csv_save_path, index=False)
    reconstructed_df.to_csv(csv_save_path.replace('features.csv', 'reconstructed.csv'), index=False)

    logger.success(f"Features and reconstructed data saved successfully to {csv_save_path}")
    return features, reconstructed

def load_physical_readings_data():
    """Load and preprocess physical readings data"""
    logger.info("Loading physical readings data")

    try:
        data = pd.read_csv('/Users/darth/Dev/stuProj/data/oversampling/train_4_task_02.csv', nrows=1000)
        original_columns = data.columns
        X = data.values.astype(np.float32)

        # Initialize and fit the scaler
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)

        X_train, X_test = train_test_split(X_normalized, test_size=0.2, random_state=42)

        logger.success(f"Data loaded and normalized. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, scaler, original_columns

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def main():
    logger.info("Starting autoencoder grid search pipeline")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Load datasets
        datasets = {
            'physical_readings': load_physical_readings_data()
        }

        # Process each dataset
        for dataset_name, (X_train, X_test, scaler, original_columns) in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")

            # Initialize trainer with grid search
            trainer = ModelTrainer()

            # Perform grid search
            best_models = trainer.grid_search(X_train, X_test)

            # Extract and save features for top 3 models
            for rank, model_config in enumerate(best_models[:3], 1):
                params = model_config['params']
                model = ConfigurableAutoencoder(
                    input_dim=X_train.shape[1],
                    compression_factor=params['compression_factor'],
                    num_hidden_layers=params['num_hidden_layers'],
                    layer_type=params['layer_type'],
                    activation=params['activation']
                )

                # Load best model state
                model.load_state_dict(model_config['best_model'])
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)  # Add this line to move model to device

                # Prepare data loaders
                train_loader = DataLoader(torch.FloatTensor(X_train).unsqueeze(1), batch_size=params['batch_size'], shuffle=True)
                test_loader = DataLoader(torch.FloatTensor(X_test).unsqueeze(1), batch_size=params['batch_size'])

                # Save directory for this configuration
                save_dir = f'features/{dataset_name}/rank_{rank}_{params["layer_type"]}_{params["activation"]}'

                # Extract and save features
                extract_and_save_features(model, train_loader, f'{save_dir}/train_features.csv', scaler, original_columns)
                extract_and_save_features(model, test_loader, f'{save_dir}/test_features.csv', scaler, original_columns)

        logger.success("Grid search pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
