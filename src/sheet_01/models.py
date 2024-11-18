# models.py
import torch
import torch.nn as nn
from .config import GANConfig

class Discriminator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, config.feature_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(config.feature_dim, config.feature_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dropout_rate),
            nn.Conv1d(config.feature_dim * 2, config.feature_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(config.feature_dim * 4, config.feature_dim * 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(config.feature_dim * 8, config.feature_dim * 16, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(config.feature_dim * 16, config.feature_dim * 32, kernel_size=2, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(config.feature_dim * 32, 1, kernel_size=2, stride=1, padding=0)
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
    def __init__(self, config: GANConfig):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(config.input_dim, config.feature_dim * 16),
            nn.ReLU(),
            nn.Linear(config.feature_dim * 16, config.feature_dim * 8),
            nn.ReLU(),
            nn.Linear(config.feature_dim * 8, config.feature_dim * 4),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(config.feature_dim * 4, config.feature_dim * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(config.feature_dim * 2, config.feature_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(config.feature_dim, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), -1, 1)
        return self.deconv_layers(x)