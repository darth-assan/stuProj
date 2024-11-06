import torch
import torch.nn as nn

FEATURE_DIM = 32
INPUT_DIM = 100
DROPOUT_RATE = 0.3

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