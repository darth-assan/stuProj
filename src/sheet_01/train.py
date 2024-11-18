import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from .models import Generator, Discriminator
from .data_utils import DataProcessor
from .config import GANConfig
import pandas as pd

class GANTrainer:
    def __init__(self, config: GANConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = DataProcessor(config)
        
    def save_model(self, model: nn.Module, filename: str) -> None:
        self.config.model_save_path.mkdir(parents=True, exist_ok=True)
        save_path = self.config.model_save_path / filename
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': self.config.input_dim,
        }, save_path)
        print(f"Model saved to {save_path}")

    def train(self) -> Tuple[Generator, pd.DataFrame, pd.DataFrame]:
        train_set_1, train_set_2 = self.data_processor.load_and_preprocess_data()
        tensor_data = self.data_processor.normalize_data(train_set_2)
        train_loader = self.data_processor.create_dataloader(tensor_data)

        discriminator = Discriminator(self.config).to(self.device)
        generator = Generator(self.config).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer_d = optim.Adam(discriminator.parameters(), lr=self.config.learning_rate_d, betas=(0.5, 0.999))
        optimizer_g = optim.Adam(generator.parameters(), lr=self.config.learning_rate_g, betas=(0.5, 0.999))

        self._train_gan(train_loader, discriminator, generator, criterion, optimizer_d, optimizer_g)
        self.save_model(generator, 'generator.pth')
        
        return generator, train_set_1, train_set_2

    def _train_gan(self, train_loader, discriminator, generator, criterion, optimizer_d, optimizer_g):
        for epoch in range(self.config.num_epochs):
            for i, batch in enumerate(train_loader):
                real_data = batch[0].to(self.device)
                batch_size = real_data.size(0)
                
                # Train Discriminator
                optimizer_d.zero_grad()
                real_labels = torch.full((batch_size, 1), 0.9).to(self.device)
                fake_labels = torch.full((batch_size, 1), 0.1).to(self.device)
                
                outputs = discriminator(real_data)
                d_loss_real = criterion(outputs, real_labels)
                
                noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
                fake_data = generator(noise)
                outputs = discriminator(fake_data.detach())
                d_loss_fake = criterion(outputs, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()
                
                # Train Generator
                optimizer_g.zero_grad()
                outputs = discriminator(fake_data)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                optimizer_g.step()
                
                if i % 50 == 0:
                    print(f"Epoch [{epoch+1}/{self.config.num_epochs}], "
                          f"Batch [{i+1}], d_loss: {d_loss.item():.4f}, "
                          f"g_loss: {g_loss.item():.4f}")