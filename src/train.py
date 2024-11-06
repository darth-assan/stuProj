import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models import Generator, Discriminator
import os

# Constants
TRAIN_SET_1_PATH = '/Users/darth/Dev/stuProj/data/GAN/train1_clean.csv'
TRAIN_SET_2_PATH = '/Users/darth/Dev/stuProj/data/GAN/train4_clean.csv'
BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0001
MODEL_SAVE_PATH = 'saved_models'
INPUT_DIM = 100

def save_model(model, filename):
    """Save the trained model to disk."""
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    
    save_path = os.path.join(MODEL_SAVE_PATH, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': INPUT_DIM,
    }, save_path)
    print(f"Model saved to {save_path}")

def load_and_preprocess_data():
    train_set_1 = pd.read_csv(TRAIN_SET_1_PATH)
    train_set_2 = pd.read_csv(TRAIN_SET_2_PATH)
    common_columns = train_set_1.columns.intersection(train_set_2.columns)
    train_set_1 = train_set_1[common_columns]
    train_set_2 = train_set_2[common_columns]
    return train_set_1, train_set_2

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = scaler.fit_transform(data[:100].values)
    tensor_data = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(1)
    return tensor_data

def train_gan(train_loader, discriminator, generator, criterion, optimizer_d, optimizer_g, device):
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(train_loader):
            real_data = batch[0].to(device)
            optimizer_d.zero_grad()
            real_labels = torch.full((real_data.size(0), 1), 0.9).to(device)
            fake_labels = torch.full((real_data.size(0), 1), 0.1).to(device)
            
            # Train Discriminator
            outputs = discriminator(real_data)
            d_loss_real = criterion(outputs, real_labels)
            noise = torch.randn(real_data.size(0), INPUT_DIM).to(device)
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
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}], "
                      f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

def main(save_model_after_training=True):
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
    
    if save_model_after_training:
        save_model(generator, 'generator.pth')
    
    return generator, train_set_1, train_set_2

if __name__ == "__main__":
    main()