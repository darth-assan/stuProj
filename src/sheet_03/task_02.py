#POC Autoencoder
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ConfigurableAutoencoder(nn.Module):
    def __init__(self, input_dim, compression_factor, num_hidden_layers, layer_type='dense', activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = int(input_dim / compression_factor)
        
        # Calculate layer sizes
        layer_sizes = np.linspace(input_dim, self.compressed_dim, num_hidden_layers + 1, dtype=int)
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
            
        # Build encoder
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            if layer_type == 'dense':
                encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                encoder_layers.append(self.activation)
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        for i in range(len(layer_sizes) - 1, 0, -1):
            if layer_type == 'dense':
                decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i-1]))
                decoder_layers.append(self.activation)
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(model, train_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    print("\nTraining autoencoder...")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            decoded, encoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.6f}")

def extract_features(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    features = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0].to(device)
            _, encoded = model(batch)
            features.append(encoded.cpu())
    return torch.cat(features, dim=0)

class AutoencoderClassifier:
    def __init__(self, autoencoder, threshold):
        self.autoencoder = autoencoder
        self.threshold = threshold
    
    def get_reconstruction_error(self, input_data):
        self.autoencoder.eval()
        with torch.no_grad():
            decoded, _ = self.autoencoder(input_data)
            # Flatten the tensors before calculating mean squared error
            input_flat = input_data.view(input_data.size(0), -1)
            decoded_flat = decoded.view(decoded.size(0), -1)
            error = torch.mean((input_flat - decoded_flat) ** 2, dim=1)
        return error
    
    def classify(self, input_data):
        errors = self.get_reconstruction_error(input_data)
        return errors > self.threshold
    
    def compute_metrics(self, true_labels, predictions):
        tp = torch.sum((true_labels == 1) & (predictions == 1)).float()
        fp = torch.sum((true_labels == 0) & (predictions == 1)).float()
        fn = torch.sum((true_labels == 1) & (predictions == 0)).float()
        tn = torch.sum((true_labels == 0) & (predictions == 0)).float()
        
        accuracy = (tp + tn) / len(true_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return accuracy.item(), precision.item(), recall.item()

def plot_roc_curve(classifier, data_loader, true_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_errors = []
    
    # Collect errors batch by batch
    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)
        errors = classifier.get_reconstruction_error(batch)
        all_errors.extend(errors.cpu().numpy())
    
    # Ensure predictions match true_labels length
    all_errors = np.array(all_errors[:len(true_labels)])
    
    thresholds = np.linspace(min(all_errors), max(all_errors), 100)
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        predictions = (all_errors > threshold).astype(int)
        tp = np.sum((true_labels == 1) & (predictions == 1))
        fp = np.sum((true_labels == 0) & (predictions == 1))
        fn = np.sum((true_labels == 1) & (predictions == 0))
        tn = np.sum((true_labels == 0) & (predictions == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    
    # Network packets
    num_packets = 1000
    packet_length = 128
    normal_packets = np.random.normal(0, 1, (num_packets, packet_length))
    anomalous_packets = np.random.normal(2, 1, (num_packets // 10, packet_length))
    
    # Combine and create labels
    all_packets = np.vstack([normal_packets, anomalous_packets])
    labels = np.zeros(len(all_packets))
    labels[num_packets:] = 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        all_packets, labels, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch datasets and move to device
    train_data = torch.FloatTensor(X_train).to(device)
    test_data = torch.FloatTensor(X_test).to(device)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    # Create and train model
    print("\nCreating model...")
    model = ConfigurableAutoencoder(
        input_dim=packet_length,
        compression_factor=4,
        num_hidden_layers=3,
        layer_type='dense',
        activation='relu'
    ).to(device)
    
    # Train the model
    train_autoencoder(model, train_loader)
    
    # Extract features
    print("\nExtracting features...")
    train_features = extract_features(model, train_loader)
    test_features = extract_features(model, test_loader)
    print(f"Extracted features shape: {train_features.shape}")
    
    # Classification
    print("\nPerforming classification...")
    classifier = AutoencoderClassifier(model, threshold=0.5)
    
    # Convert test data and labels to appropriate format
    test_data = torch.FloatTensor(X_test).to(device)
    test_labels = torch.FloatTensor(y_test).to(device)
    
    # Make predictions
    predictions = classifier.classify(test_data)
    
    # Calculate metrics
    accuracy, precision, recall = classifier.compute_metrics(test_labels, predictions)
    print(f"\nClassification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Plot ROC curve
    print("\nGenerating ROC curve...")
    plot_roc_curve(classifier, DataLoader(test_data, batch_size=32), y_test)
