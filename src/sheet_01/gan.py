import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.preprocessing import RobustScaler
from .config import GANConfig

# ------------------------
# Discriminator Model
# ------------------------
class Discriminator(Model):
    """
    The Discriminator is a Convolutional Neural Network (CNN) that classifies real and fake data.
    It consists of:
    - 8 convolutional layers with LeakyReLU activations and dropout for regularization.
    - 4 fully connected layers to produce a final real/fake classification output.
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.conv_layers = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(64, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(config.dropout_rate),
            layers.Conv1D(128, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(config.dropout_rate),
            layers.Conv1D(256, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(config.dropout_rate),
            layers.Conv1D(512, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(config.dropout_rate),
            layers.Conv1D(1024, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(config.dropout_rate),
            layers.Conv1D(2048, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(config.dropout_rate),
            layers.Conv1D(4096, kernel_size=3, strides=2, padding="same"),
            layers.LeakyReLU(0.2)
        ])
        self.fc_layers = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Output real/fake classification
        ])
    
    def call(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# ------------------------
# Generator Model
# ------------------------
class Generator(Model):
    """
    The Generator is a deconvolutional neural network (Reverse CNN) that generates synthetic data.
    It consists of:
    - 3 fully connected layers to transform noise into a higher-dimensional space.
    - 9 convolutional layers: The first 4 include upsampling to increase spatial dimensions.
    """
    def __init__(self, noise_dim, output_shape):
        super(Generator, self).__init__()
        
        self.fc_layers = tf.keras.Sequential([
            layers.Input(shape=(noise_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='relu'),
        ])
        
        self.deconv_layers = tf.keras.Sequential([
            layers.Reshape((1, 1024)),  # Reshape to a single-channel feature map
            
            # 1st Convolution with Upsampling
            layers.UpSampling1D(2),
            layers.Conv1DTranspose(512, kernel_size=3, padding="same", activation="relu"),
            
            # 2nd Convolution with Upsampling
            layers.UpSampling1D(2),
            layers.Conv1DTranspose(256, kernel_size=3, padding="same", activation="relu"),
            
            # 3rd Convolution with Upsampling
            layers.UpSampling1D(2),
            layers.Conv1DTranspose(128, kernel_size=3, padding="same", activation="relu"),
            
            # 4th Convolution with Upsampling
            layers.UpSampling1D(2),
            layers.Conv1DTranspose(64, kernel_size=3, padding="same", activation="relu"),
            
            # Remaining 5 Convolutional Layers (No Upsampling)
            layers.Conv1DTranspose(64, kernel_size=3, padding="same", activation="relu"),
            layers.Conv1DTranspose(32, kernel_size=3, padding="same", activation="relu"),
            layers.Conv1DTranspose(16, kernel_size=3, padding="same", activation="relu"),
            layers.Conv1DTranspose(8, kernel_size=3, padding="same", activation="relu"),
            layers.Conv1DTranspose(output_shape[-1], kernel_size=3, padding="same", activation="tanh"),
        ])
    
    def call(self, x):
        x = self.fc_layers(x)
        return self.deconv_layers(x)

# ------------------------
# GAN Components and Training
# ------------------------
config = GANConfig()
discriminator = Discriminator(config.input_shape)
generator = Generator(config.noise_dim, output_shape=(33, 1))

# Loss Functions
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    """
    Discriminator loss: Classify real samples as 1, fake samples as 0.
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    """
    Generator loss: Fool the discriminator into classifying fake samples as real (1).
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)

# Training Step
@tf.function
def train_step(real_data):
    """
    Perform one step of GAN training: Update generator and discriminator.
    """
    noise = tf.random.normal([real_data.shape[0], noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training Loop
def train_gan(dataset, epochs):
    """
    Train the GAN for a specified number of epochs.
    """
    for epoch in range(epochs):
        gen_loss_total = 0
        disc_loss_total = 0
        for data_batch in dataset:
            gen_loss, disc_loss = train_step(data_batch)
            gen_loss_total += gen_loss
            disc_loss_total += disc_loss
        print(f"Epoch {epoch+1}, Generator Loss: {gen_loss_total:.4f}, Discriminator Loss: {disc_loss_total:.4f}")

# ------------------------
# Preprocessing Data
# ------------------------
def preprocess_data(data):
    """
    Normalize ICS data using RobustScaler to handle outliers effectively.
    """
    scaler = RobustScaler()
    data = scaler.fit_transform(data)  # Scale to [0, 1]
    return data[..., np.newaxis]  # Add channel dimension

# Prepare Data
train4_preprocessed = preprocess_data(train4_common)  # Normalize and reshape
dataset = tf.data.Dataset.from_tensor_slices(train4_preprocessed).shuffle(10000).batch(config.batch_size)

# Train the GAN
train_gan(dataset, config.epochs)