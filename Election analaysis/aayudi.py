import os
import cv2
import matplotlib
matplotlib.use('agg')  # Use the 'agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

# Load and preprocess images
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                img = img.astype('float32') / 255.0  # Normalize images
                images.append(np.expand_dims(img, axis=-1))
    return np.array(images)

# Add noise to images
def add_noise(images):
    noisy_images = []
    for image in images:
        noise = np.random.normal(loc=0, scale=0.1, size=image.shape)
        noisy = np.clip(image + noise, 0, 1)
        noisy_images.append(noisy)
    return np.array(noisy_images)

# Build convolutional autoencoder model
def build_autoencoder():
    input_img = Input(shape=(256, 256, 1))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# Plot and compare images
def plot_comparison(original, noisy, denoised):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(original.reshape(256, 256), cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title('Noisy')
    plt.axis('off')
    plt.imshow(noisy.reshape(256, 256), cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title('Denoised')
    plt.axis('off')
    plt.imshow(denoised.reshape(256, 256), cmap='gray')
    
    plt.savefig(f'comparison_{np.random.randint(0, 10000)}.png')

# Main function
def main():
    folder = r'C:\Users\Dell\Desktop\medical'
    images = load_images(folder)
    noisy_images = add_noise(images)
 
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(noisy_images, images, test_size=0.2, random_state=42)

    # Build and train autoencoder
    autoencoder = build_autoencoder()
    autoencoder.fit(X_train, Y_train, epochs=30, batch_size=32, validation_split=0.1)

    # Denoise test images
    denoised_images = autoencoder.predict(X_test)

    # Compare original, noisy, and denoised images
    for i in range(5):
        plot_comparison(Y_test[i], X_test[i], denoised_images[i])


 
if __name__ == "__main__":
    main()
