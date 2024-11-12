## Project Overview
This is a deep learning project that classifies celebrity images using a Convolutional Neural Network (CNN). The project uses TensorFlow and Keras to build and train a model that can distinguish between three different celebrity classes.

## Data Management
**Dataset Organization**
- The code splits the dataset into training (80%) and validation sets
- Images are organized in class-specific directories
- Uses `os` module for file and directory management

**Data Augmentation**
The project implements image augmentation using `ImageDataGenerator` with several techniques:
- Rescaling pixel values to range
- Random rotation up to 40 degrees
- Width and height shifts up to 20%
- Shear and zoom transformations
- Horizontal flipping

## Model Architecture
The CNN model consists of multiple layers:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # ... multiple Conv2D and MaxPooling2D layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3)
])
```

**Key Components**:
- 5 convolutional layers with increasing filters (32 to 512)
- MaxPooling layers for dimensionality reduction
- Dropout layers (0.2) for regularization
- Dense layers for final classification

## Training Configuration
**Training Parameters**:
- Batch size: 100 images
- Image shape: 150x150 pixels
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Training duration: 50 epochs

## Visualization
The code includes visualization capabilities:
- Training progress monitoring
- Accuracy and loss plots for both training and validation sets
- Custom function to display augmented images in a grid format

## Technical Tools Used
- TensorFlow/Keras for deep learning implementation
- NumPy for numerical operations
- Matplotlib for visualization
- OS module for file system operations
