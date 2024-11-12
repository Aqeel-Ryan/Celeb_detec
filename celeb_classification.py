import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 100
IMG_SHAPE  = 150 


# Set your local data directory
data_dir = r"D:\projects\celeb_detec\dataset" 

# List all the class subfolders
classes = os.listdir(data_dir)

# Setup train and validation directories
train_dir = os.path.join(data_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(data_dir, 'validation') 
os.mkdir(validation_dir)

# Split data into train and validation
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name) 
    images = os.listdir(class_dir)
    
    # Take 80% for train
    train_images = images[:40]  
    validation_images = images[40:]
    
    train_class_dir = os.path.join(train_dir, class_name)
    os.mkdir(train_class_dir)
    
    validation_class_dir = os.path.join(validation_dir, class_name)
    os.mkdir(validation_class_dir)  
    
    # Move images to train and validation folders
    for img in train_images:
        source = os.path.join(class_dir, img)
        destination = os.path.join(train_class_dir, img)
        os.rename(source, destination)
        
    for img in validation_images:
        source = os.path.join(class_dir, img)
        destination = os.path.join(validation_class_dir, img) 
        os.rename(source, destination)
        
        


## Data Augmentation

image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='binary')


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
# Validation set
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=validation_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')


# TRAINING
# Get number of files in train and validation directories
total_train= len(os.listdir(train_dir))  
total_val = len(os.listdir(validation_dir))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()
# Model: "sequential_7"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d_28 (Conv2D)          (None, 148, 148, 32)      896       
                                                                 
#  max_pooling2d_28 (MaxPooli  (None, 74, 74, 32)        0         
#  ng2D)                                                           
                                                                 
#  conv2d_29 (Conv2D)          (None, 72, 72, 64)        18496     
                                                                 
#  max_pooling2d_29 (MaxPooli  (None, 36, 36, 64)        0         
#  ng2D)                                                           
                                                                 
#  conv2d_30 (Conv2D)          (None, 34, 34, 128)       73856     
                                                                 
#  max_pooling2d_30 (MaxPooli  (None, 17, 17, 128)       0         
#  ng2D)                                                           
                                                                 
#  conv2d_31 (Conv2D)          (None, 15, 15, 256)       295168    
                                                                 
#  max_pooling2d_31 (MaxPooli  (None, 7, 7, 256)         0         
#  ng2D)                                                           
                                                                 
#  conv2d_32 (Conv2D)          (None, 5, 5, 512)         1180160   
                                                                 
#  max_pooling2d_32 (MaxPooli  (None, 2, 2, 512)         0         
#  ng2D)                                                           
                                                                 
#  flatten_7 (Flatten)         (None, 2048)              0         
                                                                 
#  dropout_10 (Dropout)        (None, 2048)              0         
                                                                 
#  dense_14 (Dense)            (None, 512)               1049088   
                                                                 
#  dropout_11 (Dropout)        (None, 512)               0         
                                                                 
#  dense_15 (Dense)            (None, 3)                 1539      
                                                                 
# =================================================================
# Total params: 2619203 (9.99 MB)
# Trainable params: 2619203 (9.99 MB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________


epochs = 50
history = model.fit(
    train_data_gen,  
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
