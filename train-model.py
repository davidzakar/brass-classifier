import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import backend as K


import pathlib

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)

data_dir = pathlib.Path("./data/knownCasings_raspicam/")

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 8 
img_height = 960
img_width = 600
crop_x=288
crop_y=275

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

num_classes = len(class_names)

# augmentation
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.CenterCrop(crop_y, crop_x, input_shape=(img_height,img_width,3)),
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

# dropout
model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.SeparableConv2D(16, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.SeparableConv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.SeparableConv2D(64, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.SeparableConv2D(128, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

model.summary()

epochs = 30
history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
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
# uncomment this if you want to see the plots for assessing training
plt.show()

model.save('saved_model/bc_model')

# save the labels!

a_file = open("saved_model/bc_model/classes.txt", "w")
np.savetxt(a_file, class_names, delimiter=" ", newline = "\n", fmt="%s")

a_file.close()
