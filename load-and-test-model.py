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

import time

class_names=np.genfromtxt('saved_model/bc_model/classes.txt', dtype='str', delimiter="\n")

img_height=383
img_width=383

model = tf.keras.models.load_model('saved_model/bc_model')

# replace this with your own test image
testbrass_path="./data/gfl-test-case2.jpg"

start_time=time.time() * 1000

img = keras.preprocessing.image.load_img(
            testbrass_path, target_size=(img_height, img_width)
            )

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

end_time=time.time() * 1000

print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
)

print(end_time-start_time)
