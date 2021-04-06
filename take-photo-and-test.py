import io
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import backend as K

from picamera import PiCamera

import pathlib

import time

camera=PiCamera()

class_names=np.genfromtxt('saved_model/bc_model/classes.txt', dtype='str', delimiter="\n")

img_height=383
img_width=383

model = tf.keras.models.load_model('saved_model/bc_model')

# read the stream from the camera into a PIL file
stream = io.BytesIO()
camera.capture(stream,format='jpeg')
#img = Image.open(stream).convert('L').resize((img_width, img_height), Image.ANTIALIAS)
image = Image.open(stream)
image = image.resize((383,383), Image.NEAREST)
image = keras.preprocessing.image.img_to_array(image)
start_time=time.time() * 1000

img=keras.preprocessing.image.smart_resize(image, (img_height,img_width))

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
