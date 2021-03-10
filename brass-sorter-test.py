#! /usr/bin/env python3
from TurnerServo import *
from PusherServo import *
from time import sleep
import os
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if __name__ == "__main__":
    for value in range(0,20):
        print(value)
        PusherServo.value=PusherServoPositions["back"]
        sleep(PusherServoDelay)
        os.system("vgrabbj  -f /tmp/test_image.jpg")
        
        image = Image.open('/tmp/test_image.jpg')
        
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        headstamps = ["Blazer", "CBC", "CBC CR", "CCI", "GECO", "GFL", "Lapua", "Lapua Luger", "Maxxtech", "MMS", "PPU", "S&B", "S&B NT", "SPEER", "TOPSHOT", "TSC", "T SX"]

        # run the inference
        prediction = model.predict(data)
        #print("Prediction: ", prediction)
        print("Prediction max value: ", max(prediction[0]))
        #print("Index of prediction max in table: ", np.argmax(prediction[0], axis=0))
        print("Predicted headstamp: ", headstamps[np.argmax(prediction[0], axis=0)])

        if max(prediction[0]) < 0.5:
            TurnerServo.value=TurnerServoPositions[0]
            sleep(TurnerServoDelay)
            PusherServo.value=PusherServoPositions["forward"]
            sleep(PusherServoDelay)
        else:
            headstamp=headstamps[np.argmax(prediction[0], axis=0)]
            if headstamp == "Lapua":
                TurnerServo.value=TurnerServoPositions[1]
                sleep(TurnerServoDelay)
                PusherServo.value=PusherServoPositions["forward"]
                sleep(PusherServoDelay)
            elif headstamp == "GFL":
                TurnerServo.value=TurnerServoPositions[2]
                sleep(TurnerServoDelay)
                PusherServo.value=PusherServoPositions["forward"]
                sleep(PusherServoDelay)
            elif headstamp == "MMS":
                TurnerServo.value=TurnerServoPositions[3]
                sleep(TurnerServoDelay)
                PusherServo.value=PusherServoPositions["forward"]
                sleep(PusherServoDelay)
            elif headstamp == "GECO":
                TurnerServo.value=TurnerServoPositions[4]
                sleep(TurnerServoDelay)
                PusherServo.value=PusherServoPositions["forward"]
                sleep(PusherServoDelay)
            elif headstamp == "CBC":
                TurnerServo.value=TurnerServoPositions[5]
                sleep(TurnerServoDelay)
                PusherServo.value=PusherServoPositions["forward"]
                sleep(PusherServoDelay)
            else:
                TurnerServo.value=TurnerServoPositions[0]
                sleep(TurnerServoDelay)
                PusherServo.value=PusherServoPositions["forward"]
                sleep(PusherServoDelay)