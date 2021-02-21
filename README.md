# brass-classifier
Brass headstamp classification using Tensorflow

This project is an attempt to build an open-source (license TBD) brass classification and sorting machine.

The code currently makes the assumption you have a directory structure along the lines of:
/data: the dataset directory (currently knownCasings) goes here. (I also put test images in here, which may not be a great idea long-term)
/saved_model: This is where the bc_model gets saved to. In addition to the model, I also store the class names in the model directory so that they don't get lost.

Files:
train_model.py: this does simplistic training for our image classification model. Currently, it does some augmentation and dropout, and I have it run for about 25 epochs, which gets me decent enough accuracy.
load-and-test-model.py: this does what it says on the tin - loads an image, and runs it through the model to determine accuracy and speed in ms
take-photo-and-test.py: this engages the pi's camera to take a picture, and runs it through the model for speed and accuracy.

I STRONGLY recommend using GPU acceleration for training the model. It takes a bit under a minute with my 1060 GTX; it is _considerably_ longer if I go CPU only, or (worse) just use the pi4b to train it.

Current hardware target for the sorter:
Pi4b (maybe Jetson Nano later on)
pi HQ camera w/ 6mm lens

I don't know how much RAM the pi requires yet. Development is being done for the 8gb model.
