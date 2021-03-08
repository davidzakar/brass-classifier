#! /usr/bin/env python3
from gpiozero import Servo
from time import sleep
import os

myGPIO=27

# Min and Max pulse widths converted into milliseconds
# To increase range of movement:
#   increase maxPW from default of 2.0
#   decrease minPW from default of 1.0
# Change myCorrection using increments of 0.05 and
# check the value works with your servo.
myCorrection=0.45
maxPW=(2.0+myCorrection)/1000
minPW=(1.0-myCorrection)/1000
#minPW=0.00085
#maxPW=0.00215

myServo = Servo(myGPIO,min_pulse_width=minPW,max_pulse_width=maxPW)

myServo.value=-0.8
sleep(2)

for value in range(0,40):
    print value
    myServo.value=0.5
    sleep(0.2)
    os.system("vgrabbj  -f /tmp/test_{value}.jpg".format(value=value))
    myServo.value=-0.8
    sleep(0.2)
