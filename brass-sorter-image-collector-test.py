#! /usr/bin/env python3
from TurnerServo import *
from PusherServo import *
from time import sleep
import os

for value in range(0,50):
    print(value)
    PusherServo.value=PusherServoPositions["back"]
    sleep(PusherServoDelay)
    os.system("vgrabbj  -f /tmp/test_{value}.jpg".format(value=value))
    PusherServo.value=PusherServoPositions["forward"]
    sleep(PusherServoDelay)