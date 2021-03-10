#! /usr/bin/env python3
from gpiozero import Servo
from time import sleep

PusherServoGPIO=27

# Min and Max pulse widths converted into milliseconds
# To increase range of movement:
#   decrease PusherServoMinPW from default of 1.0
#   increase PusherServoMaxPW from default of 2.0
# Test what works with your servo, using increments of 0.05
#PusherServoMinPW=0.00055
#PusherServoMaxPW=0.00245
PusherServoMinPW=0.00075
PusherServoMaxPW=0.00225

# Delay after servo movement. Find value that is enough to go from min to max.
PusherServoDelay=0.4

# Servo positions for 6 buckets
PusherServoPositions = { # values between -1 and 1 work
    "forward": -0.8,
    "back": 0.5
}

PusherServo = Servo(PusherServoGPIO,min_pulse_width=PusherServoMinPW,max_pulse_width=PusherServoMaxPW)

# servo initialization routines
PusherServo.value=PusherServoPositions["forward"]
sleep(PusherServoDelay)
PusherServo.value=PusherServoPositions["back"]
sleep(PusherServoDelay)
PusherServo.value=PusherServoPositions["forward"]
sleep(PusherServoDelay)

if __name__ == "__main__":
    print("Pusher servo GPIO pin: ", PusherServoGPIO)
    print("Pusher servo minimum pulse width: ", PusherServoMinPW)
    print("Pusher servo maximum pulse width: ", PusherServoMaxPW)
    print("Delay after pusher servo movement: ", PusherServoDelay)

    while True:
        print("forward")
        PusherServo.value=PusherServoPositions["forward"]
        sleep(PusherServoDelay)
        print("back")
        PusherServo.value=PusherServoPositions["back"]
        sleep(PusherServoDelay)