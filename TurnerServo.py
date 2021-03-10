#! /usr/bin/env python3
from gpiozero import Servo
from time import sleep

TurnerServoGPIO=17

# Min and Max pulse widths converted into milliseconds
# To increase range of movement:
#   decrease TurnerServoMinPW from default of 1.0
#   increase TurnerServoMaxPW from default of 2.0
# Test what works with your servo, using increments of 0.05
TurnerServoMinPW=0.00055
TurnerServoMaxPW=0.00245

# Delay after servo movement. Find value that is enough to go from min to max.
TurnerServoDelay=0.9

# Servo positions for 6 buckets
TurnerServoPositions = { # values between -1 and 1 work
    0: -0.9,
    1: -0.6,
    2: -0.3,
    3: 0,
    4: 0.4,
    5: 0.7
}

TurnerServo = Servo(TurnerServoGPIO,min_pulse_width=TurnerServoMinPW,max_pulse_width=TurnerServoMaxPW)

# servo initialization routines
TurnerServo.value=TurnerServoPositions[0]
sleep(TurnerServoDelay)
TurnerServo.value=TurnerServoPositions[5]
sleep(TurnerServoDelay)
TurnerServo.value=TurnerServoPositions[0]
sleep(TurnerServoDelay)

if __name__ == "__main__":
    print("Turner servo GPIO pin: ", TurnerServoGPIO)
    print("Turner servo minimum pulse width: ", TurnerServoMinPW)
    print("Turner servo maximum pulse width: ", TurnerServoMaxPW)
    print("Delay after turner servo movement: ", TurnerServoDelay)

    while True:
        for position in TurnerServoPositions.keys():
            TurnerServo.value=TurnerServoPositions[position]
            print("Servo value set to "+str(TurnerServoPositions[position]))
            sleep(TurnerServoDelay)