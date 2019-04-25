import RPi.GPIO as GPIO
import time
import smbus
import FaBo9Axis_MPU9250

GPIO.setmode(GPIO.BCM)
pin = 4

bus = smbus.SMBus(1)

mpu9250 = FaBo9Axis_MPU9250.MPU9250()
accel = mpu9250.readAccel()
#print("accel Z: " + str(accel['z']))
time.sleep(0.1)

bus.write_byte_data(0x68, 0x38, 0x01)

def callBackTest(channel):
    accel = mpu9250.readAccel()
    print("accel Z: " + str(accel['z']))
    #print("callback")

GPIO.setup(pin, GPIO.IN, GPIO.PUD_UP)
GPIO.add_event_detect(pin, GPIO.FALLING, callback=callBackTest, bouncetime=300) 

try:
    while(True):
        time.sleep(1)

except KeyboardInterrupt:
    print("break")
    GPIO.cleanup()