import os
from picamera2 import Picamera2, Preview
#import RPi.GPIO as GPIO

import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import serial

sys.path.append('/home/robolectro/robocup/tflite1/tflite1-env/lib/python3.9/site-packages')


picam2 = Picamera2()
dispW=1280
dispH=720
picam2.preview_configuration.main.size = (dispW,dispH)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate=30
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(-1,cv2.CAP_V4L2)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

MODEL_NAME = 'old_good'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.95
resW = 1280
resH = 720
imW, imH = int(resW), int(resH)
use_TPU = False

# Import TensorFlow libraries

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

CWD_PATH = '/home/robolectro/robocup/tflite1'


PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5


outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

reset = False

def ball_pick(colour):
    a = ""
    flag1 = False
    xmax=0
    xmin=0
    ymin=0
    while True:
        
        if ser.inWaiting():
            a = ser.readline().decode('utf-8').rstrip()

        if a == 'reset':
            global reset
            reset = True
            break
        
        if a == 'stopped1':
            flag1 = True  # found ball
        if flag1 == True and 550 < (xmax - xmin) < 790 and 230 < ymin < 370:  # in front of ball

            print("stop")
            ser.write(bytes("stop\n", 'utf-8'))
            break

        t1 = cv2.getTickCount()
        # print(flag1)

        frame = picam2.capture_array()
        frame1 = cv2.flip(frame, -1)
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[
            0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        for index in range(10):
            if ((scores[index] > min_conf_threshold) and (scores[index] <= 1.0) and (
                    classes[index] == colour)):  # Only if silver detected, enter
                ymin = int(max(1, (boxes[index][0] * imH)))
                xmin = int(max(1, (boxes[index][1] * imW)))
                ymax = int(min(imH, (boxes[index][2] * imH)))
                xmax = int(min(imW, (boxes[index][3] * imW)))
                #print(xmax - xmin)

                position = int((xmin + xmax) / 2)

                a = bytes((str(position) + '\n'), 'utf-8')
                ser.write(a)


def drop_ball(colour):
    a = ""
    global reset
    while True:
        if ser.inWaiting():
            a = ser.readline().decode('utf-8').rstrip()

        if a == 'stopped':
            break
        
        if a == 'reset':
            reset=True
            break

        t1 = cv2.getTickCount()
        # print(flag1)

        frame = picam2.capture_array()
        frame1 = cv2.flip(frame, -1)
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[
            0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        for index in range(10):
            if ((scores[index] > min_conf_threshold) and (scores[index] <= 1.0) and (
                    classes[index] == colour)):  # Only if green detected, enter
                ymin = int(max(1, (boxes[index][0] * imH)))
                xmin = int(max(1, (boxes[index][1] * imW)))
                ymax = int(min(imH, (boxes[index][2] * imH)))
                xmax = int(min(imW, (boxes[index][3] * imW)))
                #print(xmax - xmin)

                position = int(xmax - xmin)

                a = bytes((str(position) + '\n'), 'utf-8')
                ser.write(a)


def wait_for_start(command):
    global flag1
    global a
    global reset
    
    flag1 = False
    a=""
    while a != command: #wait for command to search
        if ser.inWaiting():
            a = ser.readline().decode('utf-8').rstrip()
            
        if a == 'reset':
            reset = True
            break

'''GPIO.setmode(GPIO.BCM)
GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_UP)'''
#GPIO.setup(14, GPIO.OUT, pull_up_down=GPIO.PUD_UP)

while True:
    print('resetted')
    for i in range(2):    #2 silver pick
        
        
        if __name__=='__main__':
            ser = serial.Serial('/dev/ttyS0',9600,timeout=1)
            ser.flush()
        wait_for_start('Start')
        if reset == True:
            break
        

        ball_pick(0.0)
        if reset == True:
            break
        print("1st done")
    
    if reset == True:
        reset = False
        continue
        

    print("picked 2 silvers")

    wait_for_start('drop_silver')
    if reset == True:
        reset = False
        continue

    drop_ball(2.0)
    if reset == True:
        
        reset = False
        continue
    
    print("found green")
    
    wait_for_start('Start')
    print("Searching for black")
    if reset == True:
        reset = False
        continue

    ball_pick(1.0)
    if reset == True:
        reset = False
        continue
    
    wait_for_start('drop_silver')  #black drop
    print("Searching for red")
    if reset == True:
        reset = False
        continue

    drop_ball(3.0)
    if reset == True:
        reset = False
        continue
