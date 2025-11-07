import cv2
import numpy as np
import time
import csv

from ultralytics import YOLO

import pyk4a
from pyk4a import Config, PyK4A
from pyk4a.calibration import CalibrationType

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def saveData(data):
    with open('positions.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x_pixel', 'y_pixel'])
        writer.writerows(data) 

def main():

    #Configure Azure Kinect sensor object
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,#NFOV_UNBINNED,
            synchronized_images_only=True,
            camera_fps = pyk4a.FPS.FPS_15
        )
    )
    k4a.start()

    #Selecting pose estimation model
    #model = YOLO("yolo11n-pose.pt")

    #Using this since limited to 15fps requires 66.67Hz per loop
    model = YOLO("yolo11m-pose.pt") #halves framerate but offers better performance
    #model = YOLO("yolov8m-pose.pt")
    positions = []

    begin = time.time()

    iterations = 0
    timeSum = 0
    fpsSum = 0

    longest = 0
    shortest = 1

    while True:
        start = time.time()

        #Capture frames from specified RGB and depth cameras
        capture = k4a.get_capture()

        #captureTime = time.time()

        #Necessary for later use with cv2
        frame = np.ascontiguousarray(capture.color[:, :, :3])

        #processTime = time.time()

        #Apply the specified model to the captured colour frame
        results = model(frame, verbose=False)

        #resultsTime = time.time()
        #can increase performance
        #results = model(source=frame, imgsz=(256, 448))
   
        detection = results[0].keypoints
        keypoints = detection.xy[0]

        if detection.has_visible and not (keypoints[[5,6,11,12]] == 0).all(dim=1).any():
            #print('person detected')
            
            for idx, (x, y) in enumerate(keypoints):
                if idx in {5,6,11,12}:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            chest = 0.7 * (keypoints[5]+keypoints[6])/2 + 0.3 * (keypoints[11]+keypoints[12])/2

            cv2.circle(frame, (int(chest[0]), int(chest[1])), 5, (255, 0, 0), -1)

            chestDepth = capture.transformed_depth[int(chest[1]), int(chest[0])]
            
            if chestDepth != 0:
                chest_3d = k4a.calibration.convert_2d_to_3d(
                    coordinates=(int(chest[0]), capture.color.shape[0] - int(chest[1])),
                    depth=chestDepth,
                    source_camera=CalibrationType.COLOR,
                )
                positions.append((start, chest_3d[0], chest_3d[2]))

        end = time.time()
        ts = end - start
        if ts > longest and iterations>0:
            longest = ts
        if ts < shortest:
            shortest = ts
        #fps = 1/ts
        #print("Time per frame:", ts, "FPS:", fps)
        #if iterations != -1:
            #Accounts for delay on startup frame
            #timeSum = timeSum + ts
            #fpsSum = fpsSum + fps
        iterations = iterations + 1
        #print(iterations, timeSum)
        
        if end-begin > 60:
            break
        #print("Time per frame:", ts, "FPS:", fps)
        #print("CaptureTs:", captureTime-start, "processTs:", processTime-captureTime, "resultsTs:", resultsTime-processTime)

    k4a.stop()
    finish = time.time()
    print("Average Time per Frame:", (finish-begin)/iterations, "Average FPS:", iterations/(finish-begin), "Longest:", longest, "Shortest:", shortest)
    #print(positions)
    #saveData(positions)

if __name__ == "__main__":
    main()