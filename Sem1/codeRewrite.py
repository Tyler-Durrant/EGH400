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
    model = YOLO("yolo11n-pose.pt") #halves framerate but offers better performance

    positions = []

    while True:
        start = time.time()

        #Capture frames from specified RGB and depth cameras
        capture = k4a.get_capture()

        #Necessary for later use with cv2
        frame = np.ascontiguousarray(capture.color[:, :, :3])

        #Apply the specified model to the captured colour frame
        results = model(frame, verbose=False)

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
                print(chest_3d)
                positions.append((start, chest_3d[0], chest_3d[2]))
                
            else:
                print("Invalid depth at this pixel, skipping")
        else:
            print('no person detected')

        cv2.imshow("altered", frame)

        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break

        end = time.time()
        print("Time per frame:", end - start, "FPS:", 1/(end - start))

    k4a.stop()
    #print(positions)
    saveData(positions)

if __name__ == "__main__":
    main()