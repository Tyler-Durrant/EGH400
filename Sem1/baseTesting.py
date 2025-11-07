import cv2
import numpy as np

from ultralytics import YOLO

import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A



def main():

    #Configure Azure Kinect sensor object
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,#NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()


    #Selecting pose estimation model
    model = YOLO("yolo11n-pose.pt")

    #model = YOLO("yolo11s-pose.pt") halves framerate but offers better performance

    while True:

        #Capture frames from specified RGB and depth cameras
        capture = k4a.get_capture()

        #Necessary for later use with cv2
        frame = np.ascontiguousarray(capture.color[:, :, :3])

        #Apply the specified model to the captured colour frame
        results = model(frame, verbose = False)

        if np.any(capture.depth):
            cv2.imshow("depth", colorize(capture.depth, (None, 5000), cv2.COLORMAP_HSV))
            #cv2.imshow("colour", capture.color[:, :, :3])
            cv2.imshow("altered", frame)

        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()