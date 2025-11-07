import cv2
import numpy as np

from ultralytics import YOLO

import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A




def drawPeople(cap, colour_image, results):
    chest = []
    chestDepth = []
    for r in results:
            if r.keypoints is not None and len(r.boxes) > 0:
                keypoints_all = r.keypoints.xy  # shape: (num_people, num_keypoints, 2)
                boxes = r.boxes.xyxy  # shape: (num_people, 4)

                for i in range(len(boxes)):
                    keypoints = keypoints_all[i]
                
                    keypoints[10]
                    cv2.circle(colour_image, (int(keypoints[10][0]), int(keypoints[10][1])), 5, (255, 0, 0), -1)

                    print(keypoints[10])

    return colour_image, chest, chestDepth


def main():

    #Configure Azure Kinect sensor object
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,#NFOV_UNBINNED,
            synchronized_images_only=True,
            camera_fps = pyk4a.FPS.FPS_5
        )
    )
    k4a.start()

    #Selecting pose estimation model
    model = YOLO("yolo11s-pose.pt")

    #model = YOLO("yolo11s-pose.pt") halves framerate but offers better performance

    while True:

        #Capture frames from specified RGB and depth cameras
        capture = k4a.get_capture()

        #Necessary for later use with cv2
        frame = np.ascontiguousarray(capture.color[:, :, :3])
        print(frame.shape)

        #Apply the specified model to the captured colour frame
        results = model(frame)

        #can increase performance
        #results = model(source=frame, imgsz=(256, 448))
        
        #Draw results on the frame
        frame, coords, depth = drawPeople(capture, frame, results)


        if np.any(capture.depth):
            cv2.imshow("altered", frame)

        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
