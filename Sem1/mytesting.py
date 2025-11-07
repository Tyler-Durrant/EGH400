import cv2
import numpy as np

from ultralytics import YOLO

import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A


def drawPeople(colour_image, results):
    for r in results:
            if r.keypoints is not None and len(r.boxes) > 0:
                keypoints_all = r.keypoints.xy  # shape: (num_people, num_keypoints, 2)
                boxes = r.boxes.xyxy  # shape: (num_people, 4)

                for i in range(len(boxes)):
                    keypoints = keypoints_all[i]

                # Draw keypoints
                    for idx, (x, y) in enumerate(keypoints):

                        if idx in {5,6,11,12}:
                            colour = (0, 0, 255)
                        else:
                            colour = (0, 255, 0)
                        #cv2.circle(colour_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.circle(colour_image, (int(x), int(y)), 5, colour, -1)
                    
                    chest = 0.7 * (keypoints[5]+keypoints[6])/2 + 0.3 * (keypoints[11]+keypoints[12])/2
                    cv2.circle(colour_image, (int(chest[0]), int(chest[1])), 5, (255, 0, 0), -1)
    return colour_image



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

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    #Selecting pose estimation model
    model = YOLO("yolo11n-pose.pt")

    #model = YOLO("yolo11s-pose.pt") halves framerate but offers better performance

    while True:
        
        capture = k4a.get_capture()
        frame = np.ascontiguousarray(capture.color[:, :, :3])
        results = model(frame)

        #can increase performance
        #results = model(source=frame, imgsz=(320, 576))

        frame = drawPeople(frame, results)


        if np.any(capture.depth):
            cv2.imshow("depth", colorize(capture.depth, (None, 5000), cv2.COLORMAP_HSV))
            #cv2.imshow("colour", capture.color[:, :, :3])
            cv2.imshow("altered", frame)
            key = cv2.waitKey(10)
            if key != -1:
                cv2.destroyAllWindows()
                break
    k4a.stop()


if __name__ == "__main__":
    main()
