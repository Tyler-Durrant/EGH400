import cv2
import numpy as np

from ultralytics import YOLO

import pyk4a
from pyk4a import Config, PyK4A
from pyk4a.calibration import CalibrationType




def drawPeople(cap, colour_image, results):
    chest = (0,0)
    chestDepth = 100
    #print(results.keypoints)
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

                    #print(keypoints[5],keypoints[6],keypoints[11],keypoints[12])
                    
                    #chest = 0.7 * (keypoints[5]+keypoints[6])/2 + 0.3 * (keypoints[11]+keypoints[12])/2
                    chest = (1279,719)
                    cv2.circle(colour_image, (int(chest[0]), int(chest[1])), 5, (255, 0, 0), -1)

                    transformedDepth = cap.transformed_depth

                    #transformedDepth.shape[0] will be the y dimension, use to make adaptive to any resolution

                    ###
                    #print(type(transformedDepth))
                    #print(int(chest[0]), int(chest[1]))
                    chestDepth = transformedDepth[int(chest[1]), int(chest[0])]
                    
    return colour_image, chest, chestDepth


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
    model = YOLO("yolo11s-pose.pt")

    #model = YOLO("yolo11s-pose.pt") halves framerate but offers better performance

    while True:

        #Capture frames from specified RGB and depth cameras
        capture = k4a.get_capture()

        #Necessary for later use with cv2
        frame = np.ascontiguousarray(capture.color[:, :, :3])

        #Apply the specified model to the captured colour frame
        results = model(frame, verbose=False)
        print(results[0].keypoints.has_visible)
        #can increase performance
        #results = model(source=frame, imgsz=(256, 448))
        
        #Draw results on the frame
        frame, coords, depth = drawPeople(capture, frame, results[0])#only operate on first result, assuming only 1 person

        #chest_3d = k4a.calibration.convert_2d_to_3d(coordinates=(int(coords[1]), int(coords[0])), depth=depth, source_camera=CalibrationType.COLOR)  # or calibration.color_camera)

        if depth == 0:
            print("Invalid depth at this pixel, skipping")
        else:
            chest_3d = k4a.calibration.convert_2d_to_3d(
                coordinates=(int(coords[0]), int(coords[1])),
                depth=depth,
                source_camera=CalibrationType.COLOR,
            )
            #print(results)
        
        #only care about indexes 0 and 2

        if np.any(capture.depth):
            cv2.imshow("altered", frame)

        key = cv2.waitKey(10)
        if key != -1:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
