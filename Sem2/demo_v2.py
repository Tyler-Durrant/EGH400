import cv2
import numpy as np
import pyk4a
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, CalibrationType

from ultralytics import YOLO

import pyk4a
from helpers import colorize

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc

import math
import random
import time
marker_size = 0.2
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
marker_ids = {0,1}



def orient(cam, camera_matrix, distortion_coefficients):
    while True:
        capture = cam.get_capture()
        frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)

        if capture is None:
            continue
        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)
        
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, distortion_coefficients
            )
            cv2.aruco.drawDetectedMarkers(frame, corners)

            #Optional - drawing marker axes
            for i, marker_id in enumerate(ids.flatten()):
                #inverts the aruco markers to be aligned with world frame
                R, _ = cv2.Rodrigues(rvecs[i])
                R_flip = R @ np.array([
                        [-1, 0,  0],
                        [ 0, 1,  0],
                        [ 0, 0, -1]
                ])
                rvecs[i] = cv2.Rodrigues(R_flip)[0].T

                cv2.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvecs[i], tvecs[i], 0.03)
            
                R_M_W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                R_R_W = R_flip.T

                rvec_robot, _ = cv2.Rodrigues(R_R_W)
                print(-rvec_robot.T[0])
                #print(f"Marker {marker_id}:")
                #print(f"  tvec = {tvecs[i].flatten()} (m)")
                #print(f"  rvec = {rvecs[i].flatten()}")

            if marker_ids.issubset(ids.flatten()):
                cv2.imshow("Azure Kinect ArUco Localization", frame)
                print('System Oriented :)')
                return(tvecs.flatten())

        cv2.imshow("Azure Kinect ArUco Localization", frame)
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print('System Failed to Orient :(')
            return
        

def getCameraParams(cam):
    camera_matrix = cam.calibration.get_camera_matrix(CalibrationType.COLOR)
    distortion_coefficients = cam.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    return camera_matrix, distortion_coefficients



def updatePosition():
    pass



def main():
    #Initiate Camera Object
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            synchronized_images_only=True,
            camera_fps=pyk4a.FPS.FPS_15
        )
    )
    k4a.start()

    camera_position = np.zeros(3)
    camera_rotation = np.zeros(3)

    #Get extrinsic camera parameters
    camera_matrix, distortion_coefficients = getCameraParams(k4a)

    model = YOLO("yolov8n-pose.pt")

    hi = orient(k4a, camera_matrix, distortion_coefficients)

    oriented = 1 if hi is not None else 0

    while oriented:









        if cv2.waitKey(1) & 0xFF == ord("q"):
            break




    k4a.stop()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

