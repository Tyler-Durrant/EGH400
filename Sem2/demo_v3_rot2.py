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
marker_ids = {0,1,2,3}



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
                cv2.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvecs[i], tvecs[i], 0.03)
            
                #print(f"Marker {marker_id}:")
                #print(f"  tvec = {tvecs[i].flatten()} (m)")
                #print(f"  rvec = {rvecs[i].flatten()}")

            if marker_ids.issubset(ids.flatten()):
                cv2.imshow("Azure Kinect ArUco Localization", frame)
                print('System Oriented :)')
                return tvecs, rvecs

        cv2.imshow("Azure Kinect ArUco Localization", frame)
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print('System Failed to Orient :(')
            return None, None
        

def getCameraParams(cam):
    camera_matrix = cam.calibration.get_camera_matrix(CalibrationType.COLOR)
    distortion_coefficients = cam.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    return camera_matrix, distortion_coefficients

def createTransform(tvec, rvec):
    T = np.eye(4)
    T[:3, :3],_ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.squeeze()


    return T

def updatePosition(cam, camera_matrix, distortion_coefficients, tvecs_world, rvecs_world, last_pose):
    capture = cam.get_capture()
    frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)

    if capture is None:
        return last_pose
    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)
        
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, distortion_coefficients
        )
        cv2.aruco.drawDetectedMarkers(frame, corners)

        #Optional - drawing marker axes
        for i, marker_id in enumerate(ids.flatten()):
            cv2.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvecs[i], tvecs[i], 0.03)
            
            T_R_M = createTransform(tvecs[i], rvecs[i])
            T_M_R = np.linalg.inv(T_R_M)

            T_W_M = createTransform(tvecs_world[marker_id], rvecs_world[marker_id])
            #print(tvecs_world)
            #print(rvecs_world)
            #print(T_W_M)

            T_W_R = T_W_M @ T_M_R


            R_W_R,_ = cv2.Rodrigues(T_W_R[:3, :3])  # 3x3 rotation matrix
            t_W_R = T_W_R[:3, 3]   # 3x1 translation vector

            #print(t_W_R)
            #print(f"  tvec = {tvecs[i].flatten()} (m)")
            #print(f"  rvec = {rvecs[i].flatten()}")
            #print(R_W_R)

            #print(f"Marker {marker_id}:")
            #print(f"  tvec = {tvecs[i].flatten()} (m)")
            #print(f"  rvec = {rvecs[i].flatten()}")
            if marker_id == 0:
                cv2.imshow("Azure Kinect ArUco Localization", frame)
                return t_W_R, R_W_R

    cv2.imshow("Azure Kinect ArUco Localization", frame)
    return None, None






def main():
    #Initiate Camera Object
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            synchronized_images_only=True,
            camera_fps=pyk4a.FPS.FPS_15
        )
    )
    k4a.start()

    plt.ion()
    fig, ax = plt.subplots()
    point1, = ax.plot(0, 0, 'ro')  # red dot for current position
    point2, = ax.plot(0, 0, 'bo')  # red dot for current position
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Y position (m)")
    ax.set_ylabel("X position (m)")
    ax.set_title("Live IMU Position")
    ax.grid(True)
    ax.set_aspect('equal', 'box')

    camera_position = np.zeros(3)
    camera_rotation = np.zeros(3)

    #Get extrinsic camera parameters
    camera_matrix, distortion_coefficients = getCameraParams(k4a)

    model = YOLO("yolov8n-pose.pt")

    tvecs_world, rvecs_world = orient(k4a, camera_matrix, distortion_coefficients)

    print(tvecs_world[0])
    point2.set_data([tvecs_world[0][0][0]], [tvecs_world[0][0][2]])
    plt.pause(0.005)

    oriented = 1 if tvecs_world is not None else 0

    while oriented:
        
        t_W_R, R_W_R = updatePosition(k4a, camera_matrix, distortion_coefficients, tvecs_world, rvecs_world, (0,0,0))
        #print(t_W_R)
        if t_W_R is not None:
            point1.set_data([t_W_R[0]], [t_W_R[2]])

        plt.pause(0.005)




        if cv2.waitKey(1) & 0xFF == ord("q"):
            break




    k4a.stop()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

