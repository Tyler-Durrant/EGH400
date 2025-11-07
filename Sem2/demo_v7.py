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
marker_size = 0.202
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
                #cv2.imshow("Azure Kinect ArUco Localization", frame)
                print('System Oriented :)')
                return tvecs, rvecs

        #cv2.imshow("Azure Kinect ArUco Localization", frame)
    
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

def averageAngles(angles):
    x = np.cos(angles)
    y = np.sin(angles)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    avg_angle = np.arctan2(y_mean, x_mean)

    return avg_angle

def updatePosition(capture, camera_matrix, distortion_coefficients, tvecs_world, rvecs_world):
    frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)

    if capture is None:
        return None, frame
    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)
        
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, distortion_coefficients
        )
        cv2.aruco.drawDetectedMarkers(frame, corners)

        poses = []

        #Optional - drawing marker axes
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_ids:
                cv2.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvecs[i], tvecs[i], 0.03)
                
                T_R_M = createTransform(tvecs[i], rvecs[i])
                T_M_R = np.linalg.inv(T_R_M)

                T_W_M = createTransform(tvecs_world[marker_id], rvecs_world[marker_id])

                T_W_R = T_W_M @ T_M_R

                poses.append(T_W_R)
        
        if poses:
            translations = np.array([T[:3, 3] for T in poses])
            avg_translation = np.mean(translations, axis=0)

            rotations = np.array([cv2.Rodrigues(R[:3, :3])[0] for R in poses])
            average_rotation = averageAngles(rotations[:, 1, 0])

            #Taking smallest sideways value, as robot should be traveling straight
            axis = 0
            avg_translation[axis] = translations[np.argmin(np.abs(translations[:, axis])), axis]

            #Compile to x, y, theta
            pose = np.array([avg_translation[0], avg_translation[2], average_rotation])
            return pose, frame

    #cv2.imshow("Azure Kinect ArUco Localization", frame)
    return None, frame


def normaliseMeasured(predicted, measured):
    angle_diff = normalize_angle_rad(measured - predicted)
    return predicted + angle_diff

def normalize_angle_rad(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi



def detect(camera, capture, model, frame):

    coordinate = []

    image = np.ascontiguousarray(capture.color[:, :, :3])

    #Apply model
    results = model(source=image, verbose=False)
    detection = results[0].keypoints
    keypoints = detection.xy[0]
    
    #capture.color.shape[1] --- x
    #capture.color.shape[0] --- y
    #Both measured from top left

    if detection.has_visible and not (keypoints[[5,6,11,12]] == 0).all(dim=1).any():

        for idx, (x, y) in enumerate(keypoints):
            if idx in {5,6,11,12}:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        chest = tuple(((0.7 * ((keypoints[5] + keypoints[6]) / 2) + 0.3 * ((keypoints[11] + keypoints[12]) / 2)).int()).tolist())

        #cv2 from top left
        cv2.circle(frame, (chest[0], chest[1]), 5, (255, 0, 0), -1)
        
        #(v,u)
        chestDepth = capture.transformed_depth[chest[1], chest[0]]
        if chestDepth != 0:
            position = camera.calibration.convert_2d_to_3d(
                coordinates=(chest[0], chest[1]),
                depth=chestDepth,
                source_camera=CalibrationType.COLOR,
                target_camera=CalibrationType.COLOR,
            )
            #Convert to RH coordinate system with x forward and y left looking out from camera lens
            coordinate = (-position[0]/1000, position[2]/1000)

            #-----now need to allign world frame to be equivalent

    #cv2.imshow("altered", frame)

    return coordinate, frame




def transform(pose, object):
    #Transforms the coordinates of an object in the robot frame to the world frame
    #Takes inputs as (x,y,theta) and (x,y)

    transformedObject = (math.cos(pose[2].item()) * object[0] - math.sin(pose[2].item()) * object[1] + pose[0].item(),
                         math.sin(pose[2].item()) * object[0] + math.cos(pose[2].item()) * object[1] + pose[1].item())
    return transformedObject



def create_cv_kalman(initial_state):
    kf = cv2.KalmanFilter(3, 3)  # 2 state vars, 2 measurements

    #transition matrix models expected prediction results
    kf.transitionMatrix = np.eye(3, dtype=np.float32)

    # indicates which variables are measured and their scaling, in this case x y theta are measured directly
    kf.measurementMatrix = np.eye(3, dtype=np.float32)

    #Control input will be the previous vx and vy
    kf.controlMatrix = np.array([[1,0],
                                [0, 1],
                                [0, 0]], np.float32)

    # --- Process noise covariance (Q) ---
    # Q matrix indicates noise in the model, lower Q, more trust in model
    kf.processNoiseCov = np.array([[1,0,0],
                                [0, 1, 0],
                                [0, 0, 1]], np.float32) * 1e-3

    # R matrix indicates noise in the measurements #lower number more trust
    kf.measurementNoiseCov = np.array([[1,0,0],
                                [0, 1, 0],
                                [0, 0, 1]], np.float32) * 1e-3

    # P matrix indicates uncertainty in initial position
    kf.errorCovPost = np.eye(3, dtype=np.float32)

    # Initial pose is at 0,0,0
    kf.statePost = initial_state

    return kf




def computeControl(coord1, coord2, dt):
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
                    
    return np.array([[dx], [dy]], dtype=np.float32)








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

    last_detection = None
    no_detection = 0
    person_kf_initialized = False
    kf = None

    plt.ion()
    fig, ax = plt.subplots()
    point1, = ax.plot(0, 0, 'ro')
    point2, = ax.plot(0, 0, 'bo')
    point3, = ax.plot(0, 0, 'go')
    heading_line, = ax.plot([], [], 'b-', linewidth=2)  # blue line for heading direction

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
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

    #print(tvecs_world)
    #point2.set_data([tvecs_world[0][0][0]], [tvecs_world[0][0][2]])
    plt.pause(0.005)

    oriented = 1 if tvecs_world is not None else 0

    prev_t = k4a.get_capture().color_timestamp_usec


    while oriented:
        capture = k4a.get_capture()

        t = capture.color_timestamp_usec

        dt = (t - prev_t) / 1e6
        
        pose, frame = updatePosition(capture, camera_matrix, distortion_coefficients, tvecs_world, rvecs_world)

        if pose is not None:


            point1.set_data([pose[0]], [pose[1]])
            #print(pose)
            detection, frame = detect(k4a, capture, model, frame)
            if detection:
                #point2.set_data([detection[0]], [detection[1]])
                no_detection = 0
                transformed_detection = transform(pose, detection)

                if not person_kf_initialized:
                    #print(person_kf_initialized)
                    #Kalman filter not initialised
                    #point3.set_data([transformed_detection[0]], [transformed_detection[1]])
                    print(last_detection)
                    if last_detection is not None:
                        #Not getting to here
                        #2nd detection of person - Initialise KF
                        u_p = computeControl(last_detection,  transformed_detection, dt)

                        initial_detection = np.array([[transformed_detection[0]],
                                             [transformed_detection[1]],
                                             [math.atan2(transformed_detection[1] - last_detection[1], 
                                                         transformed_detection[0] - last_detection[0])]], np.float32)
                        

                        person_KF = create_cv_kalman(initial_detection)

                        person_kf_initialized = True

                        last_detection = transformed_detection
                    last_detection = transformed_detection
                else:
                    #Kalman filter already initialised and person detected
                    prediction_person = person_KF.predict(u_p)
                    person_KF.statePre[2,0] = normalize_angle_rad(person_KF.statePre[2,0])

                    measured_person = np.array([[transformed_detection[0]],
                         [transformed_detection[1]],
                         [math.atan2(transformed_detection[1] - last_detection[1], 
                                     transformed_detection[0] - last_detection[0])]], np.float32)
                    
                    measured_person[2, 0] = normaliseMeasured(person_KF.statePre[2, 0], measured_person[2, 0])

                    person_KF.correct(measured_person)
                    person_KF.statePost[2, 0] = normalize_angle_rad(person_KF.statePost[2, 0])

                    filtered_state = person_KF.statePost     # or however your KF stores the updated state

                    u_p = computeControl(last_detection,  transformed_detection, dt)

                    point3.set_data([filtered_state[0, 0]], [filtered_state[1, 0]])
                
                    last_detection = filtered_state

                    print(last_detection.flatten())

            else:
                #no person detected
                if person_kf_initialized:
                    #KF running but no new measurement
                    prediction_person = person_KF.predict(u_p)
                    #point3.set_data([prediction_person[0, 0]], [prediction_person[1, 0]])
                    last_detection = prediction_person

                no_detection += 1

                if no_detection > 9:
                    print("Lost person – resetting KF")
                    person_KF = None
                    person_kf_initialized = False
                    last_detection = None
                    no_detection = 0
            

        prev_t = t

     
        cv2.imshow("Live Feed", frame)

        plt.pause(0.005)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()
    k4a.stop()





if __name__ == "__main__":
    main()

