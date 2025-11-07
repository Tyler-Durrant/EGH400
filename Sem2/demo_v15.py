import cv2
import numpy as np
import pyk4a
import json
import time
from pyk4a import PyK4A, Config, CalibrationType
from ultralytics import YOLO
import math
import matplotlib.pyplot as plt




marker_size = 0.289
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
marker_ids = {0,1,2,3}
forward_vel = 0
run_number = 20
moving_test = 1



def getCameraParams(cam):
    camera_matrix = cam.calibration.get_camera_matrix(CalibrationType.COLOR)
    distortion_coefficients = cam.calibration.get_distortion_coefficients(CalibrationType.COLOR)

    return camera_matrix, distortion_coefficients

def create_cv_kalman():
    kf = cv2.KalmanFilter(3, 3)

    #Alter this depending on expected camera movement
    #transition matrix models expected prediction results
    kf.transitionMatrix = np.array([[0,0,0],
                                [0, 1, 0],
                                [0, 0, 0]], np.float32)

    #indicates which variables are measured and their scaling, in this case x y theta are measured directly
    kf.measurementMatrix = np.eye(3, dtype=np.float32)

    #Control input will be the previous vx and vy
    kf.controlMatrix = np.array([[0],
                                [1],
                                [0]], np.float32)

    #Q matrix indicates noise in the model, lower Q, more trust in model
    kf.processNoiseCov = np.array([[1,0,0],
                                    [0, 1, 0],
                                    [0, 0, 1]], np.float32) * 1e-3

  

    #R matrix indicates noise in the measurements #lower number more trust
    kf.measurementNoiseCov = np.array([[1, 0, 0],
                                         [0, 10, 0],
                                         [0, 0, 1]], np.float32) * 1e-2


    #P matrix indicates uncertainty in initial position
    kf.errorCovPost = np.eye(3, dtype=np.float32)

    #Initial pose is at 0,0,0
    kf.statePost = np.array([[0],
                            [0],
                            [np.pi/2]], np.float32)

    return kf


def orient(cam, camera_matrix, distortion_coefficients, detector):
    while True:
        capture = cam.get_capture()
        frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)

        if capture is None:
            continue


        corners, ids, _ = detector.detectMarkers(frame)
        
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
                return tvecs, rvecs, capture.color_timestamp_usec

        #cv2.imshow("Azure Kinect ArUco Localization", frame)
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print('System Failed to Orient :(')
            return None, None, None
        
def updatePosition(capture, camera_matrix, distortion_coefficients, tvecs_world, rvecs_world, detector):
    frame = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)

    if capture is None:
        return None, frame
    # Detect markers
    corners, ids, _ = detector.detectMarkers(frame)
        
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, distortion_coefficients
        )
        #cv2.aruco.drawDetectedMarkers(frame, corners)

        poses = []

        #Optional - drawing marker axes
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_ids:
                #cv2.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvecs[i], tvecs[i], 0.03)
                
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
            pose = np.array([[avg_translation[0]], [avg_translation[2]], [average_rotation]], dtype=np.float32)
            return pose, frame

    #cv2.imshow("Azure Kinect ArUco Localization", frame)
    return None, frame

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

def filterPose(pose, filter, control):

    # ----- Applying kalman filter to camera pose ----- #
    if pose is not None:
    #If new measurement available

        #Retain original pose for comparison
        measured = pose.copy()

        #Prediction based on model and control input
        filter.predict(control)

        filter.statePre[2,0] = normalize_angle_rad(filter.statePre[2,0])

        measured[2, 0] = normaliseMeasured(filter.statePre[2, 0], measured[2, 0])

        #Correction based on the measured pose
        filter.correct(measured)
        filter.statePost[2, 0] = normalize_angle_rad(filter.statePost[2, 0])
            
        #Extract filtered pose
        filtered_pose = filter.statePost

    else:
    #If no new measurement available
        #Prediction based on model and control input
        filtered_pose = filter.predict(control)
    
    return filtered_pose.copy()

def normaliseMeasured(predicted, measured):
    angle_diff = normalize_angle_rad(measured - predicted)
    return predicted + angle_diff

def normalize_angle_rad(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi



def detect(camera, capture, model, frame):

    coordinate = None

    image = np.ascontiguousarray(capture.color[:, :, :3])

    #Apply model
    results = model(source=image, verbose=False, imgsz=320)
    detection = results[0].keypoints
    keypoints = detection.xy[0]

    if detection.has_visible and not (keypoints[[5,6,11,12]] == 0).all(dim=1).any():

        for idx, (x, y) in enumerate(keypoints):
            if idx in {5,6,11,12}:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        chest = tuple(((0.7 * ((keypoints[5] + keypoints[6]) / 2) + 0.3 * ((keypoints[11] + keypoints[12]) / 2)).int()).tolist())

        #cv2 from top left
        cv2.circle(frame, (chest[0], chest[1]), 5, (255, 0, 0), -1)
        
        chestDepth = capture.transformed_depth[chest[1], chest[0]]
        if chestDepth != 0:
            position = camera.calibration.convert_2d_to_3d(
                coordinates=(chest[0], chest[1]),
                depth=chestDepth,
                source_camera=CalibrationType.COLOR,
                target_camera=CalibrationType.COLOR,
            )
            coordinate = np.array([[position[0]/1000], [position[2]/1000]], dtype=np.float32)

    return coordinate, frame

def transform(pose, object):
    #Transforms the coordinates of an object in the robot frame to the world frame
    #Takes inputs as (x,y,theta) and (x,y)

    transformedObject = np.array([np.cos(pose[2]) * object[0] - np.sin(pose[2]) * object[1] + pose[0],
                                  np.sin(pose[2]) * object[0] + np.cos(pose[2]) * object[1] + pose[1]], np.float32)
    return transformedObject



def computeControl(coord1, coord2):

    dist = np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)

    dx = dist*np.cos(coord2[2])
    dy = dist*np.sin(coord2[2])
                    
    return np.array([dx, dy], np.float32)




def create_person_kalman(first_detection, transformed_detection):

    #Let initial pose for person be based on the second detection with
    #heading determined based on the displacement between the first and second
    initial_detection = np.array([[transformed_detection[0]],
                                  [transformed_detection[1]],
                                  [np.atan2(transformed_detection[1] - first_detection[1], 
                                   transformed_detection[0] - first_detection[0])]], np.float32)

    person_kf = cv2.KalmanFilter(3, 3)

    #transition matrix models expected prediction results
    person_kf.transitionMatrix = np.array([[1,0,0],
                                [0, 1, 0],
                                [0, 0, 1]], np.float32)
    
    #indicates which variables are measured and their scaling, in this case x y theta are measured directly
    person_kf.measurementMatrix = np.eye(3, dtype=np.float32)

    #Control input will be the previous vx and vy
    person_kf.controlMatrix = np.array([[1,0],
                                [0, 1],
                                [0, 0]], np.float32)
    
    #Q matrix indicates noise in the model, lower Q, more trust in model
    person_kf.processNoiseCov = np.array([[1,0,0],
                                [0, 1, 0],
                                [0, 0, 1]], np.float32) * 1e-3

    #R matrix indicates noise in the measurements #lower number more trust
    person_kf.measurementNoiseCov = np.array([[1,0,0],
                                [0, 1, 0],
                                [0, 0, 1]], np.float32) * 1e-2

    #P matrix indicates uncertainty in initial position
    person_kf.errorCovPost = np.eye(3, dtype=np.float32)

    #Initial pose is at 0,0,0
    person_kf.statePost = initial_detection

    #Determine control input for first filtered step, 
    #necessary as persons movement is not directly modeled
    control = computeControl(first_detection,  initial_detection)
    

    return person_kf, control



def filterDetection(transformed_detection, first_detection, last_detection, person_KF, control):

    if transformed_detection is not None:
        #run prediction
        person_KF.predict(control)

        person_KF.statePre[2,0] = normalize_angle_rad(person_KF.statePre[2,0])

        #Retain original detection for comparison, and compute heading
        measured_person = np.array([[transformed_detection[0]],
                                [transformed_detection[1]],
                                [np.atan2(transformed_detection[1] - first_detection[1], 
                                 transformed_detection[0] - first_detection[0])]], np.float32)

        measured_person[2, 0] = normaliseMeasured(person_KF.statePre[2, 0], measured_person[2, 0])

        #Correct based on new detection
        person_KF.correct(measured_person)
        person_KF.statePost[2, 0] = normalize_angle_rad(person_KF.statePost[2, 0])

        #Extract filtered detection
        filtered_detection = person_KF.statePost

        #Predict next control input
        control = computeControl(last_detection, filtered_detection)

    else:
        #run prediction
        filtered_detection = person_KF.predict(control)

    return filtered_detection.copy(), control

# Helper function to convert NumPy objects to JSON-compatible types
def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# Save data_log to JSON
def save_log_to_json(data_log, filename):
    # Convert each timestep dictionary to JSON-compatible types
    json_compatible_log = [
        {k: convert_for_json(v) for k, v in entry.items()} 
        for entry in data_log
    ]
    
    with open(filename, "w") as f:
        json.dump(json_compatible_log, f, indent=2)
    
    print(f"Data log saved to {filename}")






def main(live_feed = 0, visualise = 0):


    if visualise:
        plt.ion()
        fig, ax = plt.subplots()
        measured_robot, = ax.plot([], [], 'ro', label='Robot')
        filtered_robot, = ax.plot([], [], 'go', label='Robot')

        measured_person, = ax.plot([], [], 'ro', label='Robot')
        filtered_person, = ax.plot([], [], 'go', label='Robot')

        heading_line, = ax.plot([], [], 'b-', linewidth=2)
        #point_collision_robot, = ax.plot([], [], 'co', label='Collision Robot')
        #point_collision_person, = ax.plot([], [], 'mo', label='Collision Person')
        #heading_line_person, = ax.plot([], [], 'b-', linewidth=2, label='Person Heading')
        #heading_line_robot, = ax.plot([], [], 'b-', linewidth=2, label='Robot Heading')
        #collision_circle = patches.Circle((0, 0), radius=0.5, edgecolor='red', facecolor='none', visible=False)
        #ax.add_patch(collision_circle)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("Y position (m)")
        ax.set_ylabel("X position (m)")
        ax.set_title("Replaying Logged Run")
        ax.grid(True)
        ax.set_aspect('equal', 'box')

    #Initialise data logging and recording
    data_log_name = f"data_log{run_number}.json"
    timing_log_name = f"timing_log{run_number}.json"
    video_name = f"output{run_number}.avi"

    data_log = []
    timing_log = []

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_name, fourcc, 15.0, (1280, 720))

    #Initiate Azure Camera Object
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
            synchronized_images_only=True,
            camera_fps=pyk4a.FPS.FPS_15
        )
    )
    k4a.start()

    last_detection = None
    first_detection = None
    filtered_detection = None
    person_KF = None
    no_detection = 0


    #Get extrinsic camera parameters
    camera_matrix, distortion_coefficients = getCameraParams(k4a)

    #Construct human pose detection model
    model = YOLO("yolov8n-pose.pt")

    #Create kalman filter for robot pose
    KF = create_cv_kalman()  

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    #Will return None values if aborted prior to orienting
    tvecs_world, rvecs_world, prev_t = orient(k4a, camera_matrix, distortion_coefficients, detector)
    first_time = prev_t

    oriented = 1 if tvecs_world is not None else 0

    #Initial stationary control input
    u = np.array([[0]], dtype=np.float32)
    started = 0

    while oriented:

        capture = k4a.get_capture()
        pose, frame = updatePosition(capture, camera_matrix, distortion_coefficients, tvecs_world, rvecs_world, detector)

        #Calculate timestep
        t = capture.color_timestamp_usec
        dt = (t - prev_t) / 1e6
        prev_t = t

        if not started:
            started = 1
            print('hi')
            u = np.array([[forward_vel*dt]], dtype=np.float32)

        #Filter robot pose
        filtered_pose = filterPose(pose, KF, u)

        #Uncomment for stationary testing
        #filtered_pose = np.array([[0],[0],[0]], dtype=np.float32)

        #Detect people in frame using ML model
        detection, frame = detect(k4a, capture, model, frame)

        if detection is not None:
        #If new measurement available

            #Reset count of loops without detection
            no_detection = 0

            #Transform detection from camera frame to world
            transformed_detection = transform(filtered_pose, detection)

            if person_KF is not None:
            #If filter already initialised
            
                #Filter detection
                filtered_detection, _ = filterDetection(transformed_detection, first_detection, last_detection, person_KF, u_p)
            
            else:
            #If filter not initialised
                if first_detection is not None:
                #If on second detection

                    #Initialise person KF
                    person_KF, u_p = create_person_kalman(first_detection, transformed_detection)

                    last_detection = transformed_detection

                else:
                    first_detection = transformed_detection

        else:
        #no new measurement available

            #Increment count loops since last detection
            no_detection += 1        

            if no_detection == 10:
            #If loops without detection reaches limit
                #Reset KF and detection tracking
                print("Lost person - resetting KF")
                person_KF = None
                first_detection = None
                filtered_detection = None
                last_detection = None

            elif person_KF is not None:

                filtered_detection, u_p = filterDetection(detection, first_detection, last_detection, person_KF, u_p)

        if live_feed:
            cv2.imshow("Live Feed", frame)
        out.write(frame)

        if visualise:
            if pose is not None:
                measured_robot.set_data([pose[0]], [pose[1]])
            else:
                measured_robot.set_data([], [])

            if filtered_pose is not None:
                filtered_robot.set_data([filtered_pose[0]], [filtered_pose[1]])
            else:
                filtered_robot.set_data([], [])


            if filtered_detection is not None:
                filtered_person.set_data([filtered_detection[0]], [filtered_detection[1]])
            else:
                filtered_person.set_data([], [])
            
            if detection is not None:
                measured_person.set_data([transformed_detection[0]], [transformed_detection[1]])
            else:
                measured_person.set_data([], [])
                #heading_line.set_data([first_detection[0], transformed_detection[0]], [first_detection[1], transformed_detection[1]])
                
            plt.pause(0.005)


        log_entry = {
                    'timestamp': t,
                    "dt": dt,
                    'pose': pose.flatten().tolist() if pose is not None else [None, None, None],
                    'filtered_pose': filtered_pose.flatten().tolist() if filtered_pose is not None else [None, None, None],
                    'detection': transformed_detection.flatten().tolist() if detection is not None else [None, None, None],
                    'filtered_detection': filtered_detection.flatten().tolist() if filtered_detection is not None else [None, None, None],                    
        }
        data_log.append(log_entry)

        if (t-first_time)*0.000001>20:
            break


    cv2.destroyAllWindows()
    k4a.stop()

    print('---')

    save_log_to_json(data_log, data_log_name)

    out.release()
    print('----')
    print(f"Video saved to {video_name}")
    print('----')

    



        

        






    



if __name__ == "__main__":

    main(0,0)