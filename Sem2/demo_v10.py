import cv2
import numpy as np
import pyk4a
import json
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, CalibrationType

from ultralytics import YOLO

import pyk4a

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
marker_size = 0.289
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
marker_ids = {0,1,2,3}


# Helper function to convert NumPy objects to JSON-compatible types
def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj







def project(pose, velocity, ts, N):

    # precompute per-step displacement
    step = velocity * ts
    dx = np.cos(pose[2]) * step
    dy = np.sin(pose[2]) * step
    
    # generate positions
    return [(pose[0] + i * dx, pose[1] + i * dy) for i in range(N)]

def projectRobot(pose, velocity, ts, N):

    # precompute per-step displacement
    step = velocity * ts
    dx = np.cos(np.pi/2) * step
    dy = np.sin(np.pi/2) * step
    
    # generate positions
    return [(pose[0] + i * dx, pose[1] + i * dy) for i in range(N)]


def findNormals(boxTheta, corners, cx, cy):

    boxTheta, cx, cy = boxTheta.item(), cx.item(), cy.item()

    corners = [(x.item(), y.item()) for x, y in corners]

    #normals for box
    n1 = (np.cos(boxTheta), np.sin(boxTheta))
    n2 = (-np.sin(boxTheta), np.cos(boxTheta))

    #normal for circle
    closest_corner = min(corners, key=lambda p: math.dist((cx, cy), p))

    dx = closest_corner[0] - cx
    dy = closest_corner[1] - cy
    length = math.hypot(dx, dy)
    n3 = (dx/length, dy/length)

    return (n1, n2, n3)

def project_points_onto_axis(points, axis, axis_origin=(0, 0)):

    nx, ny = axis
    ax, ay = axis_origin
    projections = []

    for px, py in points:
        # vector from axis origin to point
        vx, vy = px - ax, py - ay
        # scalar projection (dot product)
        t = vx * nx + vy * ny
        projections.append(t)
    
    #projections = (min(projections), max(projections))

    return (min(projections), max(projections))

def projected_points(projections, axis, axis_point=(0,0)):
    
    nx, ny = axis
    ax, ay = axis_point

    points = []
    
    for proj in projections:
    # projected point
        proj_x = ax + proj * nx
        proj_y = ay + proj * ny
        points.append((proj_x, proj_y))

    return points

def project_circle_onto_axis(center, radius, axis, axis_origin=(0, 0)):

    nx, ny = axis
    cx, cy = center
    ax, ay = axis_origin

    # vector from axis origin to circle center
    vx, vy = cx - ax, cy - ay
    # scalar projection of center onto axis
    t = vx * nx + vy * ny

    return [t - radius, t + radius]



def generateCollisionTestVariables(person_side, person_rear, person_front, robot_radius, ts, N):
    return (person_side, person_rear, person_front, robot_radius, ts, N)


def testCollision(robot, robotvel, object, objectvel, test_variables = (0.5, 0.5, 1, 0.5, 0.01, 200)):

    #Positional values for robot and person
    _, _, theta = object

    #---Clearances for person---#
    #Side
    s1 = test_variables[0]
    #Rear
    s21 = test_variables[1]
    #Front --- Can be scaled to adapt to velocity
    s22 = test_variables[2]

    #Radius of robot
    r = test_variables[3]

    #timestep between each projection
    ts = test_variables[4]
    #number of projections
    N = test_variables[5]

    pointsRobot = projectRobot(robot, robotvel, ts, N)
    pointsObject = project(object, objectvel, ts, N)

    for i, robotpos in enumerate(pointsRobot):
        collision = True
        ox, oy = pointsObject[i]
        rx, ry = robotpos

        corners = [(-s21 * np.cos(theta) + s1 * np.sin(theta) + ox, -s21 * np.sin(theta) - s1 * np.cos(theta) + oy),
                   (-s21 * np.cos(theta) - s1 * np.sin(theta) + ox, -s21 * np.sin(theta) + s1 * np.cos(theta) + oy),
                   (s22 * np.cos(theta) - s1 * np.sin(theta) + ox, s22 * np.sin(theta) + s1 * np.cos(theta) + oy),
                   (s22 * np.cos(theta) + s1 * np.sin(theta) + ox, s22 * np.sin(theta) - s1 * np.cos(theta) + oy)]
        
        normals = findNormals(theta, corners, rx, ry)

        for normal in normals:
            box = project_points_onto_axis(corners, normal)
            circle = project_circle_onto_axis((rx,ry), r, normal)
            if box[1] < circle[0] or circle[1] < box[0]:
                collision = False
                break

        if i == 0:
            first = corners
            first.append(first[0])
        if collision or i == len(pointsRobot) - 1:
            last = corners
            last.append(last[0])
            break
        
    if collision:

        print(f"Collision at step {i}: robot={rx.item(), ry.item()}, object={ox.item(), oy.item()}")
    final = np.array([[rx.item(), ry.item()],[ox.item(),oy.item()]], np.float32)

    return collision, final






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
                return tvecs, rvecs, capture.color_timestamp_usec

        #cv2.imshow("Azure Kinect ArUco Localization", frame)
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print('System Failed to Orient :(')
            return None, None, None


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
            pose = np.array([[avg_translation[0]], [avg_translation[2]], [average_rotation]], dtype=np.float32)
            return pose, frame

    #cv2.imshow("Azure Kinect ArUco Localization", frame)
    return None, frame


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
            #coordinate = (-position[0]/1000, position[2]/1000)
            coordinate = np.array([[position[0]/1000], [position[2]/1000]], dtype=np.float32)

            #-----now need to allign world frame to be equivalent

    #cv2.imshow("altered", frame)

    return coordinate, frame




def transform(pose, object):
    #Transforms the coordinates of an object in the robot frame to the world frame
    #Takes inputs as (x,y,theta) and (x,y)

    #transformedObject = (math.cos(pose[2].item()) * object[0] - math.sin(pose[2].item()) * object[1] + pose[0].item(),
    #                     math.sin(pose[2].item()) * object[0] + math.cos(pose[2].item()) * object[1] + pose[1].item())
    
    transformedObject = np.array([np.cos(pose[2]) * object[0] - np.sin(pose[2]) * object[1] + pose[0],
                                  np.sin(pose[2]) * object[0] + np.cos(pose[2]) * object[1] + pose[1]], np.float32)
    return transformedObject



def create_cv_kalman():
    kf = cv2.KalmanFilter(3, 3)  # 2 state vars, 2 measurements

    #Alter this depending on expected camera movement
    #transition matrix models expected prediction results
    kf.transitionMatrix = np.array([[0,0,0],
                                [0, 1, 0],
                                [0, 0, 1]], np.float32)

    #indicates which variables are measured and their scaling, in this case x y theta are measured directly
    kf.measurementMatrix = np.eye(3, dtype=np.float32)

    #Control input will be the previous vx and vy
    kf.controlMatrix = np.array([[1,0],
                                [0, 1],
                                [0, 0]], np.float32)

    #Q matrix indicates noise in the model, lower Q, more trust in model
    kf.processNoiseCov = np.array([[1,0,0],
                                    [0, 1, 0],
                                    [0, 0, 1]], np.float32) * 1e-3

  

    #R matrix indicates noise in the measurements #lower number more trust
    kf.measurementNoiseCov = np.array([[100,0,0],
                                    [0, 1, 0],
                                    [0, 0, 1]], np.float32) *0.001


    #P matrix indicates uncertainty in initial position
    kf.errorCovPost = np.eye(3, dtype=np.float32)

    #Initial pose is at 0,0,0
    kf.statePost = np.array([[0],
                            [0],
                            [np.pi/2]], np.float32)

    return kf


def create_cv_kalman_person(initial_state):
    kf = cv2.KalmanFilter(3, 3)  # 2 state vars, 2 measurements

    #transition matrix models expected prediction results
    kf.transitionMatrix = np.array([[1,0,0],
                                [0, 1, 0],
                                [0, 0, 1]], np.float32)

    #indicates which variables are measured and their scaling, in this case x y theta are measured directly
    kf.measurementMatrix = np.eye(3, dtype=np.float32)

    #Control input will be the previous vx and vy
    kf.controlMatrix = np.array([[1,0],
                                [0, 1],
                                [0, 0]], np.float32)

    #Q matrix indicates noise in the model, lower Q, more trust in model
    kf.processNoiseCov = np.array([[1,0,0],
                                [0, 1, 0],
                                [0, 0, 1]], np.float32) * 1e-3

    #R matrix indicates noise in the measurements #lower number more trust
    kf.measurementNoiseCov = np.array([[1,0,0],
                                [0, 1, 0],
                                [0, 0, 1]], np.float32) * 1e-3

    #P matrix indicates uncertainty in initial position
    kf.errorCovPost = np.eye(3, dtype=np.float32)

    #Initial pose is at 0,0,0
    kf.statePost = initial_state

    return kf




def computeControl(coord1, coord2):

    dist = np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)

    dx = dist*np.cos(coord2[2])
    dy = dist*np.sin(coord2[2])
                    
    return np.array([dx, dy], np.float32)



# Save data_log to JSON
def save_log_to_json(data_log, filename="data_log11.json"):
    # Convert each timestep dictionary to JSON-compatible types
    json_compatible_log = [
        {k: convert_for_json(v) for k, v in entry.items()} 
        for entry in data_log
    ]
    
    with open(filename, "w") as f:
        json.dump(json_compatible_log, f, indent=2)
    
    print(f"Data log saved to {filename}")

def main():

    data_log = []

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output11.avi', fourcc, 15.0/2, (1280, 720))

    #Initiate Azure Camera Object
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            synchronized_images_only=True,
            camera_fps=pyk4a.FPS.FPS_15
        )
    )
    k4a.start()


    last_detection = None
    no_detection = 0
    person_KF_initialised = False

    #Construct Live Plot
    plt.ion()
    fig, ax = plt.subplots()
    point1, = ax.plot(0, 0, 'ro')
    point2, = ax.plot(0, 0, 'bo')
    point3, = ax.plot([], [], 'go')
    point4, = ax.plot([], [], 'co')
    point5, = ax.plot([], [], 'co')
    point6, = ax.plot([], [], 'ko')
    heading_line, = ax.plot([], [], 'b-', linewidth=2)  # blue line for heading direction
    line_length = 1.0

    # create shapes
    circle = patches.Circle((0, 0), radius=0.5, edgecolor='blue', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("Y position (m)")
    ax.set_ylabel("X position (m)")
    ax.set_title("Live IMU Position")
    ax.grid(True)
    ax.set_aspect('equal', 'box')

    #Get extrinsic camera parameters
    camera_matrix, distortion_coefficients = getCameraParams(k4a)

    #Construct human pose detection model
    model = YOLO("yolov8n-pose.pt")

    KF = create_cv_kalman()

    forward_vel = 0.2

    plt.pause(0.005)

    tvecs_world, rvecs_world, prev_t = orient(k4a, camera_matrix, distortion_coefficients)

    oriented = 1 if tvecs_world is not None else 0
    #
    #while True:
    #    capture = k4a.get_capture()
    #    pos, fram = updatePosition(capture, camera_matrix, distortion_coefficients, tvecs_world, rvecs_world)
    #    cv2.imshow("Live Feed", fram)

    #    plt.pause(0.005)
    #    if cv2.waitKey(1) & 0xFF == ord("q"):
    #        break

    #Main process loop, only runs after initial orientation
    while oriented:

        #Take image
        capture = k4a.get_capture()

        #Calculate timestep
        t = capture.color_timestamp_usec
        dt = (t - prev_t) / 1e6

        #Update control input based on time pased
        u = np.array([[0],[forward_vel*dt]], dtype=np.float32)

        #Attempt to update camera pose using aruco markers
        pose, frame = updatePosition(capture, camera_matrix, distortion_coefficients, tvecs_world, rvecs_world)
        

        print('this')
        print(f"measured: {pose[1]}:")


        # ----- Applying kalman filter to camera pose ----- #
        if pose is not None:
        #If new measurement available

            point1.set_data([pose[0]], [pose[1]])

            #Prediction based on model and control input
            KF.predict(u)

            KF.statePre[2,0] = normalize_angle_rad(KF.statePre[2,0])

            #Retain original pose for comparison
            measured = pose

            measured[2, 0] = normaliseMeasured(KF.statePre[2, 0], measured[2, 0])

            #Correction based on the measured pose
            KF.correct(measured)

            KF.statePost[2, 0] = normalize_angle_rad(KF.statePost[2, 0])
            
            #Extract filtered pose
            filtered_pose = KF.statePost

        else:
        #If no new measurement available

            #Prediction based on model and control input
            filtered_pose = KF.predict(u)

        #qprint('filteredpose',filtered_pose[1])
        print(f"filtered: {filtered_pose[1]}:")

        #Detect person in frame using ML model
        #detection, frame = detect(k4a, capture, model, frame)
        detection = None

        filtered_detection = None

        # ----- Applying kalman filter to person detection ----- #
        if detection is not None:
        #If new measurement available

            #Reset count of loops without detection
            no_detection = 0

            #Transform detection from camera frame to world
            transformed_detection = transform(filtered_pose, detection)

            if person_KF_initialised:
            #If new measurement available and KF is initialised

                #run prediction
                person_KF.predict(u_p)

                person_KF.statePre[2,0] = normalize_angle_rad(person_KF.statePre[2,0])


                #print("transformed_detection:", np.shape(transformed_detection), transformed_detection,"\nlast_detection:", np.shape(last_detection), last_detection)


                #Retain original detection for comparison, and compute heading
                measured_person = np.array([[transformed_detection[0]],
                                            [transformed_detection[1]],
                                            [np.atan2(transformed_detection[1] - last_detection[1], 
                                             transformed_detection[0] - last_detection[0])]], np.float32)

                measured_person[2, 0] = normaliseMeasured(person_KF.statePre[2, 0], measured_person[2, 0])

                #Correct based on new detection
                person_KF.correct(measured_person)
                KF.statePost[2, 0] = normalize_angle_rad(KF.statePost[2, 0])

                #Extract filtered detection
                filtered_detection = person_KF.statePost

                #print(last_detection, filtered_detection)
                #Predict next control input
                u_p = computeControl(last_detection, filtered_detection)

                last_detection = np.squeeze(filtered_detection)

                point3.set_data([filtered_detection[0]], [filtered_detection[1]])




            else:
            #New measurement available but KF not initialised

                if last_detection is not None:
                #Require 2 consecutive readings to initialise KF

                    #Let initial pose for person be based on the second detection with
                    #heading determined based on the displacement between the first and second
                    initial_detection = np.array([[transformed_detection[0]],
                                                [transformed_detection[1]],
                                                [np.atan2(transformed_detection[1] - last_detection[1], 
                                                 transformed_detection[0] - last_detection[0])]], np.float32)

                    #Initialise person KF
                    person_KF = create_cv_kalman_person(initial_detection)
                    person_KF_initialised = True

                    #Determine control input for first filtered step, 
                    #necessary as persons movement is not directly modeled
                    u_p = computeControl(last_detection,  initial_detection)
            
                    dx = transformed_detection[0] - last_detection[0]
                    dy = transformed_detection[1] - last_detection[1]
                    person_speed = math.sqrt(dx.item()**2 + dy.item()**2) / dt
                
                #In the case where this is the first reading, 
                #this will increment to store as previous,
                #making above if statement execute next loop upon detection
                last_detection = transformed_detection

                point3.set_data([transformed_detection[0]], [transformed_detection[1]])
        
        else:
        #no new measurement available
            
            #Increment count loops since last detection
            no_detection += 1

            if no_detection == 10:
                #If loops without detection reaches limit

                #Reset KF and detection tracking
                print("Lost person - resetting KF")
                point3.set_data([], [])
                heading_line.set_data([], [])
                person_KF = None
                person_KF_initialised = False
                last_detection = None
            
            elif person_KF_initialised:
            #If KF is initialised

                #Run prediction based on model and last calculated control input
                #u_p not updated this loop
                filtered_detection = person_KF.predict(u_p)
            
                dx = filtered_detection[0] - last_detection[0]
                dy = filtered_detection[1] - last_detection[1]
                person_speed = math.sqrt(dx.item()**2 + dy.item()**2) / dt
                
                last_detection = filtered_detection

        point2.set_data([filtered_pose[0]], [filtered_pose[1]])

        #Add code to check for collision
        #Will need angle and velocity of both bodies

        if filtered_detection is not None:
            #Determine line length based on distance between prior points
            #line_length = np.sqrt((filtered_detection[0] - last_detection[0])**2 + (filtered_detection[1] - last_detection[1])**2)

            #Compute line endpoints for plotting heading,
            #due to how this is calculated, 
            #it indicates the model predicted next coordinate
            x_end = filtered_detection[0] + line_length * np.cos(filtered_detection[2])
            y_end = filtered_detection[1] + line_length * np.sin(filtered_detection[2])
            heading_line.set_data([filtered_detection[0].item(), x_end.item()], [filtered_detection[1].item(), y_end.item()])

            #filtered_pose is the robot body
            #filtered_detection is person but only when person_KF_initialised

            test_variables = generateCollisionTestVariables(0.5, 0.5, 1, 0.5, 0.01, 200)

            collision, collision_coords = testCollision(filtered_pose, forward_vel, filtered_detection, person_speed, test_variables)

            if collision:
            #    circle.set_edgecolor('red')
            #else:
            #    circle.set_edgecolor('blue')
                point4.set_data([collision_coords[0][0]], [collision_coords[0][1]])
                point5.set_data([collision_coords[1][0]], [collision_coords[1][1]])
                circle.center = (collision_coords[0][0], collision_coords[0][1])
                circle.set_visible(True)
            #else:
            #    point4.set_data([], [])
            #    point5.set_data([], [])
            #    circle.set_visible(False)
                fig.canvas.draw()
            else:
                point4.set_data([], [])
                point5.set_data([], [])
                circle.set_visible(False)
                fig.canvas.draw()


        log_entry = {
                    'timestamp': t,
                    "dt": dt,
                    'robot_pose': filtered_pose.flatten().tolist() if filtered_pose is not None else [None, None, None],
                    'person_pose': filtered_detection.flatten().tolist() if filtered_detection is not None else [None, None, None],
                    'collision': collision if filtered_detection is not None else False,
                    'collision_coords': collision_coords.tolist() if filtered_detection is not None and collision else None,
        }

        data_log.append(log_entry)

        out.write(frame)

        cv2.imshow("Live Feed", frame)

        plt.pause(0.005)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        prev_t = t

    cv2.destroyAllWindows()
    k4a.stop()
    out.release()

    save_log_to_json(data_log)



if __name__ == "__main__":
    main()
