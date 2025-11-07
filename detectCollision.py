import numpy as np
import math

def project(pose, velocity, ts, N):

    # precompute per-step displacementq
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
        vx, vy = px + ax, py + ay
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


def testCollision(robot, robotvel, object, objectvel, test_variables = (0.5, 0.5, 1, 0.5, 0.1, 50)):

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

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    corner_offsets = np.array([
    [-s21 * cos_theta + s1 * sin_theta, -s21 * sin_theta - s1 * cos_theta],
    [-s21 * cos_theta - s1 * sin_theta, -s21 * sin_theta + s1 * cos_theta],
    [ s22 * cos_theta - s1 * sin_theta,  s22 * sin_theta + s1 * cos_theta],
    [ s22 * cos_theta + s1 * sin_theta,  s22 * sin_theta - s1 * cos_theta]
    ])

    for i, robotpos in enumerate(pointsRobot):
        collision = True
        ox, oy = pointsObject[i]
        rx, ry = robotpos

        corners = corner_offsets + np.array([ox, oy])
        
        normals = findNormals(theta, corners, rx, ry)

        for normal in normals:
            box = project_points_onto_axis(corners, normal)
            circle = project_circle_onto_axis((rx,ry), r, normal)
            if box[1] < circle[0] or circle[1] < box[0]:
                collision = False
                break
        if collision:
            break

    #    print(f"Collision at step {i}: robot={rx.item(), ry.item()}, object={ox.item(), oy.item()}")
    final = np.array([[rx.item(), ry.item()],[ox.item(),oy.item()]], np.float32)

    return collision, final, corners