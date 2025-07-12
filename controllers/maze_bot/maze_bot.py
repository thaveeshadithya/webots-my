from controller import Robot
import math
import matplotlib.pyplot as plt

# Constants
TIME_STEP = 64
MAX_SPEED = 6
WHEEL_RADIUS = 0.015  # meters
REPLAY_THRESHOLD = 0.05  # meters
LOOP_CLOSURE_THRESHOLD = 0.1  # meters
MIN_MAPPING_TIME = 10.0  # seconds

# Initialize robot and devices
robot = Robot()

# Motors
lm = robot.getDevice("left_motor")
rm = robot.getDevice("right_motor")
lm.setPosition(float('inf'))
rm.setPosition(float('inf'))

# Distance Sensors
ds_front_right = robot.getDevice("ds_front_right")
ds_front_left = robot.getDevice("ds_front_left")
ds_side_right = robot.getDevice("ds_side_right")
ds_side_left = robot.getDevice("ds_side_left")
for sensor in [ds_front_right, ds_front_left, ds_side_right, ds_side_left]:
    sensor.enable(TIME_STEP)

# IMU
imu = robot.getDevice("inertial unit")
imu.enable(TIME_STEP)

# LiDAR
lidar = robot.getDevice("lidar")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# Variables
x, y = 0.0, 0.0
path_log = []
lidar_points_log = []
last_time = robot.getTime()

start_x, start_y = None, None
mapping = True
replay_index = 0
last_yaw_error = 0

# Wall-following thresholds
FRONT_THRESHOLD = 800
SIDE_LOWER = 500
SIDE_UPPER = 700

print("Robot controller started")

while robot.step(TIME_STEP) != -1:
    current_time = robot.getTime()
    dt = current_time - last_time
    last_time = current_time

    yaw = imu.getRollPitchYaw()[2]
    vl = lm.getVelocity()
    vr = rm.getVelocity()
    speed = (vl + vr) / 2 * WHEEL_RADIUS

    dx = speed * math.cos(yaw) * dt
    dy = speed * math.sin(yaw) * dt
    x += dx
    y += dy

    if start_x is None and start_y is None:
        start_x, start_y = x, y
        print(f"Start position recorded at x={start_x:.3f}, y={start_y:.3f}")

    if mapping:
        fr = ds_front_right.getValue()
        fl = ds_front_left.getValue()
        sr = ds_side_right.getValue()
        sl = ds_side_left.getValue()

        wall_ahead = fr < FRONT_THRESHOLD or fl < FRONT_THRESHOLD

        if wall_ahead:
            lm.setVelocity(-0.2 * MAX_SPEED)
            rm.setVelocity(0.5 * MAX_SPEED)
        elif sr < SIDE_LOWER:
            lm.setVelocity(0.3 * MAX_SPEED)
            rm.setVelocity(0.6 * MAX_SPEED)
        elif sr > SIDE_UPPER:
            lm.setVelocity(0.6 * MAX_SPEED)
            rm.setVelocity(0.3 * MAX_SPEED)
        else:
            lm.setVelocity(0.6 * MAX_SPEED)
            rm.setVelocity(0.6 * MAX_SPEED)

        path_log.append((x, y, yaw))

        # Log LiDAR data with robot pose
        points = lidar.getPointCloud()
        lidar_points_log.append((x, y, yaw, points))

        if current_time > MIN_MAPPING_TIME:
            dist_to_start = math.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
            if dist_to_start < LOOP_CLOSURE_THRESHOLD:
                mapping = False
                replay_index = 0
                print("Returned to start point — mapping complete, starting replay")

    else:
        if replay_index >= len(path_log):
            lm.setVelocity(0)
            rm.setVelocity(0)
            print("Replay finished - robot stopped")
            break

        x_target, y_target, _ = path_log[replay_index]
        dx_target = x_target - x
        dy_target = y_target - y
        dist = math.sqrt(dx_target ** 2 + dy_target ** 2)

        if dist < REPLAY_THRESHOLD:
            replay_index += 1
            last_yaw_error = 0
        else:
            desired_yaw = math.atan2(dy_target, dx_target)
            yaw_error = desired_yaw - yaw

            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi

            if abs(yaw_error) < 0.05:
                steer = 0
            else:
                Kp = 0.7
                Kd = 0.1
                yaw_error_derivative = (yaw_error - last_yaw_error) / dt
                steer = Kp * yaw_error + Kd * yaw_error_derivative
                last_yaw_error = yaw_error

            steer = max(min(steer, 1.0), -1.0)
            base_speed = 0.3 * MAX_SPEED
            lm.setVelocity(base_speed * (1 - steer))
            rm.setVelocity(base_speed * (1 + steer))

# After replay ends — generate map with matplotlib
map_x = []
map_y = []

for robot_x, robot_y, robot_yaw, points in lidar_points_log:
    for p in points:
        lx = math.cos(robot_yaw) * p.x - math.sin(robot_yaw) * p.y
        ly = math.sin(robot_yaw) * p.x + math.cos(robot_yaw) * p.y
        gx = robot_x + lx
        gy = robot_y + ly
        map_x.append(gx)
        map_y.append(gy)

plt.figure(figsize=(8, 8))
plt.scatter(map_x, map_y, s=1, c='black', label="LiDAR Points")
plt.plot([p[0] for p in path_log], [p[1] for p in path_log], c='blue', label="Robot Path")
plt.scatter([path_log[0][0]], [path_log[0][1]], c='green', label="Start")
plt.scatter([path_log[-1][0]], [path_log[-1][1]], c='red', label="End")
plt.title("2D Map from LiDAR + Path")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()           