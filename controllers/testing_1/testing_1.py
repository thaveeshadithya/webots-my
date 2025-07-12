from controller import Robot
import numpy as np

robot = Robot()
timestep = int(robot.getBasicTimeStep())
max_speed = 6.28

# Motors
lm = robot.getMotor("left_motor")
rm = robot.getMotor("right_motor")
lm.setPosition(float('inf'))
rm.setPosition(float('inf'))
lm.setVelocity(0.0)
rm.setVelocity(0.0)

# LiDAR
lidar = robot.getDevice("lidar")
lidar.enable(timestep)
resolution = lidar.getHorizontalResolution()

# Indices for 90° FOV
front_index = resolution // 2
right_index = resolution - 1

# === Tuned Thresholds (in meters) ===
FRONT_STOP_THRESHOLD = 0.22      # Stop and turn if closer than this
FRONT_SLOW_THRESHOLD = 0.32      # Slow down and turn if closer than this
SIDE_WALL_THRESHOLD = 0.18       # Desired distance from wall
MAX_RIGHT_DIST = 0.40            # No wall on right
KP = 1.0

print("Robot controller started.")

while robot.step(timestep) != -1:
    ranges = lidar.getRangeImage()
    ranges = [r if np.isfinite(r) else 10.0 for r in ranges]
    if not ranges or len(ranges) <= right_index:
        continue

    front = ranges[front_index]
    right = ranges[right_index]

    # --- Obstacle Avoidance and Wall Following ---
    if front < FRONT_STOP_THRESHOLD:
        print("Wall ahead! Sharp left turn.")
        lm.setVelocity(-max_speed)
        rm.setVelocity(max_speed)
        continue

    elif front < FRONT_SLOW_THRESHOLD:
        print("Obstacle ahead, sharp left turn.")
        lm.setVelocity(-max_speed)
        rm.setVelocity(max_speed)
        continue

    elif right > MAX_RIGHT_DIST:
        print("No wall on right, sharp right turn.")
        lm.setVelocity(max_speed)
        rm.setVelocity(-max_speed)
        continue

    elif right < SIDE_WALL_THRESHOLD:
        print("Too close to wall, adjusting left.")
        error = SIDE_WALL_THRESHOLD - right
        turn = KP * error
        lm.setVelocity(max_speed * (0.5 - turn))
        rm.setVelocity(max_speed * (0.5 + turn))

    else:
        print("Following wall, moving forward.")
        error = SIDE_WALL_THRESHOLD - right
        turn = KP * error
        lm.setVelocity(max_speed * (0.5 - turn))
        rm.setVelocity(max_speed * (0.5 + turn))