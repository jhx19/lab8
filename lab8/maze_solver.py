#!/usr/bin/env python3
"""
maze_solver.py  —  Lab 8 single-file integrated solution
=========================================================

Usage (no launch file needed):
    python3 maze_solver.py

What this file does automatically
----------------------------------
  1. Launches SLAM       (turtlebot4_navigation slam.launch.py)
  2. Polls until /map has a publisher (SLAM is ready)
  3. Launches Nav2       (turtlebot4_navigation nav2.launch.py)
     → passes your lab8 nav2.yaml so xy_goal_tolerance = 0.05 m
  4. Polls until the Nav2 costmap service appears (Nav2 is fully active)
  5. Waits 3 s for everything to settle
  6. Starts the MazeSolver ROS node + mission state machine

State machine
-------------
  INIT            poll map→base TF until SLAM publishes transforms
                  record start pose (= dock position)
    ↓
  EXPLORE         wall-follower (PID, right-wall) + SLAM builds map
                  ArUco detector in SCANNING phase runs in background
    ↓  (ArUco confirmed)
  STOP_AND_CENTER stop wall-follower; P-controller rotates robot
                  until marker is centred in image (|offset| < 0.05)
    ↓  (centred)
  MEASURE_POSE    robot stationary, facing marker
                  collect MEASURE_FRAMES solvePnP samples → median
                  compute marker position: robot_pose + forward_dist
    ↓  (samples collected)
  SAVE_AND_LABEL  save SLAM map via map_saver_cli
                  publish orange RViz cube at ArUco map position
    ↓
  RETURN_TO_START Nav2 NavigateToPose → initial_pose
    ↓  (arrived)
  GOTO_ARUCO      Nav2 NavigateToPose → GOAL_OFFSET_M in front of marker
    ↓  (arrived)
  DONE            print elapsed time

All SLAM/Nav2 subprocesses are terminated cleanly on Ctrl-C.
"""

import math
import os
import subprocess
import time
from enum import Enum, auto

import cv2
import cv2.aruco as aruco
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker

import tf2_ros


# ══════════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Wall follower ─────────────────────────────────────────────────────────────
DESIRED_WALL_DIST = 0.22    # target right-wall distance (m)
WALL_LOST_DIST    = 0.50    # right reading above this → wall lost
FRONT_DANGER_DIST = 0.45    # front reading below this → obstacle
WF_Kp             = 1.2
WF_Ki             = 0.01
WF_Kd             = 0.15
WF_INTEGRAL_MAX   = 1.0
WF_LINEAR_SPEED   = 0.25    # m/s base forward speed
WF_ANGULAR_MAX    = 1.2     # rad/s PID output clamp
WF_TURN_SPEED     = 0.8     # rad/s override turn rate
WF_SECTOR_HALF    = 0.075   # ±7.5% of scan indices per sector

# ── ArUco detector ────────────────────────────────────────────────────────────
MARKER_SIZE_M         = 0.05              # physical marker side length (m)
ARUCO_DICT_TYPE       = aruco.DICT_4X4_50
IMAGE_TOPIC           = '/oakd/right/image_raw/compressed'
CAMERA_INFO_TOPIC     = '/oakd/right/camera_info'
CONFIRM_FRAMES        = 5                 # consecutive detections to confirm
MIN_DIST              = 0.10             # reject detections closer than this
MAX_DIST              = 3.00             # reject detections farther than this
MEASURE_FRAMES        = 5                 # solvePnP samples to median-average
CAMERA_FORWARD_OFFSET = 0.10             # camera lens ahead of base_footprint (m)
PUBLISH_DEBUG_IMAGE   = True

# ── Centering P-controller ────────────────────────────────────────────────────
# pixel_offset (from aruco detector) is normalised: -1 = far left, +1 = far right
# angular_z = -Kp_CENTER * pixel_offset  (negative → turn toward marker)
Kp_CENTER        = 0.6
CENTER_THRESHOLD = 0.05    # |offset| < this → centred (≈ ±8 px on 320-wide image)
CENTER_OMEGA_MAX = 0.4     # rad/s maximum centering speed
CENTER_OMEGA_MIN = 0.08    # rad/s minimum to overcome static friction
CENTER_TIMEOUT_S = 5.0     # seconds without an offset update → FAILED

# ── Navigation ────────────────────────────────────────────────────────────────
GOAL_OFFSET_M = 0.10    # stop this far in front of the ArUco cube (m)
#   With xy_goal_tolerance = 0.05 in nav2.yaml,
#   actual stopping distance = 10 cm ± 5 cm = 5–15 cm ✓

# ── Map / RViz ────────────────────────────────────────────────────────────────
MAP_SAVE_DIR  = os.path.expanduser('~/turtlebot4_ws/src/lab8/maps')
MAP_SAVE_NAME = 'maze_map'
MAP_FRAME     = 'map'
MARKER_SCALE  = 0.15
MARKER_ORANGE = (1.0, 0.5, 0.0)    # R, G, B




# ══════════════════════════════════════════════════════════════════════════════
# STATE ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class State(Enum):
    INIT             = auto()   # wait for SLAM TF, record start pose
    EXPLORE          = auto()   # wall-following + ArUco scanning
    STOP_AND_CENTER  = auto()   # stop, rotate to centre marker in frame
    MEASURE_POSE     = auto()   # collect solvePnP samples, lock map pose
    SAVE_AND_LABEL   = auto()   # save map, publish RViz marker
    RETURN_TO_START  = auto()   # Nav2 → start position
    GOTO_ARUCO       = auto()   # Nav2 → ArUco position
    DONE             = auto()
    FAILED           = auto()


class DetPhase:
    """Sub-states of the ArUco detection pipeline (runs inside EXPLORE/CENTER/MEASURE)."""
    SCANNING  = 'SCANNING'    # looking for marker
    CENTERING = 'CENTERING'   # publishing pixel offset for P-controller
    MEASURING = 'MEASURING'   # collecting solvePnP samples
    DONE      = 'DONE'        # pose locked


# ══════════════════════════════════════════════════════════════════════════════
# PID CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class PIDController:
    """Discrete PID with anti-windup integral clamping."""

    def __init__(self, kp: float, ki: float, kd: float, integral_max: float = 1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_max = integral_max
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, error: float, now: float) -> float:
        if self._prev_time is None:
            self._prev_time = now
            self._prev_error = error
            return self.kp * error
        dt = now - self._prev_time
        if dt <= 0.0:
            return self.kp * error
        p = self.kp * error
        self._integral = max(-self.integral_max,
                             min(self.integral_max,
                                 self._integral + error * dt))
        i = self.ki * self._integral
        d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error
        self._prev_time = now
        return p + i + d

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None


# ══════════════════════════════════════════════════════════════════════════════
# MAZE SOLVER NODE
# ══════════════════════════════════════════════════════════════════════════════

class MazeSolver(Node):
    """
    Single ROS2 node that integrates:
      - Wall follower (PID, right-wall)
      - ArUco detector (3-phase: SCANNING → CENTERING → MEASURING)
      - Centering P-controller
      - Nav2 NavigateToPose action client
      - Full state machine orchestrator
    """

    def __init__(self):
        super().__init__('maze_solver')

        # Sensor callbacks can run concurrently (image, scan, camera_info).
        # Timer / action callbacks use a separate exclusive group so they
        # never run concurrently with each other, preventing state corruption.
        self._sensor_cbg = ReentrantCallbackGroup()
        self._ctrl_cbg   = MutuallyExclusiveCallbackGroup()

        # ── TF2 ───────────────────────────────────────────────────────────────
        # TransformListener must be created after the node so it can
        # subscribe to /tf and /tf_static using this node's executor.
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Mission bookkeeping ───────────────────────────────────────────────
        self.state              = State.INIT
        self._mission_start     = None
        self._initial_pose: PoseStamped = None   # dock / start position
        self._aruco_pose:   PoseStamped = None   # locked ArUco map-frame pose

        # ── Wall follower state ───────────────────────────────────────────────
        self._wf_running = False
        self._wf_pid     = PIDController(WF_Kp, WF_Ki, WF_Kd, WF_INTEGRAL_MAX)

        # ── ArUco detector state ──────────────────────────────────────────────
        self._det_phase      = DetPhase.SCANNING
        self._cam_matrix     = None
        self._dist_coeffs    = None
        self._cam_frame      = None
        self._aruco_detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(ARUCO_DICT_TYPE),
            aruco.DetectorParameters())
        # Phase A — confirmation counter
        self._confirm_id    = None
        self._confirm_count = 0
        # Phase C — measurement samples: list of (x, y, yaw)
        self._meas_samples  = []

        # ── Centering state ───────────────────────────────────────────────────
        self._pixel_offset      = None   # latest normalised offset from image cb
        self._last_offset_time  = None
        self._centering_timer   = None

        # ── Nav2 action client ────────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._marker_pub  = self.create_publisher(Marker, '/visualization_marker', 10)
        self._debug_pub   = (
            self.create_publisher(CompressedImage, '/aruco_debug_image', 10)
            if PUBLISH_DEBUG_IMAGE else None)
        self._rviz_marker = None   # set in _publish_aruco_marker, republished at 2 Hz

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._camera_info_cb, 1,
            callback_group=self._sensor_cbg)
        self.create_subscription(
            CompressedImage, IMAGE_TOPIC, self._image_cb, 10,
            callback_group=self._sensor_cbg)
        self.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10,
            callback_group=self._sensor_cbg)

        # ── Background timer — republish RViz marker at 2 Hz ─────────────────
        self.create_timer(0.5, self._republish_marker,
                          callback_group=self._ctrl_cbg)

        # ── Kick off INIT polling ─────────────────────────────────────────────
        self.get_logger().info('MazeSolver ready. Waiting for map→base TF...')
        self._init_timer = self.create_timer(
            1.0, self._poll_for_initial_pose,
            callback_group=self._ctrl_cbg)

    # ══════════════════════════════════════════════════════════════════════════
    # INIT — poll TF until SLAM is publishing transforms
    # ══════════════════════════════════════════════════════════════════════════

    def _poll_for_initial_pose(self):
        """
        Fires at 1 Hz until map→base_footprint TF is available.
        Records the robot's start position, then transitions to EXPLORE.

        We use the map frame (not odom) because:
          • map is SLAM-corrected and globally consistent
          • odom drifts over time due to wheel slip
        """
        pose = self._get_robot_pose_in_map()
        if pose is None:
            self.get_logger().warn(
                'map→base TF not yet available. Retrying...',
                throttle_duration_sec=3.0)
            return

        self._init_timer.cancel()
        self._initial_pose  = pose
        self._mission_start = time.time()
        self.get_logger().info(
            f'Start pose recorded: x={pose.pose.position.x:.3f}  '
            f'y={pose.pose.position.y:.3f}')
        self._transition_to(State.EXPLORE)

    # ══════════════════════════════════════════════════════════════════════════
    # STATE MACHINE — transitions
    # ══════════════════════════════════════════════════════════════════════════

    def _transition_to(self, new_state: State):
        self.get_logger().info(
            f'─── {self.state.name} → {new_state.name} ───')
        self.state = new_state
        dispatch = {
            State.EXPLORE:         self._enter_explore,
            State.STOP_AND_CENTER: self._enter_stop_and_center,
            State.MEASURE_POSE:    self._enter_measure_pose,
            State.SAVE_AND_LABEL:  self._enter_save_and_label,
            State.RETURN_TO_START: self._enter_return_to_start,
            State.GOTO_ARUCO:      self._enter_goto_aruco,
            State.DONE:            self._enter_done,
            State.FAILED:          self._enter_failed,
        }
        if new_state in dispatch:
            dispatch[new_state]()

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: EXPLORE
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_explore(self):
        """Start wall-follower and reset ArUco detector to SCANNING."""
        self.get_logger().info(
            '[EXPLORE] Wall-follower started. ArUco detector scanning.')
        self._wf_pid.reset()
        self._wf_running = True
        # Reset detection sub-state
        self._det_phase      = DetPhase.SCANNING
        self._confirm_id     = None
        self._confirm_count  = 0
        self._meas_samples   = []

    # ── Wall follower (LiDAR scan callback) ───────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        """PID right-wall follower. Only active during EXPLORE."""
        if not self._wf_running or self.state != State.EXPLORE:
            return

        right = self._sector_min(msg, centre_frac=0.00, half_frac=WF_SECTOR_HALF)
        front = self._sector_min(msg, centre_frac=0.25, half_frac=WF_SECTOR_HALF)
        now   = self.get_clock().now().nanoseconds * 1e-9

        twist = Twist()

        # Priority 1: obstacle ahead → turn left in place
        if front < FRONT_DANGER_DIST:
            # Slow down proportionally as obstacle approaches
            ratio = max(front - FRONT_DANGER_DIST * 0.2, 0.0) / (FRONT_DANGER_DIST * 0.8)
            twist.linear.x  = WF_LINEAR_SPEED * 0.6 * ratio
            twist.angular.z = WF_TURN_SPEED
            self._wf_pid.reset()
            self.get_logger().info(
                f'[WF] Obstacle {front:.2f}m → L-turn', throttle_duration_sec=1.0)

        # Priority 2: right wall lost → slow right turn to reacquire
        elif right > WALL_LOST_DIST:
            twist.linear.x  = WF_LINEAR_SPEED * 0.6
            twist.angular.z = -WF_TURN_SPEED * 0.7
            self._wf_pid.reset()
            self.get_logger().info(
                f'[WF] Wall lost ({right:.2f}m) → reacquiring', throttle_duration_sec=1.0)

        # Normal: PID maintains desired distance to right wall
        else:
            # error > 0 → too far from wall → steer right (negative ω)
            error     = right - DESIRED_WALL_DIST
            pid_out   = self._wf_pid.compute(error, now)
            angular_z = max(-WF_ANGULAR_MAX, min(WF_ANGULAR_MAX, -pid_out))

            # Reduce forward speed on sharp turns for smoother cornering
            if abs(angular_z) > WF_ANGULAR_MAX * 0.6:
                twist.linear.x = WF_LINEAR_SPEED * max(0.2, 1.0 - abs(angular_z) / WF_ANGULAR_MAX)
            else:
                twist.linear.x = WF_LINEAR_SPEED
            twist.angular.z = angular_z
            self.get_logger().info(
                f'[WF] r={right:.3f} err={error:+.3f} ω={angular_z:+.3f}',
                throttle_duration_sec=0.5)

        self._cmd_vel_pub.publish(twist)

    def _sector_min(self, msg: LaserScan,
                    centre_frac: float, half_frac: float) -> float:
        """
        Minimum valid range in an index-defined angular sector.

        Index conventions (LiDAR spec):
          centre_frac = 0.00 → right  (index 0)
          centre_frac = 0.25 → front  (index n/4)
          centre_frac = 0.50 → left   (index n/2)
          centre_frac = 0.75 → back   (index 3n/4)
        """
        n  = len(msg.ranges)
        ci = int(centre_frac * n)
        hi = int(half_frac   * n)
        idx = [(ci + o) % n for o in range(-hi, hi + 1)]
        valid = [
            msg.ranges[i] for i in idx
            if (not math.isnan(msg.ranges[i])
                and not math.isinf(msg.ranges[i])
                and msg.range_min < msg.ranges[i] < msg.range_max)
        ]
        return min(valid) if valid else msg.range_max

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: STOP_AND_CENTER
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_stop_and_center(self):
        """
        Stop wall-follower immediately, then rotate in-place until the
        ArUco marker is centred in the camera image.

        Why centre before measuring?
          solvePnP's translation vector (tx, ty) carries angular error
          when the marker is off-axis. After centring, the robot faces
          the marker directly, so tx≈ty≈0 and tz = true forward distance.
          This makes the map-frame projection far more accurate.
        """
        self.get_logger().info('[STOP_AND_CENTER] Stopping. Starting centering.')
        self._wf_running = False
        self._stop_motors()
        # Switch detector to publish pixel offsets instead of confirming
        self._det_phase        = DetPhase.CENTERING
        self._pixel_offset     = None
        self._last_offset_time = None
        # 20 Hz centering loop
        self._centering_timer = self.create_timer(
            0.05, self._centering_loop,
            callback_group=self._ctrl_cbg)

    def _centering_loop(self):
        """
        Runs at 20 Hz during STOP_AND_CENTER.
        Reads latest pixel_offset (set by _image_cb) and commands rotation.

        Control law:
          ω = clip(-Kp * offset, -MAX…-MIN or +MIN…+MAX)
          offset > 0 (marker right) → ω < 0 → CW → marker moves left
          offset < 0 (marker left)  → ω > 0 → CCW → marker moves right
        """
        if self.state != State.STOP_AND_CENTER:
            self._centering_timer.cancel()
            return

        now = time.time()

        # Safety: if no offset update for CENTER_TIMEOUT_S → marker lost
        if (self._last_offset_time is not None
                and now - self._last_offset_time > CENTER_TIMEOUT_S):
            self.get_logger().warn('[CENTER] Timeout — marker lost. → FAILED')
            self._stop_motors()
            self._centering_timer.cancel()
            self._transition_to(State.FAILED)
            return

        if self._pixel_offset is None:
            return   # waiting for first image with the marker

        offset = self._pixel_offset

        if abs(offset) < CENTER_THRESHOLD:
            self.get_logger().info(
                f'[CENTER] Centred! offset={offset:+.4f} → MEASURE_POSE')
            self._stop_motors()
            self._centering_timer.cancel()
            self._transition_to(State.MEASURE_POSE)
            return

        # Proportional rotation toward marker
        omega = -Kp_CENTER * offset
        # Apply minimum speed (overcome static friction), preserve sign
        if 0.0 < abs(omega) < CENTER_OMEGA_MIN:
            omega = math.copysign(CENTER_OMEGA_MIN, omega)
        omega = max(-CENTER_OMEGA_MAX, min(CENTER_OMEGA_MAX, omega))

        twist = Twist()
        twist.angular.z = omega
        self._cmd_vel_pub.publish(twist)
        self.get_logger().info(
            f'[CENTER] offset={offset:+.4f}  ω={omega:+.3f}',
            throttle_duration_sec=0.2)

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: MEASURE_POSE
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_measure_pose(self):
        """
        Robot is stationary, facing the marker. Switch detector to MEASURING.
        The image callback will call _handle_measuring() on each frame.
        When MEASURE_FRAMES samples are collected, it calls _on_pose_locked().
        """
        self.get_logger().info(
            f'[MEASURE_POSE] Collecting {MEASURE_FRAMES} solvePnP samples...')
        self._meas_samples = []
        self._det_phase    = DetPhase.MEASURING

    def _on_pose_locked(self, final_pose: PoseStamped):
        """Called by image callback once median pose is computed."""
        self._aruco_pose = final_pose
        self._det_phase  = DetPhase.DONE
        if self.state == State.MEASURE_POSE:
            self._transition_to(State.SAVE_AND_LABEL)

    # ══════════════════════════════════════════════════════════════════════════
    # ArUco IMAGE CALLBACK (runs in all three detection phases)
    # ══════════════════════════════════════════════════════════════════════════

    def _camera_info_cb(self, msg: CameraInfo):
        """Load camera intrinsics once (first message only)."""
        if self._cam_matrix is not None:
            return
        self._cam_matrix  = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.get_logger().info('Camera intrinsics loaded.')

    def _image_cb(self, msg: CompressedImage):
        """
        Main ArUco detection callback. Routes to the correct phase handler.
        Active during EXPLORE, STOP_AND_CENTER, and MEASURE_POSE.
        """
        if self._cam_matrix is None:
            return
        if self._det_phase == DetPhase.DONE:
            return
        if self.state not in (State.EXPLORE, State.STOP_AND_CENTER,
                              State.MEASURE_POSE):
            return

        if self._cam_frame is None:
            self._cam_frame = msg.header.frame_id

        # Decode JPEG → grayscale
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        gray   = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return
        _, w = gray.shape[:2]

        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if self._debug_pub else None

        # Detect ArUco markers
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        detection        = self._best_detection(corners, ids)

        # Route to phase handler
        if self._det_phase == DetPhase.SCANNING:
            self._handle_scanning(detection, debug)
        elif self._det_phase == DetPhase.CENTERING:
            self._handle_centering(detection, debug, w)
        elif self._det_phase == DetPhase.MEASURING:
            self._handle_measuring(detection, debug)

        if debug is not None:
            cv2.putText(debug, self._det_phase, (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
            self._publish_debug(debug, msg.header)

    # ── Phase A: SCANNING ─────────────────────────────────────────────────────

    def _handle_scanning(self, detection, debug):
        """
        Accumulate CONFIRM_FRAMES consecutive detections of the same marker.
        Once confirmed, tell the state machine to transition to STOP_AND_CENTER.

        Why require consecutive frames?
          Single-frame detections can be false positives (reflections, noise).
          Requiring N consecutive detections of the same ID at a plausible
          distance filters these out reliably.
        """
        if detection is None:
            if self._confirm_count > 0:
                self.get_logger().info(
                    f'[SCAN] ID {self._confirm_id} lost — '
                    f'reset ({self._confirm_count}→0)')
            self._confirm_id    = None
            self._confirm_count = 0
            return

        marker_id, corners_i, tvec, _, dist = detection

        # Distance plausibility check
        if not (MIN_DIST < dist < MAX_DIST):
            return

        if marker_id != self._confirm_id:
            self._confirm_id    = marker_id
            self._confirm_count = 1
            self.get_logger().info(
                f'[SCAN] New candidate ID {marker_id}  '
                f'dist={dist:.2f}m  need {CONFIRM_FRAMES} frames')
        else:
            self._confirm_count += 1
            self.get_logger().info(
                f'[SCAN] ID {marker_id}: '
                f'{self._confirm_count}/{CONFIRM_FRAMES}  dist={dist:.2f}m')

        if debug is not None:
            cv2.aruco.drawDetectedMarkers(debug, [corners_i])

        if self._confirm_count >= CONFIRM_FRAMES:
            self.get_logger().info(f'[SCAN] ID {marker_id} CONFIRMED.')
            # det_phase change happens inside _enter_stop_and_center
            if self.state == State.EXPLORE:
                self._transition_to(State.STOP_AND_CENTER)

    # ── Phase B: CENTERING ────────────────────────────────────────────────────

    def _handle_centering(self, detection, debug, image_w: int):
        """
        Compute normalised pixel offset of the marker centre and store it
        for the centering loop timer to consume.

        offset = (marker_cx − image_cx) / (image_w / 2)
          -1.0  → marker fully LEFT  → robot must turn LEFT  (+ω)
          +1.0  → marker fully RIGHT → robot must turn RIGHT (-ω)
           0.0  → centred

        Normalising by half-width makes the gain camera-resolution-independent.
        """
        if detection is None:
            self.get_logger().warn(
                '[CENTER] Marker lost.', throttle_duration_sec=1.0)
            return

        _, corners_i, _, _, dist = detection
        pts = corners_i.reshape(4, 2)
        cx  = float(pts[:, 0].mean())

        image_cx = image_w / 2.0
        offset   = (cx - image_cx) / (image_w / 2.0)

        # Store for _centering_loop to read
        self._pixel_offset     = offset
        self._last_offset_time = time.time()

        self.get_logger().info(
            f'[CENTER] offset={offset:+.3f}  dist={dist:.2f}m',
            throttle_duration_sec=0.3)

        if debug is not None:
            cv2.aruco.drawDetectedMarkers(debug, [corners_i])
            cv2.line(debug, (int(image_cx), 0), (int(image_cx), debug.shape[0]),
                     (0, 100, 255), 1)
            cv2.line(debug, (int(cx), 0), (int(cx), debug.shape[0]),
                     (0, 255, 100), 1)

    # ── Phase C: MEASURING ────────────────────────────────────────────────────

    def _handle_measuring(self, detection, debug):
        """
        Collect MEASURE_FRAMES solvePnP samples and compute the ArUco
        map-frame position without relying on the camera TF chain.

        Why bypass camera TF?
          The camera→base transform depends on physical mounting calibration.
          Even a small error propagates into a large position error at distance.
          Instead, after centring the robot directly faces the marker, so:

            marker_x = robot_x + (tz + CAMERA_FORWARD_OFFSET) * cos(robot_yaw)
            marker_y = robot_y + (tz + CAMERA_FORWARD_OFFSET) * sin(robot_yaw)

          where tz is solvePnP's forward distance (camera Z-axis = forward),
          robot_x/y/yaw come from the well-calibrated map→base_footprint TF,
          and CAMERA_FORWARD_OFFSET corrects for the camera being ahead of
          the robot's center (≈ 10 cm on TurtleBot4 with OAK-D cradle).

        Median over MEASURE_FRAMES samples rejects outlier frames (motion blur,
        JPEG artefacts) while √N averaging reduces random pixel-level noise.
        """
        # Guard: stop collecting once we have enough (race condition safety)
        if len(self._meas_samples) >= MEASURE_FRAMES:
            return
        if detection is None:
            self.get_logger().warn(
                '[MEASURE] Marker not visible.', throttle_duration_sec=1.0)
            return

        _, corners_i, tvec, _, _ = detection
        tz = float(tvec[2])   # camera Z = forward distance to marker

        robot_pose = self._get_robot_pose_in_map()
        if robot_pose is None:
            return

        rx  = robot_pose.pose.position.x
        ry  = robot_pose.pose.position.y
        q   = robot_pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        fwd      = tz + CAMERA_FORWARD_OFFSET
        marker_x = rx + fwd * math.cos(yaw)
        marker_y = ry + fwd * math.sin(yaw)

        self._meas_samples.append((marker_x, marker_y, yaw))
        n = len(self._meas_samples)

        self.get_logger().info(
            f'[MEASURE] sample {n}/{MEASURE_FRAMES}  '
            f'tz={tz:.3f}m  robot=({rx:.3f},{ry:.3f},{math.degrees(yaw):.1f}°)  '
            f'→ marker=({marker_x:.4f},{marker_y:.4f})')

        if debug is not None:
            cv2.aruco.drawDetectedMarkers(debug, [corners_i])
            cv2.putText(debug, f'Measuring {n}/{MEASURE_FRAMES}',
                        (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if n < MEASURE_FRAMES:
            return

        # ── All samples collected: compute median pose ────────────────────────
        xs   = np.array([s[0] for s in self._meas_samples])
        ys   = np.array([s[1] for s in self._meas_samples])
        yaws = np.array([s[2] for s in self._meas_samples])

        xm  = float(np.median(xs))
        ym  = float(np.median(ys))
        yaw = float(np.median(yaws))

        self.get_logger().info(
            f'[MEASURE] LOCKED: x={xm:.4f}  y={ym:.4f}  '
            f'(spread x=[{xs.min():.4f},{xs.max():.4f}]  '
            f'y=[{ys.min():.4f},{ys.max():.4f}])')

        final = PoseStamped()
        final.header.frame_id    = MAP_FRAME
        final.header.stamp       = self.get_clock().now().to_msg()
        final.pose.position.x    = xm
        final.pose.position.y    = ym
        final.pose.position.z    = 0.0
        final.pose.orientation.z = math.sin(yaw / 2.0)
        final.pose.orientation.w = math.cos(yaw / 2.0)

        self._on_pose_locked(final)

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: SAVE_AND_LABEL
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_save_and_label(self):
        self.get_logger().info('[SAVE_AND_LABEL] Saving map and labelling ArUco.')
        self._save_done = False
        # One-shot timer: the cancel() inside _do_save ensures it fires only once
        self._save_timer = self.create_timer(
            1.0, self._do_save,
            callback_group=self._ctrl_cbg)

    def _do_save(self):
        # Cancel immediately — prevents this from firing again
        self._save_timer.cancel()
        if self._save_done:
            return
        self._save_done = True

        # Save SLAM map
        os.makedirs(MAP_SAVE_DIR, exist_ok=True)
        map_path = os.path.join(MAP_SAVE_DIR, MAP_SAVE_NAME)
        try:
            subprocess.Popen(
                ['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                 '-f', map_path, '--ros-args', '-p', 'save_map_timeout:=5.0'])
            self.get_logger().info(f'Map saving → {map_path}.pgm/.yaml')
        except Exception as e:
            self.get_logger().error(f'Map save failed: {e}')

        # Publish orange RViz cube at ArUco map position
        if self._aruco_pose is not None:
            self._publish_aruco_marker(self._aruco_pose)

        self._transition_to(State.RETURN_TO_START)

    def _publish_aruco_marker(self, pose: PoseStamped):
        m            = Marker()
        m.header     = pose.header
        m.ns         = 'aruco_target'
        m.id         = 0
        m.type       = Marker.CUBE
        m.action     = Marker.ADD
        m.pose       = pose.pose
        m.pose.position.z = MARKER_SCALE / 2.0
        m.scale.x = m.scale.y = m.scale.z = MARKER_SCALE
        m.color.r, m.color.g, m.color.b = MARKER_ORANGE
        m.color.a    = 0.85
        m.lifetime.sec = 0   # never expire
        self._rviz_marker = m
        self._marker_pub.publish(m)

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: RETURN_TO_START
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_return_to_start(self):
        """
        Navigate back to the recorded start (dock) position.
        A 2 s settle delay lets Nav2's local costmap stabilise after the
        robot stopped moving — avoids the planner rejecting goals immediately.
        """
        self.get_logger().info(
            '[RETURN_TO_START] Waiting 2 s before sending Nav2 goal...')
        self._ret_timer = self.create_timer(
            2.0, self._send_goto_start,
            callback_group=self._ctrl_cbg)

    def _send_goto_start(self):
        self._ret_timer.cancel()
        if self._initial_pose is None:
            self.get_logger().error('No start pose stored!')
            self._transition_to(State.FAILED)
            return
        # Refresh timestamp so Nav2 doesn't reject a stale PoseStamped
        goal = PoseStamped()
        goal.header.frame_id = MAP_FRAME
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.pose            = self._initial_pose.pose
        self._nav_to(goal, success_tag='at_start')

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: GOTO_ARUCO
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_goto_aruco(self):
        """Navigate to GOAL_OFFSET_M in front of the ArUco marker."""
        self.get_logger().info('[GOTO_ARUCO] Navigating to ArUco cube...')
        if self._aruco_pose is None:
            self.get_logger().error('No ArUco pose stored!')
            self._transition_to(State.FAILED)
            return
        goal = self._compute_aruco_goal(self._aruco_pose)
        self._nav_to(goal, success_tag='at_goal')

    def _compute_aruco_goal(self, aruco_pose: PoseStamped) -> PoseStamped:
        """
        Compute goal pose GOAL_OFFSET_M in front of the ArUco marker.

        The stored orientation encodes the robot's yaw AT measurement time
        — i.e. pointing FROM the robot TOWARD the marker.
        "In front of the marker" = step back GOAL_OFFSET_M along that yaw.

        Diagram (top-down):
          [start] ──yaw──> [ArUco]
                    [goal] ← GOAL_OFFSET_M ←
        """
        mx  = aruco_pose.pose.position.x
        my  = aruco_pose.pose.position.y
        q   = aruco_pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        gx = mx - GOAL_OFFSET_M * math.cos(yaw)
        gy = my - GOAL_OFFSET_M * math.sin(yaw)

        goal                     = PoseStamped()
        goal.header.frame_id     = MAP_FRAME
        goal.header.stamp        = self.get_clock().now().to_msg()
        goal.pose.position.x     = gx
        goal.pose.position.y     = gy
        goal.pose.orientation.z  = math.sin(yaw / 2.0)
        goal.pose.orientation.w  = math.cos(yaw / 2.0)

        self.get_logger().info(
            f'ArUco goal: marker=({mx:.3f},{my:.3f})  '
            f'yaw={math.degrees(yaw):.1f}°  goal=({gx:.3f},{gy:.3f})')
        return goal

    # ══════════════════════════════════════════════════════════════════════════
    # STATE: DONE / FAILED
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_done(self):
        elapsed = time.time() - self._mission_start
        self.get_logger().info(
            f'\n╔══════════════════════════════╗\n'
            f'║   MISSION COMPLETE  ✓         ║\n'
            f'║   Total time: {elapsed:>6.1f} s       ║\n'
            f'╚══════════════════════════════╝')

    def _enter_failed(self):
        elapsed = time.time() - (self._mission_start or time.time())
        self.get_logger().error(
            f'[FAILED] Mission failed after {elapsed:.1f} s.')
        self._stop_motors()

    # ══════════════════════════════════════════════════════════════════════════
    # NAVIGATION — Nav2 NavigateToPose action
    # ══════════════════════════════════════════════════════════════════════════

    def _nav_to(self, pose: PoseStamped, success_tag: str):
        """Send a NavigateToPose goal. success_tag identifies which state to enter on success."""
        if not self._nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 action server not available.')
            self._transition_to(State.FAILED)
            return

        nav_goal      = NavigateToPose.Goal()
        nav_goal.pose = pose
        future        = self._nav_client.send_goal_async(
            nav_goal, feedback_callback=self._nav_feedback_cb)
        future.add_done_callback(
            lambda f, tag=success_tag: self._nav_goal_cb(f, tag))

    def _nav_feedback_cb(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        self.get_logger().info(
            f'Nav2 distance remaining: {dist:.2f} m',
            throttle_duration_sec=5.0)

    def _nav_goal_cb(self, future, success_tag: str):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('Nav2 goal rejected.')
            self._transition_to(State.FAILED)
            return
        gh.get_result_async().add_done_callback(
            lambda f, tag=success_tag: self._nav_result_cb(f, tag))

    def _nav_result_cb(self, future, success_tag: str):
        status = future.result().status
        if status == 4:   # GoalStatus.STATUS_SUCCEEDED
            self.get_logger().info(f'Nav2 SUCCEEDED → {success_tag}')
            if success_tag == 'at_start':
                self._transition_to(State.GOTO_ARUCO)
            elif success_tag == 'at_goal':
                self._transition_to(State.DONE)
        else:
            self.get_logger().error(f'Nav2 FAILED (status={status})')
            self._transition_to(State.FAILED)

    # ══════════════════════════════════════════════════════════════════════════
    # TF HELPER
    # ══════════════════════════════════════════════════════════════════════════

    def _get_robot_pose_in_map(self) -> PoseStamped:
        """
        Look up map → base_footprint (falls back to base_link).
        Returns PoseStamped or None if the TF is unavailable.
        """
        for frame in ('base_footprint', 'base_link'):
            try:
                tf = self._tf_buffer.lookup_transform(
                    MAP_FRAME, frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5))
                pose                  = PoseStamped()
                pose.header.frame_id  = MAP_FRAME
                pose.header.stamp     = self.get_clock().now().to_msg()
                pose.pose.position.x  = tf.transform.translation.x
                pose.pose.position.y  = tf.transform.translation.y
                pose.pose.position.z  = tf.transform.translation.z
                pose.pose.orientation = tf.transform.rotation
                return pose
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # ARUCO solvePnP HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _best_detection(self, corners, ids):
        """Return (marker_id, corners_i, tvec, rvec, dist) for the first valid marker, or None."""
        if ids is None or len(ids) == 0:
            return None
        for i, mid in enumerate(ids.flatten()):
            result = self._solve_pnp(corners[i])
            if result is not None:
                return int(mid), corners[i], *result
        return None

    def _solve_pnp(self, corners_i):
        """Run solvePnP. Returns (tvec, rvec, dist) or None."""
        half    = MARKER_SIZE_M / 2.0
        obj_pts = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)
        img_pts = corners_i.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts,
            self._cam_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        tvec = tvec.flatten()
        rvec = rvec.flatten()
        return tvec, rvec, float(np.linalg.norm(tvec))

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def _stop_motors(self):
        self._cmd_vel_pub.publish(Twist())

    def _publish_debug(self, frame, header):
        if self._debug_pub is None:
            return
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        out        = CompressedImage()
        out.header = header
        out.format = 'jpeg'
        out.data   = buf.tobytes()
        self._debug_pub.publish(out)

    def _republish_marker(self):
        """Republish the RViz ArUco marker at 2 Hz so RViz never loses it."""
        if self._rviz_marker is not None:
            self._rviz_marker.header.stamp = self.get_clock().now().to_msg()
            self._marker_pub.publish(self._rviz_marker)


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _wait_for_topic(topic: str, timeout: float = 120.0):
    """
    Block until `topic` has at least one publisher.
    Uses `ros2 topic info` (no ROS node required — CLI only).
    """
    print(f'[startup] Waiting for {topic} ...', flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ['ros2', 'topic', 'info', topic],
            capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if 'Publisher count:' in line:
                    try:
                        if int(line.split(':')[1].strip()) > 0:
                            print(f'[startup] {topic} is live ✓', flush=True)
                            return
                    except ValueError:
                        pass
        time.sleep(1.0)
    raise RuntimeError(f'Timeout waiting for topic {topic}')


def _wait_for_service(substr: str, timeout: float = 120.0):
    """
    Block until a service whose name contains `substr` appears in
    `ros2 service list`. Used to detect Nav2 fully active.
    """
    print(f'[startup] Waiting for Nav2 service ({substr}) ...', flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ['ros2', 'service', 'list'],
            capture_output=True, text=True)
        if substr in result.stdout:
            print('[startup] Nav2 is active ✓', flush=True)
            return
        time.sleep(2.0)
    raise RuntimeError(f'Timeout waiting for Nav2 service matching "{substr}"')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    procs = []
    try:
        # ── 1. Launch SLAM ────────────────────────────────────────────────────
        print('[startup] Launching SLAM ...', flush=True)
        slam_proc = subprocess.Popen(
            ['ros2', 'launch', 'turtlebot4_navigation', 'slam.launch.py',
             'use_sim_time:=false'])
        procs.append(slam_proc)

        # ── 2. Wait for SLAM to publish /map ──────────────────────────────────
        _wait_for_topic('/map')

        # ── 3. Launch Nav2 (default params) ──────────────────────────────────
        print('[startup] Launching Nav2 ...', flush=True)
        nav2_proc = subprocess.Popen(
            ['ros2', 'launch', 'turtlebot4_navigation', 'nav2.launch.py',
             'use_sim_time:=false'])
        procs.append(nav2_proc)

        # ── 4. Wait for Nav2 to become fully active ───────────────────────────
        _wait_for_service('global_costmap/clear_entirely_global_costmap')

        # ── 5. Short settle — let costmaps initialise ─────────────────────────
        print('[startup] Settling 3 s ...', flush=True)
        time.sleep(3.0)

        # ── 6. Start the maze solver node ─────────────────────────────────────
        print('[startup] Starting MazeSolver — MISSION BEGINS', flush=True)
        rclpy.init()
        node     = MazeSolver()
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        try:
            executor.spin()
        except KeyboardInterrupt:
            print('\n[startup] KeyboardInterrupt — shutting down.', flush=True)
        finally:
            node.destroy_node()
            rclpy.shutdown()

    finally:
        # ── Clean up SLAM and Nav2 subprocesses ───────────────────────────────
        print('[startup] Terminating SLAM and Nav2 ...', flush=True)
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                p.kill()
        print('[startup] Done.', flush=True)


if __name__ == '__main__':
    main()