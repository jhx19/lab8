"""
orchestrator.py
---------------
Phase 2: Central state machine — wall following + SLAM + ArUco centering.

Full state machine:
──────────────────────────────────────────────────────────────────────
  INIT
    │  Poll TF map→base until SLAM is ready.
    │  Record initial pose (= dock position for later return).
    ▼
  EXPLORE
    │  Wall-follower running + SLAM building map.
    │  ArUco detector running in SCANNING phase.
    │  Transition: /aruco_confirmed fires.
    ▼
  STOP_AND_CENTER
    │  Stop wall-follower (robot halts).
    │  Read /aruco_pixel_offset and rotate in place with a P-controller
    │  until the marker is centred in the image (|offset| < threshold).
    │  Centering ensures the camera optical axis points directly at the
    │  marker, minimising angular error in the solvePnP translation.
    │  Once centred: stop rotating, stop SLAM, tell detector to measure.
    ▼
  MEASURE_POSE
    │  Robot is stationary + facing marker.
    │  Detector collects 15 solvePnP samples → median → map-frame pose.
    │  Transition: /aruco_found fires.
    ▼
  SAVE_AND_LABEL
    │  Save SLAM map to disk (map_saver_cli).
    │  Publish orange RViz cube marker at ArUco map position.
    ▼
  RETURN_TO_DOCK  →  GOTO_ARUCO  →  DONE
──────────────────────────────────────────────────────────────────────
"""

import math
import os
import subprocess
from enum import Enum, auto
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Bool, String, Float32
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import Marker

import tf2_ros


# ── Map save location ─────────────────────────────────────────────────────────
MAP_SAVE_DIR  = os.path.expanduser('~/turtlebot4_ws/src/lab8/maps')
MAP_SAVE_NAME = 'maze_map'

# TF frames
MAP_FRAME = 'map'

# RViz marker appearance
MARKER_SCALE   = 0.15
MARKER_COLOR_R = 1.0
MARKER_COLOR_G = 0.5
MARKER_COLOR_B = 0.0    # orange

# ── Centering P-controller ────────────────────────────────────────────────────
# pixel_offset is normalised -1..+1 (from aruco_detector).
# angular_z = -Kp_center * pixel_offset
#   offset > 0 (marker RIGHT of centre) → negative ω → turn right (CW)
#   offset < 0 (marker LEFT  of centre) → positive ω → turn left  (CCW)
Kp_CENTER        = 0.6    # proportional gain for centering rotation
CENTER_THRESHOLD = 0.05   # |offset| below this → centred (±5% of half-width ≈ ±8 px on 320)
CENTER_OMEGA_MAX = 0.4    # cap angular speed during centering (rad/s)
CENTER_OMEGA_MIN = 0.08   # minimum angular speed to overcome static friction (rad/s)

# How long (seconds) to wait with no offset update before assuming marker lost
CENTER_TIMEOUT_S = 5.0

# ─────────────────────────────────────────────────────────────────────────────


class State(Enum):
    INIT            = auto()   # polling TF until SLAM is ready
    EXPLORE         = auto()   # wall-following + SLAM
    STOP_AND_CENTER = auto()   # stop robot, rotate to centre marker in frame
    MEASURE_POSE    = auto()   # stationary, detector averaging 15 samples
    SAVE_AND_LABEL  = auto()   # save map + RViz marker
    RETURN_TO_DOCK  = auto()   # navigate back to dock
    GOTO_ARUCO      = auto()   # navigate from dock to ArUco
    DONE            = auto()
    FAILED          = auto()


class Orchestrator(Node):

    def __init__(self):
        super().__init__('orchestrator')

        # ── Mission timing ────────────────────────────────────────────────────
        self.mission_start_time = None

        # ── Saved poses ───────────────────────────────────────────────────────
        self.initial_pose: PoseStamped = None
        self.aruco_pose:   PoseStamped = None

        # ── TF2 ──────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Centering state ────────────────────────────────────────────────────
        self._pixel_offset        = None
        self._last_offset_time    = None
        self._centering_timer     = None

        # ── RViz marker (set once in _publish_aruco_marker) ───────────────────
        self._rviz_marker     = None
        self._save_label_done = False   # one-shot guard for save callback

        # ── Publishers ────────────────────────────────────────────────────────
        self.wall_follower_cmd_pub = self.create_publisher(
            String, '/wall_follower/cmd', 10)
        self.navigator_cmd_pub = self.create_publisher(
            String, '/navigator/cmd', 10)
        self.aruco_detector_cmd_pub = self.create_publisher(
            String, '/aruco_detector/cmd', 10)
        # Direct cmd_vel for in-place rotation during centering
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(
            Marker, '/visualization_marker', 10)
        # Publish start pose so navigator can use it for goto_start
        self.start_pose_pub = self.create_publisher(
            PoseStamped, '/start_map_pose', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        # Phase A complete: marker confirmed, stop wall-follower
        self.aruco_confirmed_sub = self.create_subscription(
            Bool, '/aruco_confirmed', self._aruco_confirmed_cb, 10)
        # Phase B: pixel offset for centering P-controller
        self.pixel_offset_sub = self.create_subscription(
            Float32, '/aruco_pixel_offset', self._pixel_offset_cb, 10)
        # Phase C complete: averaged pose ready
        self.aruco_found_sub = self.create_subscription(
            Bool, '/aruco_found', self._aruco_found_cb, 10)
        self.aruco_pose_sub = self.create_subscription(
            PoseStamped, '/aruco_map_pose', self._aruco_pose_cb, 10)
        self.nav_status_sub = self.create_subscription(
            String, '/navigator/status', self._nav_status_cb, 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.state = State.INIT

        self.get_logger().info('Orchestrator started. Waiting for SLAM map frame...')
        self._init_timer = self.create_timer(1.0, self._poll_for_initial_pose)
        # Single persistent timer — republishes RViz marker at 2 Hz once set.
        # This survives state transitions and never creates duplicate timers.
        self.create_timer(0.5, self._republish_marker)

    # ── SLAM warm-up: poll until map→base TF is available ─────────────────────
    def _poll_for_initial_pose(self):
        """
        Fires at 1 Hz. Attempts to look up the robot's pose in the map frame.
        Cancels itself and advances to EXPLORE once the pose is available.

        TurtleBot4 publishes base_footprint, not base_link, as the robot
        body frame — we try both so the code works regardless of config.
        """
        pose = self._lookup_robot_pose()
        if pose is None:
            self.get_logger().warn(
                'map → base TF not yet available. Retrying...', throttle_duration_sec=3.0)
            return   # timer keeps firing — no new timer created

        # Cancel the polling timer — no more retries needed
        self._init_timer.cancel()

        self.initial_pose = pose
        self.mission_start_time = time.time()

        self.get_logger().info(
            f'Start pose recorded: '
            f'x={pose.pose.position.x:.3f} '
            f'y={pose.pose.position.y:.3f} '
            f'(map frame)')

        # Publish start pose immediately and keep re-publishing at 1 Hz
        # so the navigator node (which may start slightly later) always
        # receives it regardless of startup timing.
        self.start_pose_pub.publish(self.initial_pose)
        self.create_timer(1.0, lambda: self.start_pose_pub.publish(self.initial_pose))

        self._transition_to(State.EXPLORE)

    # ── State machine ─────────────────────────────────────────────────────────
    def _transition_to(self, new_state: State):
        self.get_logger().info(
            f'─── State: {self.state.name} → {new_state.name} ───')
        self.state = new_state

        if   new_state == State.EXPLORE:          self._enter_explore()
        elif new_state == State.STOP_AND_CENTER:  self._enter_stop_and_center()
        elif new_state == State.MEASURE_POSE:     self._enter_measure_pose()
        elif new_state == State.SAVE_AND_LABEL:   self._enter_save_and_label()
        elif new_state == State.RETURN_TO_DOCK:   self._enter_return_to_dock()
        elif new_state == State.GOTO_ARUCO:       self._enter_goto_aruco()
        elif new_state == State.DONE:             self._enter_done()
        elif new_state == State.FAILED:           self._enter_failed()

    # ── EXPLORE ───────────────────────────────────────────────────────────────
    def _enter_explore(self):
        self.get_logger().info('[EXPLORE] Wall-follower started. SLAM building map.')
        self._send_wall_follower_cmd('start')

    # ── STOP_AND_CENTER ───────────────────────────────────────────────────────
    def _enter_stop_and_center(self):
        """
        Stop the wall-follower immediately, then spin the robot in-place
        using a P-controller on the normalised pixel offset published by
        aruco_detector until the marker is centred.

        Control law:
          angular_z = clip(-Kp * pixel_offset, -MAX, -MIN or +MIN, +MAX)

        The minimum angular speed prevents the robot stalling due to
        static friction when the offset is very small but non-zero.
        """
        self.get_logger().info(
            '[STOP_AND_CENTER] Stopping wall-follower. Starting centering.')
        self._send_wall_follower_cmd('stop')

        # Reset pixel offset so we don't act on a stale value
        self._pixel_offset     = None
        self._last_offset_time = None

        # 20 Hz centering loop
        self._centering_timer = self.create_timer(0.05, self._centering_loop)

    def _centering_loop(self):
        """
        Runs at 20 Hz during STOP_AND_CENTER.
        Reads the latest pixel_offset and publishes a cmd_vel rotation.
        """
        if self.state != State.STOP_AND_CENTER:
            self._centering_timer.cancel()
            return

        now = time.time()

        # Safety: if we haven't received an offset for CENTER_TIMEOUT_S seconds,
        # the marker may have drifted out of frame during our stop manoeuvre.
        if (self._last_offset_time is not None and
                now - self._last_offset_time > CENTER_TIMEOUT_S):
            self.get_logger().warn(
                '[CENTER] Pixel offset timeout — marker lost. Aborting centering.')
            self._stop_motors()
            self._transition_to(State.FAILED)
            return

        if self._pixel_offset is None:
            # Waiting for first offset message — hold still
            return

        offset = self._pixel_offset

        # Check if centred
        if abs(offset) < CENTER_THRESHOLD:
            self.get_logger().info(
                f'[CENTER] Centred! offset={offset:+.4f} < {CENTER_THRESHOLD}. '
                f'Stopping rotation.')
            self._stop_motors()
            self._centering_timer.cancel()
            self._transition_to(State.MEASURE_POSE)
            return

        # P-controller: turn toward the marker
        omega = -Kp_CENTER * offset

        # Apply minimum speed (overcome static friction) while preserving sign
        if 0 < abs(omega) < CENTER_OMEGA_MIN:
            omega = math.copysign(CENTER_OMEGA_MIN, omega)

        # Clamp to maximum
        omega = max(-CENTER_OMEGA_MAX, min(CENTER_OMEGA_MAX, omega))

        twist = Twist()
        twist.angular.z = omega
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(
            f'[CENTER] offset={offset:+.4f}  ω={omega:+.3f} rad/s',
            throttle_duration_sec=0.2)

    def _pixel_offset_cb(self, msg: Float32):
        """Receive normalised pixel offset from aruco_detector during centering."""
        self._pixel_offset     = msg.data
        self._last_offset_time = time.time()

    # ── MEASURE_POSE ──────────────────────────────────────────────────────────
    def _enter_measure_pose(self):
        """
        Robot is now stationary and facing the marker.
        Tell the detector to start its averaged measurement phase.
        Also stop SLAM at this point — the map is as good as it will get,
        and stopping prevents SLAM from updating based on the stationary view.
        """
        self.get_logger().info(
            '[MEASURE_POSE] Robot centred on marker. Starting pose measurement.')

        # Tell aruco_detector to begin Phase C (averaging 15 frames)
        self._send_aruco_cmd('start_measuring')

        # Stop SLAM map updates — send map_saver_cli as a placeholder;
        # in practice slam_toolbox responds to a /slam_toolbox/pause_new_scans
        # service, but saving is simpler and gives us the same map file.
        # We will save properly in SAVE_AND_LABEL after pose is measured.
        self.get_logger().info('[MEASURE_POSE] Waiting for detector to lock pose...')

    # ── SAVE_AND_LABEL ────────────────────────────────────────────────────────
    def _enter_save_and_label(self):
        """
        Pose is confirmed. Save the SLAM map and label the ArUco on the map.
        Uses a one-shot timer (cancelled after first fire) to avoid the
        repeating-timer bug that would cause multiple RETURN_TO_DOCK transitions.
        """
        self.get_logger().info('[SAVE_AND_LABEL] Saving map and labelling ArUco.')
        self._send_aruco_cmd('shutdown')   
        self._save_label_done = False   # guard: only execute once
        self._save_label_timer = self.create_timer(1.0, self._save_map_and_label)

    def _save_map_and_label(self):
        """Save the SLAM map and publish the ArUco RViz marker. Fires once."""
        # Cancel immediately — this must be a one-shot, not repeating
        self._save_label_timer.cancel()

        if self._save_label_done:
            return
        self._save_label_done = True

        # ── Save SLAM map via map_saver_cli ───────────────────────────────────
        os.makedirs(MAP_SAVE_DIR, exist_ok=True)
        map_path = os.path.join(MAP_SAVE_DIR, MAP_SAVE_NAME)
        cmd = ['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
               '-f', map_path, '--ros-args', '-p', 'save_map_timeout:=5.0']
        try:
            subprocess.Popen(cmd)
            self.get_logger().info(f'Map saving to: {map_path}.pgm / .yaml')
        except Exception as e:
            self.get_logger().error(f'Map save failed: {e}')

        # ── Publish RViz marker at ArUco map position ─────────────────────────
        if self.aruco_pose is not None:
            self._publish_aruco_marker(self.aruco_pose)
            self.get_logger().info(
                f'ArUco labelled on map at: '
                f'x={self.aruco_pose.pose.position.x:.3f} '
                f'y={self.aruco_pose.pose.position.y:.3f}')
        else:
            self.get_logger().warn('ArUco pose not received — marker not published.')

        # ── Transition (only once) ────────────────────────────────────────────
        self._transition_to(State.RETURN_TO_DOCK)

    def _publish_aruco_marker(self, pose: PoseStamped):
        """
        Publish a coloured cube marker in RViz at the ArUco map-frame position.
        Stores the marker on self so the 2 Hz republish timer always sends
        the latest version (started once in __init__).
        """
        self._rviz_marker = Marker()
        self._rviz_marker.header.frame_id = MAP_FRAME
        self._rviz_marker.header.stamp    = self.get_clock().now().to_msg()
        self._rviz_marker.ns              = 'aruco_target'
        self._rviz_marker.id              = 0
        self._rviz_marker.type            = Marker.CUBE
        self._rviz_marker.action          = Marker.ADD

        self._rviz_marker.pose            = pose.pose
        self._rviz_marker.pose.position.z = MARKER_SCALE / 2.0

        self._rviz_marker.scale.x = MARKER_SCALE
        self._rviz_marker.scale.y = MARKER_SCALE
        self._rviz_marker.scale.z = MARKER_SCALE

        self._rviz_marker.color.r = MARKER_COLOR_R
        self._rviz_marker.color.g = MARKER_COLOR_G
        self._rviz_marker.color.b = MARKER_COLOR_B
        self._rviz_marker.color.a = 0.85

        self._rviz_marker.lifetime.sec = 0   # never expire

        self.marker_pub.publish(self._rviz_marker)

    # ── RETURN_TO_DOCK ────────────────────────────────────────────────────────
    def _enter_return_to_dock(self):
        """
        Navigate back to the start position using Nav2.
        Small delay before sending the command gives Nav2 time to finish
        any costmap updates after the robot stopped moving.
        """
        self.get_logger().info(
            '[RETURN_TO_DOCK] Navigating back to start position via Nav2...')
        # 2 s settle: lets Nav2 rebuild local costmap after robot stopped,
        # and ensures navigator has received /start_map_pose before the cmd.
        self._return_timer = self.create_timer(2.0, self._send_goto_start)

    def _send_goto_start(self):
        self._return_timer.cancel()
        self._send_navigator_cmd('goto_start')

    # ── GOTO_ARUCO ────────────────────────────────────────────────────────────
    def _enter_goto_aruco(self):
        self.get_logger().info(
            '[GOTO_ARUCO] Navigating to ArUco cube...')
        self._send_navigator_cmd('goto_aruco')

    # ── DONE ──────────────────────────────────────────────────────────────────
    def _enter_done(self):
        elapsed = time.time() - self.mission_start_time
        self.get_logger().info(
            f'╔══════════════════════════════╗\n'
            f'║   MISSION COMPLETE ✓          ║\n'
            f'║   Total time: {elapsed:>6.1f} s       ║\n'
            f'╚══════════════════════════════╝')

    # ── FAILED ────────────────────────────────────────────────────────────────
    def _enter_failed(self):
        elapsed = time.time() - (self.mission_start_time or time.time())
        self.get_logger().error(
            f'[FAILED] Mission failed after {elapsed:.1f} s.')
        self._send_wall_follower_cmd('stop')

    # ── Topic callbacks ───────────────────────────────────────────────────────
    def _aruco_confirmed_cb(self, msg: Bool):
        """
        Phase A complete: detector has confirmed the marker.
        Stop wall-follower and begin centering.
        Only fires once (checked by state guard).
        """
        if msg.data and self.state == State.EXPLORE:
            elapsed = time.time() - self.mission_start_time
            self.get_logger().info(
                f'[EXPLORE] ArUco confirmed at t={elapsed:.1f} s. '
                f'Transitioning to STOP_AND_CENTER.')
            self._transition_to(State.STOP_AND_CENTER)

    def _aruco_found_cb(self, msg: Bool):
        """
        Phase C complete: detector has locked the averaged map-frame pose.
        Proceed to save the map and label it.
        """
        if msg.data and self.state == State.MEASURE_POSE:
            self.get_logger().info(
                '[MEASURE_POSE] Pose measurement complete. Saving map.')
            self._transition_to(State.SAVE_AND_LABEL)

    def _aruco_pose_cb(self, msg: PoseStamped):
        """Save the ArUco map-frame pose (arrives with /aruco_found)."""
        if self.aruco_pose is None:
            self.aruco_pose = msg
            self.get_logger().info(
                f'ArUco map pose saved: '
                f'x={msg.pose.position.x:.4f} '
                f'y={msg.pose.position.y:.4f}')

    def _nav_status_cb(self, msg: String):
        """Navigator reports completion → advance state machine."""
        status = msg.data

        if status == 'at_start' and self.state == State.RETURN_TO_DOCK:
            self.get_logger().info('[RETURN_TO_DOCK] Reached start position.')
            self._transition_to(State.GOTO_ARUCO)

        elif status == 'at_goal' and self.state == State.GOTO_ARUCO:
            self.get_logger().info('[GOTO_ARUCO] Reached ArUco position.')
            self._transition_to(State.DONE)

        elif status == 'failed':
            self.get_logger().error(
                f'Navigator failed in state {self.state.name}.')
            self._transition_to(State.FAILED)

    # ── TF2 robot pose lookup ─────────────────────────────────────────────────
    def _lookup_robot_pose(self) -> PoseStamped:
        """
        Look up the robot's current pose in the map frame.
        Returns a PoseStamped, or None if TF is not yet available.

        We use map frame (not odom) because:
          - map is SLAM-corrected and globally consistent
          - odom drifts over time due to wheel slip

        TurtleBot4 uses base_footprint as the robot body frame.
        We try base_footprint first, then fall back to base_link.
        """
        for frame in ('base_footprint', 'base_link'):
            try:
                transform = self.tf_buffer.lookup_transform(
                    MAP_FRAME, frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5)
                )
                # Log which frame worked (only the first time)
                if not hasattr(self, '_base_frame_confirmed'):
                    self._base_frame_confirmed = frame
                    self.get_logger().info(f'Using robot body frame: "{frame}"')

                pose = PoseStamped()
                pose.header.frame_id = MAP_FRAME
                pose.header.stamp    = self.get_clock().now().to_msg()
                pose.pose.position.x = transform.transform.translation.x
                pose.pose.position.y = transform.transform.translation.y
                pose.pose.position.z = transform.transform.translation.z
                pose.pose.orientation = transform.transform.rotation
                return pose

            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue   # try next frame name

        return None   # neither frame available yet

    # ── RViz marker republish ─────────────────────────────────────────────────
    def _republish_marker(self):
        """Republish the ArUco RViz marker at 2 Hz so RViz never loses it."""
        if self._rviz_marker is not None:
            # Refresh stamp so RViz doesn't warn about stale markers
            self._rviz_marker.header.stamp = self.get_clock().now().to_msg()
            self.marker_pub.publish(self._rviz_marker)

    # ── Command helpers ───────────────────────────────────────────────────────
    def _send_wall_follower_cmd(self, cmd: str):
        msg = String(); msg.data = cmd
        self.wall_follower_cmd_pub.publish(msg)

    def _send_navigator_cmd(self, cmd: str):
        msg = String(); msg.data = cmd
        self.navigator_cmd_pub.publish(msg)

    def _send_aruco_cmd(self, cmd: str):
        msg = String(); msg.data = cmd
        self.aruco_detector_cmd_pub.publish(msg)

    def _stop_motors(self):
        """Publish zero velocity immediately."""
        self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = Orchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()