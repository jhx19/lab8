"""
aruco_detector.py
-----------------
Phase 1-B: ArUco detection with pixel-centering + averaged pose measurement.

Detection pipeline (3 phases):
─────────────────────────────────────────────────────────────────────
  Phase A — SCANNING (normal wall-following is running)
    Continuously run ArUco detector on every frame.
    Apply false-positive suppression:
      - Must see same marker ID for CONFIRM_FRAMES_REQUIRED consecutive frames
      - Computed distance must be within [MIN_DIST, MAX_DIST]
    Once confirmed → publish /aruco_confirmed (Bool) so orchestrator
    can stop the wall-follower and start centering.

  Phase B — CENTERING (orchestrator rotates robot)
    Continuously publish /aruco_pixel_offset (Float32):
      value = (marker_center_x - image_center_x) / image_half_width
      range: -1.0 (marker fully left) … +1.0 (marker fully right)
      target: 0.0 (marker is centered)
    Orchestrator uses this to drive a P-controller on angular_z.
    When orchestrator signals "centered", move to Phase C.

  Phase C — MEASURING (robot is stationary and facing marker)
    Collect MEASURE_FRAMES_REQUIRED solvePnP tvec samples.
    Average them (median per axis for outlier robustness).
    Transform averaged pose → map frame via TF2.
    Publish /aruco_map_pose (PoseStamped) and /aruco_found (Bool).
    Orchestrator can now proceed to save map and navigate.
─────────────────────────────────────────────────────────────────────

Why centering improves accuracy:
  solvePnP solves for pose from 2D corner pixels. When the marker is
  off-centre, there is a lever-arm angular error: the translation vector
  tx, ty carry cross-axis contamination. When the robot turns to face
  the marker directly, tx ≈ ty ≈ 0 and tz = true forward distance.
  This makes the map-frame projection far more accurate.

Why averaging improves accuracy:
  JPEG compression introduces pixel-level noise on corner locations.
  Averaging 15 independent solvePnP solutions reduces random error
  by ~√15 ≈ 4×. Median is used instead of mean to reject outlier
  frames (e.g. a motion blur frame during the last micro-movement).

Published topics:
  /aruco_confirmed    (std_msgs/Bool)      Phase A complete
  /aruco_pixel_offset (std_msgs/Float32)   Phase B: centering error (-1..+1)
  /aruco_found        (std_msgs/Bool)      Phase C complete, pose is ready
  /aruco_map_pose     (geometry_msgs/PoseStamped)  final map-frame pose
  /aruco_debug_image  (sensor_msgs/CompressedImage) annotated debug feed

Subscribed commands from orchestrator (/aruco_detector/cmd, String):
  "start_measuring"  → begin Phase C (robot is now centred and still)
  "reset"            → clear all state, return to Phase A
"""

import math

import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, String

import tf2_ros
import tf2_geometry_msgs


# ── Parameters ────────────────────────────────────────────────────────────────

MARKER_SIZE_M    = 0.05             # physical side length of marker (m) — confirmed 5 cm
ARUCO_DICT_TYPE  = aruco.DICT_4X4_50
TARGET_MARKER_ID = None             # None = any; set to int to filter a specific ID

IMAGE_TOPIC       = '/oakd/right/image_raw/compressed'
CAMERA_INFO_TOPIC = '/oakd/right/camera_info'
MAP_FRAME         = 'map'

# ── Phase A: false-positive suppression ──────────────────────────────────────
CONFIRM_FRAMES_REQUIRED = 5         # consecutive frames before confirming
MIN_DETECTION_DIST      = 0.10      # reject if closer than this (m)
MAX_DETECTION_DIST      = 3.00      # reject if farther than this (m)

# ── Phase B: centering ────────────────────────────────────────────────────────
# IMAGE_WIDTH must match the actual camera resolution (confirmed: 320 px)
IMAGE_WIDTH = 320

# ── Phase C: pose measurement ─────────────────────────────────────────────────
MEASURE_FRAMES_REQUIRED = 5        # frames to average for final pose

# Forward distance from base_footprint to the OAK-D camera lens (meters).
# This is added to tvec[2] so the marker position accounts for the camera
# not being at the robot center. Measure from robot center to camera face.
# TurtleBot4 with OAK-D cradle: approx 0.10 m. Set 0.0 if unsure.
CAMERA_FORWARD_OFFSET_M = 0.10

# ── Debug image ───────────────────────────────────────────────────────────────
PUBLISH_DEBUG_IMAGE = True
DEBUG_IMAGE_TOPIC   = '/aruco_debug_image'

# ─────────────────────────────────────────────────────────────────────────────


class DetectionPhase:
    SCANNING   = 'SCANNING'     # Phase A: looking for marker
    CENTERING  = 'CENTERING'    # Phase B: publishing pixel offset for centering
    MEASURING  = 'MEASURING'    # Phase C: collecting averaged pose samples
    DONE       = 'DONE'         # pose locked, stop processing


class ArucoDetector(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        # ── ArUco ─────────────────────────────────────────────────────────────
        self.detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(ARUCO_DICT_TYPE),
            aruco.DetectorParameters()
        )

        # ── Camera intrinsics ─────────────────────────────────────────────────
        self.camera_matrix = None
        self.dist_coeffs   = None
        self.camera_frame  = None
        self.image_width   = IMAGE_WIDTH   # updated from first frame if available

        # ── TF2 ──────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Internal state ────────────────────────────────────────────────────
        self.phase = DetectionPhase.SCANNING

        # Phase A state
        self._confirm_id    = None
        self._confirm_count = 0

        # Phase C state: list of (tvec, rvec) tuples collected while measuring
        self._measure_samples = []

        # Final result
        self.aruco_map_pose = None

        # ── Publishers ────────────────────────────────────────────────────────
        self.confirmed_pub    = self.create_publisher(Bool,        '/aruco_confirmed',    10)
        self.pixel_offset_pub = self.create_publisher(Float32,     '/aruco_pixel_offset', 10)
        self.found_pub        = self.create_publisher(Bool,        '/aruco_found',        10)
        self.pose_pub         = self.create_publisher(PoseStamped, '/aruco_map_pose',     10)

        self.debug_pub = (
            self.create_publisher(CompressedImage, DEBUG_IMAGE_TOPIC, 10)
            if PUBLISH_DEBUG_IMAGE else None
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.info_sub = self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._camera_info_cb, 1)
        self.image_sub = self.create_subscription(
            CompressedImage, IMAGE_TOPIC, self._image_cb, 10)
        self.cmd_sub = self.create_subscription(
            String, '/aruco_detector/cmd', self._cmd_cb, 10)

        # Re-publish final pose at 1 Hz once locked
        self.create_timer(1.0, self._republish_timer)

        self.get_logger().info(
            f'ArucoDetector ready.  Phase: {self.phase}  '
            f'Confirm={CONFIRM_FRAMES_REQUIRED}  Measure={MEASURE_FRAMES_REQUIRED}  '
            f'Dist=[{MIN_DETECTION_DIST},{MAX_DETECTION_DIST}] m')

    # ── Camera info ───────────────────────────────────────────────────────────
    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_matrix is not None:
            return
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs   = np.array(msg.d, dtype=np.float64)
        if msg.width > 0:
            self.image_width = msg.width
        self.get_logger().info(
            f'Camera intrinsics loaded.  '
            f'Resolution={self.image_width}×{msg.height}  '
            f'D-coeffs={len(self.dist_coeffs)}')

    # ── Orchestrator commands ─────────────────────────────────────────────────
    def _cmd_cb(self, msg: String):
        cmd = msg.data.strip()

        if cmd == 'start_measuring':
            # Orchestrator says robot is now centred and still — begin Phase C
            if self.phase == DetectionPhase.CENTERING:
                self.get_logger().info(
                    'Command: start_measuring → entering MEASURING phase.')
                self._measure_samples = []
                self.phase = DetectionPhase.MEASURING
            else:
                self.get_logger().warn(
                    f'start_measuring ignored: current phase is {self.phase}')

        elif cmd == 'reset':
            self.get_logger().info('Command: reset → returning to SCANNING.')
            self._reset()
        elif cmd == 'shutdown':
            self.get_logger().info('Command: shutdown → destroying node.')
            self.destroy_node()
            rclpy.shutdown()

    # ── Main image callback ───────────────────────────────────────────────────
    def _image_cb(self, msg: CompressedImage):
        if self.camera_matrix is None:
            return
        if self.phase == DetectionPhase.DONE:
            return   # pose already locked, nothing to do

        # Grab camera frame id once from image header (camera_info has empty frame_id)
        if self.camera_frame is None:
            self.camera_frame = msg.header.frame_id
            self.get_logger().info(f'Camera frame: "{self.camera_frame}"')

        # Decode JPEG → grayscale
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        gray   = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return

        # Actual image width from the decoded frame (most reliable source)
        h, w = gray.shape[:2]

        # Build debug canvas (BGR for colour overlays)
        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if self.debug_pub else None

        # ── Detect ────────────────────────────────────────────────────────────
        corners, ids, _ = self.detector.detectMarkers(gray)

        # Find first valid marker (matching TARGET_MARKER_ID if set)
        detection = self._best_detection(corners, ids, w)

        # ── Route to correct phase handler ────────────────────────────────────
        if self.phase == DetectionPhase.SCANNING:
            self._handle_scanning(detection, msg, debug, w)
        elif self.phase == DetectionPhase.CENTERING:
            self._handle_centering(detection, debug, w)
        elif self.phase == DetectionPhase.MEASURING:
            self._handle_measuring(detection, msg, debug)

        # Publish annotated debug image
        if debug is not None:
            # Phase label in top-left corner
            cv2.putText(debug, self.phase, (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
            self._publish_debug(debug, msg.header)

    # ── Phase A: SCANNING ─────────────────────────────────────────────────────
    def _handle_scanning(self, detection, msg, debug, image_w):
        """
        Accumulate consecutive detections of the same marker.
        Once CONFIRM_FRAMES_REQUIRED is reached, transition to CENTERING.
        """
        if detection is None:
            # Lost the marker — reset counter
            if self._confirm_count > 0:
                self.get_logger().info(
                    f'ID {self._confirm_id} lost — reset confirmation '
                    f'({self._confirm_count} → 0).')
            self._confirm_id    = None
            self._confirm_count = 0
            return

        marker_id, corners_i, tvec, rvec, dist = detection

        # Distance guard
        if not (MIN_DETECTION_DIST < dist < MAX_DETECTION_DIST):
            self.get_logger().warn(
                f'ID {marker_id} dist={dist:.2f} m outside valid range — ignored.',
                throttle_duration_sec=2.0)
            return

        # Confirmation counting
        if marker_id != self._confirm_id:
            self._confirm_id    = marker_id
            self._confirm_count = 1
            self.get_logger().info(
                f'New candidate: ID {marker_id}  dist={dist:.2f} m  '
                f'need {CONFIRM_FRAMES_REQUIRED} consecutive frames.')
        else:
            self._confirm_count += 1
            self.get_logger().info(
                f'Confirming ID {marker_id}: '
                f'{self._confirm_count}/{CONFIRM_FRAMES_REQUIRED}  '
                f'dist={dist:.2f} m')

        # Annotate debug
        if debug is not None:
            cv2.aruco.drawDetectedMarkers(debug, [corners_i])
            cx, cy = _marker_centre(corners_i)
            cv2.putText(debug,
                        f'ID:{marker_id} {dist:.2f}m '
                        f'[{self._confirm_count}/{CONFIRM_FRAMES_REQUIRED}]',
                        (cx - 55, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Confirmed — transition to CENTERING
        if self._confirm_count >= CONFIRM_FRAMES_REQUIRED:
            self.get_logger().info(
                f'ID {marker_id} CONFIRMED. Transitioning to CENTERING.')
            self.phase = DetectionPhase.CENTERING
            # Tell orchestrator immediately so it can stop the wall-follower
            confirmed_msg      = Bool()
            confirmed_msg.data = True
            self.confirmed_pub.publish(confirmed_msg)

    # ── Phase B: CENTERING ────────────────────────────────────────────────────
    def _handle_centering(self, detection, debug, image_w):
        """
        Publish normalised pixel offset of the marker centre from the image
        centre. Orchestrator uses this to rotate the robot until offset ≈ 0.

        offset = (marker_cx - image_cx) / (image_w / 2)
          -1.0  → marker is at the far LEFT  → robot must turn LEFT  (+ω)
          +1.0  → marker is at the far RIGHT → robot must turn RIGHT (-ω)
           0.0  → marker is centred          → stop rotating

        Why normalise by half-width?
          Normalising makes the value camera-resolution-independent and
          gives the orchestrator's P-gain a consistent meaning regardless
          of whether we switch cameras in the future.
        """
        if detection is None:
            # Marker temporarily out of view during rotation — publish
            # last known direction to keep rotating the right way.
            self.get_logger().warn(
                'Marker lost during centering.', throttle_duration_sec=1.0)
            return

        marker_id, corners_i, tvec, rvec, dist = detection
        cx, cy = _marker_centre(corners_i)

        image_cx    = image_w / 2.0
        half_width  = image_w / 2.0
        pixel_offset = (cx - image_cx) / half_width   # normalised, -1..+1

        offset_msg       = Float32()
        offset_msg.data  = float(pixel_offset)
        self.pixel_offset_pub.publish(offset_msg)

        self.get_logger().info(
            f'[CENTERING] marker_cx={cx:.1f}  '
            f'offset={pixel_offset:+.3f}  dist={dist:.2f} m',
            throttle_duration_sec=0.3)

        # Annotate debug image
        if debug is not None:
            cv2.aruco.drawDetectedMarkers(debug, [corners_i])
            # Draw vertical line at image centre and marker centre
            cv2.line(debug, (int(image_cx), 0), (int(image_cx), debug.shape[0]),
                     (0, 100, 255), 1)
            cv2.line(debug, (int(cx), 0), (int(cx), debug.shape[0]),
                     (0, 255, 100), 1)
            cv2.putText(debug, f'offset={pixel_offset:+.3f}',
                        (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

    # ── Phase C: MEASURING ────────────────────────────────────────────────────
    def _handle_measuring(self, detection, msg, debug):
        """
        Compute the marker's map-frame position WITHOUT relying on the
        camera TF chain.

        Why bypass the camera TF chain?
        ─────────────────────────────────────────────────────────────────
        The previous approach used:
            tvec (camera frame)  →  TF(camera→base)  →  TF(base→map)

        TF(camera→base) is a static transform that depends on how the
        OAK-D is physically mounted and calibrated. If this transform is
        even slightly off — wrong translation or rotation — the projected
        marker position ends up nowhere near the real location.

        After centering, the robot is pointing directly at the marker.
        We can use a much simpler, more reliable calculation:

            marker_x = robot_x + forward_dist * cos(robot_yaw)
            marker_y = robot_y + forward_dist * sin(robot_yaw)

        where:
            robot_x, robot_y, robot_yaw  ← TF(map → base_footprint)
            forward_dist                 ← tvec[2]  (camera Z = forward)
                                           + CAMERA_FORWARD_OFFSET

        This uses ONLY the well-calibrated base_footprint TF (which SLAM
        maintains), and the clean solvePnP forward distance. No camera
        mounting offsets involved.

        CAMERA_FORWARD_OFFSET accounts for the camera being mounted ahead
        of base_footprint (approx 0.10 m on TurtleBot4 with OAK-D cradle).
        Set to 0.0 if unsure — it's a small correction.
        ─────────────────────────────────────────────────────────────────
        """
        if detection is None:
            self.get_logger().warn(
                'Marker not visible during measurement — waiting...',
                throttle_duration_sec=1.0)
            return

        marker_id, corners_i, tvec, rvec, dist = detection
        tx, ty, tz = tvec   # camera frame: x=right, y=down, z=forward

        if debug is not None:
            cv2.aruco.drawDetectedMarkers(debug, [corners_i])
            cv2.putText(debug,
                        f'Measuring {len(self._measure_samples)+1}/{MEASURE_FRAMES_REQUIRED}',
                        (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(debug, f'dist={tz:.3f}m', (4, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # ── Get robot pose in map frame ───────────────────────────────────────
        robot_pose = self._lookup_robot_pose_in_map()
        if robot_pose is None:
            self.get_logger().warn(
                'Cannot get robot pose in map — skipping sample.',
                throttle_duration_sec=1.0)
            return

        robot_x, robot_y, robot_yaw = robot_pose

        # ── Compute marker position in map frame ──────────────────────────────
        # After centering, the robot is facing the marker.
        # tz (camera Z) = forward distance from camera to marker.
        # Add the camera's forward offset from base_footprint.
        forward_dist = tz + CAMERA_FORWARD_OFFSET_M

        marker_x = robot_x + forward_dist * math.cos(robot_yaw)
        marker_y = robot_y + forward_dist * math.sin(robot_yaw)

        self.get_logger().info(
            f'[MEASURING] sample {len(self._measure_samples)+1}/{MEASURE_FRAMES_REQUIRED}  '
            f'tz={tz:.3f} m  robot=({robot_x:.3f},{robot_y:.3f},{math.degrees(robot_yaw):.1f}°)  '
            f'→ marker=({marker_x:.4f},{marker_y:.4f})')

        self._measure_samples.append((marker_x, marker_y))

        n = len(self._measure_samples)
        if n < MEASURE_FRAMES_REQUIRED:
            return

        # ── All samples collected — median x, y ───────────────────────────────
        xs = np.array([s[0] for s in self._measure_samples])
        ys = np.array([s[1] for s in self._measure_samples])

        x_med = float(np.median(xs))
        y_med = float(np.median(ys))

        self.get_logger().info(
            f'Median over {n} samples: x={x_med:.4f}  y={y_med:.4f}')
        self.get_logger().info(
            f'  x spread [{xs.min():.4f}, {xs.max():.4f}]  '
            f'  y spread [{ys.min():.4f}, {ys.max():.4f}]')

        # ── Build final locked pose ───────────────────────────────────────────
        final_pose = PoseStamped()
        final_pose.header.frame_id = MAP_FRAME
        final_pose.header.stamp    = self.get_clock().now().to_msg()
        final_pose.pose.position.x = x_med
        final_pose.pose.position.y = y_med
        final_pose.pose.position.z = 0.0
        # Orientation: robot yaw (marker is facing toward robot)
        final_pose.pose.orientation.x = 0.0
        final_pose.pose.orientation.y = 0.0
        final_pose.pose.orientation.z = math.sin(robot_yaw / 2.0)
        final_pose.pose.orientation.w = math.cos(robot_yaw / 2.0)

        # ── Lock and publish ──────────────────────────────────────────────────
        self.aruco_map_pose = final_pose
        self.phase          = DetectionPhase.DONE

        self.get_logger().info(
            f'ArUco LOCKED in map: x={x_med:.4f}  y={y_med:.4f}  '
            f'yaw={math.degrees(robot_yaw):.1f}°')

        self.pose_pub.publish(self.aruco_map_pose)
        found_msg      = Bool()
        found_msg.data = True
        self.found_pub.publish(found_msg)

    # ── Robot pose in map frame ───────────────────────────────────────────────
    def _lookup_robot_pose_in_map(self):
        """
        Look up map → base_footprint (or base_link as fallback).
        Returns (x, y, yaw) as Python floats, or None if TF unavailable.

        This is the only TF lookup we need for position calculation.
        It uses the well-maintained SLAM TF and avoids any camera mounting
        calibration issues.
        """
        for base_frame in ('base_link', 'base_footprint'):
            try:
                tf = self.tf_buffer.lookup_transform(
                    MAP_FRAME, base_frame,
                    rclpy.time.Time(seconds=0, nanoseconds=0),  # latest, ignore clock mismatch
                    timeout=Duration(seconds=1.0)
                )
                x = tf.transform.translation.x
                y = tf.transform.translation.y
                q = tf.transform.rotation
                # Extract yaw from quaternion
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                self.get_logger().info(
                    f'Robot pose: x={x:.4f}  y={y:.4f}  yaw={math.degrees(yaw):.2f}°  '
                    f'(frame: {base_frame})',
                    throttle_duration_sec=0.5)
                return x, y, yaw
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
        self.get_logger().warn('Robot pose TF unavailable (tried base_footprint, base_link).')
        return None

    # ── TF2 ───────────────────────────────────────────────────────────────────
    def _to_map_frame(self, pose_stamped: PoseStamped):
        """
        Transform PoseStamped from camera frame to map frame.

        We request the transform at the pose's header stamp (which we set
        to now() during measuring). rclpy.time.Time() with no args means
        "latest available" — equivalent to Time(0) in ROS2 — which gives
        us the most recent SLAM-corrected robot pose.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                MAP_FRAME,
                pose_stamped.header.frame_id,
                rclpy.time.Time(),         # latest available transform
                timeout=Duration(seconds=1.0)
            )
            # Log the robot translation at transform time for debugging
            t = transform.transform.translation
            self.get_logger().info(
                f'TF robot→map at measurement: '
                f'x={t.x:.4f}  y={t.y:.4f}  z={t.z:.4f}')
            return tf2_geometry_msgs.do_transform_pose_stamped(pose_stamped, transform)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    # ── Debug image ───────────────────────────────────────────────────────────
    def _publish_debug(self, frame, header):
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        out        = CompressedImage()
        out.header = header
        out.format = 'jpeg'
        out.data   = buf.tobytes()
        self.debug_pub.publish(out)

    # ── 1 Hz re-publisher ─────────────────────────────────────────────────────
    def _republish_timer(self):
        if self.phase != DetectionPhase.DONE or self.aruco_map_pose is None:
            return
        self.pose_pub.publish(self.aruco_map_pose)
        found_msg      = Bool()
        found_msg.data = True
        self.found_pub.publish(found_msg)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset(self):
        self.phase             = DetectionPhase.SCANNING
        self._confirm_id       = None
        self._confirm_count    = 0
        self._measure_samples  = []
        self.aruco_map_pose    = None

    # ── solvePnP helper ───────────────────────────────────────────────────────
    def _solve_pnp(self, corners_i):
        """
        Run solvePnP on detected corners. Returns (tvec, rvec, dist) or None.
        tvec is (3,1), rvec is (3,1), dist is scalar float.
        """
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
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            return None
        # Flatten (3,1) → (3,) so callers can index tvec[0], tvec[1], tvec[2]
        # directly as Python floats without extra unpacking.
        tvec = tvec.flatten()
        rvec = rvec.flatten()
        dist = float(np.linalg.norm(tvec))
        return tvec, rvec, dist

    # ── Detection helper ──────────────────────────────────────────────────────
    def _best_detection(self, corners, ids, image_w):
        """
        From the raw detector output, return the first valid detection as:
          (marker_id, corners_i, tvec, rvec, dist)
        or None if nothing valid was found.
        """
        if ids is None or len(ids) == 0:
            return None

        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if TARGET_MARKER_ID is not None and mid != TARGET_MARKER_ID:
                continue
            result = self._solve_pnp(corners[i])
            if result is None:
                continue
            tvec, rvec, dist = result
            return mid, corners[i], tvec, rvec, dist

        return None


# ── Module-level helpers ──────────────────────────────────────────────────────

def _marker_centre(corners_i) -> tuple:
    """Return (cx, cy) integer pixel coordinates of a marker's centre."""
    pts = corners_i.reshape(4, 2)
    return int(pts[:, 0].mean()), int(pts[:, 1].mean())


def _rotation_matrix_to_quaternion(R: np.ndarray) -> list:
    """Convert 3×3 rotation matrix to quaternion [x, y, z, w] (Shepperd's method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [x, y, z, w]


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()