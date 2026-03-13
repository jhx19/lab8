"""
navigator.py
------------
Navigation node with loop-wait for Nav2 + built-in stuck recovery.

Commands on /navigator/cmd (String):
  "goto_start"    → navigate to the saved start pose
  "goto_aruco"    → navigate to GOAL_OFFSET_M in front of the ArUco marker
  "dock"          → TurtleBot4 auto-dock
  "undock"        → TurtleBot4 undock

Status on /navigator/status (String):
  "at_start", "at_goal", "docked", "undocked", "failed"

Stuck-recovery logic (internal, transparent to orchestrator):
─────────────────────────────────────────────────────────────
  A 2 Hz watchdog checks robot progress while a Nav2 goal is active.

  STUCK declared when BOTH are true over STUCK_TIME_S seconds:
    • translation < STUCK_DIST_THRESHOLD metres
    • heading change < STUCK_YAW_THRESHOLD radians
  (checking both avoids false positives when robot is turning in place)

  Recovery sequence:
    1. Cancel current Nav2 goal (release /cmd_vel control)
    2. Read /scan → find sector with lowest avg range (nearest obstacle)
    3. Rotate in-place toward the OPPOSITE direction (escape heading)
    4. Drive forward RECOVERY_FWD_TIME seconds at RECOVERY_LINEAR m/s
    5. Re-send the same Nav2 goal
    6. Repeat up to MAX_RECOVERY_ATTEMPTS before reporting "failed"
─────────────────────────────────────────────────────────────
"""

import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan

from nav2_msgs.action import NavigateToPose
from irobot_create_msgs.action import Dock, Undock

import tf2_ros


# ── Navigation parameters ─────────────────────────────────────────────────────
GOAL_OFFSET_M = 0.10      # stop this far in front of the ArUco cube (m)

# ── Stuck detection ───────────────────────────────────────────────────────────
STUCK_TIME_S         = 6.0    # observation window (s)
STUCK_DIST_THRESHOLD = 0.08   # metres of travel required in window
STUCK_YAW_THRESHOLD  = 0.12   # radians of heading change required in window
STUCK_GRACE_S        = 4.0    # ignore first N seconds after goal is sent
WATCHDOG_HZ          = 2      # watchdog check rate

# ── Recovery motion ───────────────────────────────────────────────────────────
RECOVERY_OMEGA        = 0.6   # rotation speed during alignment (rad/s)
RECOVERY_ALIGN_THRESH = 0.15  # stop rotating when |error| < this (rad)
RECOVERY_LINEAR       = 0.15  # forward speed during escape (m/s)
RECOVERY_FWD_TIME     = 1.5   # seconds of forward escape motion
RECOVERY_SCAN_WINDOW  = 30    # ± degrees around each candidate heading
MAX_RECOVERY_ATTEMPTS = 5     # give up after this many failed recoveries

# Number of evenly-spaced candidate headings evaluated by the scan analyser
ESCAPE_CANDIDATES = 12
# ─────────────────────────────────────────────────────────────────────────────


class Navigator(Node):

    def __init__(self):
        super().__init__('navigator')

        # ── Saved poses ───────────────────────────────────────────────────────
        self.start_map_pose: PoseStamped = None
        self.aruco_map_pose: PoseStamped = None

        # ── TF2 ──────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Action clients ────────────────────────────────────────────────────
        self._nav_client    = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._dock_client   = ActionClient(self, Dock,   '/dock')
        self._undock_client = ActionClient(self, Undock, '/undock')

        # ── Cached scan ───────────────────────────────────────────────────────
        self._latest_scan: LaserScan = None
        self._scan_lock = threading.Lock()

        # ── Active goal state ─────────────────────────────────────────────────
        self._active_gh = None
        self._active_goal_pose: PoseStamped = None
        self._active_success_status: str = None
        self._goal_send_time: float = None
        self._recovery_count: int = 0
        self._in_recovery: bool = False

        # ── Watchdog ──────────────────────────────────────────────────────────
        self._pose_history: list = []
        self._watchdog_timer = None

        # ── Publishers / Subscribers ──────────────────────────────────────────
        self.status_pub  = self.create_publisher(String, '/navigator/status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist,  '/cmd_vel', 10)

        self.cmd_sub = self.create_subscription(
            String, '/navigator/cmd', self._cmd_cb, 10)
        self.start_pose_sub = self.create_subscription(
            PoseStamped, '/start_map_pose', self._start_pose_cb, 10)
        self.aruco_sub = self.create_subscription(
            PoseStamped, '/aruco_map_pose', self._aruco_pose_cb, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10)

        self.get_logger().info(
            f'Navigator ready.  offset={GOAL_OFFSET_M} m  '
            f'stuck_window={STUCK_TIME_S} s  max_recovery={MAX_RECOVERY_ATTEMPTS}')

    # ═══════════════════════════════════════════════════════════════════════════
    # Nav2 server wait
    # ═══════════════════════════════════════════════════════════════════════════

    def _wait_for_nav2(self):
        """Block until Nav2 action server is available, retrying every 5 s."""
        self.get_logger().info('Waiting for Nav2 action server...')
        while not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 not ready yet, retrying...')
        self.get_logger().info('Nav2 ready.')

    # ═══════════════════════════════════════════════════════════════════════════
    # Pose / scan storage
    # ═══════════════════════════════════════════════════════════════════════════

    def _start_pose_cb(self, msg: PoseStamped):
        self.start_map_pose = msg
        self.get_logger().info(
            f'Start pose stored: x={msg.pose.position.x:.3f}  '
            f'y={msg.pose.position.y:.3f}')

    def _aruco_pose_cb(self, msg: PoseStamped):
        self.aruco_map_pose = msg
        self.get_logger().info(
            f'ArUco pose stored: x={msg.pose.position.x:.3f}  '
            f'y={msg.pose.position.y:.3f}')

    def _scan_cb(self, msg: LaserScan):
        with self._scan_lock:
            self._latest_scan = msg

    # ═══════════════════════════════════════════════════════════════════════════
    # Command handler
    # ═══════════════════════════════════════════════════════════════════════════

    def _cmd_cb(self, msg: String):
        cmd = msg.data.strip()
        self.get_logger().info(f'Navigator command: "{cmd}"')
        if   cmd == 'goto_start': self._do_goto_start()
        elif cmd == 'goto_aruco': self._do_goto_aruco()
        elif cmd == 'dock':       self._do_dock()
        elif cmd == 'undock':     self._do_undock()
        else:
            self.get_logger().warn(f'Unknown command: {cmd}')

    # ═══════════════════════════════════════════════════════════════════════════
    # Navigation commands
    # ═══════════════════════════════════════════════════════════════════════════

    def _do_goto_start(self):
        if self.start_map_pose is None:
            self.get_logger().error('No start pose stored — cannot navigate.')
            self._publish_status('failed')
            return

        self._wait_for_nav2()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp    = self.get_clock().now().to_msg()
        goal_pose.pose            = self.start_map_pose.pose

        self.get_logger().info(
            f'[goto_start] x={goal_pose.pose.position.x:.3f}  '
            f'y={goal_pose.pose.position.y:.3f}')
        self._send_nav_goal(goal_pose, success_status='at_start')

    def _do_goto_aruco(self):
        if self.aruco_map_pose is None:
            self.get_logger().error('No ArUco pose stored — cannot navigate.')
            self._publish_status('failed')
            return

        self._wait_for_nav2()

        goal_pose = self._compute_aruco_goal(self.aruco_map_pose)
        self.get_logger().info(
            f'[goto_aruco] goal x={goal_pose.pose.position.x:.3f}  '
            f'y={goal_pose.pose.position.y:.3f}')
        self._send_nav_goal(goal_pose, success_status='at_goal')

    def _compute_aruco_goal(self, aruco_pose: PoseStamped) -> PoseStamped:
        mx = aruco_pose.pose.position.x
        my = aruco_pose.pose.position.y
        q  = aruco_pose.pose.orientation
        approach_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        gx = mx - GOAL_OFFSET_M * math.cos(approach_yaw)
        gy = my - GOAL_OFFSET_M * math.sin(approach_yaw)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp    = self.get_clock().now().to_msg()
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(approach_yaw / 2.0)
        goal.pose.orientation.w = math.cos(approach_yaw / 2.0)
        self.get_logger().info(
            f'ArUco offset goal: marker=({mx:.3f},{my:.3f})  '
            f'yaw={math.degrees(approach_yaw):.1f}°  '
            f'goal=({gx:.3f},{gy:.3f})')
        return goal

    # ═══════════════════════════════════════════════════════════════════════════
    # Core Nav2 send / callbacks
    # ═══════════════════════════════════════════════════════════════════════════

    def _send_nav_goal(self, goal_pose: PoseStamped, success_status: str,
                       reset_recovery: bool = True):
        """Send a NavigateToPose goal and arm the stuck watchdog."""
        if reset_recovery:
            self._recovery_count = 0

        self._active_goal_pose      = goal_pose
        self._active_success_status = success_status
        self._active_gh             = None
        self._goal_send_time        = time.time()
        self._pose_history.clear()
        self._in_recovery           = False

        nav_goal      = NavigateToPose.Goal()
        nav_goal.pose = goal_pose
        future = self._nav_client.send_goal_async(
            nav_goal, feedback_callback=self._nav_feedback_cb)
        future.add_done_callback(self._nav_accepted_cb)

        self._stop_watchdog()
        self._watchdog_timer = self.create_timer(
            1.0 / WATCHDOG_HZ, self._watchdog_tick)

    def _nav_accepted_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('Nav2 goal rejected.')
            self._stop_watchdog()
            self._publish_status('failed')
            return
        self._active_gh = gh
        gh.get_result_async().add_done_callback(self._nav_result_cb)

    def _nav_feedback_cb(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        self.get_logger().info(
            f'  Nav2 distance remaining: {dist:.2f} m',
            throttle_duration_sec=5.0)

    def _nav_result_cb(self, future):
        # Ignore result callbacks that fire after a recovery cancel
        if self._in_recovery:
            return
        self._stop_watchdog()
        status = future.result().status
        if status == 4:   # SUCCEEDED
            self.get_logger().info(
                f'Navigation SUCCEEDED → {self._active_success_status}')
            self._publish_status(self._active_success_status)
        else:
            self.get_logger().error(f'Navigation FAILED (status={status})')
            self._publish_status('failed')

    # ═══════════════════════════════════════════════════════════════════════════
    # Stuck watchdog
    # ═══════════════════════════════════════════════════════════════════════════

    def _watchdog_tick(self):
        if self._in_recovery or self._active_gh is None:
            return

        pose = self._get_robot_pose()
        if pose is None:
            return
        x, y, yaw = pose
        now = time.time()

        if now - self._goal_send_time < STUCK_GRACE_S:
            return

        self._pose_history.append((now, x, y, yaw))
        cutoff = now - STUCK_TIME_S
        self._pose_history = [
            p for p in self._pose_history if p[0] >= cutoff]

        if len(self._pose_history) < 3:
            return

        oldest = self._pose_history[0]
        dist = math.sqrt((x - oldest[1]) ** 2 + (y - oldest[2]) ** 2)
        dyaw = abs(_angle_diff(yaw, oldest[3]))

        if dist < STUCK_DIST_THRESHOLD and dyaw < STUCK_YAW_THRESHOLD:
            self.get_logger().warn(
                f'[STUCK] dist={dist:.3f} m  Δyaw={math.degrees(dyaw):.1f}° '
                f'over {STUCK_TIME_S} s — triggering recovery '
                f'(attempt {self._recovery_count + 1}/{MAX_RECOVERY_ATTEMPTS})')
            self._trigger_recovery()

    def _stop_watchdog(self):
        if self._watchdog_timer is not None:
            self._watchdog_timer.cancel()
            self._watchdog_timer = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Recovery
    # ═══════════════════════════════════════════════════════════════════════════

    def _trigger_recovery(self):
        if self._recovery_count >= MAX_RECOVERY_ATTEMPTS:
            self.get_logger().error(
                f'Exceeded {MAX_RECOVERY_ATTEMPTS} recovery attempts — FAILED.')
            self._stop_watchdog()
            self._publish_status('failed')
            return

        self._recovery_count += 1
        self._in_recovery = True
        self._stop_watchdog()

        if self._active_gh is not None:
            self.get_logger().info('[RECOVERY] Cancelling Nav2 goal...')
            cancel_future = self._active_gh.cancel_goal_async()
            cancel_future.add_done_callback(self._recovery_after_cancel)
        else:
            threading.Thread(target=self._escape_thread, daemon=True).start()

    def _recovery_after_cancel(self, future):
        self.get_logger().info('[RECOVERY] Goal cancelled. Starting escape...')
        # Small pause to let Nav2 fully release /cmd_vel
        time.sleep(0.3)
        threading.Thread(target=self._escape_thread, daemon=True).start()

    def _escape_thread(self):
        """Blocking escape sequence: rotate away from obstacle, drive forward."""
        self.get_logger().info('[RECOVERY] Starting escape manoeuvre...')

        # ── 1. Find escape heading (opposite of nearest obstacle) ─────────────
        escape_yaw = self._find_escape_heading()

        # ── 2. Rotate toward escape heading ───────────────────────────────────
        pose = self._get_robot_pose()
        if pose is not None:
            heading_error = _angle_diff(escape_yaw, pose[2])
            self.get_logger().info(
                f'[RECOVERY] Escape yaw={math.degrees(escape_yaw):.1f}°  '
                f'current={math.degrees(pose[2]):.1f}°  '
                f'error={math.degrees(heading_error):.1f}°')

            deadline = time.time() + 5.0
            while abs(heading_error) > RECOVERY_ALIGN_THRESH \
                    and time.time() < deadline:
                omega = math.copysign(
                    min(RECOVERY_OMEGA, max(0.15, abs(heading_error) * 0.8)),
                    heading_error)
                self._pub_vel(0.0, omega)
                time.sleep(0.05)
                updated = self._get_robot_pose()
                if updated:
                    heading_error = _angle_diff(escape_yaw, updated[2])

            self._pub_vel(0.0, 0.0)
            time.sleep(0.1)

        # ── 3. Drive forward to escape inflation zone ─────────────────────────
        self.get_logger().info(
            f'[RECOVERY] Driving forward for {RECOVERY_FWD_TIME} s...')
        deadline = time.time() + RECOVERY_FWD_TIME
        while time.time() < deadline:
            self._pub_vel(RECOVERY_LINEAR, 0.0)
            time.sleep(0.05)

        self._pub_vel(0.0, 0.0)
        time.sleep(0.2)

        # ── 4. Re-send original Nav2 goal ─────────────────────────────────────
        self.get_logger().info('[RECOVERY] Escape complete. Re-sending Nav2 goal...')
        self._in_recovery = False
        self._send_nav_goal(
            self._active_goal_pose,
            self._active_success_status,
            reset_recovery=False)

    def _find_escape_heading(self) -> float:
        """
        Return the world-frame heading pointing AWAY from the nearest obstacle.
        Finds the scan sector with the lowest average range (nearest obstacle),
        then returns the opposite direction as the escape heading.
        """
        with self._scan_lock:
            scan = self._latest_scan

        pose = self._get_robot_pose()
        robot_yaw = pose[2] if pose else 0.0

        if scan is None or len(scan.ranges) == 0:
            self.get_logger().warn(
                '[RECOVERY] No scan available — reversing along current heading.')
            return _normalise_angle(robot_yaw + math.pi)

        n    = len(scan.ranges)
        half = int(RECOVERY_SCAN_WINDOW / 360.0 * n)

        worst_avg     = float('inf')
        obstacle_frac = 0.0

        step = n // ESCAPE_CANDIDATES
        for k in range(ESCAPE_CANDIDATES):
            centre  = k * step
            indices = [(centre + offset) % n
                       for offset in range(-half, half + 1)]
            valid   = [scan.ranges[i] for i in indices
                       if not math.isnan(scan.ranges[i])
                       and not math.isinf(scan.ranges[i])
                       and scan.range_min < scan.ranges[i] < scan.range_max]
            if not valid:
                continue
            avg = sum(valid) / len(valid)
            if avg < worst_avg:
                worst_avg     = avg
                obstacle_frac = centre / n

        # LiDAR index 0 = right = robot_yaw − π/2 in world frame
        obstacle_robot_angle = obstacle_frac * 2.0 * math.pi
        obstacle_world_angle = _normalise_angle(
            robot_yaw - math.pi / 2.0 + obstacle_robot_angle)
        escape_world_angle   = _normalise_angle(obstacle_world_angle + math.pi)

        self.get_logger().info(
            f'[RECOVERY] Nearest obstacle: frac={obstacle_frac:.2f}  '
            f'world={math.degrees(obstacle_world_angle):.1f}°  '
            f'avg_range={worst_avg:.2f} m  '
            f'→ escape={math.degrees(escape_world_angle):.1f}°')
        return escape_world_angle

    # ═══════════════════════════════════════════════════════════════════════════
    # Dock / Undock
    # ═══════════════════════════════════════════════════════════════════════════

    def _do_dock(self):
        self.get_logger().info('Docking...')
        if not self._dock_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Dock action server not available.')
            self._publish_status('failed')
            return
        future = self._dock_client.send_goal_async(Dock.Goal())
        future.add_done_callback(self._dock_goal_cb)

    def _dock_goal_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self._publish_status('failed')
            return
        gh.get_result_async().add_done_callback(
            lambda f: self._publish_status('docked'))

    def _do_undock(self):
        self.get_logger().info('Undocking...')
        if not self._undock_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Undock action server not available.')
            self._publish_status('failed')
            return
        future = self._undock_client.send_goal_async(Undock.Goal())
        future.add_done_callback(self._undock_goal_cb)

    def _undock_goal_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self._publish_status('failed')
            return
        gh.get_result_async().add_done_callback(
            lambda f: self._publish_status('undocked'))

    # ═══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_robot_pose(self):
        """Return (x, y, yaw) in map frame, or None if TF unavailable."""
        for frame in ('base_footprint', 'base_link'):
            try:
                tf = self.tf_buffer.lookup_transform(
                    'map', frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5))
                x   = tf.transform.translation.x
                y   = tf.transform.translation.y
                q   = tf.transform.rotation
                yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                return x, y, yaw
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
        return None

    def _pub_vel(self, linear: float, angular: float):
        twist = Twist()
        twist.linear.x  = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Navigator status → {status}')


# ── Module-level helpers ──────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    """Signed difference a − b, wrapped to (−π, π]."""
    return _normalise_angle(a - b)


def _normalise_angle(a: float) -> float:
    while a >  math.pi: a -= 2.0 * math.pi
    while a <= -math.pi: a += 2.0 * math.pi
    return a


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()