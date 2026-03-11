"""
wall_follower.py
----------------
Phase 1-A: PID-controlled right-wall-following exploration node.

LiDAR index convention (as specified):
  index 0         → right   (  0°)
  index n * 1/4   → front   ( 90°, CCW from right)
  index n * 1/2   → left    (180°)
  index n * 3/4   → back    (270°)

  Where n = total number of readings in one scan.
  Angle increases counter-clockwise when viewed from above.

Right-wall PID logic:
  error = measured_right_dist - DESIRED_WALL_DIST

  error > 0  →  robot drifted away from right wall  →  turn right (negative angular_z)
  error < 0  →  robot too close to right wall        →  turn left  (positive angular_z)

  angular_z = -(Kp * error + Ki * integral + Kd * derivative)

  The negative sign converts "distance error" into "steering correction":
    too far from wall  (error > 0) → steer right → negative angular_z (CW)
    too close to wall  (error < 0) → steer left  → positive angular_z (CCW)

Priority overrides (pre-empt PID output):
  1. Front obstacle < FRONT_DANGER_DIST  → turn left in place (corner handling)
  2. Right wall lost > WALL_LOST_DIST    → slow down and turn right to reacquire

Orchestrator interface:
  Subscribe: /wall_follower/cmd  (std_msgs/String)  "start" | "stop"
  Publish:   /exploration_done   (std_msgs/Bool)     True once stopped
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


# ── Tunable parameters ────────────────────────────────────────────────────────

DESIRED_WALL_DIST = 0.37    # target distance to right wall (meters)
WALL_LOST_DIST    = 0.7    # if right reading > this, wall is lost → reacquire
FRONT_DANGER_DIST = 0.5    # if front reading < this, obstacle → turn left

# PID gains  (tune on real robot — start with Kp only, then add Kd, then Ki)
Kp = 1.2
Ki = 0.01
Kd = 0.15

INTEGRAL_MAX = 1.0          # anti-windup clamp on the integral term

LINEAR_SPEED  = 0.46      # base forward speed (m/s)
ANGULAR_MAX   = 1.5         # PID output clamped to ±this (rad/s)
TURN_SPEED    = 1.0        # fixed turn rate for priority overrides (rad/s)

# Sector half-width as a fraction of total scan indices.
# 0.07 → ±7% of n indices, which is roughly ±25° for a 360° LiDAR.
SECTOR_HALF_FRAC = 0.07

# ─────────────────────────────────────────────────────────────────────────────


class PIDController:
    """Discrete PID controller with anti-windup integral clamping."""

    def __init__(self, kp: float, ki: float, kd: float,
                 integral_max: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max

        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None   # float seconds; None until first call

    def compute(self, error: float, current_time: float) -> float:
        """
        Return PID output for the given error and current timestamp.

        Args:
            error:        setpoint − measurement
                          (positive = below target, needs positive correction)
            current_time: wall-clock time in seconds (float)
        """
        # First call: no dt available yet, return P-only output
        if self._prev_time is None:
            self._prev_time  = current_time
            self._prev_error = error
            return self.kp * error

        dt = current_time - self._prev_time
        if dt <= 0.0:
            return self.kp * error   # guard against duplicate timestamps

        # P term
        p = self.kp * error

        # I term — accumulate and clamp to prevent windup
        self._integral = max(-self.integral_max,
                             min(self.integral_max,
                                 self._integral + error * dt))
        i = self.ki * self._integral

        # D term — on error difference to avoid derivative kick on setpoint change
        d = self.kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time  = current_time

        return p + i + d

    def reset(self):
        """Clear integral and history. Call before each new run."""
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None


class WallFollower(Node):

    def __init__(self):
        super().__init__('wall_follower')

        # ── PID ───────────────────────────────────────────────────────────────
        self.pid = PIDController(Kp, Ki, Kd, INTEGRAL_MAX)

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.done_pub    = self.create_publisher(Bool,  '/exploration_done', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)

        # Orchestrator sends "start" or "stop" here
        self.cmd_sub = self.create_subscription(
            String, '/wall_follower/cmd', self._cmd_callback, 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.running = False

        self.get_logger().info(
            f'WallFollower ready (right-wall PID, Kp={Kp} Ki={Ki} Kd={Kd}). '
            'Waiting for "start".')

    # ── Orchestrator command handler ──────────────────────────────────────────
    def _cmd_callback(self, msg: String):
        if msg.data == 'start':
            self.get_logger().info('Wall-following STARTED.')
            self.pid.reset()   # fresh integrator for every run
            self.running = True

        elif msg.data == 'stop':
            self.get_logger().info('Wall-following STOPPED.')
            self.running = False
            self._publish_stop()
            done = Bool()
            done.data = True
            self.done_pub.publish(done)

    # ── LiDAR scan callback ───────────────────────────────────────────────────
    def _scan_callback(self, msg: LaserScan):
        if not self.running:
            return

        # Read relevant sectors (index-fraction based, with wrap-around)
        right_dist = self._sector_min(msg, centre_frac=0.0,  half_frac=SECTOR_HALF_FRAC)
        front_dist = self._sector_min(msg, centre_frac=0.25, half_frac=SECTOR_HALF_FRAC)

        twist = Twist()
        now   = self.get_clock().now().nanoseconds * 1e-9   # seconds (float)

        # ── Priority 1: obstacle ahead → turn left (CCW) in place ────────────
        if front_dist < FRONT_DANGER_DIST: # when there is obstacle in front, turn left, the speed is determined by front_dist, the closer the obstacle, the faster the turn, and the slower the forward speed
            twist.linear.x  = LINEAR_SPEED * 0.6 * (max(front_dist-FRONT_DANGER_DIST*0.2,0) / FRONT_DANGER_DIST*0.80)  # slow down as obstacle gets closer
            twist.angular.z = TURN_SPEED     # positive = CCW = left in ROS
            self.get_logger().info(
                f'[OVERRIDE] Front obstacle {front_dist:.2f} m → turning left, s {twist.linear.x:.2f} m/s, w {twist.angular.z:.2f} rad/s',
                throttle_duration_sec=1.0)
            self.pid.reset()   # prevent derivative spike after the pause

        # ── Priority 2: right wall lost → turn right slowly to reacquire ─────
        elif right_dist > WALL_LOST_DIST:
            twist.linear.x  = LINEAR_SPEED * 0.7 
            twist.angular.z = -TURN_SPEED * 0.7  # negative = CW = right
            self.get_logger().info(
                f'[OVERRIDE] Wall lost ({right_dist:.2f} m) → reacquiring, s {twist.linear.x:.2f} m/s, w {twist.angular.z:.2f} rad/s',
                throttle_duration_sec=1.0)
            self.pid.reset()

        # ── Normal: PID steers the robot to maintain desired wall distance ────
        else:
            # Positive error → too far from right wall → steer right (negative ω)
            error = right_dist - DESIRED_WALL_DIST

            pid_raw = self.pid.compute(error, now)

            # Negate so positive error produces rightward correction
            angular_z = max(-ANGULAR_MAX, min(ANGULAR_MAX, -pid_raw))

            # if the anglular correction is large, we can reduce the forward speed to make the turn smoother
            if abs(angular_z) > ANGULAR_MAX * 0.6:
                # it depends on the angular_z, the larger the angular_z, the slower the forward speed, but it will not be less than 30% of the original speed
                twist.linear.x = LINEAR_SPEED * max(0.3, 1 - abs(angular_z) / ANGULAR_MAX)
            else:
                twist.linear.x = LINEAR_SPEED
            
            twist.angular.z = angular_z

            self.get_logger().info(
                f'right={right_dist:.3f} m  err={error:+.3f}  '
                f'PID={pid_raw:+.3f} s {twist.linear.x:.2f} m/s, w {twist.angular.z:+.3f} rad/s',
                throttle_duration_sec=0.5)

        self.cmd_vel_pub.publish(twist)

    # ── Sector extraction ─────────────────────────────────────────────────────
    def _sector_min(self, msg: LaserScan,
                    centre_frac: float, half_frac: float) -> float:
        """
        Return the minimum valid range in an index-defined angular sector.

        Sector centre and width are expressed as fractions of the total
        number of scan readings (n), so the result is independent of
        the LiDAR's angular resolution.

        Index-to-direction mapping (as given in the spec):
          centre_frac = 0.00  → right   (index 0)
          centre_frac = 0.25  → front   (index n/4)
          centre_frac = 0.50  → left    (index n/2)
          centre_frac = 0.75  → back    (index 3n/4)

        Args:
            msg:         LaserScan message
            centre_frac: fractional index of sector centre  (0.0 – 1.0)
            half_frac:   half-width of sector as a fraction of n

        Returns:
            Minimum valid range (m), or msg.range_max if no valid readings.
        """
        n          = len(msg.ranges)
        centre_idx = int(centre_frac * n)
        half_idx   = int(half_frac   * n)

        # Build index list with modulo wrap-around
        indices = [(centre_idx + offset) % n
                   for offset in range(-half_idx, half_idx + 1)]

        valid = [
            msg.ranges[i] for i in indices
            if (not math.isnan(msg.ranges[i])
                and not math.isinf(msg.ranges[i])
                and msg.range_min < msg.ranges[i] < msg.range_max)
        ]

        return min(valid) if valid else msg.range_max

    # ── Stop helper ───────────────────────────────────────────────────────────
    def _publish_stop(self):
        """Publish zero velocity to halt the robot immediately."""
        self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
