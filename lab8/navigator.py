"""
navigator.py
------------
Navigation node. Receives commands from orchestrator and executes
Nav2 NavigateToPose goals.

Commands on /navigator/cmd (String):
  "goto_start"    → navigate to the saved start pose (initial robot position)
  "goto_aruco"    → navigate to GOAL_OFFSET_M in front of the ArUco marker
  "dock"          → TurtleBot4 auto-dock action (IR guided)
  "undock"        → TurtleBot4 undock

Status on /navigator/status (String):
  "at_start", "at_goal", "docked", "undocked", "failed"

Subscribed poses:
  /start_map_pose  (PoseStamped) — published by orchestrator at INIT
  /aruco_map_pose  (PoseStamped) — published by aruco_detector at DONE

Goal offset geometry (goto_aruco):
  After centering, the ArUco pose orientation stored is the robot's
  own yaw at measurement time — i.e. pointing FROM the robot TOWARD
  the marker. So "10 cm in front of the marker" means backing off
  10 cm OPPOSITE to that yaw:

      goal_x = marker_x - GOAL_OFFSET_M * cos(approach_yaw)
      goal_y = marker_y - GOAL_OFFSET_M * sin(approach_yaw)
      goal_orientation = approach_yaw  (robot faces marker)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from nav2_msgs.action import NavigateToPose
from irobot_create_msgs.action import Dock, Undock


# ── Parameters ────────────────────────────────────────────────────────────────
GOAL_OFFSET_M = 0.10      # stop this far in front of the ArUco cube (m)
NAV_TIMEOUT_S = 120.0     # Nav2 goal timeout
# ─────────────────────────────────────────────────────────────────────────────


class Navigator(Node):

    def __init__(self):
        super().__init__('navigator')

        # ── Saved poses ───────────────────────────────────────────────────────
        self.start_map_pose: PoseStamped = None   # recorded before exploration
        self.aruco_map_pose: PoseStamped = None   # locked after measuring

        # ── Action clients ───────────────────────────────────────────────────
        self._nav_client    = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._dock_client   = ActionClient(self, Dock,   '/dock')
        self._undock_client = ActionClient(self, Undock, '/undock')

        # ── Publishers / Subscribers ─────────────────────────────────────────
        self.status_pub = self.create_publisher(String, '/navigator/status', 10)

        self.cmd_sub = self.create_subscription(
            String, '/navigator/cmd', self._cmd_cb, 10)

        self.start_pose_sub = self.create_subscription(
            PoseStamped, '/start_map_pose', self._start_pose_cb, 10)

        self.aruco_sub = self.create_subscription(
            PoseStamped, '/aruco_map_pose', self._aruco_pose_cb, 10)

        self.get_logger().info(f'Navigator ready.  offset={GOAL_OFFSET_M} m')

    # ── Pose storage ─────────────────────────────────────────────────────────
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

    # ── Command handler ──────────────────────────────────────────────────────
    def _cmd_cb(self, msg: String):
        cmd = msg.data.strip()
        self.get_logger().info(f'Navigator command: "{cmd}"')
        if   cmd == 'goto_start': self._do_goto_start()
        elif cmd == 'goto_aruco': self._do_goto_aruco()
        elif cmd == 'dock':       self._do_dock()
        elif cmd == 'undock':     self._do_undock()
        else:
            self.get_logger().warn(f'Unknown command: {cmd}')

    # ── goto_start ───────────────────────────────────────────────────────────
    def _do_goto_start(self):
        """
        Navigate back to the recorded start pose using Nav2.
        Uses the full Nav2 global planner + costmap for obstacle avoidance,
        unlike the TurtleBot4 dock IR approach which is line-of-sight only.
        """
        if self.start_map_pose is None:
            self.get_logger().error('No start pose stored — cannot navigate.')
            self._publish_status('failed')
            return

        if not self._nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 action server not available.')
            self._publish_status('failed')
            return

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp    = self.get_clock().now().to_msg()
        goal_pose.pose            = self.start_map_pose.pose

        self.get_logger().info(
            f'[goto_start] x={goal_pose.pose.position.x:.3f}  '
            f'y={goal_pose.pose.position.y:.3f}')

        nav_goal       = NavigateToPose.Goal()
        nav_goal.pose  = goal_pose
        future         = self._nav_client.send_goal_async(
            nav_goal, feedback_callback=self._nav_feedback_cb)
        future.add_done_callback(
            lambda f: self._nav_goal_cb(f, success_status='at_start'))

    # ── goto_aruco ───────────────────────────────────────────────────────────
    def _do_goto_aruco(self):
        """
        Navigate to GOAL_OFFSET_M in front of the ArUco marker.

        The stored ArUco orientation encodes the robot's yaw AT MEASUREMENT
        TIME — the direction pointing FROM the robot TOWARD the marker.
        "In front of the marker" = back off GOAL_OFFSET_M along that yaw
        so the robot stops just short of the marker, facing it.
        """
        if self.aruco_map_pose is None:
            self.get_logger().error('No ArUco pose stored — cannot navigate.')
            self._publish_status('failed')
            return

        if not self._nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 action server not available.')
            self._publish_status('failed')
            return

        goal_pose = self._compute_aruco_goal(self.aruco_map_pose)

        self.get_logger().info(
            f'[goto_aruco] goal x={goal_pose.pose.position.x:.3f}  '
            f'y={goal_pose.pose.position.y:.3f}')

        nav_goal       = NavigateToPose.Goal()
        nav_goal.pose  = goal_pose
        future         = self._nav_client.send_goal_async(
            nav_goal, feedback_callback=self._nav_feedback_cb)
        future.add_done_callback(
            lambda f: self._nav_goal_cb(f, success_status='at_goal'))

    def _compute_aruco_goal(self, aruco_pose: PoseStamped) -> PoseStamped:
        """
        Compute goal pose GOAL_OFFSET_M in front of the ArUco marker.

        Orientation convention:
          aruco_pose.orientation encodes the robot's approach yaw, i.e.
          the angle pointing FROM start TOWARD marker. To place the goal
          just in front of the marker, we step BACK along that direction.

        Diagram (top view):
          [start/dock]  ──approach_yaw──>  [ArUco marker]
                                    [goal] ← GOAL_OFFSET_M ←

          goal = marker − GOAL_OFFSET_M × (cos θ, sin θ)
          robot faces θ at goal (already pointing at marker)
        """
        mx = aruco_pose.pose.position.x
        my = aruco_pose.pose.position.y

        q = aruco_pose.pose.orientation
        approach_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

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

    # ── Dock ─────────────────────────────────────────────────────────────────
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

    # ── Undock ───────────────────────────────────────────────────────────────
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

    # ── Nav2 shared callbacks ─────────────────────────────────────────────────
    def _nav_feedback_cb(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        self.get_logger().info(
            f'  Nav2 distance remaining: {dist:.2f} m',
            throttle_duration_sec=5.0)

    def _nav_goal_cb(self, future, success_status: str):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('Nav2 goal rejected.')
            self._publish_status('failed')
            return
        gh.get_result_async().add_done_callback(
            lambda f: self._nav_result_cb(f, success_status))

    def _nav_result_cb(self, future, success_status: str):
        status = future.result().status
        if status == 4:   # SUCCEEDED
            self.get_logger().info(f'Navigation SUCCEEDED → {success_status}')
            self._publish_status(success_status)
        else:
            self.get_logger().error(f'Navigation FAILED (status={status})')
            self._publish_status('failed')

    # ── Utility ──────────────────────────────────────────────────────────────
    def _publish_status(self, status: str):
        msg = String(); msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Navigator status → {status}')


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