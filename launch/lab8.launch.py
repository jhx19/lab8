"""
lab8.launch.py
--------------
Single-terminal launch for the full Lab 8 maze mission.

Startup logic (event-driven, no fixed timers):
  1. SLAM starts immediately.
  2. A background process polls `ros2 topic info /map` every second.
     As soon as SLAM publishes /map (publisher count > 0), it exits.
  3. Nav2 launches in response to that exit event.
  4. Another background process polls `ros2 service list` for the
     Nav2 costmap service that only appears when Nav2 is fully active.
     When it appears, the app nodes + orchestrator launch.

This means startup is as fast as the hardware allows — no wasted waiting,
no race conditions from under-estimated fixed delays.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    LogInfo,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Custom Nav2 params ────────────────────────────────────────────────────
    # lab8_nav2_yaml   = os.path.join(
    #     get_package_share_directory('lab8'), 'config', 'nav2.yaml')
    # system_nav2_yaml = os.path.join(
    #     get_package_share_directory('turtlebot4_navigation'), 'config', 'nav2.yaml')
    # nav2_params = lab8_nav2_yaml if os.path.isfile(lab8_nav2_yaml) else system_nav2_yaml

    # ── SLAM ──────────────────────────────────────────────────────────────────
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot4_navigation'),
                'launch', 'slam.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': 'false'}.items()
    )

    # ── Nav2 ──────────────────────────────────────────────────────────────────
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot4_navigation'),
                'launch', 'nav2.launch.py'
            ])
        ]),
        # launch_arguments={
        #     'use_sim_time': 'false',
        #     'params_file':  nav2_params,
        # }.items()
        launch_arguments={'use_sim_time': 'false'}.items()
    )

    # ── App nodes ─────────────────────────────────────────────────────────────
    aruco_detector_node = Node(
        package='lab8', executable='aruco_detector',
        name='aruco_detector', output='screen')
    wall_follower_node = Node(
        package='lab8', executable='wall_follower',
        name='wall_follower', output='screen')
    navigator_node = Node(
        package='lab8', executable='navigator',
        name='navigator', output='screen')
    orchestrator_node = Node(
        package='lab8', executable='orchestrator',
        name='orchestrator', output='screen')

    # ── Readiness probe 1: wait for SLAM to publish /map ─────────────────────
    # Polls every second. Exits with code 0 as soon as /map has a publisher.
    # This process exit event triggers Nav2 to start.
    wait_for_map = ExecuteProcess(
        name='wait_for_map',
        cmd=[
            'bash', '-c',
            'echo "[lab8] Waiting for SLAM to publish /map..."; '
            'until ros2 topic info /map 2>/dev/null | grep -q "Publisher count: [1-9]"; '
            'do sleep 1; done; '
            'echo "[lab8] /map is live — starting Nav2"'
        ],
        output='screen'
    )

    # ── Readiness probe 2: wait for Nav2 costmap service to be active ─────────
    # The global costmap clear service only exists once Nav2 lifecycle is fully
    # active. We use it as the "Nav2 is ready" signal.
    # Polls every 2 seconds. Exits when the service appears.
    wait_for_nav2 = ExecuteProcess(
        name='wait_for_nav2',
        cmd=[
            'bash', '-c',
            'echo "[lab8] Waiting for Nav2 to become active..."; '
            'until ros2 service list 2>/dev/null | grep -q "global_costmap/clear_entirely_global_costmap"; '
            'do sleep 2; done; '
            'echo "[lab8] Nav2 is active — starting app nodes"'
        ],
        output='screen'
    )

    # ── Event: SLAM /map ready → start Nav2 + probe 2 ────────────────────────
    on_map_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_map,
            on_exit=[
                LogInfo(msg='[lab8] SLAM map ready. Launching Nav2...'),
                nav2_launch,
                wait_for_nav2,   # start the Nav2 readiness probe simultaneously
            ]
        )
    )

    # ── Event: Nav2 active → start app nodes + orchestrator ──────────────────
    on_nav2_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_nav2,
            on_exit=[
                LogInfo(msg='[lab8] Nav2 active. Launching app nodes...'),
                aruco_detector_node,
                wall_follower_node,
                navigator_node,
                # Small delay so sub-nodes are subscribed before orchestrator
                # sends its first commands
                TimerAction(period=3.0, actions=[
                    LogInfo(msg='[lab8] Starting orchestrator — MISSION BEGINS'),
                    orchestrator_node,
                ]),
            ]
        )
    )

    return LaunchDescription([
        LogInfo(msg='[lab8] Starting SLAM...'),
        slam_launch,
        wait_for_map,    # probe 1 starts alongside SLAM
        on_map_ready,    # registers the SLAM→Nav2 trigger
        on_nav2_ready,   # registers the Nav2→app trigger
    ])