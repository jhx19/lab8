"""
lab8.launch.py
--------------
Single-terminal launch for the full Lab 8 maze mission.

Startup chain (fully sequential, event-driven):

  1. SLAM launches immediately.

  2. wait_for_map: polls until /map has a publisher.
     → SLAM process has started and is publishing.

  3. wait_for_slam_tf: polls until slam_toolbox services appear AND
     the map→odom TF transform is being broadcast.
     This is the *true* SLAM-ready signal: TF tree is established,
     the map frame exists, and localisation is running.
     Adds a 3 s stabilisation buffer after TF appears.

  4. Nav2 launches (map TF is stable → costmaps can initialise correctly).

  5. wait_for_nav2: polls until Nav2's global costmap clear service exists.
     This service only appears once all Nav2 lifecycle nodes are ACTIVE.

  6. App nodes launch (aruco_detector, wall_follower, navigator).

  7. After a 3 s settle (so sub-nodes subscribe before orchestrator publishes),
     the orchestrator launches and begins the mission.

Why this order matters:
  Nav2's costmap nodes subscribe to /map and need the map→odom TF to
  initialise their transform listeners.  If Nav2 starts before SLAM
  has published even one transform on that chain, the costmap lifecycle
  transitions fail silently or the planner gets stuck waiting for TF.

  The orchestrator also has its own guard (_poll_for_initial_pose at 1 Hz)
  that blocks exploration until map→base_footprint TF is confirmed, but
  relying on that alone meant Nav2 was already in a broken state by the
  time the orchestrator started moving the robot.
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

    # ── Probe 1: /map topic has a publisher ───────────────────────────────────
    # Fires as soon as slam_toolbox starts publishing the occupancy grid.
    # This is just "SLAM process is alive", NOT "SLAM TF is ready".
    wait_for_map = ExecuteProcess(
        name='wait_for_map',
        cmd=[
            'bash', '-c',
            'echo "[lab8] Waiting for SLAM to publish /map..."; '
            'until ros2 topic info /map 2>/dev/null '
            '      | grep -q "Publisher count: [1-9]"; '
            'do sleep 1; done; '
            'echo "[lab8] /map is live — checking SLAM TF next"'
        ],
        output='screen'
    )

    # ── Probe 2: SLAM TF is established (map → odom transform exists) ─────────
    #
    # slam_toolbox publishes the map→odom transform on /tf only after it has
    # successfully localised the robot for the first time.  We check two
    # things in sequence:
    #
    #   a) slam_toolbox service list includes 'save_map' — the toolbox node
    #      is fully initialised (its lifecycle is ACTIVE).
    #
    #   b) ros2 topic echo /tf_static or a brief spin shows a transform
    #      whose header.frame_id is 'map'.  We use a small python snippet
    #      because bash string matching on /tf is unreliable (binary msgs).
    #
    # After both pass we sleep 3 s to let the TF tree fully stabilise before
    # handing off to Nav2.  This 3 s is short compared to SLAM startup time
    # (typically 10–30 s on TurtleBot4) and eliminates the race condition.
    wait_for_slam_tf = ExecuteProcess(
        name='wait_for_slam_tf',
        cmd=[
            'bash', '-c',
            # Step a: 静态 TF 用 /tf_static
            'echo "[lab8] Waiting for robot TF (base_footprint→rplidar_link)..."; '
            'until ros2 topic echo --once /tf_static 2>/dev/null '
            '      | grep -q "rplidar_link"; '
            'do sleep 1; done; '
            'echo "[lab8] Robot TF ready. Waiting for slam_toolbox service..."; '
            # Step b: slam_toolbox node active
            'until ros2 service list 2>/dev/null '
            '      | grep -q "slam_toolbox/save_map"; '
            'do sleep 1; done; '
            'echo "[lab8] slam_toolbox service found — waiting for map→odom TF..."; '
            # Step c: map→odom 是动态 TF，在 /tf 上
            'until ros2 topic echo --once /tf 2>/dev/null '
            '      | grep -q "frame_id: map"; '
            'do sleep 1; done; '
            'echo "[lab8] map→odom TF confirmed — stabilising 3 s"; '
            'sleep 3; '
            'echo "[lab8] SLAM ready → launching Nav2"'
        ],
        output='screen'
    )
    # ── Probe 3: Nav2 costmap service is active ───────────────────────────────
    # The global costmap clear service only appears once all Nav2 lifecycle
    # nodes have successfully transitioned to ACTIVE state.  Polling for it
    # is the most reliable "Nav2 is ready" signal available without writing
    # a custom lifecycle client.
    wait_for_nav2 = ExecuteProcess(
        name='wait_for_nav2',
        cmd=[
            'bash', '-c',
            'echo "[lab8] Waiting for Nav2 to become active..."; '
            'until ros2 service list 2>/dev/null '
            '      | grep -q "global_costmap/clear_entirely_global_costmap"; '
            'do sleep 2; done; '
            'echo "[lab8] Nav2 is active — starting app nodes"'
        ],
        output='screen'
    )

    # ── Event chain ───────────────────────────────────────────────────────────
    #
    #   SLAM start
    #     └─► wait_for_map (probe 1)
    #               │ exit
    #               └─► wait_for_slam_tf (probe 2)
    #                         │ exit
    #                         └─► nav2_launch
    #                         └─► wait_for_nav2 (probe 3)
    #                                   │ exit
    #                                   └─► aruco_detector
    #                                   └─► wall_follower
    #                                   └─► navigator
    #                                   └─► [3 s timer]
    #                                             └─► orchestrator  ← MISSION START

    # Step 1 → 2: /map live → start SLAM TF probe
    on_map_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_map,
            on_exit=[
                LogInfo(msg='[lab8] SLAM /map ready. Checking TF...'),
                wait_for_slam_tf,
            ]
        )
    )

    # Step 2 → 3: SLAM TF confirmed → start Nav2 + Nav2 probe
    on_slam_tf_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_slam_tf,
            on_exit=[
                LogInfo(msg='[lab8] SLAM TF stable. Launching Nav2 + early nodes...'),
                nav2_launch,
                wait_for_nav2,
                # 这两个不依赖 Nav2，立刻启动
                aruco_detector_node,
                wall_follower_node,
            ]
        )
    )

    # Step 3 → 4: Nav2 active → start app nodes, then orchestrator
    on_nav2_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_nav2,
            on_exit=[
                LogInfo(msg='[lab8] Nav2 active. Launching navigator + orchestrator...'),
                navigator_node,
                TimerAction(period=3.0, actions=[
                    LogInfo(msg='[lab8] ─── MISSION BEGINS ───'),
                    orchestrator_node,
                ]),
            ]
        )
    )

    return LaunchDescription([
        LogInfo(msg='[lab8] Launching SLAM...'),
        slam_launch,
        wait_for_map,       # probe 1 starts immediately alongside SLAM
        on_map_ready,       # registers SLAM-map → probe-2 trigger
        on_slam_tf_ready,   # registers probe-2 → Nav2 trigger
        on_nav2_ready,      # registers Nav2 → app-nodes trigger
    ])