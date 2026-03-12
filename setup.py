from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lab8'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Install config files
        # (os.path.join('share', package_name, 'config'),
        #     glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@uw.edu',
    description='Lab 8: Maze solving with ArUco navigation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Each module can be run standalone for testing (Phase 1)
            'wall_follower   = lab8.wall_follower:main',
            'aruco_detector  = lab8.aruco_detector:main',
            'navigator       = lab8.navigator:main',
            # Central orchestrator (Phase 2)
            'orchestrator    = lab8.orchestrator:main',
            'maze_solver = lab8.maze_solver:main',
        ],
    },
)