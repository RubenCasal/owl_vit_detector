import os
import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration



def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='owl_vit_detector',
            executable='theta_node',
            name='theta_node',
            output='screen'
        ),

        launch_ros.actions.Node(
            package='owl_vit_detector',
            executable='owl_detector.py',
            name='owl_detector',
            output='screen',
            parameters=[],
            arguments=[os.path.join(os.getenv('ROS_WS', '/home/rcasal/ros2_ws'), 'install/owl_vit_detector/lib/owl_vit_detector/owl_detector.py')]
        )


    ])
