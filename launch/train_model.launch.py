from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='brover_recognizer',
            executable='train_model',
            name='train_model',
            output='screen',
        ),
    ])
