from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_path = Path(
        get_package_share_directory('brover_recognizer'),
        'config',
        'recognizer.yaml',
    )

    return LaunchDescription([
        Node(
            package='brover_recognizer',
            executable='prepare_dataset',
            name='prepare_dataset',
            output='screen',
            parameters=[str(config_path)],
        ),
    ])
