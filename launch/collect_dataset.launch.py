from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from pathlib import Path


def generate_launch_description():
    config_path = Path(
        get_package_share_directory('brover_recognizer'),
        'config',
        'recognizer.yaml',
    )

    return LaunchDescription([
        Node(
            package='brover_recognizer',
            executable='dataset_collector',
            name='dataset_collector',
            output='screen',
            emulate_tty=True,
            parameters=[str(config_path)],
        ),
    ])
