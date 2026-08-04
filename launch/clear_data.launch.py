from pathlib import Path

from ament_index_python.packages import get_package_prefix
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def get_config_path():
    install_prefix = Path(get_package_prefix('brover_recognizer'))
    source_config_path = (
        install_prefix.parent.parent
        / 'src'
        / 'brover_recognizer'
        / 'config'
        / 'recognizer.yaml'
    )
    if source_config_path.exists():
        return source_config_path

    return Path(get_package_share_directory('brover_recognizer'), 'config', 'recognizer.yaml')


def generate_launch_description():
    config_path = get_config_path()

    return LaunchDescription([
        Node(
            package='brover_recognizer',
            executable='clear_data',
            name='clear_data',
            output='screen',
            parameters=[str(config_path)],
        ),
    ])
