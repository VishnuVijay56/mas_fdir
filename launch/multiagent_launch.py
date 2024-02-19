from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_drones',
            default_value=1),
        Node(
            package='agent_measurements',
	    executable='interagent_measurements',
	    name='interagent_measurements'),
	Node(
	    package='agent_measurements',
	    executable='fixed_frames_tf2_broadcaster',
	    name='fixed_frames_tf2_broadcaster'),
	Node(
	    package='agent_measurements',
	    executable='dynamic_frames_tf2_broadcaster',
	    name='dynamic_frames_tf2_broadcaster'),
	])
