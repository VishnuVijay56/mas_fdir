import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():

	ros_ns_str = "[px4_1, px4_2, px4_3, px4_4, px4_5, px4_6, px4_7]"

	interagent_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gz_interagent_meas'), 'launch', 'interagent_meas.launch.py')),
        launch_arguments={
            'gz_world_name': "AbuDhabi",
            'gz_model_names': "[x500_1, x500_2, x500_3, x500_4, x500_5, x500_6, x500_7]",
			'ros_ns': ros_ns_str,
        }.items(),
    )

	fdir_node = Node(
		package='mas_fdir',
	    executable='fdir_node_centralized',
	    name='fdir_node_centralized',
		parameters = [
			{'ros_ns' : ros_ns_str}
		])
	
	return LaunchDescription([
		fdir_node,
		interagent_pub
	])
