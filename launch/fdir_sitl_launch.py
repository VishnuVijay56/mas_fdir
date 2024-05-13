import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():

	ros_ns_str = ['px4_1', 'px4_2', 'px4_3', 'px4_4', 'px4_5', 'px4_6', 'px4_7']

	ros_ns_str_cpp = '['
	for str in ros_ns_str:
		ros_ns_str_cpp += str
		ros_ns_str_cpp += ','
	ros_ns_str_cpp += ']'

	interagent_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gz_interagent_meas'), 'launch', 'interagent_meas.launch.py')),
        launch_arguments={
            'gz_world_name': "AbuDhabi",
            'gz_model_names': "[x500g_1, x500g_2, x500r_3, x500g_4, x500r_5, x500g_6, x500g_7]",
			'ros_ns': ros_ns_str_cpp,
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
