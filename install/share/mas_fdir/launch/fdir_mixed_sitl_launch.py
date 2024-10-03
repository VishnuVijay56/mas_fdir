import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

import yaml
from pathlib import Path
import numpy as np

def generate_launch_description():
    
	ros_ns_str = ['px4_1', 'px4_2', 'px4_3', 'px4_4', 'px4_5', 'px4_6', 'px4_7']

	# yaml_dict 	= 	yaml.safe_load(Path(os.path.join("/home/user/work/ros2_ws/src/px4-multiagent-offboard","config/mixedexp_3d_10a.yaml")).read_text())

	# ros_ns_str 	=	yaml_dict['ros_ns_str']

	ros_ns_str_cpp = '['
	for str in ros_ns_str:
		ros_ns_str_cpp += str
		ros_ns_str_cpp += ','
	ros_ns_str_cpp += ']'

	# formation   =   np.zeros((len(yaml_dict['ros_ns_str'])*3),dtype=np.float64)

	# for index, x in np.ndenumerate(formation):
	# 	if isinstance(yaml_dict['formation'][index[0]], str):
	# 		formation[index[0]]	=   eval(yaml_dict['formation'][index[0]])

	# 	elif isinstance(yaml_dict['formation'][index[0]], float) or isinstance(yaml_dict['formation'][index[0]], int):
	# 		formation[index[0]]	=   np.float64(yaml_dict['formation'][index[0]])

	# formation 	=	np.ndarray.tolist(formation)

	# adjacency   =   np.zeros((len(yaml_dict['ros_ns_str'])*len(yaml_dict['ros_ns_str'])),dtype=np.float64)

	# for index, x in np.ndenumerate(adjacency):
	# 	if isinstance(yaml_dict['adjacency'][index[0]], float) or isinstance(yaml_dict['adjacency'][index[0]], int):
	# 		adjacency[index[0]]	=   np.float64(yaml_dict['adjacency'][index[0]])

	# adjacency 	=	np.ndarray.tolist(adjacency)


	interagent_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gz_interagent_meas'), 'launch', 'interagent_meas.launch.py')),
        launch_arguments={
            'gz_world_name': "AbuDhabi",
            'gz_model_names': "[x500g_1, x500g_2, x500r_3, x500g_4, x500g_5, x500g_6, x500r_7]",
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
	
	foxglove_studio = ExecuteProcess(cmd=["foxglove-studio"])

	foxglove_bridge = IncludeLaunchDescription(XMLLaunchDescriptionSource(
        os.path.join(
        get_package_share_directory("mas_fdir"),
        "launch/foxglove_bridge.launch",))
    )
	return LaunchDescription([
		fdir_node,
		interagent_pub,
		foxglove_bridge,
		foxglove_studio
	])