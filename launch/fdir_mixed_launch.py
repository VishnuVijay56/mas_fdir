from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    
	fdir_node = Node(
	    gz_world_name 
		
		package='mas_fdir',
	    executable='fdir_node_centralized',
	    name='fdir_node_centralized')
	
	return LaunchDescription([
		fdir_node
	])
