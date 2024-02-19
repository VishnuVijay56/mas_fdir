#!/usr/bin/env python

__author__ = "Vishnu Vijay"
__contact__ = "@purdue.edu"

import argparse
import sys

import rclpy
import numpy as np
from functools import partial

from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import navpy

from px4_msgs.msg import VehicleLocalPosition, VehicleGlobalPosition

from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import UInt8, Bool, Float32MultiArray

from tf2_ros import TransformBroadcaster


class FixedFramesBroadcaster(Node):

    def __init__(self, num_drones):

        super().__init__("fixed_frames_tf2_broadcaster")

        # Node Parameters
        self.num_drones = num_drones
        self.ref_origins_ned = [None] * num_drones # these will be wrt original drone 1 pos
        self.ref_origins_ned[0] = np.array([0, 0, 0])
        self.ref_origins_lla = [None] * num_drones # lla coordinates of each drone


        # set publisher and subscriber quality of service profile
        qos_profile_pub = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability = QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = 1
        )

        qos_profile_sub = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability = QoSDurabilityPolicy.VOLATILE,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = 5
        )


        # Define subscribers and publishers
        self.global_pos_sub = [None] * num_drones
        self.tf_fixed_broadcaster = [None] * num_drones
        for i in range(num_drones):
            sub_topic_name = "/px4_" + str(i+1) + "/fmu/out/vehicle_local_position"
            self.global_pos_sub[i] = self.create_subscription(
                VehicleLocalPosition,
                sub_topic_name,
                partial(self.local_position_callback, drone_ind=i),
                qos_profile_sub)
            
            self.tf_fixed_broadcaster[i] = TransformBroadcaster(self)





    ### Subscriber callbacks
        
    def local_position_callback(self,msg,drone_ind):
        try:
            if (self.ref_origins_lla[drone_ind] is None): # assign origin of each drone
                self.ref_origins_lla[drone_ind] = np.array([msg.ref_lat, msg.ref_lon, msg.ref_alt])
        
            if (self.ref_origins_ned[drone_ind] is None) and (self.ref_origins_lla[0] is not None): # if drone 1 origin is set, calc local pos wrt drone 1 origin
                ref_pos_ned = navpy.lla2ned(msg.ref_lat, msg.ref_lon, msg.ref_alt, 
                                            self.ref_origins_lla[0][0], self.ref_origins_lla[0][1], self.ref_origins_lla[0][2],
                                            latlon_unit='deg', alt_unit='m', model='wgs84')
                # self.get_logger().info("Drone #" + str(drone_ind) + ": " + str(ref_pos_ned))
                self.ref_origins_ned[drone_ind] = np.array(ref_pos_ned, dtype=np.float64).flatten()

        except:
           self.get_logger().info("Exception: Issue with getting position of drone #" + str(drone_ind))

        if (self.ref_origins_ned[drone_ind] is not None): # transmit ned coordinates of drone origin
                t = TransformStamped()

                # self.get_logger().info(str(self.ref_origins_ned[drone_ind][0]))

                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'world'
                t.child_frame_id = "drone_" + str(drone_ind + 1) + "_origin"
                t.transform.translation.x = float(self.ref_origins_ned[drone_ind][0])
                t.transform.translation.y = float(self.ref_origins_ned[drone_ind][1])
                t.transform.translation.z = float(self.ref_origins_ned[drone_ind][2])
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0

                self.tf_fixed_broadcaster[drone_ind].sendTransform(t)



### Main Func
    
def main():

    # Parse Arguments
    main_args = sys.argv[1:]
    num_drones = 1
    if (len(main_args) == 2) and (main_args[0] == "-n"):
        num_drones = int(main_args[1])
    
    # Node init
    rclpy.init(args=None)
    origins_broadcaster = FixedFramesBroadcaster(num_drones=num_drones)
    origins_broadcaster.get_logger().info("Initialized")

    # Spin Node
    rclpy.spin(origins_broadcaster)

    # Explicitly destroy node
    origins_broadcaster.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()