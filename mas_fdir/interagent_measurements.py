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


class Interagent_Measurements(Node):

    def __init__(self, num_drones):

        super().__init__("interagent_measurements")

        # Node Parameters
        self.timer_period = 0.5  # seconds
        self.num_drones = num_drones
        self.local_pos_ned = [None] * num_drones # these will be wrt original drone 1 pos
        self.ref_origins_lla = [None] * num_drones # lla coordinates of each drone
        

        # Plotting measurements
        self.window_size = 10
        self.meas_window = np.zeros((self.num_drones, self.num_drones, self.window_size))
        self.meas_head = [0] * num_drones # points to ind of oldest data


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
        self.local_pos_pub = [None] * num_drones
        self.publisher_timers = [None] * num_drones
        for i in range(num_drones):
            sub_topic_name = "/px4_" + str(i+1) + "/fmu/out/vehicle_global_position"
            self.global_pos_sub[i] = self.create_subscription(
                VehicleGlobalPosition,
                sub_topic_name,
                partial(self.global_position_callback, drone_ind=i),
                qos_profile_sub)
            
            pub_topic_name = "/px4_" + str(i+1) + "/fmu/out/interagent_distances"
            self.local_pos_pub[i] = self.create_publisher(
                Float32MultiArray,
                pub_topic_name,
                qos_profile_pub)
            
            self.publisher_timers[i] = self.create_timer(self.timer_period, 
                                            partial(self.local_measurements_callback, drone_ind=i))



    ### Subscriber callbacks
        
    def global_position_callback(self,msg,drone_ind):
        try:
            if (self.ref_origins_lla[drone_ind] is None): # assign origin of each drone
                self.ref_origins_lla[drone_ind] = np.array([msg.lat, msg.lon, msg.alt])
        
            if (self.ref_origins_lla[0] is not None): # if drone 1 origin is set, calc local pos wrt drone 1 origin
                ref_pos_ned = navpy.lla2ned(msg.lat, msg.lon, msg.alt, 
                                            self.ref_origins_lla[0][0], self.ref_origins_lla[0][1], self.ref_origins_lla[0][2],
                                            latlon_unit='deg', alt_unit='m', model='wgs84')
                # self.get_logger().info("Drone #" + str(drone_ind) + ": " + str(ref_pos_ned))
                self.local_pos_ned[drone_ind] = np.array(ref_pos_ned).flatten()
        except:
            self.get_logger().info("Exception: Issue with getting position of drone #" + str(drone_ind))



    ### Publisher callbacks
        
    def local_measurements_callback(self, drone_ind):
        # Init msg
        msg = Float32MultiArray()
        msg_array = [0.0] * self.num_drones

        # Create measurement array
        for i in range(self.num_drones):
            if i is drone_ind: # skip self
                continue

            try: # calc distance between drone [i] and drone [drone_ind]
                dist_i = self.distance(self.local_pos_ned[i], self.local_pos_ned[drone_ind])
            except:
                dist_i = 0
                self.get_logger().info("Exception: A variable is None type")
            
            self.meas_window[drone_ind, i, self.meas_head] = dist_i
            # self.get_logger().info("Drone " + str(drone_ind+1) + " to Drone " + str(i+1) + ": " + str(self.meas_window[drone_ind, i, :].flatten()))
            msg_array[i] = np.mean(self.meas_window[drone_ind, i, :].flatten())
        
        # Change head pointer
        self.meas_head[drone_ind] = (self.meas_head[drone_ind] + 1) % self.window_size

        # Send off measurements
        # self.get_logger().info("drone #" + str(drone_ind) + ": " + str(msg_array))
        msg.data = msg_array
        self.local_pos_pub[drone_ind].publish(msg)


    ### Measurement functions
    
    def displacement(self, pos1, pos2):
        return pos1 - pos2
    
    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2, 2)
    
    def bearing(self, pos1, pos2):
        return ((pos1 - pos2) / np.linalg.norm(pos1 - pos2))



### Main Func
    
def main():
    num_drones = 6
    # # Parse Arguments
    # main_args = sys.argv[1:]
    # num_drones = 1
    # if (len(main_args) == 2) and (main_args[0] == "-n"):
    #     num_drones = int(main_args[1])
    
    # Node init
    rclpy.init(args=None)
    interagent_measurer = Interagent_Measurements(num_drones=num_drones)
    # interagent_measurer.get_logger().info("Initialized")

    # Spin Node
    rclpy.spin(interagent_measurer)

    # Explicitly destroy node
    interagent_measurer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()