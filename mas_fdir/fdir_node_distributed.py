#!/usr/bin/env python

__author__ = "Vishnu Vijay"
__contact__ = "@purdue.edu"

import argparse
import sys
import rclpy
import numpy as np
import cvxpy as cp

from functools import partial

from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import navpy

from px4_msgs.msg import VehicleLocalPosition, VehicleGlobalPosition

from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import UInt8, Bool, Float32MultiArray


class Fault_Detector(Node):

    def __init__(self, num_drones):

        super().__init__("fault_detector")

        # Initialize Graph Structure Here
        self.num_neighbors = np.zeros(num_drones)


        # Node Parameters
        self.timer_period = 0.01  # seconds
        self.num_drones = num_drones
        self.local_pos_ned = [None] * num_drones # these will be wrt original drone 1 pos
        self.ref_origins_lla = [None] * num_drones # lla coordinates of each drone
        self.update_step = [0] * num_drones # Indicates the update step the node is on


        # Set publisher and subscriber quality of service profile
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
            depth = 1
        )


        # Define subscribers and publishers
        self.primal1_sub = [None] * num_drones
        self.primal2_sub = [None] * num_drones
        
        self.primal1_pub = [None] * num_drones
        self.primal2_pub = [None] * num_drones

        self.primal1_pub_timers = [None] * num_drones
        self.primal2_pub_timers = [None] * num_drones
        self.dual_update_timers = [None] * num_drones
        
        for i in range(num_drones):
            # Subscribers
            sub_topic1_name = "/px4_" + str(i+1) + "/fmu/out/primal1_variables"
            self.primal1_sub[i] = self.create_subscription(
                Float32MultiArray,
                sub_topic1_name,
                partial(self.sub_primal1_callback, drone_ind=i),
                qos_profile_sub)
            sub_topic2_name = "/px4_" + str(i+1) + "/fmu/out/primal2_variables"
            self.primal2_sub[i] = self.create_subscription(
                Float32MultiArray,
                sub_topic2_name,
                partial(self.sub_primal2_callback, drone_ind=i),
                qos_profile_sub)
            
            # Publishers
            pub_topic1_name = "/px4_" + str(i+1) + "/fmu/out/primal1_variables"
            self.primal1_pub[i] = self.create_publisher(
                Float32MultiArray,
                pub_topic1_name,
                qos_profile_pub)
            pub_topic2_name = "/px4_" + str(i+1) + "/fmu/out/primal2_variables"
            self.primal2_pub[i] = self.create_publisher(
                Float32MultiArray,
                pub_topic2_name,
                qos_profile_pub)
            
            # Callback Timers
            self.primal1_pub_timers[i] = self.create_timer(self.timer_period, 
                                                partial(self.pub_primal1_callback, drone_ind=i))
            self.primal2_pub_timers[i] = self.create_timer(self.timer_period, 
                                                partial(self.pub_primal2_callback, drone_ind=i))
            self.dual_update_timers[i] = self.create_timer(self.timer_period, 
                                                partial(self.dual_var_update, drone_ind=i))



    ### Subscriber callbacks
    
    # Step 1
    def sub_primal1_callback(self, msg, drone_ind):
        # Return if previous steps not done
        if self.update_step[drone_ind] != 1:
            return
        
        try:
            print()
        except:
            self.get_logger().info("Exception: Issue with getting Primal Update 1 of drone #" + str(drone_ind))


    # Step 3
    def sub_primal2_callback(self, msg, drone_ind):
        # Return if previous steps not done
        if self.update_step[drone_ind] != 3:
            return

        try:
            print()
        except:
            self.get_logger().info("Exception: Issue with getting Primal Update 2 of drone #" + str(drone_ind))


    ### Publisher callbacks

    # Step 0
    def pub_primal1_callback(self, drone_ind):
        # Return if previous steps not done
        if self.update_step[drone_ind] != 0:
            return

        # Init msg
        msg = Float32MultiArray()
        msg_array = [0.0] * self.num_neighbors[drone_ind]

        # SOLVE Primal Problem 1
        self.primal1_update()

        # SEND out the necessary variables

        # self.get_logger().info("drone #" + str(drone_ind) + ": " + str(msg_array))
        msg.data = msg_array
        self.local_pos_pub[drone_ind].publish(msg)
        
        return

    
    # Step 2
    def pub_primal2_callback(self, drone_ind):
        # Return if previous steps not done
        if self.update_step[drone_ind] != 2:
            return

        # Init msg
        msg = Float32MultiArray()
        msg_array = [0.0] * self.num_neighbors[drone_ind]

        # SOLVE Primal Problem 2
        self.primal2_update()

        # SEND out the necessary variables

        # self.get_logger().info("drone #" + str(drone_ind) + ": " + str(msg_array))
        msg.data = msg_array
        self.local_pos_pub[drone_ind].publish(msg)
        
        return
    

    # Step 4 - Doesn't actually publish anything
    def dual_var_update(self, drone_ind):
        # Return if previous steps not done
        if self.update_step[drone_ind] != 4:
            return

        print()
        return



    ### Helper Functions

    def primal1_update(self):
        print()
        return
    

    def primal2_update(self):
        print()
        return



### Main Func
    
def main():

    # Parse Arguments
    main_args = sys.argv[1:]
    num_drones = 1
    if (len(main_args) == 2) and (main_args[0] == "-n"):
        num_drones = int(main_args[1])
    
    # Node init
    rclpy.init(args=None)
    fault_detector = Fault_Detector(num_drones=num_drones)
    # interagent_measurer.get_logger().info("Initialized")

    # Spin Node
    rclpy.spin(fault_detector)

    # Explicitly destroy node
    fault_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()