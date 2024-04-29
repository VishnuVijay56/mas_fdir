#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Kartik Anand Pant, Minhyun CHo"
__contact__ = "kpant, mhcho@purdue.edu"

from re import M
import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from px4_msgs.msg import VehicleLocalPosition, VehicleGlobalPosition
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from geometry_msgs.msg import  TransformStamped
from tf2_ros import TransformBroadcaster

import navpy

from functools import partial

class DroneRelativePoseVisualizer(Node):
    def __init__(self,N):
        super().__init__("multidrone_relative_visualizer")
        # Configure subscritpions
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1,
        )

        # define number of drones
        self.N_drone    =   N

        self.array_subscribers    =   [{'local_pos_sub':None, 'global_pos_sub':None} for _ in range(self.N_drone)]

        for i in range(self.N_drone):

            self.ns             =   f'/px4_{i+1}'

            self.array_subscribers[i]['local_pos_sub']  =   self.create_subscription(
                VehicleLocalPosition,
                f'{self.ns}/fmu/out/vehicle_local_position',
                partial(self.local_position_callback,id=i),
                qos_profile)

            self.array_subscribers[i]['global_pos_sub'] =   self.create_subscription(
                VehicleGlobalPosition,
                f'{self.ns}/fmu/out/vehicle_global_position',
                partial(self.global_position_callback,id=i),
                qos_profile)

        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)


        # variables for subscribing navigation information
        self.local_pos_ned_list     =   [None for _ in range(self.N_drone)]
        self.global_ref_lla_list    =   [None for _ in range(self.N_drone)]

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)


    def local_position_callback(self,msg,id):
        self.local_pos_ned_list[id]     =   np.array([msg.x,msg.y,msg.z],dtype=np.float64)

    def global_position_callback(self,msg,id):
        self.global_ref_lla_list[id]    =   np.array([msg.lat,msg.lon,msg.alt],dtype=np.float64)

    def cmdloop_callback(self):

        if all([self.local_pos_ned_list[i] is not None for i in range(self.N_drone)]) and \
            all([self.global_ref_lla_list[i] is not None for i in range(self.N_drone)]):
                           
            for i in range(self.N_drone-1):
                t = TransformStamped()

                # Read message content and assign it to
                # corresponding tf variables
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'Drone 1'
                t.child_frame_id = f'Drone {i+2}'

                # Turtle only exists in 2D, thus we get x and y translation
                # coordinates from the message and set the z coordinate to 0
                temp    =   navpy.lla2ned(self.global_ref_lla_list[i+1][0], self.global_ref_lla_list[i+1][1],self.global_ref_lla_list[i+1][2],
                                          self.global_ref_lla_list[0][0], self.global_ref_lla_list[0][1], self.global_ref_lla_list[0][2],
                                          latlon_unit='deg', alt_unit='m', model='wgs84')

                t.transform.translation.x = temp[1]
                t.transform.translation.y = temp[0]
                t.transform.translation.z = temp[2]

                # For the same reason, turtle can only rotate around one axis
                # and this why we set rotation in x and y to 0 and obtain
                # rotation in z axis from the message
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0

                # Send the transformation
                self.tf_broadcaster.sendTransform(t)
    
    
       
def main(args=None):

    N_drone     =   6

    rclpy.init(args=args)

    mdrone_visualizer = DroneRelativePoseVisualizer(N_drone)

    rclpy.spin(mdrone_visualizer)

    mdrone_visualizer.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()