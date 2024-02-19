#!/usr/bin/env python

__author__ = "Vishnu Vijay"
__contact__ = "@purdue.edu"

import argparse
import sys

import rclpy
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import navpy

from px4_msgs.msg import VehicleLocalPosition, VehicleGlobalPosition

from geometry_msgs.msg import PointStamped
from std_msgs.msg import UInt8, Bool, Float32MultiArray

class Measurement_Plotter(Node):

    def __init__(self, num_drones):

        super().__init__("interagent_measurements")

        # Node Parameters
        self.timer_period = 0.5  # seconds
        self.timer = self.create_timer(self.timer_period, self.local_measurements_callback)
        self.num_drones = num_drones
        self.local_pos_ned = [None] * num_drones # these will be wrt original drone 1 pos
        self.ref_pos_ned = [None] * num_drones # position of drone ref wrt original drone 1 pos
        self.ref_pos_ned[0] = np.array([0, 0, 0], dtype=np.float64) # drone 1 pos is origin
        

        # Plotting measurements
        self.window_size = 10
        self.meas_window = np.zeros((self.num_drones, self.num_drones, self.window_size))
        self.meas_head = 0 # points to ind of oldest data
        
        self.fig = plt.figure()
        self.ax = [ [None]*self.num_drones for i in range(self.num_drones)]
        # self.meas_window = [ [[]]*self.num_drones for i in range(self.num_drones)]
        for i in range(self.num_drones):
            for j in range(self.num_drones):
                ind = i + j*self.num_drones + 1
                self.ax[i][j] = self.fig.add_subplot(self.num_drones, self.num_drones, ind)
        



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


        # Define subscribers
        self.local_pos_sub = [None] * num_drones
        for i in range(num_drones):
            topic_name = "/px4_" + str(i+1) + "/fmu/out/vehicle_local_position"
            
            self.local_pos_sub[i] = self.create_subscription(
                VehicleLocalPosition,
                topic_name,
                partial(self.local_position_callback, drone_label=i),
                qos_profile_sub)

        
        # Define publishers
        self.publisher_measurements = self.create_publisher(
            Float32MultiArray,
            'interagent_measurements',
            qos_profile_pub
        )



    ### Subscriber callbacks

    def local_position_callback(self,msg,drone_label):
        
        if (self.ref_pos_ned[drone_label] is None) and (self.ref_pos_ned[0] is not None):
            self.ref_pos_ned[drone_label] = 0

        ref_pos_ned = navpy.lla2ned(msg.ref_lat, msg.ref_lon, msg.ref_alt, 
                                    msg.ref_lat, msg.ref_lon, msg.ref_alt,
                                    latlon_unit='deg', alt_unit='m', model='wgs84')
        self.local_pos_ned[drone_label] = np.array(ref_pos_ned).flatten() + \
                                np.array([msg.x, msg.y, msg.z],dtype=np.float64)



    ### Publisher callbacks
        
    def measurements_callback(self):
        # Init msg
        msg = Float32MultiArray()

        # Create measurement matrix
        # meas_arr = np.zeros((self.num_drones, self.num_drones))
        for i in range(self.num_drones):
            for j in range(i+1, self.num_drones):
                try:
                    dist_ij = self.distance(self.local_pos_ned[i], self.local_pos_ned[j])
                except:
                    dist_ij = 0
                    self.get_logger().info("Exception: A variable is None type")
                self.meas_window[i, j, self.meas_head] = dist_ij
                self.meas_window[j, i, self.meas_head] = dist_ij
        # Change head pointer
        self.meas_head = (self.meas_head + 1) % self.window_size

        # Measurement Window
        self.plot_measurements(1)
        


    ### Measurement functions
    
    def displacement(self, pos1, pos2):
        return pos1 - pos2
    
    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2, 2)
    
    def bearing(self, pos1, pos2):
        return ((pos1 - pos2) / np.linalg.norm(pos1 - pos2))
    

    ### Plotting measurements

    def plot_measurements(self, i):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.renderer
        xs = np.arange(self.window_size)
        for i in range(self.num_drones):
            for j in range(self.num_drones):
                ys = np.hstack((self.meas_window[i, j, self.meas_head:].flatten(), 
                               self.meas_window[i, j, :self.meas_head].flatten()))
                
                out_str = "(" + str(i) + ", " + str(j) + ")"
                # self.get_logger().info(out_str)
                # self.get_logger().info(str(ys))
                # self.get_logger().info(str(self.ax))
                # this_list = self.ax[i]
                # this_list = this_list[j]
                self.ax[i][j].clear()
                self.ax[i][j].bar(xs, ys)
                ax_title = "Distance between drone " + str(i) + " and drone " + str(j)
                self.ax[i][j].set_title(ax_title)
                self.ax[i][j].set_xlabel("Window Index")
                self.ax[i][j].set_ylabel("Distance (m)")
                # self.ax[i][j].set_ylim(bottom=0, top = 2)
                self.ax[i][j].draw(renderer)
        plt.draw()
        plt.pause(0.001)



### Main Func
    
def main():

    # Parse Arguments
    main_args = sys.argv[1:]
    num_drones = 1
    if (len(main_args) == 2) and (main_args[0] == "-n"):
        num_drones = int(main_args[1])
    
    # Node init
    rclpy.init(args=None)
    interagent_measurer = Interagent_Measurements(num_drones=num_drones)
    interagent_measurer.get_logger().info("Initialized")

    # Spin Node
    plt.ion()
    plt.show()
    rclpy.spin(interagent_measurer)

    # Explicitly destroy node
    interagent_measurer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()