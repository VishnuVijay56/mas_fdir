#!/usr/bin/env python

__author__ = "Vishnu Vijay"
__contact__ = "@purdue.edu"

import argparse
import sys
import rclpy
import navpy
import numpy as np
import cvxpy as cp
import time

from functools import partial
from copy import deepcopy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from mas_fdir.my_agent import MyAgent

from px4_msgs.msg import VehicleLocalPosition, VehicleGlobalPosition, TrajectorySetpoint
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import UInt8, Bool, Float32MultiArray, Float32

import yaml
from pathlib import Path
import os


class Fault_Detector(Node):

    def __init__(self, debug, dim):

        super().__init__("fault_detector")


        ##  Arguments
        
        self.debug = debug
        self.dim = dim


        ## ROS2 Parameters
        
        # Declare
        self.declare_parameter('ros_ns', rclpy.Parameter.Type.STRING_ARRAY)
        
        # Get
        self.model_ns = self.get_parameter('ros_ns').value
        for ns in self.model_ns:
            self.get_logger().info(f"ROS Namespace: {ns}")


        ## Initialization - General Parameters and Empty Variables
        
        # General Parameters
        self.num_agents = len(self.model_ns)
        self.timer_period = 0.5  # seconds
        
        # Received information
        self.centroid_pos = None
        self.agent_local_pos = [None] * self.num_agents
        self.formation_msg = [None] * self.num_agents
        self.spawn_offset_pos = [None] * self.num_agents
        
        # Agents and Network
        self.agents = [None] * self.num_agents
        for id, _ in enumerate(self.agents):
            this_agent = MyAgent(agent_id=id)
            self.agents[id] = this_agent
        self.adj_matrix = None
        self.edge_list = None
        self.edges_from_adj()


        ##  Initialization - Measurements and Positions
        
        # Positions
        self.p_est = [self.agents[i].get_pos() for i in range(self.num_agents)]
        self.p_reported = [None] * self.num_agents
        
        # Error
        self.x_star = [np.zeros((self.dim, 1)) for i in range(self.num_agents)]
        # Measurements
        self.exp_meas = self.measurement_model()
        # Residuals
        self.residuals = [None] * self.num_agents
        

        ##  Initialization - Thresholds and Related Variables
        
        # Check Residual for Optimization Skip
        self.use_res_thresh = False
        # Extra Residual Threshold
        self.alpha = 5.0
        # Error Threshold
        self.err_thresh = 0.6
        
        # Dual Variable Threshold
        self.check_dual_var = False
        self.lam_lim = 2.0
        self.mu_lim = 3e-5
        self.lam_reset = [False] * self.num_agents
        self.mu_reset = [False] * self.num_agents

        # Linearized Measurement Model Threshold
        self.check_R_diff = False
        self.R_old = self.get_Jacobian_matrix()     # Older Jacobian of measurement model
        self.R = deepcopy(self.R_old)               # Newer Jacobian of measurement model
        self.R_norm_diff = 0.0
        self.R_diff_thresh = 3.0


        ##  Initialization - Optimization Parameters
        
        self.n_admm = 20
        self.curr_iter = 0
        self.rho = 0.5
        
        for id, agent in enumerate(self.agents):
            # CVX variables
            agent.init_x_cp(cp.Variable((self.dim, 1)))
            agent.init_w_cp(cp.Variable((self.dim, 1)), np.arange(self.num_agents))

            # Parameters
            agent.init_x_bar(np.zeros((self.dim, 1)))
            agent.init_lam(np.zeros((1, 1)), np.arange(self.num_agents))
            agent.init_mu(np.zeros((self.dim, 1)), np.arange(self.num_agents))
            agent.init_x_star(np.zeros((self.dim, 1)), np.arange(self.num_agents)) # own err is last elem
            agent.init_w(np.zeros((self.dim, 1)), np.arange(self.num_agents))


        ## Set publisher and subscriber quality of service profile
        
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

        ## Define - Different Callback Groups
        
        client_cb_group = MutuallyExclusiveCallbackGroup()
        timer_cb_group = MutuallyExclusiveCallbackGroup()


        ##  Define - Single Subscribers

        self.centroid_sub = self.create_subscription(
            PointStamped,
            '/px4_0/detector/vleader_position',
            self.sub_centroid_callback,
            qos_profile = qos_profile_sub,
            callback_group = client_cb_group)
        
        self.adj_matrix_sub = self.create_subscription(
            Float32MultiArray,
            '/px4_0/detector/adjacency',
            self.sub_adj_matrix,
            qos_profile = qos_profile_sub,
            callback_group = client_cb_group)
        
        self.formation_sub = self.create_subscription(
            Float32MultiArray,
            '/px4_0/detector/formation_config',
            self.sub_formation_callback,
            qos_profile = qos_profile_sub,
            callback_group = client_cb_group)
        

        ## Define - Single Publishers
                
        threshold_name = f"/px4_0/detector/threshold"
        self.thres_pub = self.create_publisher(
            Float32,
            threshold_name,
            qos_profile_pub)
        
        pub_avg_err_name = f"/px4_0/detector/avg_err"
        self.avg_err_pub = self.create_publisher(
            Float32,
            pub_avg_err_name,
            qos_profile_pub)


        ## Define - Lists of Subscribers/Publishers for Each Drone

        # Initialize List of Subs and Pubs for drones
        self.measurements_sub = [None] * self.num_agents
        self.local_pos_sub = [None] * self.num_agents
        self.spawn_offset_sub = [None] * self.num_agents
        self.err_pub = [None] * self.num_agents
        self.residual_pub = [None] * self.num_agents

        for i, name_space in enumerate(self.model_ns):
            # Subscribers
            sub_measurements_name = f"/{name_space}/detector/interagent_distances"
            self.measurements_sub[i] = self.create_subscription(
                Float32MultiArray,
                sub_measurements_name,
                partial(self.sub_measurements_callback, drone_ind=i),
                qos_profile = qos_profile_sub,
                callback_group = client_cb_group)
            
            # sub_local_pos_name = f"/px4_{i+1}/fmu/out/vehicle_local_position"
            sub_local_pos_name = f"/{name_space}/fmu/in/trajectory_setpoint"
            self.local_pos_sub[i] = self.create_subscription(
                TrajectorySetpoint,
                sub_local_pos_name,
                partial(self.sub_local_pos_callback, drone_ind=i),
                qos_profile = qos_profile_sub,
                callback_group = client_cb_group)

            sub_spawn_offset_name = f"/{name_space}/detector/spawn_offset"
            self.spawn_offset_sub[i] = self.create_subscription(
                PointStamped,
                sub_spawn_offset_name,
                partial(self.sub_spawn_offset_callback, drone_ind=i),
                qos_profile = qos_profile_sub,
                callback_group = client_cb_group)
            
            # Publishers
            pub_err_name = f"/{name_space}/detector/reconstructed_error"
            self.err_pub[i] = self.create_publisher(
                Float32MultiArray,
                pub_err_name,
                qos_profile_pub)
            
            pub_res_name = f"/{name_space}/detector/residual"
            self.residual_pub[i] = self.create_publisher(
                Float32,
                pub_res_name,
                qos_profile_pub
            )


        ##  Define: Callback Timer(s)
        
        self.admm_update_timer = self.create_timer(self.timer_period, 
                                            # self.admm_update)
                                            self.admm_update, callback_group=timer_cb_group)



    ### Subscriber callbacks
    
    # Sub: Inter-Agent Measurements
    def sub_measurements_callback(self, msg, drone_ind):
        
        try:
            iam_array = deepcopy(np.array(msg.data).flatten())
            self.agents[drone_ind].set_measurements(iam_array)
            #self.get_logger().info(f"{iam_array}: Agent" + str(drone_ind))

        except:
            self.get_logger().info("Exception: Issue with getting Inter-Agent Measurements of drone #" + str(drone_ind))
            
        return


    # Sub: Local Drone Position
    def sub_local_pos_callback(self, msg, drone_ind):
        
        try: # Extract msg
            self.agent_local_pos[drone_ind] = np.array([[msg.position[0]], [msg.position[1]], [msg.position[2]]]).reshape((self.dim, -1))

            # self.agent_local_pos[drone_ind] = np.array([[msg.x], [msg.y], [msg.z]]).reshape((self.dim, -1))
            # self.get_logger().info(f"Agent {drone_ind} Local Pos: {self.agent_local_pos[drone_ind].flatten()}")
        except:
            self.get_logger().info("Exception: Issue with getting Relative Position of drone #" + str(drone_ind))
            
        return
    

    # Sub: Spawn Offset Position
    def sub_spawn_offset_callback(self, msg, drone_ind):

        if self.spawn_offset_pos[drone_ind] is not None: # If spawn position is set, don't do anything
            return
        
        try:
            self.spawn_offset_pos[drone_ind] = np.array([[msg.point.x], [msg.point.y], [msg.point.z]]).reshape((self.dim, -1))
        except:
            self.get_logger().info("Exception: Issue with getting the spawn offset position of drone #" + str(drone_ind))
            
        return


    # Sub: Formation of Swarm
    def sub_formation_callback(self, msg):
        
        try:
            formation_arr = msg.data
            for id, _ in enumerate(self.agents):
                this_formation_arr = np.array(formation_arr[(id*self.dim):(self.dim*(id+1))]).reshape((self.dim, -1))
                self.formation_msg[id] = this_formation_arr
                self.agents[id].position = this_formation_arr
                
            self.p_est = [self.agents[i].get_pos() for i in range(self.num_agents)]
        except:
            self.get_logger().info("Exception: Issue with getting the formation config of the swarm")
            
        return


    # Sub: Virtual Leader NED Position
    def sub_centroid_callback(self, msg):
        
        try: # Not in debugging mode
            self.centroid_pos = np.array([[msg.point.x], [msg.point.y], [msg.point.z]]).reshape((self.dim, -1))
        except:
            self.get_logger().info("Exception: Issue with getting Centroid of Drone Swarm")
        
        return

    
    # Sub: Adjacency Matrix of the graph
    def sub_adj_matrix(self, msg):
        
        try: # Not in debugging mode
            self.adj_matrix = np.array(msg.data).reshape(self.num_agents, -1)
            self.edges_from_adj()
        except:
            self.get_logger().info("Exception: Issue with getting the adjacency matrix of the system")
        
        return



    ### Helper Functions

    # Help: Returns expected inter-agent measurements given position estimate and reconstructed error
    def measurement_model(self):
        # Check if edge list is set
        if (self.edge_list is None):
            self.get_logger().info("Cannot compute measurement model - edge list not set")
            return None
        
        est_meas = []

        for edge in self.edge_list:
            recons_pos1 = self.p_est[edge[0]] + self.x_star[edge[0]]
            recons_pos2 = self.p_est[edge[1]] + self.x_star[edge[1]]
            dist = self.distance(recons_pos1, recons_pos2)
            
            est_meas.append(dist)

        return est_meas
    

    # Help: Returns true inter-agent measurements according to Phi
    def true_measurements(self):
        # Check if edge list is set
        if (self.edge_list is None):
            self.get_logger().info("Cannot compute measurements - edge list not set")
            return
        
        true_meas = []
        
        if (self.debug): # If in debugging mode
            for edge in self.edge_list:
                id1 = edge[0]
                id2 = edge[1]

                true_pos1 = self.agents[id1].position
                true_pos2 = self.agents[id2].position

                dist = self.distance(true_pos1, true_pos2)
                true_meas.append(dist)
            
            return true_meas

        # print(self.edge_list)
        # print('here')
        for edge in self.edge_list: # Not in debugging mode
            try:
                true_meas.append(self.agents[edge[0]].measurements[edge[1]])
            except:
                self.get_logger().info(f"Exception: Measurement not set on edge {edge}")
                self.all_meas_set = False
                break

        return true_meas
    

    # Help: Computes distance between 2 points
    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2, 2)


    # Help: Computes row of R
    def get_Jacobian_row(self, edge):
        agent1_id = edge[0]
        agent2_id = edge[1]
        pos1 = self.p_est[agent1_id] + self.x_star[agent1_id]
        pos2 = self.p_est[agent2_id] + self.x_star[agent2_id]
        
        disp = (pos2 - pos1)
        R_k = np.zeros((1, self.dim*self.num_agents))

        dist = self.distance(pos1, pos2)
        R_k[:, self.dim*agent2_id:self.dim*(agent2_id + 1)] = disp.T  / dist
        R_k[:, self.dim*agent1_id:self.dim*(agent1_id + 1)] = -disp.T / dist

        return R_k


    # Help: Computes whole R matrix
    def get_Jacobian_matrix(self):
        # Check if edge list is set
        if (self.edge_list is None):
            self.get_logger().info("Cannot compute Jacobian matrix - edge list is not set")
            return None
             
        R = []
        for edge_ind, edge in enumerate(self.edge_list):
            R.append(self.get_Jacobian_row(edge))
        
        return R
    

    # Help: Computes matrix norm of the difference between old and new R matrices
    def get_Jacobian_matrix_norm_diff(self):

        old_matrix = self.R_old[0]
        for row in self.R_old[1:]:
            old_matrix = np.vstack((old_matrix, row))

        new_matrix = self.R[0]
        for row in self.R[1:]:
            new_matrix = np.vstack((new_matrix, row))

        self.R_norm_diff = np.linalg.norm(new_matrix - old_matrix)
        
        return
    

    # Help: Computes relative position of drones wrt centroid
    def get_rel_pos(self, id):
        # Compute
        rel_pos = self.agent_local_pos[id] - self.centroid_pos + self.spawn_offset_pos[id]
        # log_message = f"Agent {id} - " + \
        #               f"\n\tRadius: {np.linalg.norm(rel_pos)}" + \
        #               f"\n\tLocal Pos\t: {self.agent_local_pos[id].flatten()}" + \
        #               f"\n\tCentroid Pos\t: {self.centroid_pos.flatten()}" + \
        #               f"\n\tSpawn Offset\t: {self.spawn_offset_pos[id].flatten()}"                      
        # self.get_logger().info(log_message)
        #if id == 2:
        #self.get_logger().info(f"Agent {id} : Rel Pos {rel_pos.flatten()}")

        # Assign
        self.agents[id].position = rel_pos
        self.p_reported[id] = rel_pos
        
        return

    
    # Help: Reconstructs edge list from adjacency matrix and assigns these edges to the agents and the global edge list
    def edges_from_adj(self):
        # Check if adjacency matrix is set
        if (self.adj_matrix is None):
            self.get_logger().info("Cannot reconstruct edge list - adjacency matrix is not set")
            return
        
        new_global_edge_list = []

        # Iterate over rows of adjacency matrix
        for i in range(self.num_agents):
            nbr_id_list = []
            agent_edge_list = []
            for j in range(self.num_agents):
                if (i == j): # skip if on diagonal
                    continue

                elif (self.adj_matrix[i, j] == 1.0): # If adjacent
                    new_global_edge_list.append((i, j)) # global edge list
                    agent_edge_list.append(len(new_global_edge_list)-1) # agent edge list
                    nbr_id_list.append(j) # nbr list
            
            # Set neighbors and edges
            self.agents[i].set_neighbors(nbr_id_list)
            self.agents[i].set_edge_indices(agent_edge_list)
        
        # Set global edge list
        self.edge_list = new_global_edge_list
        
        return



    ### Publisher callbacks

    # Publish error vector
    def publish_err(self, id):
        # Init msg
        msg = Float32MultiArray()
        
        # Current error = outer loop error + inner loop error
        this_x = self.x_star[id].flatten() + self.agents[id].x_bar.flatten()
        this_x_norm = np.linalg.norm(this_x)
        this_x = np.hstack((this_x,np.array(this_x_norm/self.err_thresh)))
        # if id == 2:
        #     self.get_logger().info(f"Agent {id} - Error\t: {this_x}")

        # Send off error
        msg.data = this_x.tolist()
        self.err_pub[id].publish(msg)
        
        return

    
    # Publish agent residual
    def publish_residual(self, id):
        msg = Float32()
        # msg.data = self.residuals[id]
        # self.residual_pub[id].publish(msg)
        if id == 2:
            self.get_logger().info(f"Agent {id} - Residual\t: {self.residuals[id]}")
        
        return
    

    # Publish swarm residual
    # def publish_swarm_residual(self):
    #     msg = Float32()
    #     msg.data = np.average(np.array(self.residuals))
    #     self.avg_residual_pub.publish(msg)
    #     return


    def publish_avg_norm_err(self):
        msg = Float32()
        sum_norm_err = 0
        for id, _ in enumerate(self.agents):
            sum_norm_err += np.linalg.norm(self.x_star[id].flatten() + self.agents[id].x_bar.flatten())
        msg.data = (sum_norm_err / (self.err_thresh))
        self.avg_err_pub.publish(msg)
        
        return
        
    
    # Publish residual threshold
    def publish_threshold(self):
        msg = Float32()
        msg.data = 1.0#self.err_thresh #1/self.rho + self.alpha
        self.thres_pub.publish(msg)
        
        return
    

    # Calls the ADMM Update Step
    def admm_update(self):
        time_stamp = Clock().now()
        
        ##      Check           - See if variables are set before proceeding

        unset_var = False
        
        # Check Positions
        for id, agent in enumerate(self.agents):
            if (self.agent_local_pos[id] is None):
                self.get_logger().info(f"A local position is not set at {id}")
                unset_var = True
            if (self.spawn_offset_pos[id] is None):
                self.get_logger().info(f"A spawn offset is not set at {id}")
                unset_var = True
        if self.centroid_pos is None:
            self.get_logger().info("The centroid pos is not set")
            unset_var = True
        
        # Return if a position variable is not set
        if unset_var:
            return
        
        ##      Initialization  - Compute the relative distance of the agents wrt centroid
        for id, agent in enumerate(self.agents):
            self.get_rel_pos(id)


        ##      Initialization  - Get the true inter-agent measurements
        self.all_meas_set = True
        y = self.true_measurements()
        if not self.all_meas_set:
            return
        z = [(y[i] - self.exp_meas[i]) for i, _ in enumerate(y)]  


        ##      Minimization    - Primal Variable 1
        
        for id, agent in enumerate(self.agents):
            
            # Compute residual value if residual flag is set
            if self.use_res_thresh:
                # Thresholding: Summation over edges
                term1 = 0
                for i, edge_ind in enumerate(agent.get_edge_indices()):
                    R_k = self.R[edge_ind]
                    constr_c = R_k[:, self.dim*id:self.dim*(id+1)] @ (-agent.x_star[id]) - z[edge_ind]
                    for nbr_id in agent.get_neighbors():
                        constr_c += R_k[:, self.dim*id:self.dim*(id+1)] @ agent.w[nbr_id]
                    
                    term1 += R_k[:, self.dim*id:self.dim*(id+1)].T @ (constr_c + (agent.lam[self.edge_list[edge_ind]] / self.rho))
                
                # Thresholding: Summation over neighbors
                term2 = 0
                for nbr_id in agent.get_neighbors():
                    constr_d = -agent.x_star[id] - agent.w[nbr_id]
                    term2 += constr_d + (agent.mu[nbr_id] / self.rho)
                
                # Thresholding: Check that residual is under threshold
                this_res = np.linalg.norm(term1 + term2)
                

                self.residuals[id] = this_res
            
            # Check if:     (1) residual flag is set
            #               (2) value of residual falls under threshold
            # If either is false, solve optimization problem
            if (self.use_res_thresh) and (this_res <= ((1/self.rho) + self.alpha)): # skip optimization
                agent.x_bar = deepcopy(-agent.x_star[id])
                # self.get_logger().info(f"case: skip optimization")
            else:
            # Optimization: Solve minimization problem for x_bar if over threshold
                # self.get_logger().info(f"Optimization for agent {id}")
                objective = cp.norm(agent.x_star[id] + agent.x_cp)
                
                # Summation for c() constraint
                for _, edge_ind in enumerate(agent.get_edge_indices()): 
                    constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ agent.x_cp - z[edge_ind]
                    for nbr_id in agent.get_neighbors():
                        constr_c += self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.agents[nbr_id].w[id]
                    
                    objective += ((self.rho/2)*cp.power(cp.norm(constr_c), 2)
                                    + agent.lam[self.edge_list[edge_ind]].T @ (constr_c))
                
                # Summation for d() constraint
                for _, nbr_id in enumerate(agent.get_neighbors()): 
                    constr_d = agent.x_cp - agent.w[nbr_id]
                    objective += ((self.rho/2)*cp.power(cp.norm(constr_d), 2)
                                + agent.mu[nbr_id].T @ (constr_d))
                    
                prob1 = cp.Problem(cp.Minimize(objective), [])
                prob1.solve(verbose=False)
                # if prob1.status != cp.OPTIMAL:
                #     self.get_logger().warning(f"~ERROR~ Problem 1: Agent {id} - Optimization Status {prob1.status} @ {self.curr_iter}th iteration")

                agent.x_bar = deepcopy(np.array(agent.x_cp.value).reshape((-1, 1)))
                # self.get_logger().info(f"case: do optimization")


        ##      Minimization    - Primal Variable 2
        for id, agent in enumerate(self.agents):
            objective = cp.norm(agent.x_star[id] + agent.x_bar)

            # Summation for c() constraint
            for edge_ind in agent.get_edge_indices(): 
                constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c = constr_c + self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.agents[nbr_id].w_cp[id]
                
                objective += ((self.rho/2)*cp.power(cp.norm(constr_c), 2)
                                + agent.lam[self.edge_list[edge_ind]].T @ (constr_c))
            
            # Summation for d() constraint
            for nbr_id in agent.get_neighbors():
                constr_d = agent.x_bar - agent.w_cp[nbr_id]
                objective += ((self.rho/2)*cp.power(cp.norm(constr_d), 2)
                              + agent.mu[nbr_id].T @ (constr_d))
                
            prob2 = cp.Problem(cp.Minimize(objective), [])
            prob2.solve(verbose=False)
            #if prob2.status != cp.OPTIMAL:
                #self.get_logger().warning(f"~ERROR~ Problem 2: Agent {id} - Optimization Status {prob2.status} @ {self.curr_iter}th iteration")

            for _, nbr_id in enumerate(agent.get_neighbors()):
                agent.w[nbr_id] = deepcopy(np.array(agent.w_cp[nbr_id].value).reshape((-1, 1)))


        ##      Multipliers     - Update Lagrangian Multipliers of Minimization Problem
        for id, agent in enumerate(self.agents):
            
            # Summation for c() constraint
            for _, edge_ind in enumerate(agent.get_edge_indices()):
                constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.agents[nbr_id].w[id]
                new_lam = agent.lam[self.edge_list[edge_ind]] + self.rho * constr_c
                agent.lam[self.edge_list[edge_ind]] = deepcopy(new_lam)
            
                #self.get_logger().info(f" ---> Agent: {id}, EdgeIDX: {edge_ind}, Constraints c: {constr_c}")
                    
                # Check if cold start is required
                if (np.linalg.norm(new_lam) >= self.lam_lim):
                    self.lam_reset[id] = True

            # Summation for d() constraint
            for _, nbr_id in enumerate(agent.get_neighbors()):
                constr_d = agent.x_bar - agent.w[nbr_id]
                new_mu = agent.mu[nbr_id] + self.rho * constr_d
                agent.mu[nbr_id] = deepcopy(new_mu)
                
                #self.get_logger().info(f" ---> Agent: {id}, Neighbor: {nbr_id}, Constraints d: {constr_d}")

                # Check if cold start is required
                if (np.linalg.norm(new_mu) >= self.mu_lim):
                    self.mu_reset[id] = True


        ##      Update          - SCP Outer Loop Handling
        if ((self.curr_iter % self.n_admm) == 0) and ((self.curr_iter - self.n_admm) >= 0):
            #self.get_logger().info(" ---> SCP Step: Relinearization, Error Vector Updating, and Primal Variable w Resetting")

            ##          Update          - Post ADMM Subroutine Handling

            # Linearized Measurement Model
            self.exp_meas = self.measurement_model()
            self.R_old = deepcopy(self.R)
            self.R = self.get_Jacobian_matrix()
            self.get_Jacobian_matrix_norm_diff()
            
            for agent_id, agent in enumerate(self.agents): 
                
                # Update Error Vectors
                for list_ind, nbr_id in enumerate(agent.get_neighbors()):
                    agent.x_star[nbr_id] = agent.x_star[nbr_id] + self.agents[nbr_id].x_bar
                
                agent.x_star[agent_id] = agent.x_star[agent_id] + agent.x_bar
                self.x_star[agent_id] = agent.x_star[agent_id]
                
                # Update position
                self.p_est[agent_id] = self.p_reported[agent_id] + self.x_star[agent_id]
                # print(f" -> Agent {agent_id} Pos: {self.p_est[agent_id].flatten()}")

                # Check if: (1) the norm difference in R exceed prescribed threshold OR 
                #           (2) a reset flag for dual variables was set
                if (self.check_R_diff and (self.R_norm_diff >= self.R_diff_thresh)) or \
                        (self.check_dual_var and (self.lam_reset[agent_id] or self.mu_reset[agent_id])):
                    #self.get_logger().info(f"RESET DUAL: Agent {agent_id} at Iteration {self.curr_iter}")
                    agent.init_lam(np.zeros((1, 1)), np.arange(self.num_agents))
                    agent.init_mu(np.zeros((self.dim, 1)), np.arange(self.num_agents))
                    self.mu_reset[agent_id] = False
                    self.lam_reset[agent_id] = False
                    

            # Reset primal variables w after relinearization
            for agent in self.agents:
                agent.init_w(np.zeros((self.dim, 1)), agent.get_neighbors())


        ##      End         - Publish error and residuals, increment current iteration, and return
        for id, agent in enumerate(self.agents):
            self.publish_err(id)
            self.publish_residual(id)
        self.publish_threshold()
        self.publish_avg_norm_err()

        self.curr_iter += 1
        total_time  = Clock().now() - time_stamp
        self.get_logger().info(f"ADMM time: {total_time}")
        return
    

    
### Main Func
    
def main():
    # Initializations
    DEBUG = False
    dim = 3
    # num_agents = 7
    # Agents = [None] * num_agents
    # Formation =     [
    #                  np.array([[3.0*np.cos(np.pi/180*0),     3.0*np.sin(np.pi/180*0),    0]]).T,
    #                  np.array([[3.0*np.cos(np.pi/180*60),    3.0*np.sin(np.pi/180*60),   0]]).T,
    #                  np.array([[3.0*np.cos(np.pi/180*120),   3.0*np.sin(np.pi/180*120),  0]]).T,
    #                  np.array([[3.0*np.cos(np.pi/180*180),   3.0*np.sin(np.pi/180*180),  0]]).T,
    #                  np.array([[3.0*np.cos(np.pi/180*240),   3.0*np.sin(np.pi/180*240),  0]]).T,
    #                  np.array([[3.0*np.cos(np.pi/180*300),   3.0*np.sin(np.pi/180*300),  0]]).T]
    
    # Formation   =   [
    #                  np.array([[4.0,   0.0,  2.0]]).T,
    #                  np.array([[-4.0,  0.0,  1.0]]).T,
    #                  np.array([[2.0,   2.0,  -0.5]]).T,
    #                  np.array([[2.0,  -2.0,  -0.5]]).T,
    #                  np.array([[-2.0, -2.0,  0.5]]).T,
    #                  np.array([[-2.0,  2.0,  0.5]]).T]

    # Formation =     [
    #                  np.array([[2.0*np.cos(np.pi/180*60),     2.0*np.sin(np.pi/180*60),    0.0]]).T,
    #                  np.array([[2.0*np.cos(np.pi/180*180),    2.0*np.sin(np.pi/180*180),   0.0]]).T,
    #                  np.array([[2.0*np.cos(np.pi/180*300),   2.0*np.sin(np.pi/180*300),  0.0]]).T,
    #                  np.array([[1.0*np.cos(np.pi/180*0),   1.0*np.sin(np.pi/180*0),  -0.1]]).T,
    #                  np.array([[1.0*np.cos(np.pi/180*120),   1.0*np.sin(np.pi/180*120),  -0.1]]).T,
    #                  np.array([[1.0*np.cos(np.pi/180*240),   1.0*np.sin(np.pi/180*240),  -0.1]]).T,
    #                  np.array([[0.0,  0.0, 0.1]]).T]
    
    # Formation =     [
    #                  np.array([[2.0*np.cos(np.pi/180*60),     2.0*np.sin(np.pi/180*60),    0.0]]).T,
    #                  np.array([[2.0*np.cos(np.pi/180*180),    2.0*np.sin(np.pi/180*180),   0.0]]).T,
    #                  np.array([[2.0*np.cos(np.pi/180*300),   2.0*np.sin(np.pi/180*300),  0.0]]).T,
    #                  np.array([[2.0*np.cos(np.pi/180*0),     2.0*np.sin(np.pi/180*0),    -0.25]]).T,
    #                  np.array([[2.0*np.cos(np.pi/180*120),    2.0*np.sin(np.pi/180*120),   -0.25]]).T,
    #                  np.array([[2.0*np.cos(np.pi/180*240),   2.0*np.sin(np.pi/180*240),  -0.25]]).T,
    #                  np.array([[1.0*np.cos(np.pi/180*0),   1.0*np.sin(np.pi/180*0),  0.0]]).T,
    #                  np.array([[1.0*np.cos(np.pi/180*120),   1.0*np.sin(np.pi/180*120),  0.0]]).T,
    #                  np.array([[1.0*np.cos(np.pi/180*240),   1.0*np.sin(np.pi/180*240),  0.0]]).T,
    #                  np.array([[0.0,  0.0, 0.1]]).T]


    # Adjacency =     np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    
    # Adjacency   =   np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    
    # Adjacency   =   np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    #                           [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],], dtype=np.float64)
    

    # yaml_dict 	= 	yaml.safe_load(Path(os.path.join("/home/user/work/ros2_ws/src/px4-multiagent-offboard","config/mixedexp_3d_10a.yaml")).read_text())

    # formation   =   [np.zeros((3,1),dtype=np.float64) for i in range(len(yaml_dict['ros_ns_str']))]

    # for index, x in enumerate(formation):
    #     if isinstance(yaml_dict['formation'][3*index:3], str):
        #     print(formation[index[0]])
			# formation[index[0]]	=   eval(yaml_dict['formation'][index[0]])

	# 	elif isinstance(yaml_dict['formation'][index[0]], float) or isinstance(yaml_dict['formation'][index[0]], int):
	# 		formation[index[0]]	=   np.float64(yaml_dict['formation'][index[0]])

	# formation 	=	np.ndarray.tolist(formation)

	# adjacency   =   np.zeros((len(yaml_dict['ros_ns_str'])*len(yaml_dict['ros_ns_str'])),dtype=np.float64)

	# for index, x in np.ndenumerate(adjacency):
	# 	if isinstance(yaml_dict['adjacency'][index[0]], float) or isinstance(yaml_dict['adjacency'][index[0]], int):
	# 		adjacency[index[0]]	=   np.float64(yaml_dict['adjacency'][index[0]])

	# adjacency 	=	np.ndarray.tolist(adjacency)



    
    # Edges =         [(0,1), (0,2), (0,3), 
    #                  (0,4), (0,5), (1,2),
    #                  (1,3), (1,4), (1,5),
    #                  (2,3), (2,4), (2,5),
    #                  (3,4), (3,5), (4,5),
                    
    #                  (1,0), (2,0), (3,0), 
    #                  (4,0), (5,0), (2,1),
    #                  (3,1), (4,1), (5,1),
    #                  (3,2), (4,2), (5,2),
    #                  (4,3), (5,3), (5,4)] # these edges are directed
    
    # Edges =         [(0,1), (0,2), (0,3), 
    #                  (0,4), (0,5), (0,6),
    #                  (1,2), (1,3), (1,4),
    #                  (1,5), (1,6), (2,3),
    #                  (2,4), (2,5), (2,6),
    #                  (3,4), (3,5), (3,6),
    #                  (4,5), (4,6), (5,6),
                    
    #                  (1,0), (2,0), (3,0), 
    #                  (4,0), (5,0), (6,0),
    #                  (2,1), (3,1), (4,1),
    #                  (5,1), (6,1), (3,2),
    #                  (4,2), (5,2), (6,2),
    #                  (4,3), (5,3), (6,3),
    #                  (5,4), (6,4), (6,5)] # these edges are directed
    
    # Edges =         [(0,1), (0,2), (0,3), 
    #                  (0,4), (0,5), (0,6),
    #                  (0,7), (0,8), (0,9),
    #                  (1,2), (1,3), (1,4),
    #                  (1,5), (1,6), (1,7),
    #                  (1,8), (1,9), (2,3),
    #                  (2,4), (2,5), (2,6),
    #                  (2,7), (2,8), (2,9),
    #                  (3,4), (3,5), (3,6),
    #                  (3,7), (3,8), (3,9),
    #                  (4,5), (4,6), (4,7),
    #                  (4,8), (4,9), (5,6),
    #                  (5,7), (5,8), (5,9),
    #                  (6,7), (6,8), (6,9),
    #                  (7,8), (7,9), (8,9)] # these edges are directed
    
    # Edges_flip  =   deepcopy(Edges)
    # for idx, dir_edge in enumerate(Edges_flip):
    #     Edges_flip[idx]    =   dir_edge[::-1]

    # Edges   =   Edges+Edges_flip

    # Graph
    # for id, _ in enumerate(Agents): # Create agent objects with nbr and edge lists
    #     this_agent = MyAgent(agent_id=id,
    #                          init_position=Formation[id])
        
    #     nbr_id_list = []
    #     nbr_ptr_list = []
    #     edge_list = []

    #     for edge_ind, edge in enumerate(Edges):
    #         if id == edge[0]:
    #             nbr_id_list.append(edge[1])
    #             edge_list.append(edge_ind)
        
    #     this_agent.set_neighbors(nbr_id_list)
    #     this_agent.set_edge_indices(edge_list)
        
    #     Agents[id] = this_agent

    # # Add error vector
    # if DEBUG:
    #     faulty_id   =   np.random.randint(0, high=num_agents)
    #     fault_vec   =   0.5*np.random.rand(dim, 1) # np.array([[0.0, 0, 0]]).T #
    #     Agents[faulty_id].faulty = True
    #     Agents[faulty_id].error_vector = fault_vec
    #     print("\n\n================================================================")
    #     print(f"Faulty Agent:   {faulty_id}")
    #     print(f"Faulty Vector:  {fault_vec.flatten()}")
    #     for id, pos in enumerate(Formation):
    #         print(f" -> Agent {id} True Pos: {pos.flatten()}")
    #     print("================================================================\n\n")

    
    # Node init
    rclpy.init(args=None)
    fault_detector = Fault_Detector(debug=DEBUG, dim=dim)#, agents=Agents, adj_matrix=Adjacency)
    # interagent_measurer.get_logger().info("Initialized")
    executer = MultiThreadedExecutor()
    executer.add_node(fault_detector)

    # Spin Node
    try:
        # rclpy.spin(fault_detector)
        executer.spin()
    except KeyboardInterrupt:
        print("\n ~~~ Keyboard Interrupt Called ~~~")
        # if (DEBUG):
        #     print("\n\nKeyboardInterrupt Called")
        #     print("\n\n================================================================")
        #     print(f"Faulty Agent:   {faulty_id}")
        #     print(f"Faulty Vector:  {fault_vec.flatten()}")
        #     for id, pos in enumerate(Formation):
        #         print(f" -> Agent {id} True Pos: {pos.flatten()}")
        #     print("================================================================\n\n")

    # Explicitly destroy node
    fault_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()