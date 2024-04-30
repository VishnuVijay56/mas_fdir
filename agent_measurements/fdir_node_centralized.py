#!/usr/bin/env python

__author__ = "Vishnu Vijay"
__contact__ = "@purdue.edu"

import argparse
import sys
import rclpy
import navpy
import numpy as np
import cvxpy as cp

from functools import partial
from copy import deepcopy
from tqdm import tqdm
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from my_agent import MyAgent

from px4_msgs.msg import VehicleLocalPosition, VehicleGlobalPosition, TrajectorySetpoint
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import UInt8, Bool, Float32MultiArray


class Fault_Detector(Node):

    def __init__(self, debug, dim, agents, edge_list):

        super().__init__("fault_detector")


        ##  Arguments
        self.debug = debug
        self.agents = agents
        self.num_agents = len(agents)
        self.edge_list = edge_list
        self.dim = dim


        ##  Initialization - Optimization Parameters
        self.n_admm = 20
        self.curr_iter = 0
        self.rho = 1.0
        for agent_id, agent in enumerate(agents):
            # CVX variables
            agent.init_x_cp(cp.Variable((self.dim, 1)))
            agent.init_w_cp(cp.Variable((self.dim, 1)), agent.get_neighbors())

            # Parameters
            agent.init_x_bar(np.zeros((self.dim, 1)))
            agent.init_lam(np.zeros((1, 1)), agent.get_edge_indices())
            agent.init_mu(np.zeros((self.dim, 1)), agent.get_neighbors())
            agent.init_x_star(np.zeros((self.dim, 1)), agent.get_neighbors()) # own err is last elem
            agent.init_w(np.zeros((self.dim, 1)), agent.get_neighbors())


        ##  Initialization - Measurements and Positions
        self.x_star = [np.zeros((self.dim, 1)) for i in range(self.num_agents)]         # Contains reconstructed error vector from localized SCP problem
        self.p_est = [agents[i].get_pos() for i in range(self.num_agents)]              # Contains reconstructed position vector
        self.p_reported = deepcopy(self.p_est)                                          # Reported positions of agents
        self.exp_meas = self.measurement_model()                                        # Expected measurements given positions and error
        self.R = self.get_Jacobian_matrix()                                             # Jacobian of measurement model


        # Node Parameters
        self.timer_period = 0.01  # seconds
        self.centroid_pos = np.array([[0], [0], [0]]).T
        self.agent_rel_pos = self.p_est # [None] * self.num_agents


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
        self.centroid_sub = self.create_subscription(
                TrajectorySetpoint,
                '/px4_0/fmu/in/trajectory_centroid',
                self.sub_centroid_callback,
                qos_profile_sub)
        
        self.measurements_sub = [None] * self.num_agents
        self.pos_sub = [None] * self.num_agents
        self.err_pub = [None] * self.num_agents

        for i in range(self.num_agents):
            # Subscribers
            sub_measurements_name = "/px4_" + str(i+1) + "/fmu/out/interagent_distances"
            self.measurements_sub[i] = self.create_subscription(
                Float32MultiArray,
                sub_measurements_name,
                partial(self.sub_measurements_callback, drone_ind=i),
                qos_profile_sub)
            
            sub_pos_name = "/px4_" + str(i+1) + "/fmu/in/trajectory_setpoint"
            self.pos_sub[i] = self.create_subscription(
                TrajectorySetpoint,
                sub_pos_name,
                partial(self.sub_pos_callback, drone_ind=i),
                qos_profile_sub)
            
            # Publishers
            pub_err_name = "/px4_" + str(i+1) + "/fmu/out/reconstructed_error"
            self.err_pub[i] = self.create_publisher(
                Float32MultiArray,
                pub_err_name,
                qos_profile_pub)
            
        # Callback Timers
        self.admm_update_timer = self.create_timer(self.timer_period, 
                                            self.admm_update)



    ### Subscriber callbacks
    
    # Inter-Agent Measurements
    def sub_measurements_callback(self, msg, drone_ind):
        
        try:
            iam_array = np.array(msg.data).flatten()
            self.agents[drone_ind].set_measurements(iam_array)
        except:
            self.get_logger().info("Exception: Issue with getting Inter-Agent Measurements of drone #" + str(drone_ind))


    # Relative Position of drone wrt centroid
    def sub_pos_callback(self, msg, drone_ind):
        if (self.debug): # If in debugging mode
            self.agent_rel_pos[drone_ind] = self.p_reported
            return
        
        try: # Extract msg and make pos rel to centroid
            my_ned_pos = msg.position.flatten()
            self.agent_rel_pos[drone_ind] = my_ned_pos - self.centroid_pos
            self.agent_rel_pos[drone_ind].reshape(self.dim, -1)
        except:
            self.get_logger().info("Exception: Issue with getting Relative Position of drone #" + str(drone_ind))


    # Position of Centroid of swarm
    def sub_centroid_callback(self, msg):
        if (self.debug): # If in debugging mode
            self.centroid_pos = np.array([[0], [0], [0]]).T
        
        try: # Not in debugging mode
            self.centroid_pos = msg.position.flatten()
        except:
            self.get_logger().info("Exception: Issue with getting Centroid of Drone Swarm")

    
    # Adjacency matrix of the graph
    def sub_adj_matrix(self, msg):
        if self.debug: # If in debugging mode
            return
        
        try: # Not in debugging mode
            self.adj_matrix = np.array(msg.data).reshape(self.num_agents, -1)
            new_edge_list = []
            for i in range(self.num_agents):
                nbr_id_list = []
                this_edge_list = []
                for j in range(self.num_agents):
                    if (i == j): # skip if on diagonal
                        continue

                    elif (self.adj_matrix[i, j] == 1): # If adjacent
                        new_edge_list.append([i, j]) # global edge list
                        this_edge_list.append(len(new_edge_list)) # agent edge list
                        nbr_id_list.append(j) # nbr list
                    
                self.agents[i].set_neighbors(nbr_id_list)
                self.agents[i].set_edge_indices(new_edge_list)
            
            # Set global edge list
            self.edge_list = new_edge_list

        except:
            self.get_logger().info("Exception: Issue with getting the adjacency matrix of the system")



    ### Publisher callbacks

    # Calls the ADMM Update Step
    def admm_update(self):
        ##      Check           - See if all variables are set before proceeding
        
        if (any(pos is None for pos in self.agent_rel_pos) or (self.centroid_pos is None)):
            print("\n ERROR: ")
            for id, pos in enumerate(self.agent_rel_pos):
                if pos is None:
                    print(f"\tAgent {id} has an invalid relative position")
            
            if self.centroid_pos is None:
                print(f"\tCentroid has invalid value")
            
            return


        ##      Initialization  - Get the true inter-agent measurements
        y = self.true_measurements()
        z = [(y[i] - self.exp_meas[i]) for i, _ in enumerate(y)]       


        ##      Minimization    - Primal Variable 1
        
        for id, agent in enumerate(self.agents):
            objective = cp.norm(agent.x_star[id] + agent.x_cp)
            
            # Summation for c() constraint
            for _, edge_ind in enumerate(agent.get_edge_indices()): 
                constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ agent.x_cp - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.agents[nbr_id].w[id]
                
                objective += ((self.rho/2)*cp.power(cp.norm(constr_c), 2)
                                + agent.lam[edge_ind].T @ (constr_c))
            
            # Summation for d() constraint
            for _, nbr_id in enumerate(agent.get_neighbors()): 
                constr_d = agent.x_cp - agent.w[nbr_id]
                objective += ((self.rho/2)*cp.power(cp.norm(constr_d), 2)
                              + agent.mu[nbr_id].T @ (constr_d))
                
            prob1 = cp.Problem(cp.Minimize(objective), [])
            prob1.solve(verbose=False)
            if prob1.status != cp.OPTIMAL:
                print("\nERROR Problem 1: Optimization problem not solved @ (%d)" % (self.curr_iter))

            agent.x_bar = deepcopy(np.array(agent.x_cp.value).reshape((-1, 1)))


        ##      Minimization    - Thresholding Parameter
        # TODO: Implement this

        ##      Minimization    - Primal Variable 2
        for id, agent in enumerate(self.agents):
            objective = cp.norm(agent.x_star[id] + agent.x_bar)

            # Summation for c() constraint
            for edge_ind in agent.get_edge_indices(): 
                constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c = constr_c + self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.agents[nbr_id].w_cp[id]
                
                objective += ((self.rho/2)*cp.power(cp.norm(constr_c), 2)
                                + agent.lam[edge_ind].T @ (constr_c))
            
            # Summation for d() constraint
            for nbr_id in agent.get_neighbors():
                constr_d = agent.x_bar - agent.w_cp[nbr_id]
                objective += ((self.rho/2)*cp.power(cp.norm(constr_d), 2)
                              + agent.mu[nbr_id].T @ (constr_d))
                
            prob2 = cp.Problem(cp.Minimize(objective), [])
            prob2.solve(verbose=False)
            if prob2.status != cp.OPTIMAL:
                print("\nERROR Problem 2: Optimization problem not solved @ (%d)" % (self.curr_iter))

            for _, nbr_id in enumerate(agent.get_neighbors()):
                agent.w[nbr_id] = deepcopy(np.array(agent.w_cp[nbr_id].value).reshape((-1, 1)))


        ##      Multipliers     - Update Lagrangian Multipliers of Minimization Problem
        for id, agent in enumerate(self.agents):
            
            # Summation for c() constraint
            for _, edge_ind in enumerate(agent.get_edge_indices()):
                constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.agents[nbr_id].w[id]
                
                agent.lam[edge_ind] = deepcopy(agent.lam[edge_ind] + self.rho * constr_c)

            # Summation for d() constraint
            for _, nbr_id in enumerate(agent.get_neighbors()):
                constr_d = agent.x_bar - agent.w[nbr_id]
                agent.mu[nbr_id] = deepcopy(agent.mu[nbr_id] + self.rho * constr_d)


        ##      Update          - SCP Outer Loop Handling
        if (self.curr_iter % self.n_admm):
            print("SCP Step")
            ##          Update          - Error Vectors after ADMM Subroutine
            for agent_id, agent in enumerate(self.agents): 
                for list_ind, nbr_id in enumerate(agent.get_neighbors()):
                    agent.x_star[nbr_id] = agent.x_star[nbr_id] + self.agents[nbr_id].x_bar
                
                agent.x_star[agent_id] = agent.x_star[agent_id] + agent.x_bar
                self.x_star[agent_id] = agent.x_star[agent_id]
                
                # Update position and x_dev
                self.p_est[agent_id] = self.p_reported[agent_id] + self.x_star[agent_id]
                print(f" -> Agent {agent_id} Pos: {self.p_est[agent_id].flatten()}")
            
            ##          Update          - Relinearize Measurement Model and Reset Primal Variables w_i
            if (self.curr_iter - self.n_admm) >= 0:

                # Linearized Measurement Model
                self.exp_meas = self.measurement_model()
                self.R = self.get_Jacobian_matrix()
                
                # Reset primal variables w after relinearization
                for agent in self.agents:
                    agent.init_w(np.zeros((self.dim, 1)), agent.get_neighbors())


        ##      End         - Publish error, increment current iteration, and return
        for id, agent in enumerate(self.agents):
            self.publish_err(id)

        self.curr_iter += 1
        return
    

    # Publish error vector
    def publish_err(self, id):
        # Init msg
        msg = Float32MultiArray()
        
        # Current error = outer loop error + inner loop error
        this_x = self.x_star[id].flatten() + self.agents[id].x_bar.flatten()

        # Send off error
        msg.data = this_x.tolist()
        self.err_pub[id].publish(msg)

        

    ### Helper Functions

    # Returns expected inter-agent measurements given position estimate and reconstructed error
    def measurement_model(self):
        est_meas = []

        for edge in self.edge_list:
            recons_pos1 = self.p_est[edge[0]] + self.x_star[edge[0]]
            recons_pos2 = self.p_est[edge[1]] + self.x_star[edge[1]]
            dist = self.distance(recons_pos1, recons_pos2)
            
            est_meas.append(dist)

        return est_meas
    

    # Returns true inter-agent measurements according to Phi
    def true_measurements(self):
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

        for edge in self.edge_list: # Not in debugging mode
            true_meas.append(self.agents[edge[0]].measurements[edge[1]])

        return true_meas
    

    # Computes distance between 2 points
    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2, 2)


    # Computes row of R
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


    # Computes whole R matrix
    def get_Jacobian_matrix(self):
        R = []

        for edge_ind, edge in enumerate(self.edge_list):
            R.append(self.get_Jacobian_row(edge))
        
        return R



### Main Func
    
def main():
    # Initializations
    DEBUG = False
    dim = 3
    num_agents = 6
    Agents = [None] * num_agents
    Formation =     [
                     np.array([[3.0*np.cos(np.pi/180*0),     3.0*np.sin(np.pi/180*0),    0]]).T,
                     np.array([[3.0*np.cos(np.pi/180*60),    3.0*np.sin(np.pi/180*60),   0]]).T,
                     np.array([[3.0*np.cos(np.pi/180*120),   3.0*np.sin(np.pi/180*120),  0]]).T,
                     np.array([[3.0*np.cos(np.pi/180*180),   3.0*np.sin(np.pi/180*180),  0]]).T,
                     np.array([[3.0*np.cos(np.pi/180*240),   3.0*np.sin(np.pi/180*240),  0]]).T,
                     np.array([[3.0*np.cos(np.pi/180*300),   3.0*np.sin(np.pi/180*300),  0]]).T]
    Edges =         [[0,1], [0,2], [0,3], 
                     [0,4], [0,5], [1,2],
                     [1,3], [1,4], [1,5],
                     [2,3], [2,4], [2,5],
                     [3,4], [3,5], [4,5],
                    
                     [1,0], [2,0], [3,0], 
                     [4,0], [5,0], [2,1],
                     [3,1], [4,1], [5,1],
                     [3,2], [4,2], [5,2],
                     [4,3], [5,3], [5,4]] # these edges are directed
    
    # Graph
    for id, _ in enumerate(Agents): # Create agent objects with nbr and edge lists
        this_agent = MyAgent(agent_id=id,
                             init_position=Formation[id])
        
        nbr_id_list = []
        nbr_ptr_list = []
        edge_list = []

        for edge_ind, edge in enumerate(Edges):
            if id == edge[0]:
                nbr_id_list.append(edge[1])
                edge_list.append(edge_ind)
        
        this_agent.set_neighbors(nbr_id_list)
        this_agent.set_edge_indices(edge_list)
        
        Agents[id] = this_agent

    # Add error vector
    if DEBUG:
        faulty_id   =   np.random.randint(0, high=num_agents)
        fault_vec   =   0.5*np.random.rand(dim, 1) # np.array([[0.0, 0, 0]]).T #
        Agents[faulty_id].faulty = True
        Agents[faulty_id].error_vector = fault_vec
        print("\n\n================================================================")
        print(f"Faulty Agent:   {faulty_id}")
        print(f"Faulty Vector:  {fault_vec.flatten()}")
        for id, pos in enumerate(Formation):
            print(f" -> Agent {id} True Pos: {pos.flatten()}")
        print("================================================================\n\n")

    
    # Node init
    rclpy.init(args=None)
    fault_detector = Fault_Detector(debug=DEBUG, dim=dim, agents=Agents, edge_list=Edges)
    # interagent_measurer.get_logger().info("Initialized")

    # Spin Node
    try:
        rclpy.spin(fault_detector)
    except KeyboardInterrupt:
        if (DEBUG):
            print("\n\nKeyboardInterrupt Called")
            print("\n\n================================================================")
            print(f"Faulty Agent:   {faulty_id}")
            print(f"Faulty Vector:  {fault_vec.flatten()}")
            for id, pos in enumerate(Formation):
                print(f" -> Agent {id} True Pos: {pos.flatten()}")
            print("================================================================\n\n")

    # Explicitly destroy node
    fault_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()