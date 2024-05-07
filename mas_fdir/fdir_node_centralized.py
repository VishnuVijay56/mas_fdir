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
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from mas_fdir.my_agent import MyAgent

from px4_msgs.msg import VehicleLocalPosition, VehicleGlobalPosition, TrajectorySetpoint
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import UInt8, Bool, Float32MultiArray, Float32


class Fault_Detector(Node):

    def __init__(self, debug, dim, agents, adj_matrix):

        super().__init__("fault_detector")


        ##  Arguments
        self.debug = debug
        self.agents = agents
        self.num_agents = len(agents)
        self.adj_matrix = adj_matrix
        self.dim = dim


        ##  Initialization - Optimization Parameters
        self.n_admm = 20
        self.curr_iter = 0
        self.rho = 1.0
        for agent_id, agent in enumerate(agents):
            # CVX variables
            agent.init_x_cp(cp.Variable((self.dim, 1)))
            agent.init_w_cp(cp.Variable((self.dim, 1)), np.arange(self.num_agents)) #agent.get_neighbors())

            # Parameters
            agent.init_x_bar(np.zeros((self.dim, 1)))
            agent.init_lam(np.zeros((1, 1)), np.arange(self.num_agents)) #agent.get_edge_indices())
            agent.init_mu(np.zeros((self.dim, 1)), np.arange(self.num_agents)) #agent.get_neighbors())
            agent.init_x_star(np.zeros((self.dim, 1)), np.arange(self.num_agents)) #agent.get_neighbors()) # own err is last elem
            agent.init_w(np.zeros((self.dim, 1)), np.arange(self.num_agents)) #agent.get_neighbors())


        ##  Initialization - Measurements and Positions
        self.x_star = [np.zeros((self.dim, 1)) for i in range(self.num_agents)]         # Contains reconstructed error vector from localized SCP problem
        self.p_est = [agents[i].get_pos() for i in range(self.num_agents)]              # Contains reconstructed position vector
        self.p_reported = [None] * self.num_agents                                      # Reported positions of agents
        self.edges_from_adj()
        self.exp_meas = self.measurement_model()                                        # Expected measurements given positions and error
        self.R = self.get_Jacobian_matrix()                                             # Jacobian of measurement model
        self.residuals = [None] * self.num_agents                                       # Residuals of each agent to be checked against the threshold


        ## Initialization - Specific to ROS2 Implementation
        self.timer_period = 0.1  # seconds
        self.centroid_pos = None
        self.agent_local_pos = [None] * self.num_agents
        self.formation_msg = [None] * self.num_agents
        self.spawn_offset_pos = [None] * self.num_agents
        self.admm_running = False


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


        ##  Define - Subscribers and Publishers

        self.centroid_sub = self.create_subscription(
            PointStamped,
            '/px4_0/detector/in/vleader_position',
            self.sub_centroid_callback,
            qos_profile_sub)
        
        self.adj_matrix_sub = self.create_subscription(
            Float32MultiArray,
            '/px4_0/detector/in/adjacency',
            self.sub_adj_matrix,
            qos_profile_sub)
        
        self.formation_sub = self.create_subscription(
            Float32MultiArray,
            '/px4_0/detector/in/formation_config',
            self.sub_formation_callback,
            qos_profile_sub)

        # Initialize List of Subs and Pubs for drones
        self.measurements_sub = [None] * self.num_agents
        self.local_pos_sub = [None] * self.num_agents
        self.spawn_offset_sub = [None] * self.num_agents
        self.err_pub = [None] * self.num_agents
        self.residual_pub = [None] * self.num_agents

        for i in range(self.num_agents):
            # Subscribers
            sub_measurements_name = f"/px4_{i+1}/fmu/out/interagent_distances"
            self.measurements_sub[i] = self.create_subscription(
                Float32MultiArray,
                sub_measurements_name,
                partial(self.sub_measurements_callback, drone_ind=i),
                qos_profile_sub)
            
            sub_local_pos_name = f"/px4_{i+1}/fmu/out/vehicle_local_position"
            self.local_pos_sub[i] = self.create_subscription(
                VehicleLocalPosition,
                sub_local_pos_name,
                partial(self.sub_local_pos_callback, drone_ind=i),
                qos_profile_sub)
            
            sub_spawn_offset_name = f"/px4_{i+1}/fmu/in/vehicle_spawn_offset"
            self.spawn_offset_sub[i] = self.create_subscription(
                Float32MultiArray,
                sub_spawn_offset_name,
                partial(self.sub_spawn_offset_callback, drone_ind=i),
                qos_profile_sub)
            
            # Publishers
            pub_err_name = f"/px4_{i+1}/fmu/out/reconstructed_error"
            self.err_pub[i] = self.create_publisher(
                Float32MultiArray,
                pub_err_name,
                qos_profile_pub)
            
            pub_res_name = f"/px4_{i+1}/fmu/out/residual"
            self.residual_pub[i] = self.create_publisher(
                Float32,
                pub_res_name,
                qos_profile_pub
            )
            

        ##  Define: Callback Timer(s)
        self.admm_update_timer = self.create_timer(self.timer_period, 
                                            self.admm_update)



    ### Subscriber callbacks
    
    # Sub: Inter-Agent Measurements
    def sub_measurements_callback(self, msg, drone_ind):
        if self.admm_running:
            return
        
        try:
            iam_array = np.array(msg.data).flatten()
            self.agents[drone_ind].set_measurements(iam_array)
        except:
            self.get_logger().info("Exception: Issue with getting Inter-Agent Measurements of drone #" + str(drone_ind))


    # Sub: Local Drone Position
    def sub_local_pos_callback(self, msg, drone_ind):
        if self.admm_running:
            return
        
        try: # Extract msg
            self.agent_local_pos[drone_ind] = np.array([[msg.x], [msg.y], [msg.z]]).reshape((self.dim, -1))
            # self.get_logger().info(f"Agent {drone_ind} Local Pos: {self.agent_local_pos[drone_ind].flatten()}")
        except:
            self.get_logger().info("Exception: Issue with getting Relative Position of drone #" + str(drone_ind))

    
    # Sub: Spawn Offset Position
    def sub_spawn_offset_callback(self, msg, drone_ind):
        if self.admm_running:
            return

        if self.spawn_offset_pos[drone_ind] is not None: # If spawn position is set, don't do anything
            return
        
        try:
            arr = msg.data
            self.spawn_offset_pos[drone_ind] = np.array(arr).reshape((self.dim, -1))
        except:
            self.get_logger().info("Exception: Issue with getting the spawn offset position of drone #" + str(drone_ind))


    # Sub: Formation of Swarm
    def sub_formation_callback(self, msg):
        if self.admm_running:
            return
        
        try:
            formation_arr = msg.data
            for id, _ in enumerate(self.agents):
                this_formation_arr = np.array(formation_arr[(id*self.dim):(self.dim*(id+1))]).reshape((self.dim, -1))
                self.formation_msg[id] = this_formation_arr
        except:
            self.get_logger().info("Exception: Issue with getting the formation config of the swarm")


    # Sub: Virtual Leader NED Position
    def sub_centroid_callback(self, msg):
        if self.admm_running:
            return
        
        try: # Not in debugging mode
            self.centroid_pos = np.array([[msg.point.x], [msg.point.y], [msg.point.z]]).reshape((self.dim, -1))
        except:
            self.get_logger().info("Exception: Issue with getting Centroid of Drone Swarm")

    
    # Sub: Adjacency Matrix of the graph
    def sub_adj_matrix(self, msg):
        if self.admm_running:
            return
        
        try: # Not in debugging mode
            self.adj_matrix = np.array(msg.data).reshape(self.num_agents, -1)
            self.edges_from_adj()
        except:
            self.get_logger().info("Exception: Issue with getting the adjacency matrix of the system")



    ### Helper Functions

    # Help: Returns expected inter-agent measurements given position estimate and reconstructed error
    def measurement_model(self):
        est_meas = []

        for edge in self.edge_list:
            recons_pos1 = self.p_est[edge[0]] + self.x_star[edge[0]]
            recons_pos2 = self.p_est[edge[1]] + self.x_star[edge[1]]
            dist = self.distance(recons_pos1, recons_pos2)
            
            est_meas.append(dist)

        return est_meas
    

    # Help: Returns true inter-agent measurements according to Phi
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
        R = []

        for edge_ind, edge in enumerate(self.edge_list):
            R.append(self.get_Jacobian_row(edge))
        
        return R
    

    # Help: Computes relative position of drones wrt centroid
    def get_rel_pos(self, id):
        # Compute
        rel_pos = self.agent_local_pos[id] - self.centroid_pos + self.spawn_offset_pos[id]
        log_message = f"Agent {id} - " + \
                      f"\n\tRadius: {np.linalg.norm(rel_pos)}" + \
                      f"\n\tLocal Pos\t: {self.agent_local_pos[id].flatten()}" + \
                      f"\n\tCentroid Pos\t: {self.centroid_pos.flatten()}" + \
                      f"\n\tSpawn Offset\t: {self.spawn_offset_pos[id].flatten()}"                      
        self.get_logger().info(log_message)

        # Assign
        self.agents[id].position = rel_pos
        self.p_reported[id] = rel_pos

    
    # Help: Reconstructs edge list from adjacency matrix and assigns these edges to the agents and the global edge list
    def edges_from_adj(self):
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



    ### Publisher callbacks

    # Publish error vector
    def publish_err(self, id):
        # Init msg
        msg = Float32MultiArray()
        
        # Current error = outer loop error + inner loop error
        this_x = self.x_star[id].flatten() + self.agents[id].x_bar.flatten()
        # self.get_logger().info(f"Agent {id} - Error\t: {this_x}")

        # Send off error
        msg.data = this_x.tolist()
        self.err_pub[id].publish(msg)
        return

    
    # Publish agent residual
    def publish_residual(self, id):
        msg = Float32()
        msg.data = self.residuals[id]
        self.residual_pub[id].publish(msg)
        # self.get_logger().info(f"Agent {id} - Residual\t: {self.residuals[id]}")
        return
    

    # Calls the ADMM Update Step
    def admm_update(self):
        
        ##      Check           - See if variables are set before proceeding

        unset_var = False
        # Check
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
        
        # Return if variable is not set
        if unset_var:
            return
        
        self.admm_running = True
        ##      Initialization  - Compute the relative distance of the agents wrt centroid
        for id, agent in enumerate(self.agents):
            self.get_rel_pos(id)


        ##      Initialization  - Get the true inter-agent measurements
        self.all_meas_set = True
        y = self.true_measurements()
        if not self.all_meas_set:
            return
        z = [(y[i] - self.exp_meas[i]) for i, _ in enumerate(y)]  

        self.admm_running = False     


        ##      Minimization    - Primal Variable 1
        
        for id, agent in enumerate(self.agents):
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
            if (this_res*self.rho <= 1): # skip optimization
                agent.x_bar = deepcopy(-agent.x_star[id])
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
                if prob1.status != cp.OPTIMAL:
                    self.get_logger().warning(f"~ERROR~ Problem 1: Optimization problem not solved @ {self.curr_iter}th iteration")

                agent.x_bar = deepcopy(np.array(agent.x_cp.value).reshape((-1, 1)))


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
            if prob2.status != cp.OPTIMAL:
                self.get_logger().warning(f"~ERROR~ Problem 2: Optimization problem not solved @ {self.curr_iter}th iteration")

            for _, nbr_id in enumerate(agent.get_neighbors()):
                agent.w[nbr_id] = deepcopy(np.array(agent.w_cp[nbr_id].value).reshape((-1, 1)))


        ##      Multipliers     - Update Lagrangian Multipliers of Minimization Problem
        for id, agent in enumerate(self.agents):
            
            # Summation for c() constraint
            for _, edge_ind in enumerate(agent.get_edge_indices()):
                constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.agents[nbr_id].w[id]
                
                agent.lam[self.edge_list[edge_ind]] = deepcopy(agent.lam[self.edge_list[edge_ind]] + self.rho * constr_c)

            # Summation for d() constraint
            for _, nbr_id in enumerate(agent.get_neighbors()):
                constr_d = agent.x_bar - agent.w[nbr_id]
                agent.mu[nbr_id] = deepcopy(agent.mu[nbr_id] + self.rho * constr_d)


        ##      Update          - SCP Outer Loop Handling
        if ((self.curr_iter % self.n_admm) == 0) and ((self.curr_iter - self.n_admm) >= 0):
            self.get_logger().info(" ---> SCP Step: Relinearization, Error Vector Updating, and Primal Variable w Resetting")

            ##          Update          - Post ADMM Subroutine Handling
            
            # Update Error Vectors
            self.admm_running = True
            for agent_id, agent in enumerate(self.agents): 
                for list_ind, nbr_id in enumerate(agent.get_neighbors()):
                    agent.x_star[nbr_id] = agent.x_star[nbr_id] + self.agents[nbr_id].x_bar
                
                agent.x_star[agent_id] = agent.x_star[agent_id] + agent.x_bar
                self.x_star[agent_id] = agent.x_star[agent_id]
                
                # Update position and x_dev
                self.p_est[agent_id] = self.p_reported[agent_id] + self.x_star[agent_id]
                print(f" -> Agent {agent_id} Pos: {self.p_est[agent_id].flatten()}")
            
            # Linearized Measurement Model
            self.exp_meas = self.measurement_model()
            self.R = self.get_Jacobian_matrix()
            self.admm_running = False

            # Reset primal variables w after relinearization
            for agent in self.agents:
                agent.init_w(np.zeros((self.dim, 1)), agent.get_neighbors())


        ##      End         - Publish error and residuals, increment current iteration, and return
        for id, agent in enumerate(self.agents):
            self.publish_err(id)
            self.publish_residual(id)

        self.curr_iter += 1
        return
    

    
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
    Adjacency =     np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    Edges =         [(0,1), (0,2), (0,3), 
                     (0,4), (0,5), (1,2),
                     (1,3), (1,4), (1,5),
                     (2,3), (2,4), (2,5),
                     (3,4), (3,5), (4,5),
                    
                     (1,0), (2,0), (3,0), 
                     (4,0), (5,0), (2,1),
                     (3,1), (4,1), (5,1),
                     (3,2), (4,2), (5,2),
                     (4,3), (5,3), (5,4)] # these edges are directed
    
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
    fault_detector = Fault_Detector(debug=DEBUG, dim=dim, agents=Agents, adj_matrix=Adjacency)
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