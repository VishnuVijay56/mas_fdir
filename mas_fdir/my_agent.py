"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
            - Represents generic agent object
"""

import numpy as np
import cvxpy as cp

from copy import deepcopy

class MyAgent():

    ##      Constructor
    
    def __init__(self, agent_id=0, init_position=np.zeros((3,1)), faulty=False, err_vector=None):
        # Prescribed information about the agent
        self.agent_id       =   agent_id
        self.pos_reported   =   init_position
        self.pos_est        =   init_position
        self.dim            =   init_position.shape[0]

        # Used for debugging of script
        self.faulty         =   faulty 
        self.error_vector   =   err_vector if (err_vector is not None) else np.zeros(self.pos_est.shape)
        
        # Graph information
        self.neighbor_ids   =   []
        self.neighbor_ptrs  =   {}
        self.edge_idx       =   []
        self.true_meas      =   {}
        self.exp_meas       =   {}
        self.z              =   {}
        self.R              =   {}
        
        # CVXPY variables
        self.x_cp           =   {}
        self.w_cp           =   {}

        # Variables updated over the course of optimization
        self.x_bar          =   []
        self.lam            =   {}
        self.mu             =   {}
        self.x_star         =   {}
        self.w              =   {}

        # Parameters
        self.rho            =   None
        self.curr_iter      =   0
        self.threshold_lhs  =   0

        # Miscellaneous items
        self.misc_dict      =   {}



    ###     Setting Functions

    # Sets ids of neighbors of current agent
    def set_neighbors(self, id_list, ptr_list=None):
        self.neighbor_ids   =   id_list
        self.neighbor_ptrs  =   ptr_list
        return None
       
    # Sets elements in agent dictionary
    def set_dict_elem(self, keys, values):
        self.misc_dict[keys]        =   values
        return None
    
    # Sets indices of edges this vertex is involved in
    def set_edge_indices(self, edge_idx):
        self.edge_idx = edge_idx
        return None
    
    # Sets true measurements and the difference between true and expected measurements
    def set_true_meas(self, iam_array):
        for id in self.neighbor_ids:
            # Set true measurements
            self.true_meas[id] = iam_array[id]

            # Set difference between the true and expected
            self.z[id] = self.true_meas[id] - self.exp_meas[id]

        return None
    


    ###     Getting Functions

    # Returns estimated position (true pos + error vector)
    def get_pos(self):
        return (self.position + self.error_vector)

    # Returns list of agent ids that are neighbors
    def get_neighbors(self):
        return self.neighbor_ids
    
    # Returns values from misc dictionary
    def get_dict_elem(self, key):
        return self.misc_dict[key]
    
    # Gets lists of edge indices this vertex is involved in
    def get_edge_indices(self):
        return self.edge_idx
    
    

    ###     Optimization Functions

    # Primal Update 1
    def primal1_update(self):
        id = self.agent_id

        if (self.threshold_lhs*self.rho) <= 1:
            self.x_bar = -self.x_star[id]
        
        else:
        ## Solve CVX problem if over threshold
            objective = cp.norm(self.x_star[id] + self.x_cp)
                
            # Summation for c() constraint
            for _, edge_ind in enumerate(self.get_edge_indices()): 
                constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ self.x_cp - self.z[edge_ind]
                for nbr_id in self.get_neighbors():
                    constr_c += self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.neighbor_ptrs[nbr_id].w[self.id]
                
                objective += ((self.rho/2)*cp.power(cp.norm(constr_c), 2)
                                + self.lam[edge_ind].T @ (constr_c))
            
            # Summation for d() constraint
            for _, nbr_id in enumerate(self.get_neighbors()): 
                constr_d = self.x_cp - self.w[nbr_id]
                objective += ((self.rho/2)*cp.power(cp.norm(constr_d), 2)
                                + self.mu[nbr_id].T @ (constr_d))
                
            prob1 = cp.Problem(cp.Minimize(objective), [])
            prob1.solve(verbose=False)
            if prob1.status != cp.OPTIMAL:
                print("\nERROR Problem 1: Optimization problem not solved @ (%d)" % (self.curr_iter))

            self.x_bar = deepcopy(np.array(self.x_cp.value).reshape((-1, 1)))
        ## END CVX problem

        return


    # Thresholding Parameter
    def thresholding(self):
        id = self.agent_id

        # Summation over neighbors
        term2 = 0
        for nbr_id in self.get_neighbors():
            constr_d = (-self.x_star) - self.w[nbr_id]
            term2 += constr_d + (self.mu[nbr_id]/self.rho)


        # Summation over edges
        term1 = 0
        for edge_ind in self.get_edge_indices():
            R_row = self.R[edge_ind]
            constr_c = R_row[:, self.dim*id:self.dim*(id+1)] @ (-self.x_star) - self.z[edge_ind]
            for nbr_id in self.get_neighbors():
                constr_c += R_row[:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.w[nbr_id]
            
            term1 += R_row[:, self.dim*id:self.dim*(id+1)].T @ (constr_c + self.lam[edge_ind])
        
        self.threshold_lhs = np.linalg.norm(term1 + term2)
        
        return
            

    # Primal Update 2
    def primal2_update(self):
        id = self.agent_id

        objective = cp.norm(self.x_star[id] + self.x_bar)

        # Summation for c() constraint
        for edge_ind in self.get_edge_indices(): 
            constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ self.x_bar - self.z[edge_ind]
            for nbr_id in self.get_neighbors():
                constr_c += self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.neighbor_ptrs[nbr_id].w_cp[self.id]
            
            objective += ((self.rho/2)*cp.power(cp.norm(constr_c), 2)
                            + self.lam[edge_ind].T @ (constr_c))
        
        # Summation for d() constraint
        for nbr_id in self.get_neighbors():
            constr_d = self.x_bar - self.w_cp[nbr_id]
            objective += ((self.rho/2)*cp.power(cp.norm(constr_d), 2)
                            + self.mu[nbr_id].T @ (constr_d))
            
        prob2 = cp.Problem(cp.Minimize(objective), [])
        prob2.solve(verbose=False)
        if prob2.status != cp.OPTIMAL:
            print("\nERROR Problem 2: Optimization problem not solved @ (%d)" % (self.curr_iter))

        for _, nbr_id in enumerate(self.get_neighbors()):
            self.w[nbr_id] = deepcopy(np.array(self.w_cp[nbr_id].value).reshape((-1, 1)))

        return


    # Lagrangian Multipliers
    def multipliers_update(self):
        id = self.agent_id

        # Summation for c() constraint
        for _, edge_ind in enumerate(self.get_edge_indices()):
            constr_c = self.R[edge_ind][:, self.dim*id:self.dim*(id+1)] @ self.x_bar - self.z[edge_ind]
            for nbr_id in self.get_neighbors():
                constr_c += self.R[edge_ind][:, self.dim*nbr_id:self.dim*(nbr_id+1)] @ self.neighbor_ptrs[nbr_id].w[self.id]
            
            self.lam[edge_ind] = deepcopy(self.lam[edge_ind] + self.rho * constr_c)

        # Summation for d() constraint
        for _, nbr_id in enumerate(self.get_neighbors()):
            constr_d = self.x_bar - self.w[nbr_id]
            self.mu[nbr_id] = deepcopy(self.mu[nbr_id] + self.rho * constr_d)

        # Update Current Iteration
        self.curr_iter += 1

        return
    

    # Outer Loop Handling
    def outer_loop_handling(self):
        id = self.agent_id

        # Update error vectors
        for _, nbr_id in enumerate(self.get_neighbors()):
            self.x_star[nbr_id] = self.x_star[nbr_id] + self.neighbor_ptrs[nbr_id].x_bar
            self.x_star[id] = self.x_star[id] + self.x_bar
            
            # Update position and x_dev
            self.pos_est[id] = self.pos_reported[id] + self.x_star[id]
            print(f" -> Agent {id} Pos: {self.pos_est[id].flatten()}")

        ## Linearized Measurement Model

        # Expected distance measurement according to graph structure
        for _, nbr_id in enumerate(self.get_neighbors()):
            nbr_pos_est = self.neighbor_ptrs[nbr_id].pos_est
            nbr_rel_pos = nbr_pos_est - self.pos_est
            est_dist = np.linalg.norm(nbr_rel_pos)
            self.exp_meas[nbr_id] = est_dist
        
        # Update Jacobian matrix
        for ind, edge in enumerate(self.get_edge_indices()):
            self.R[ind] = nbr_rel_pos.T / est_dist

        # Reset Primal Variables w After Relinearization
        self.init_w(np.zeros((self.dim, 1)), self.get_neighbors)

        return



    ###     Initializing Functions

    # 
    def init_x_cp(self, var):
        self.x_cp = var
        return None
    
    #
    def init_x_bar(self, var):
        self.x_bar = var
        return None

    # 
    def init_w_cp(self, var, nbr_ids):
        self.w_cp = {id: var for id in nbr_ids}
        return None
    
    #
    def init_lam(self, var, edge_inds):
        self.lam = {ind: var for ind in edge_inds}
        return None
    
    #
    def init_mu(self, var, nbr_ids):
        self.mu = {id: var for id in nbr_ids}
        return None
    
    #
    def init_x_star(self, var, nbr_ids):
        self.x_star = {id: var for id in nbr_ids}
        self.x_star[self.agent_id] = var
        return None
    
    #
    def init_w(self, var, nbr_ids):
        self.w = {id: var for id in nbr_ids}
        return None