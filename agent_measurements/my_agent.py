"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
            - Represents generic agent object
"""

import numpy as np

class MyAgent():
    # Constructor
    def __init__(self, agent_id=0, init_position=np.zeros((3,1)), faulty=False, err_vector=None):
        self.agent_id       =   agent_id
        self.position       =   init_position
        self.faulty         =   faulty 
        self.error_vector   =   err_vector if (err_vector is not None) else np.zeros(self.position.shape)
        self.neighbor_ids   =   []
        
        self.edge_idx       =   []
        
        self.x_cp           =   {}
        self.w_cp           =   {}

        self.x_bar          =   []
        self.lam            =   {}
        self.mu             =   {}
        self.x_star         =   {}
        self.w              =   {}

        self.misc_dict      =   {}

    # Returns estimated position (true pos + error vector)
    def get_estimated_pos(self):
        if not self.faulty:
            return      self.position
        else:
            return      self.position + self.error_vector
    
    # Returns true position
    def get_true_pos(self):
        return self.position

    # Sets neighbors of current agent
    def set_neighbors(self, neighbor_list):
        self.neighbor_ids   =   neighbor_list
        return None

    # Returns list of agent ids that are neighbors
    def get_neighbors(self):
        return self.neighbor_ids
    
    # Sets elements in agent dictionary
    def set_dict_elem(self, keys, values):
        self.misc_dict[keys]        =   values
        return None
        
    # Returns values from misc dictionary
    def get_dict_elem(self, key):
        return self.misc_dict[key]
    
    # Sets indices of edges this vertex is involved in
    def set_edge_indices(self, edge_idx):
        self.edge_idx = edge_idx
        return None
    
    # Gets lists of edge indices this vertex is involved in
    def get_edge_indices(self):
        return self.edge_idx
    
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
    
