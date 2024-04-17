"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
            - Represents generic agent object
"""

import numpy as np

class MyAgent():

    ##      Constructor
    
    def __init__(self, agent_id=0, init_position=np.zeros((3,1)), faulty=False, err_vector=None):
        # Prescribed information about the agent
        self.agent_id       =   agent_id
        self.position       =   init_position

        # Used for debugging of script
        self.faulty         =   faulty 
        self.error_vector   =   err_vector if (err_vector is not None) else np.zeros(self.position.shape)
        
        # Graph information
        self.neighbor_ids   =   []
        self.neighbor_ptrs  =   {}
        self.edge_idx       =   []
        self.measurements   =   {}
        
        # CVXPY variables
        self.x_cp           =   {}
        self.w_cp           =   {}

        # Variables updated over the course of optimization
        self.x_bar          =   []
        self.lam            =   {}
        self.mu             =   {}
        self.x_star         =   {}
        self.w              =   {}

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
    
    #
    def set_measurements(self, iam_array):
        for id in self.neighbor_ids:
            self.measurements[id] = iam_array[id]
        
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
