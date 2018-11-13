"""
Created By Manish S. Devana

Generates Self Organizing Maps and gives options for training and testing with the model

"""

import numpy as np 
import xarray as xr 
import scipy 
from som_core import *
import dask as dk 




class SOM(object):
    """
    Creates a Self Organizing Map as an object
    """

    def __init__(self, grid_size, nfeatures):
        """


        :param grid_size: (m X n feature map size)
        :param nfeatures: (number of features/attirbutes for each input (also the size of the weights))

        """

        self.n = grid_size[0]
        self.m = grid_size[1]

        self.map = np.zeros((self.m, self.n, nfeatures))

    def train(self, epochs=1000):
        """
        :

        :param  epochs: Number of Training Epochs


        """
        





