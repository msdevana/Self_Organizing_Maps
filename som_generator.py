"""
Created By Manish S. Devana

Generates Self Organizing Maps and gives options for training and testing with the model

"""

import numpy as np 
import xarray as xr 
import scipy 
from som_core import *
import dask as dk 
from numpy import random
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from numba import jit
from tqdm import tqdm



class SOM(object):
    """
    Creates a Self Organizing Map as an object
    """

    def __init__(self, nfeatures, grid_size=(10,10), learning_rate=.01):
        """


        :param grid_size: (m X n feature map size)
        :param nfeatures: (number of features/attirbutes for each input (also the size of the weights))

        """

        self.n = grid_size[0]
        self.m = grid_size[1]
        self.grid_size = grid_size
        self.alpha = learning_rate

        self.net = np.zeros((self.m, self.n, nfeatures))
        self.trained = False


    

    def optimize_map_size(self, data):
        """
        Option for optimizing the network size 
        FOR NOW : follow the law 5*sqrt(N) where N = # of samples


        """
        length = int(np.sqrt(5*np.sqrt(data.shape[0])))
        self.m = length
        self.n = length
        self.net = np.zeros((self.m, self.n, self.net.shape[2]))



    # @jit(nopython=True, parallel=True)
    def randomize_weights(self, data, normalize=False, nmin=0, nmax=1):
        """
        Generate randomized weights using data



        """
        if normalize:
            data = preprocessing.MinMaxScaler(feature_range=(nmin, nmax)).fit_transform(data)
            print('NOTE: normalizing data will make weights no longer resemble original data')

        
        
        # This is assumes data has already been vectorized
        idx = random.choice(data.shape[0], size=self.grid_size)

        for i in range(self.n):
            for k in range(self.m):
                self.net[i, k, :] = data[idx[i, k], :]
        

    @jit( parallel=True)
    def train(self, data, epochs=1000, learning_rate=None, radius=None, norm=False):
        """
        Training the Self Organizing Map, 
        Changes to training parameters can be tuned here


        :param  epochs: Number of Training Epochs


        """

        # Update Learning Rate
        if learning_rate:
            alpha = learning_rate
        else:
            init_alpha = np.copy(self.alpha)
        
        # set initial learning Radius
        if not learning_rate:
            init_radius = np.max((self.m, self.n)) / 2 # i.e. start with a large learning radius

        # Set decay of learning radius (start with a time constant decay rate)
        decay = epochs / np.log(init_radius)

        # Normalize data if chosen to
        if norm:
            pass # Add in the normalizing later

        # Make an array with the same size of the net except with indexs instead of weights
        # This comes in handy when calculating the influence and updating wieghts
        xx, yy = np.meshgrid(np.arange(self.n), np.arange(self.m))
        net_idx = np.dstack((yy, xx))
        
        # Load in the net (make sure its a copy otherwise weird shit happens)
        net = np.copy(self.net)
        
        # RUN TRAINING + Convergence checks
        self.radii = []
        self.alphas = []
        for i in tqdm(range(epochs)):
            data = data[np.random.permutation(np.arange(data.shape[0])), :]
            for vec in data:
                
                # Get devayed learning rates and radius
                radius =  decay_radius(init_radius, i, decay)
                alpha = decay_learning_rate(init_alpha, i, epochs)
                self.radii.append(radius)
                self.alphas.append(alpha)


                # Find the best matching unit
                bmu = find_bmu_training(vec, net)

                # Calculate the distance 
                dist2_from_bmu = np.sum(
                    (np.broadcast_to(bmu, net_idx.shape) - net_idx)**2,
                    axis=2
                )

                # mask the values outside the radius
                mask = dist2_from_bmu <= radius**2
                # dist2_from_bmu[mask] = 0
                
                # Calculate influence
                influence = calculate_influence(dist2_from_bmu, radius)
                influence[mask] = 0

                # update the weights within the neighborhod
                net = update_net(vec, net, influence, alpha)

        print('Training Complete: {} Epochs'.format(epochs))

        # Flag to check if data has been trained.
        self.net = net
        self.trained = True



        





