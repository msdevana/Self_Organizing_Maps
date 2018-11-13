"""
Created By: Manish Devana

Core functions for a self organizing map oriented at oceanographic purposes.

"""


import numpy as np 
import xarray as xr 
from numba import jit


# Functions for som class



def euclid_dist(data, net):
    """
    euclidian distance function for SOM (maybe experiment with more later)

    """
    dist = 34

@jit(parallel=True)
def find_bmu_training(vec, net):
    """ 
    Use the square of euclidean distance to find the closest matching codebook vector
    *Using the square saves the squareroot computation

    """
    vec = np.broadcast_to(vec, net.shape) # make the shapes match

    # find the index of the closest neuron
    bmu = np.unravel_index(
            np.argmin(np.sum((vec  - net)**2, axis=2)),
            net.shape[:-1])

    return bmu # an index tuple for the location of the BMU

    

def decay_radius(init_radius, i, decay):
    return init_radius * np.exp(-i / decay)

def decay_learning_rate(init_alpha, i, n_iterations):
    return init_alpha * np.exp(-i / n_iterations)


def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))

def update_net(vec, net, influence, alpha):
    """
    Update the SOM network with formula:
    new weight = old_weight  + learning_rate * influence * delta 
    delta  = vector(i) 
    """

    vec2 = np.broadcast_to(vec, net.shape)
    influence = np.expand_dims(influence, axis=2)
    influence = np.repeat(influence, net.shape[2], axis=2)
    new_net = net + alpha * influence * (vec2 - net)

    return new_net










def decay_func():
    """
    The decay function for reducing effects of learning

    """
    fe =1

def convergence_test():


    f =1 
    

def normalize(data):
    """
    Normalize data 
    """


