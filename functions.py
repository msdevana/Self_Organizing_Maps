"""
Created: by Manish S. Devana

Functions for looking at argo data
"""

import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from collections import OrderedDict
import gsw
import cmocean

default_parameters ={
    'storage location': os.path.join(os.path.expanduser("~"), "Research/data_storage") + '/'
}

def load_argo(fname='~/Research/data_storage/argo_2005-2018_grd.nc'):
    """

    :param fname:
    :return:
    """
    argo_grid = xr.open_dataset(fname, decode_times=False)


    return argo_grid

def load_bathy_gebco(fname='gebco.nc'):
    """
    loads gebco bathymetry
    """
    bathy = xr.open_dataset(default_parameters['storage location'] + fname)
    return bathy


def order_moorings(moorings):
    """


    :param moorings:
    :return:
    """
    mooring_names = list(moorings.keys())
    lons = []
    ordered = []
    lats = []
    for m in moorings.values():
        lons.append(m.lon)
        lats.append(m.lat)
    for i in np.argsort(lons):
        ordered.append((mooring_names[i], moorings[mooring_names[i]]))

    moorings = OrderedDict(ordered)
    return moorings


def load_moorings(fname='osnap_gridded_simple'):
    """
    load in my time gridded depth profiles of each mooring
    :param fname:
    :return:
    """

    files = os.listdir(default_parameters['storage location'] + fname)
    moorings = {}
    for file in files:
        data = xr.open_dataset(default_parameters['storage location'] + fname + '/' + file)
        moorings[data.station] = data

    return order_moorings(moorings)


def load_gridded_adt(fname='merged_adt_data/merged_adt_aviso_13_16.nc'):
    """

    :param fname:
    :return:
    """
    data = xr.open_dataset(default_parameters['storage location'] + fname)
    return data

def mooring_currents(fname='full_depth_osnap_currents'):
    """
    
    
    """
    files = os.listdir(default_parameters['storage location'] + fname)
    moorings = {}
    for file in files:
        data = xr.open_dataset(default_parameters['storage location'] + fname + '/' + file)
        moorings[data.station] = data

    return moorings


def matdate2pydate(mat_dates):
    """ Convert matlab datenum into python datetime
    """
    python_datetime = []
    for matlab_datenum in mat_dates:
        python_datetime.append(pd.to_datetime(datetime.fromordinal(int(matlab_datenum))
                                         + timedelta(days=matlab_datenum%1)
                                         - timedelta(days = 366)))

    return np.array(python_datetime)


def merge_daily_adt(fname='adt_daily_13-16'):
    """

    :return:
    """
    path = default_parameters['storage location'] + fname
    files = os.listdir(path)
    path2 = path + '/'

    adt = []
    u = []
    v = []
    lon = []
    lat = []

    # for file in files:


def eke(adt_array):
    """
    Calculate Eddy Kinetic Energy from AVISO geostrophic Velocity

    :param adt_array:
    :return:
    """
    
    means = adt_array.mean(dim='time')
    uu = adt_array.ugeo - means.ugeo
    vv = adt_array.vgeo - means.vgeo

    eke = 0.5 * (uu**2 + vv**2)
    
    return eke


def ts_diagram(trange=(0,15), srange=(30,39),nlevels=10, 
               figsize=(6,6), **kwargs):
    """
    Function to autogenerate ts diagrams
    
    """
    

    trange = np.linspace(trange[0], trange[1], 100)
    srange = np.linspace(srange[0], srange[1], 100)
    srange, trange = np.meshgrid(srange, trange)

    sigma0 = gsw.density.sigma0(srange, trange)

    ts_fig = plt.figure(figsize=figsize)
    levels = np.linspace(sigma0.min(), sigma0.max(), nlevels)
    rho_contours = plt.contour(srange,
                               trange,
                               sigma0,
                               levels=levels,
                               cmap=cmocean.cm.dense,
                               zorder=0)

    plt.clabel(rho_contours,
               rho_contours.levels,colors='k',
               inline=True, fmt='%1.2f', fontsize=10)
    plt.ylabel(r'Temperature ($^{\circ}C$)')
    plt.xlabel(r'Absolute Salinity')
    
    return ts_fig


def make_plots():
    """


    :return:
    """
