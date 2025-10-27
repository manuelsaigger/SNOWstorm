#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

import os
import json
from netCDF4 import Dataset
#from wrf import getvar, interplevel







def calc_dp_from_dz(dz, T0, p0, const):
    # use hypsometric equation to calculate pressure difference for given height difference
    
    rho = p0/(T0*const['Rd']['value'])
    
    dp = - dz / (const['g']['value'] * rho)
    
    return dp
    
def calc_dT_from_dz_gamma(dz, gamma, const):
    # use lapse rate to calculate temperature difference for given height difference
    
    dT = dz*gamma
    
    return dT
    
def calc_N2(th1, th2, dz, const):
    # calculate dry Brunt-Väisällä Frequcency using standard formula
    
    N2 = (const['g']['value'] * (th1 - th2)) / (dz * 0.5*(th1+th2))
    
    return N2

    
def calc_cosddasp(dem, dd_in):
    # calcualte cosine of difference angle between slope aspect and ambient wind direction
    
    gr_tt = np.gradient(np.abs(dem))                                 # calculate gradient vector of terrain height
    
    aspect = np.rad2deg(np.arctan2(gr_tt[1], gr_tt[0])) + 180        # calculate aspect angle based on gradient vector
    
    cosddasp = np.cos(np.deg2rad(aspect-dd_in))                      # calculate cosine of difference angle

    return cosddasp


def filter_121(field, weight_filter=2):
    # apply 1-2-1 filter on field, weight_filter: weight of center point, default 2
    
    # get dimenstion of field in x and y
    nx = np.shape(field)[0]
    ny = np.shape(field)[1]

    # predefine emty field to fill up
    field_filter = np.zeros_like(field)
    
    # use unfiltered values on edge points of field
    field_filter[0,0] = field[0,0]
    field_filter[0,-1] = field[0,-1]
    field_filter[-1,0] = field[-1,0]
    field_filter[-1,-1] = field[-1,-1]

    # loop over field in x and y
    for ix in range(1,nx-1):
        # border points: only use half filter matrix
        field_filter[ix, 0] = (field[ix-1, 0] + weight_filter*field[ix, 0] + field[ix+1, 0] + field[ix, 1]) / (weight_filter+3)
        field_filter[ix, -1] = (field[ix-1, -1] + weight_filter*field[ix, -1] + field[ix+1, -1] + field[ix, -2]) / (weight_filter+3)

        for iy in range(1, ny-1):
            # inner points: use full 2D filter matrix
            field_filter[ix, iy] = (field[ix-1, iy] + weight_filter*field[ix, iy] + field[ix+1, iy] + field[ix, iy-1] + field[ix, iy+1]) / (weight_filter+4)

    for iy in range(1, ny-1):
        # border points in y: only use half filter matrix
        field_filter[0, iy] = (weight_filter*field[0, iy] + field[0, iy+1] + field[0, iy-1] + field[1, iy]) / (weight_filter+3)
        field_filter[-1, iy] = (weight_filter*field[-1, iy] + field[-1, iy+1] + field[-1, iy-1] + field[1, iy]) / (weight_filter+3)


    return field_filter

