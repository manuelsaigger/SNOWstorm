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
from wrf import getvar, interplevel


def unnorm_data(pred, normfactors, var_pred='wind'):
    """
    Transfer data from normalized space into regular space based on given normalization factors
    
    Parameters
    ----------
    
    pred: torch tensor
            torch tensor of prediction to be unnormalized
    normfactors: list
            list of normalization factors
    var_pred: str, optional
            output variable of model ('wind', 'dswe', 'subl', 'phi') 
            
    
    Returns
    -------
    data_unnorm_out: ndarray, list for var_pred == 'wind' and 'phi'
            unnormalized prediction (numpy array for single channel predictions, list of numpy arrays for components in wind and phi)
    
    """
    
    if var_pred in ['wind', 'phi']: 
        # extract components and turn into numpy array
        print('wind or phi!')
        component_x = np.squeeze(pred[:, 0, :, :].detach().numpy())
        component_y = np.squeeze(pred[:, 1, :, :].detach().numpy())

        # unnormalize: multiply with std, add mean
        component_x_unnorm = component_x*normfactors['norm_factors_out'][0][1] + normfactors['norm_factors_out'][0][0]
        component_y_unnorm = component_y*normfactors['norm_factors_out'][1][1] + normfactors['norm_factors_out'][1][0]
        
        # make list for output
        data_unnorm_out = [component_x_unnorm, component_y_unnorm]    
    else:
        # turn into numpy array
        data_norm = pred.detach().numpy()
        
        # unnormalize: multiply with std, add mean
        data_unnorm_out = data_norm*normfactors['norm_factors_out'][0][1] + normfactors['norm_factors_out'][0][0]
    
    return data_unnorm_out

def untrans_data(data, offset_trans, var_pred='dswe'):
    """
    Transfer data from log-transformed space into regular space given offsets
    
    Parameters
    ----------
    
    data: torch tensor or ndarray
            data to be back-transfered
    offset_trans: float
            constant offset used in transformation to avoid 0
    var_pred: str, optional
            output variable of model ('dswe', 'subl')
    
    Returns
    -------
    
    data_untrans: torch tensor or ndarray
            data in regular space, same type as data
    """


    # use torch syntax if data is tensor
    if type(data) == torch.Tensor:
        if var_pred == 'dswe':
            data_untrans = torch.exp(data) + offset_trans        # backtransform dswe
        
        elif var_pred == 'subl':
            data_untrans = -(torch.exp(data) - offset_trans)     # backtransform subl  
               
        
    # use numpy syntax if data is ndarray
    else:
        if var_pred == 'dswe':
            data_untrans = np.exp(data) + offset_trans           # backtransform dswe
        
        elif var_pred == 'subl':
            data_untrans = -(np.exp(data) - offset_trans)        # backtransform subl
        
        elif var_pred == 'phi':
            phix = data[0]
            phiy = data[1]
    
            phix_untrans = np.empty_like(phix)
            phiy_untrans = np.empty_like(phix)
    
            phix_untrans[phix == 0] = 0
            phiy_untrans[phiy == 0] = 0
    
            phix_untrans[phix > 0] = phix[phix > 0]**2
            phiy_untrans[phiy > 0] = phiy[phiy > 0]**2
    
            phix_untrans[phix < 0] = -(phix[phix < 0]**2)
            phiy_untrans[phiy < 0] = -(phiy[phiy < 0]**2)
    
            data_untrans = [phix_untrans, phiy_untrans] 
        
    return data_untrans
    
def write_ds_pred(pred, dem_glacier_np, dem_glacier_xr, vars_pred, type_dem):
    """
    Write predictions into xarray dataset with same grid as input dem
    
    Parameters
    ----------
    
    pred: list
            list of numpy arrays with snowstorm predictions of all variables
    dem_glacier_np: ndarray
            numpy array with terrain height
    dem_glacier_xr: xarray dataset
            xarray dataset of terrain height with coordinates
    vars_pred: list
            list of predicted variables
    type_dem: str, optional
            type of input dem (currently: 'tdx': Tandem-X DEM, 'wrfles': same topography as WRF-LES, 'fourierland': synthetic fourier topography)
     
     Returns
     -------
     ds_out: xr dataset
             xarray dataset of predictions with same coordinate grid as input dem    
    """

    if type_dem == 'tdx':        # GLO-30 topography
        # get grid coordinates of input dem 
        lat = dem_glacier_xr.lat.values
        lon = dem_glacier_xr.lon.values
    
        if 'wind' in vars_pred:        # get wind direction and speed from predicted wind components
            dd, ff = util_funct_meteo.uv2ddff(pred[0][0], pred[0][1])
        
        # create xr dataset to fill up
        ds_out = xr.Dataset(coords={'lat':lat,
                                    'lon':lon})
        
        # write output fields in dataset
        if 'wind' in vars_pred:
            ds_out['u'] = (['lon', 'lat'], pred[0][0])
            ds_out['v'] = (['lon', 'lat'], pred[0][1])
            ds_out['dd'] = (['lon', 'lat'], dd)
            ds_out['ff'] = (['lon', 'lat'], ff)
            ds_out['ter'] = (['lon', 'lat'], dem_glacier_np)
    
        if 'dswe' in vars_pred:
            #print(len(pred[1]))
            #print(pred[1][0].shape)
            ds_out['dswe'] = (['lon', 'lat'], np.squeeze(pred[1][0]))
    
        if 'subl' in vars_pred:
            ds_out['subl'] = (['lon', 'lat'], np.squeeze(pred[2][0]))
        
        if 'phi' in vars_pred:
            ds_out['phi_x'] = (['lon', 'lat'], pred[3][0])
            ds_out['phi_y'] = (['lon', 'lat'], pred[3][1])
            
    elif type_dem == 'wrfles':        # WRF LES topography
    
        ds_out = dem_glacier_xr       # use wrf topography data set as template
        
        # write output fields in data set, use WRF grid coordinates
        if 'wind' in vars_pred:
            dd, ff = util_funct_meteo.uv2ddff(pred[0][0], pred[0][1])
            ds_out['u'] = (['west_east', 'south_north'], pred[0][0].T)
            ds_out['v'] = (['west_east', 'south_north'], pred[0][1].T)
            ds_out['dd'] = (['west_east', 'south_north'], dd.T)
            ds_out['ff'] = (['west_east', 'south_north'], ff.T)
            ds_out['ter'] = (['west_east', 'south_north'], dem_glacier_np.T)
    
        if 'dswe' in vars_pred:
            #print(len(pred[1]))
            #print(pred[1][0].shape)
            ds_out['dswe'] = (['west_east', 'south_north'], np.squeeze(pred[1][0]).T)
    
        if 'subl' in vars_pred:
            ds_out['subl'] = (['west_east', 'south_north'], np.squeeze(pred[2][0]).T)
        
        if 'phi' in vars_pred:
            ds_out['phi_x'] = (['west_east', 'south_north'], pred[3][0].T)
            ds_out['phi_y'] = (['west_east', 'south_north'], pred[3][1].T)
    
    elif type_dem == 'fourierland':        # synthetic fourierland topography
    
        # create coordinates and emtpy data set, use WRF grid coordinates
        sn = np.arange(256)
        we = np.arange(256)
        ds_out = xr.Dataset(coords={'south_north':sn,
                                    'west_east':we})
                                   
        # write output fields in data set, use WRF grid coordinates
        if 'wind' in vars_pred:
            dd, ff = util_funct_meteo.uv2ddff(pred[0][0], pred[0][1])
            ds_out['u'] = (['west_east', 'south_north'], pred[0][0].T)
            ds_out['v'] = (['west_east', 'south_north'], pred[0][1].T)
            ds_out['dd'] = (['west_east', 'south_north'], dd.T)
            ds_out['ff'] = (['west_east', 'south_north'], ff.T)
            ds_out['ter'] = (['west_east', 'south_north'], dem_glacier_np.T)
    
        if 'dswe' in vars_pred:
            #print(len(pred[1]))
            #print(pred[1][0].shape)
            ds_out['dswe'] = (['west_east', 'south_north'], np.squeeze(pred[1][0]).T)
    
        if 'subl' in vars_pred:
            ds_out['subl'] = (['west_east', 'south_north'], np.squeeze(pred[2][0]).T)
        
        if 'phi' in vars_pred:
            #print(pred[3].shape)
            ds_out['phi_x'] = (['west_east', 'south_north'], pred[3][0].T)
            ds_out['phi_y'] = (['west_east', 'south_north'], pred[3][1].T)
               
    return ds_out


