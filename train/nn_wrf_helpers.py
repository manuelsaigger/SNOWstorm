#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:53:47 2024

@author: manuel

Helper functions for training and data preparation of SNOWstorm 

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torch
from scipy import interpolate

import os
import json
import time

import nn_wrf_main
import nn_wrf_diag




##############
# Data Extraction
##############

def get_data_wrf(path_to_wrf, infile_wrf, nx, layers_in, layers_out, data_in, data_out, ii_data):
    """
    Fill up input and output data tensor based on wrf fields for single simulation
     
    Parameters
    ----------
    path_to_wrf: str
            path to wrf file
    infile_wrf: str
            filename of wrf file
    nx: int
            number of grid points in x and y direction in data tensor
    layers_in: list
            list of input layer names (from namelist)
    layers_out: list
            list of ouput layer names (from namelist)
    data_in: tensor
            input data tensor to be filled up
    data_out: tensor
            output data tensor to be filled up
    ii_data: int
            index of data set
    
    
    Returns
    -------
    data_in: tensor
            input data tensor with filled up fields
    data_out: tensor
            output data tensor with filled up fields    
    """
       
    # open training data wrf field
    ds_tmp = xr.open_dataset(os.path.join(path_to_wrf, infile_wrf))
    

    # loop over input layers, fill up data_in    
    ii_tensor = 0
    for layer_in in layers_in:
        
        if layer_in == 'uv_in': # get wind components separately
            dd_in_tmp = ds_tmp['dd0'].values
            ff_in_tmp = ds_tmp['ff0'].values
            
            u_in1_tmp, v_in1_tmp = ddff2uv(dd_in_tmp, ff_in_tmp)
            u_in_tmp = np.ones((nx, nx)) * u_in1_tmp
            v_in_tmp = np.ones((nx, nx)) * v_in1_tmp
            
            u_in_tensor = torch.reshape(torch.from_numpy(u_in_tmp), (1, 1, nx, nx))
            v_in_tensor = torch.reshape(torch.from_numpy(v_in_tmp), (1, 1, nx, nx))
            
            data_in[ii_data, ii_tensor, 0:nx, 0:nx] = u_in_tensor
            data_in[ii_data, ii_tensor+1, 0:nx, 0:nx] = v_in_tensor
            ii_tensor += 2
        
        elif layer_in in ['N0', 'p0', 'z0', 'rho0', 'znt', 'T0', 'rh0']: # single value layers: get value, fill up entire domain
            
            value_tmp = ds_tmp[layer_in].values
            layer_tmp = np.ones((nx, nx)) * value_tmp
                        
            layer_tmp_tensor = torch.reshape(torch.from_numpy(layer_tmp), (1, 1, nx, nx))
        
            data_in[ii_data, ii_tensor, :, :] = layer_tmp_tensor
            ii_tensor += 1
            
            
        elif layer_in == 'slope': # calculate slope angle from terrain height
            ter = ds_tmp['ter']
            ter = check_for_nan(ter)
            slope = get_slope_angle(ter.values, dx=50)
                        
            slope_tensor = torch.reshape(torch.from_numpy(slope), (1, 1, nx, nx))
            data_in[ii_data, ii_tensor, 0:nx, 0:nx] = slope_tensor
            ii_tensor += 1
        
        
        elif layer_in == 'cos_dd_asp': # calculate cosine of difference angle between slope aspect and ambient wind direction
            ter = ds_tmp['ter']
            ter = check_for_nan(ter)
            dd_in = ds_tmp['dd0'].values
            
            asp = get_aspect(ter.values)
            cosddasp = np.cos(np.deg2rad(asp-dd_in))
            
            cosddasp_tensor = torch.reshape(torch.from_numpy(cosddasp), (1, 1, nx, nx))
            
            data_in[ii_data, ii_tensor, 0:nx, 0:nx] = cosddasp_tensor
            ii_tensor += 1
            
            
        elif layer_in == 'ter_log': # terrain height with log-filter
            ter = ds_tmp['ter'].values
            
            ter_log = np.log(ter + 10)
            
            layer_tmp_tensor = torch.reshape(torch.from_numpy(ter_log), (1, 1, nx, nx))
        
            data_in[ii_data, ii_tensor, :, :] = layer_tmp_tensor
            ii_tensor += 1
        
        elif layer_in == 'ter_sqrt': # terrain height with sqrt-filter
            ter = ds_tmp['ter'].values
            
            ter_log = np.sqrt(ter + 1)
            
            layer_tmp_tensor = torch.reshape(torch.from_numpy(ter_log), (1, 1, nx, nx))
        
            data_in[ii_data, ii_tensor, :, :] = layer_tmp_tensor
            ii_tensor += 1
        
        else:
            layer_tmp = ds_tmp[layer_in]
            
            layer_tmp = check_for_nan(layer_tmp)
            
            layer_tmp_tensor = torch.reshape(torch.from_numpy(layer_tmp.values), (1, 1, nx, nx))
            
            data_in[ii_data, ii_tensor, 0:nx, 0:nx] = layer_tmp_tensor
            ii_tensor += 1
            
    
    # loop over output layers, fill up data_out   
    ii_tensor = 0
    for layer_out in layers_out:
        
        if layer_out == 'ff': # calculate full wind speed from components if separate training for ff
            u_tmp = ds_tmp['um_10'].values
            v_tmp = ds_tmp['vm_10'].values
            
            dd_tmp, ff_tmp = uv2ddff(u_tmp, v_tmp)
            layer_tmp_tensor = torch.reshape(torch.from_numpy(ff_tmp), (1, 1, nx, nx))
            
        elif layer_out in ['dswe', 'subl_vertint']:
            layer_tmp = ds_tmp[layer_out]
            layer_tmp = check_for_nan(layer_tmp)
            
            layer_tmp_tensor = torch.reshape(torch.from_numpy(layer_tmp.values), (1, 1, nx, nx))
        
        else:
            
            layer_tmp = ds_tmp[layer_out]
            layer_tmp = check_for_nan(layer_tmp)
        
            layer_tmp_tensor = torch.reshape(torch.from_numpy(layer_tmp.values), (1, 1, nx, nx))
        
        data_out[ii_data, ii_tensor, 0:nx, 0:nx] = layer_tmp_tensor
        ii_tensor += 1
            
            
    return data_in, data_out
        

def check_for_nan(field_in):
    # searches for nan values and interpolates nearest value, to avoid nan values in training data
    if field_in.isnull().sum().values == 0:
        field_out = field_in
    else: 
        field_out = field_in.interpolate_na(dim='west_east', method='nearest')
        if field_out.isnull().sum().values != 0:
            field_out = field_out.interpolate_na(dim='south_north', method='nearest')
            # print('Number of nan: {}'.format(field_out.isnull().sum()))
    
    return field_out
    
    

def build_last_row(data, nx):
    # mirrors first row of data set to compensate for smaller wrf domain
    
    shape_in = data.shape
    shape_out = (shape_in[0], shape_in[1], shape_in[2]+1, shape_in[3]+1) # build larger output shape
    
    # build new data set, fill up and mirror last row/column
    data_new = torch.empty(shape_out)
        
    data_new[:, :, 0:nx, 0:nx] = data
 
    data_new[:, :, -1, :] = data_new[:, :, 0, :]
    data_new[:, :, :, -1] = data_new[:, :, :, 0]
    data_new[:, :, -1, -1] = data_new[:, :, 0, 0]
    
    return data_new


def get_prediction_dswe(namelist, data_in, ii_train):
    # ms: maybe delete
    ds_dswe_subl = xr.open_dataset('snow_predicted_90_91.nc')
    
    
    dswe_tmp = ds_dswe_subl.dswe.sel(sim_i=ii_train).values
    subl_tmp = ds_dswe_subl.subl.sel(sim_i=ii_train).values
    
    pos_dswe = int(np.argwhere(np.array(namelist_dswe['input_layers']) == 'dswe')) + 1
    pos_subl = int(np.argwhere(np.array(namelist_dswe['input_layers']) == 'subl_vertint')) + 1
      
    data_in[:, pos_dswe, :, :] = torch.from_numpy(dswe_tmp)
    data_in[:, pos_subl, :, :] = torch.from_numpy(subl_tmp)
    
    return data_in



def make_prediction_wind(namelist_dswe, data_in_dswe, data_out_dswe, ii_train, path_to_wrf, prediction_mode='online'):
    # pre-calculated wind field for stacking of u-nets: 
    # make calculation (prediction_mode='online'), save fields (prediction_mode='precalc') or load precalculated fields (prediction_mode='offline')
    print(prediction_mode)
    
    # make calculation of wind field
    if prediction_mode in ['online', 'precalc']:
        # get wind model
        savename_model_uv = 'unet_uv_{}_e2000.pth'.format(namelist_dswe['nn_wind_model_number'])
    
        path_to_namelist = 'nn_wrf_namelists'
        namelist_name_uv = 'unet_wind_{}.json'.format(namelist_dswe['nn_wind_model_number'])
        with open(os.path.join(path_to_namelist, namelist_name_uv), 'r') as fp:
            namelist_uv = json.load(fp)

        path_to_model = 'models_nn'
        model_uv = torch.load(os.path.join(path_to_model, savename_model_uv))
    
        # prepare input data for wind model
        data_in_uv, data_out_uv = nn_wrf_main.get_all_data(namelist_uv, path_to_wrf)
        
        # cut out training data set
        data_in_uv_train = data_in_uv[ii_train, :, :, :]
        data_out_uv_train = data_out_uv[ii_train, :, :, :]
        
        # normalize
        data_in_norm, data_out_norm, norm_factors_in, norm_factors_out = nn_wrf_main.normalize_all_data(namelist_uv, data_in_uv_train, data_out_uv_train)
    
        # save disc space, delete obsolete variables
        del data_in_uv, data_out_uv
        del data_out_uv_train, data_in_uv_train
    
        
        # run wind model, fill up input data set (for online prediction mode) or output data field (for precalculation mode)
        print('make prediction for wind...')
        model_uv.eval()
    
        pos_u = int(np.argwhere(np.array(namelist_dswe['input_layers']) == 'um_10'))
        pos_v = int(np.argwhere(np.array(namelist_dswe['input_layers']) == 'vm_10'))
              
        data_uv_out = np.empty(shape=(len(ii_train), 2, namelist_uv['nx_data'], namelist_uv['nx_data']))
        
        # loop over all, make prediction, fill up
        for ii in range(namelist_dswe['n_data']):
            pred_i, ter_i, label_i = nn_wrf_diag.make_prediction_test(ii, model_uv, data_in_norm, data_out_norm, namelist_uv, norm_factors_in, norm_factors_out, unnorm_out=True)
    
            if prediction_mode == 'online':
                data_in_dswe[ii, pos_u, :, :] = pred_i[:, 0, :, :]
                data_in_dswe[ii, pos_v, :, :] = pred_i[:, 1, :, :]
            else:
                data_uv_out[ii, 0, :, :] = pred_i[:, 0, :, :].cpu().detach().numpy()
                data_uv_out[ii, 1, :, :] = pred_i[:, 1, :, :].cpu().detach().numpy()
    
    # offline mode: load precalculated wind fields, fill up input data
    elif prediction_mode == 'offline':
        ds_uv = xr.open_dataset('uv_predicted_{}.nc'.format(namelist_dswe['nn_wind_model_number']))
        
        pos_u = int(np.argwhere(np.array(namelist_dswe['input_layers']) == 'um_10')) + 1
        pos_v = int(np.argwhere(np.array(namelist_dswe['input_layers']) == 'vm_10')) + 1
                
        u_tmp = ds_uv.uv.sel(u_v='u').sel(sim_i=ii_train).values
        u_tensor = torch.from_numpy(u_tmp)
        
        v_tmp = ds_uv.uv.sel(u_v='v').sel(sim_i=ii_train).values
        v_tensor = torch.from_numpy(v_tmp)
        
        data_in_dswe[:, pos_u, :, :] = u_tensor
        data_in_dswe[:, pos_v, :, :] = v_tensor
    
    # precalculation mode: write predicted wind fields in data set and save
    if prediction_mode == 'precalc':
        print('saving...')
        ds_uv_out = xr.Dataset(
            coords=dict(
                x=('x', np.arange(namelist_uv['nx_data'])),
                y=('y', np.arange(namelist_uv['nx_data'])),
                u_v=('u_v', ['u', 'v']),
                sim_i=('sim_i', np.arange(namelist_dswe['n_data']))),
            data_vars=dict(
                uv=(['sim_i', 'u_v', 'x', 'y'], data_uv_out))
            )
        ds_uv_out.to_netcdf('uv_predicted_{}.nc'.format(namelist_dswe['nn_wind_model_number']))
        
    return data_in_dswe
        
    
def make_logtrans_subl(data_out, namelist, offset_trans=0.0001):
    # make log-transformation for sublimation field
    
    # get position of sublimation field
    pos_subl = int(np.argwhere(np.array(namelist['output_layers']) == 'subl_vertint'))
    subl_tmp = data_out[:, pos_subl, :, :]
        
    subl_trans = torch.log(-subl_tmp + offset_trans) # log-transform sublimation, use offset to avoid divergence at 0
   
    data_out[:, pos_subl, :, :] = subl_trans # write into output field
    
    return data_out, offset_trans

def unmake_logtrans_subl(data_out, namelist, offset_trans=0.0001):
    # make backtransformation of log-transform for sublimation field
    
    # get position of sublimation field
    pos_subl = int(np.argwhere(np.array(namelist['output_layers']) == 'subl_vertint'))
    
    if data_out.dim() == 4:
        subl_tmp = data_out[:, pos_subl, :, :]
    else:
        subl_tmp = data_out
    
    # backtransform: use same offset as during transformation (saved in normfactors field) 
    subl_untrans = -(torch.exp(subl_tmp) - offset_trans)
    
    # write into output field
    if data_out.dim() == 4:
        data_out[:, pos_subl, :, :] = subl_untrans
        return data_out
    else:
        return subl_untrans
     
   
def make_logtrans_dswe(data_out, namelist):
    # make log-transformation for dswe field
    
    # get position of dswe
    pos_dswe = int(np.argwhere(np.array(namelist['output_layers']) == 'dswe'))
       
    dswe_tmp = data_out[:, pos_dswe, :, :]    # extract dswe field
        
    # calculate offset value to avoid divergence at 0: minimum value - small offset
    offset_trans = torch.min(dswe_tmp)-0.0001
    
    # make transformation
    dswe_trans = torch.log(dswe_tmp - offset_trans)
    
    # fill up output field
    data_out[:, pos_dswe, :, :] = dswe_trans
    
    return data_out, offset_trans


def unmake_logtrans_dswe(data_out, namelist, offset_trans):
    # make backtransformation of log-transform for dswe field
    
    # get position of dswe
    pos_dswe = int(np.argwhere(np.array(namelist['output_layers']) == 'dswe'))
       
    if data_out.dim() == 4:
        dswe_tmp = data_out[:, pos_dswe, :, :]
    else:
        dswe_tmp = data_out
    
    # backtransform: use same offset as during transformation (saved in normfactors field) 
    dswe_untrans = torch.exp(dswe_tmp) + offset_trans
    
    # fill up output field
    if data_out.dim() == 4:
        data_out[:, pos_dswe, :, :] = dswe_untrans
        return data_out
    else:
        return dswe_untrans
        
    
    
        
        
        


def blow_up_data_in(data, namelist):
    # mirror data set to double size for rotation augmentation
    
    data_big = torch.empty(data.shape[0], data.shape[1], 2*data.shape[2], 2*data.shape[3])
     
    print(data_big.shape)
    
    i_mid1 = int(data.shape[2]/2)
    i_mid2 = i_mid1 + data.shape[2]
    
    print(i_mid1)
    print(i_mid2)
    
    data_big[:, :, i_mid1:i_mid2, i_mid1:i_mid2] = data
    
    data_big[:, :, :i_mid1, i_mid1:i_mid2] = data[:, :, i_mid1:, :]
    data_big[:, :, i_mid2:, i_mid1:i_mid2] = data[:, :, :i_mid1, :]
    data_big[:, :, i_mid1:i_mid2, :i_mid1] = data[:, :, :, i_mid1:]
    data_big[:, :, i_mid1:i_mid2, i_mid2:] = data[:, :, :, :i_mid1]
    data_big[:, :, :i_mid1, :i_mid1] = data[:, :, i_mid1:, i_mid1:]
    data_big[:, :, :i_mid1, i_mid2:] = data[:, :, i_mid1:, :i_mid1]
    data_big[:, :, i_mid2:, :i_mid1] = data[:, :, :i_mid1, i_mid1:]
    data_big[:, :, i_mid2:, i_mid2:] = data[:, :, :i_mid1, :i_mid1]
    
    return data_big
    
##############
# Normalization functions
##############

def normalize_field_meanstd(field, calc_new_factors=True, field_mean=None, field_std=None):
    # z-score normalization: calculate mean and std or use pre-existing factors
    
    if calc_new_factors:
        field_mean = torch.mean(field)
        field_std = torch.std(field)
    
    field_norm = (field - field_mean) / field_std

    return field_norm, field_mean, field_std


def unnormalize_field_meanstd(field, field_mean, field_std):
    # backtransform from normalized space after z-score normalization
    
    field_unnorm = (field * field_std) + field_mean

    return field_unnorm

    
##############
# Augmentation functions
##############

def main_augment(data_in, data_out, namelist, augmentation_online=False, crossvalid=False):
    # run data augmentation: build empty augmented data tensor, call subfunctions to fill up
    
    if augmentation_online:
        data_in_augment = torch.empty_like(data_in)
        data_out_augment = torch.empty_like(data_out)
        n_data_augment = data_in_augment.shape[0]
        n_data_use = 0
    
    else:
        if crossvalid:
            n_data_use = data_in.shape[0]
        else:
            n_testtopo = len(namelist['topo_test'])
            loops_topo = int(namelist['n_data'] / 72)
    
            n_data_use = namelist['n_data'] - (n_testtopo*loops_topo)
    
        n_data_augment = n_data_use * namelist['factor_augment']
    
        data_in_augment = torch.empty(n_data_augment, data_in.shape[1], namelist['nx_data'], namelist['nx_data'])
        data_out_augment = torch.empty(n_data_augment, data_out.shape[1], namelist['nx_data'], namelist['nx_data'])
    
        data_in_augment[:n_data_use, : ,:, :] = data_in
        data_out_augment[:n_data_use, :, :, :] = data_out

    ii_augment = n_data_use
    ii_data = 0
    if namelist['augment_type'] == 'rot_arb': # rotation by random angle
        rotangles = np.random.rand(n_data_augment - n_data_use) * 360
        
        for rotang in rotangles:
            data_in_augment, data_out_augment = make_rotation_all_fields(data_in, data_out, data_in_augment, data_out_augment,
                                                                     rotang, namelist, ii_data, ii_augment)
            ii_augment += 1
            ii_data += 1
            if ii_data >= n_data_use:
                ii_data -= n_data_use
    
    elif namelist['augment_type'] == 'rot_90': # rotation by 90 degree angles
        rotangles = np.ones(n_data_augment - n_data_use)
        if namelist['factor_augment'] == 3:
            rotangles[2*n_data_use:] = 2
        elif namelist['factor_augment'] == 4:
            rotangles[2*n_data_use:3*n_data_use] = 2
            rotangles[3*n_data_use:] = 3
        
        for rotang in rotangles:
            data_in_augment, data_out_augment = make_rot90_all_fields(data_in, data_out, data_in_augment, data_out_augment,
                                                                      rotang, namelist, ii_data, ii_augment)
            ii_augment += 1
            ii_data += 1
            if ii_data >= n_data_use:
                ii_data -= n_data_use    
                
    elif namelist['augment_type'] == 'shift': # shift by random distance and in random direction
        shift_dims = np.random.choice(np.array([0, 1]), n_data_augment - n_data_use)
        shift_vals = np.random.rand(n_data_augment - n_data_use) * ((namelist['nx_data']-1))
        
        
        for (shift_dim_i, shift_val_i) in zip(shift_dims, shift_vals):
            shift_val_i = int(shift_val_i)
            data_in_augment, data_out_augment = make_shift_all_fields(data_in, data_out, data_in_augment, data_out_augment,
                                                                      shift_dim_i, shift_val_i, namelist, ii_data, ii_augment)
            
            ii_augment += 1
            ii_data += 1
            if ii_data >= n_data_use:
                ii_data -= n_data_use
                
    
    elif namelist['augment_type'] == 'noise': # add random noise
        
        for ii_field in range(data_in_augment.shape[1]):
            
            data_in_augment = make_some_noise_field(data_in_augment, ii_field, n_data_use, namelist)
        
    return data_in_augment, data_out_augment


def make_some_noise_field(data, ii_1, n_data_use, namelist):
    # add random noise over field
    
    data_extract = data[:n_data_use, ii_1, :, :].detach().numpy()
            
    range_max_field = data_extract.max()
    range_min_field = data_extract.min()
        
    range_field = np.max([np.abs(range_max_field), np.abs(range_min_field)])
    
    field_noise = np.random.rand(data_extract.shape[0], data_extract.shape[1], data_extract.shape[2])
    data_noisy = data_extract + (field_noise * 2 * range_field / namelist['factor_noise']) - (range_field/namelist['factor_noise'])
    
    data_noisy_tensor = torch.reshape(torch.from_numpy(data_noisy.copy()), (data_noisy.shape[0], namelist['nx_data'], namelist['nx_data']))
    
    data[n_data_use:, ii_1, :, :] = data_noisy_tensor

    return data    
    

def make_shift_all_fields(data_in, data_out, data_in_augment, data_out_augment, shift_dim_i, shift_val_i, namelist, ii_data, ii_augment):
    # shift augmentation: shift field by random distance in x or y, let re-enter on other side
    
    # loop over all input data layers
    ii_in = 0
    for field_name in namelist['input_layers']:
        if field_name == 'uv_in':
            u_tmp = data_in[ii_data, ii_in, :, :].cpu().detach().numpy()
            v_tmp = data_in[ii_data, ii_in+1, :, :].cpu().detach().numpy()
            
            u_tmp_shift = make_shift_field(u_tmp, shift_dim_i, shift_val_i, namelist)
            v_tmp_shift = make_shift_field(v_tmp, shift_dim_i, shift_val_i, namelist)
            
            u_shift_tensor = torch.reshape(torch.from_numpy(u_tmp_shift.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))
            v_shift_tensor = torch.reshape(torch.from_numpy(v_tmp_shift.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))
            
            data_in_augment[ii_augment, ii_in, :, :] = u_shift_tensor
            data_in_augment[ii_augment, ii_in+1, :, :] = v_shift_tensor
            
            ii_in += 2   
            
        else:
            field_tmp = data_in[ii_data, ii_in, :, :].cpu().detach().numpy()
            
            field_shift = make_shift_field(field_tmp, shift_dim_i, shift_val_i, namelist)
            
            field_shift_tensor = torch.reshape(torch.from_numpy(field_shift.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))

            data_in_augment[ii_augment, ii_in, :, :] = field_shift_tensor

            ii_in += 1
        
    # loop over all output data layers, make shift, fill up augmented data tensor
    ii_out = 0
    for field_name in namelist['output_layers']:
        field_tmp = data_out[ii_data, ii_out, :, :].cpu().detach().numpy()
        
        field_shift = make_shift_field(field_tmp, shift_dim_i, shift_val_i, namelist)
                        
        field_shift_tensor = torch.reshape(torch.from_numpy(field_shift.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))
            
        data_out_augment[ii_augment, ii_out, :, :] = field_shift_tensor
            
        ii_out += 1
        
    return data_in_augment, data_out_augment
        
    
def make_shift_field(field, shift_dim, shift_val, namelist):
    # shift single field
    
    field_shift = np.empty_like(field)
    
    shift_val_out = namelist['nx_data'] - shift_val
    
    if shift_dim == 0:
        field_shift[:shift_val, :] = field[shift_val_out:, :]
        field_shift[shift_val:, :] = field[:shift_val_out, :]

    else:
        field_shift[:, :shift_val] = field[:, shift_val_out:]
        field_shift[:, shift_val:] = field[:, :shift_val_out]
        
    return field_shift
        


def make_rot90_all_fields(data_in, data_out, data_in_augment, data_out_augment, rotang, namelist, ii_data, ii_augment):
    # rotation of fiels by random times 90 degree, keep wind components consistent
    
    # loop over all input layers
    ii_in = 0
    for field_name in namelist['input_layers']:
        if field_name == 'uv_in':
            u_tmp = data_in[ii_data, ii_in, :, :].detach().numpy()
            v_tmp = data_in[ii_data, ii_in+1, :, :].detach().numpy()
            
            u_r_tmp = np.rot90(u_tmp, k=-rotang)
            v_r_tmp = np.rot90(v_tmp, k=-rotang)
            
            dd_r_tmp, ff_r_tmp = uv2ddff(u_r_tmp, v_r_tmp)
            
            dd_r_tmp = dd_r_tmp + 90*rotang
            dd_r_tmp[dd_r_tmp >= 360] -= 360
            
            u_r, v_r = ddff2uv(dd_r_tmp, ff_r_tmp)
                        
            u_r_tensor = torch.reshape(torch.from_numpy(u_r.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))
            v_r_tensor = torch.reshape(torch.from_numpy(v_r.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))
            
            data_in_augment[ii_augment, ii_in, :, :] = u_r_tensor
            data_in_augment[ii_augment, ii_in+1, :, :] = v_r_tensor
            
            ii_in += 2         
            
        else:
            field_tmp = data_in[ii_data, ii_in, :, :].detach().numpy()
            field_r = np.rot90(field_tmp, k=-rotang)
                                    
            field_r_tensor = torch.reshape(torch.from_numpy(field_r.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))
            
            data_in_augment[ii_augment, ii_in, :, :] = field_r_tensor
            
            ii_in += 1
           
    # loop over output layers 
    ii_out = 0
    while ii_out > len(namelist['output_layers']):
        if namelist['output_layers'][ii_out] in ['um_10', 'vm_10']:
            u_tmp = data_out[ii_data, ii_out, :, :].detach().numpy()
            v_tmp = data_out[ii_data, ii_out+1, :, :].detach().numpy()
            
            u_r_tmp = np.rot90(u_tmp, k=-rotang)
            v_r_tmp = np.rot90(v_tmp, k=-rotang)
            
            dd_r_tmp, ff_r_tmp = uv2ddff(u_r_tmp, v_r_tmp)
            
            dd_r_tmp = dd_r_tmp + 90*rotang
            dd_r_tmp[dd_r_tmp >= 360] -= 360
            
            u_r, v_r = ddff2uv(dd_r_tmp, ff_r_tmp)
            
            u_r_tensor = torch.reshape(torch.from_numpy(u_r.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))
            v_r_tensor = torch.reshape(torch.from_numpy(v_r.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))
            
            data_out_augment[ii_augment, ii_out, :, :] = u_r_tensor
            data_out_augment[ii_augment, ii_out+1, :, :] = v_r_tensor
            
            ii_out += 2         
            
            
        else:
            field_tmp = data_out[ii_data, ii_out, :, :].detach().numpy()
            field_r = np.rot90(field_tmp, k=-rotang)
                        
            field_r_tensor = torch.reshape(torch.from_numpy(field_r.copy()), (1, 1, namelist['nx_data'], namelist['nx_data']))
            
            data_out_augment[ii_augment, ii_out, :, :] = field_r_tensor
            
            ii_out += 1
    
    return data_in_augment, data_out_augment
    

def make_rotation_all_fields(data_in, data_out, data_in_augment, data_out_augment, rotang, namelist, ii_data, ii_augment):
    # rotation by random angle, keep wind components consistent:
    # mirror field to enlarge, rotate coordintates, interpolate to new, rotated coordinates 
    
    nx = namelist['nx_data']
    x = np.arange(nx)
    y = np.arange(nx)

    xmid = nx/2
    xm = x - xmid
    ym = y - xmid
    xx, yy = np.meshgrid(xm, ym)

    # unit vectors of rotated coordinate system
    e1r = np.array([np.cos(np.deg2rad(rotang)), -np.sin(np.deg2rad(rotang))])
    e2r = np.array([np.sin(np.deg2rad(rotang)), np.cos(np.deg2rad(rotang))])

    # coodinates of rotated field in un-rotated space
    xxr = xx * e1r[0] + yy * e2r[0]
    yyr = xx * e1r[1] + yy * e2r[1]
    
    xl = np.arange(3*nx) - 1.5*nx
    yl = np.arange(3*nx) - 1.5*nx
    
    # loop over all input fields, make rotation and fill up
    ii_in = 0
    for field_name in namelist['input_layers']:
        if field_name == 'uv_in':
            u_tmp = data_in[ii_data, ii_in, :, :].detach().numpy()
            v_tmp = data_in[ii_data, ii_in+1, :, :].detach().numpy()
            
            ur_tmp, vr_tmp = make_rotation_field(u_tmp, xxr, yyr, xl, yl, rotang, namelist, make_uv=True, field2=v_tmp)
            
            ur_tmp_tensor = torch.reshape(torch.from_numpy(ur_tmp), (1, 1, nx, nx))
            vr_tmp_tensor = torch.reshape(torch.from_numpy(vr_tmp), (1, 1, nx, nx))
            
            data_in_augment[ii_augment, ii_in, :, :] = ur_tmp_tensor
            data_in_augment[ii_augment, ii_in+1, :, :] = vr_tmp_tensor
            
            ii_in += 2
        else:
            field_tmp = data_in[ii_data, ii_in, :, :].detach().numpy()
            field_r = make_rotation_field(field_tmp, xxr, yyr, xl, yl, rotang, namelist)
            
            field_r_tensor = torch.reshape(torch.from_numpy(field_r), (1, 1, nx, nx))
            
            data_in_augment[ii_augment, ii_in, :, :] = field_r_tensor
            
            ii_in += 1
       
    # loop over all output fields, make roation and fill up
    ii_out = 0
    while ii_out > len(namelist['output_layers']):
        if namelist['output_layers'][ii_out] in ['um_10', 'vm_10']:
            u_tmp = data_out[ii_data, ii_out, :, :].detach().numpy()
            v_tmp = data_out[ii_data, ii_out+1, :, :].detach().numpy()
            
            ur_tmp, vr_tmp = make_rotation_field(u_tmp, xxr, yyr, xl, yl, rotang, namelist, make_uv=True, field2=v_tmp)
            
            ur_tmp_tensor = torch.reshape(torch.from_numpy(ur_tmp), (1, 1, nx, nx))
            vr_tmp_tensor = torch.reshape(torch.from_numpy(vr_tmp), (1, 1, nx, nx))
            
            data_out_augment[ii_augment, ii_out, :, :] = ur_tmp_tensor
            data_out_augment[ii_augment, ii_out+1, :, :] = vr_tmp_tensor
            
            ii_out += 2
        else:
            field_tmp = data_out[ii_data, ii_out, :, :].detach().numpy()
            field_r = make_rotation_field(field_tmp, xxr, yyr, xl, yl, rotang, namelist)
            
            field_r_tensor = torch.reshape(torch.from_numpy(field_r), (1, 1, nx, nx))
            
            data_out_augment[ii_augment, ii_out, :, :] = field_r_tensor
            
            ii_in += 1
            
    return data_in_augment, data_out_augment
    

def make_rotation_field(field, xxr, yyr, xl, yl, rotang, namelist, make_uv=False, field2=None):
    # make rotaion of single field: mirror field to double size, interpolate big field to coordinates on rotated field
    xyr = np.stack((xxr, yyr), axis=2)
    
    if make_uv:
        u_big = make_matrix_big(field, len(xl), namelist['nx_data'])
        v_big = make_matrix_big(field2, len(xl), namelist['nx_data'])                       
        
        ur = np.transpose(interpolate.interpn((xl, yl), u_big, xyr))
        vr = np.transpose(interpolate.interpn((xl, yl), v_big, xyr))
    
        ddr0, ffr = uv2ddff(ur, vr)
        ddr1 = ddr0 + rotang
        ddr1[ddr1 >= 360] = ddr1[ddr1 >= 360] - 360 
        ur1, vr1 = ddff2uv(ddr1, ffr)
        
        return ur1, vr1
        
    else: 
        field_big = make_matrix_big(field, len(xl), namelist['nx_data'])
        
        field_r = np.transpose(interpolate.interpn((xl, yl), field_big, xyr))
        
        return field_r
    

def make_matrix_big(data, lenbig, nx):
    # mirror data to enlarge matrix
    
    dl = np.zeros((lenbig, lenbig))

    dl[:nx, :nx] = data
    dl[nx:2*nx, :nx] = data
    dl[2*nx:, :nx] = data

    dl[:nx, nx:2*nx] = data
    dl[:nx, 2*nx:] = data
    dl[2*nx:, 2*nx:] = data

    dl[nx:2*nx, nx:2*nx] = data
    dl[nx:2*nx, 2*nx:] = data
    dl[2*nx:, nx:2*nx] = data
    
    return dl


##############
# Geometric helping functions
##############

def make_all_sx():
    
    path_to_topo = '../../env/topo_synth_256_2_smooth'
    path_to_save = 'sx300_all'
    
    for ii in range(1, 73):
        tstart = time.time()
        print('--------')
        print(ii)
        
        if ii < 10:
            filename_topo = 'topo_0{}_smooth'.format(ii)
            filename_sx = 'sx300_topo_0{}.nc'.format(ii)
        else:
            filename_topo = 'topo_{}_smooth'.format(ii)
            filename_sx = 'sx300_topo_{}.nc'.format(ii)
        
        dem_tmp = np.loadtxt(os.path.join(path_to_topo, filename_topo), skiprows=1, delimiter=',')
        make_fields_sx(dem_tmp, 5, 300, path_to_save, filename_sx)
        
        tend = time.time()
        dt = tend - tstart
        dt_m = np.floor(dt / 60)
        dt_s = np.floor(dt - (dt_m*60))
        print('Time to process: {} min {} s'.format(int(dt_m), int(dt_s)))
    

def make_fields_sx(dem, dalpha, dist_rad, path_to_save, savename):
    
    # transpose
    dem_t = dem.T
    
    # expand
    dx = 50
    x_pl = int(dist_rad / dx)
    
    nx = np.shape(dem)[0]
      
    dem_g = np.empty((nx+2*x_pl, nx+2*x_pl))
    
    dem_g[x_pl:nx+x_pl, x_pl:nx+x_pl] = dem_t
    dem_g[:x_pl, x_pl:nx+x_pl] = dem_t[nx-x_pl:, :]
    dem_g[nx+x_pl:, x_pl:nx+x_pl] = dem_t[:x_pl, :]
    
    dem_g[x_pl:nx+x_pl, :x_pl] = dem_t[:, nx-x_pl:]
    dem_g[x_pl:nx+x_pl:, nx+x_pl:] = dem_t[:, :x_pl]
    
    dem_g[:x_pl, :x_pl] = dem_t[nx-x_pl:, nx-x_pl:]
    
    dem_g[nx+x_pl:, :x_pl] = dem_t[:x_pl, nx-x_pl:]
    dem_g[nx+x_pl:, nx+x_pl:] = dem_t[:x_pl, :x_pl]
    dem_g[:x_pl, nx+x_pl:] = dem_t[nx-x_pl:, :x_pl]
    
    xx, yy = np.meshgrid(np.arange(0, nx+2*x_pl)*dx, np.arange(0, nx+2*x_pl)*dx)
    # print(np.shape(xx))
    
        
    # predefine out
    sx_dd = []
    for dd_tmp in range(0, 360, dalpha):
        sx_tmp_single = np.empty_like(dem_t)
        sx_dd.append(sx_tmp_single)       
    
    
    # print(len(sx_dd))
    # sx
    for xi, xi_g in enumerate(range(x_pl, nx+x_pl)):
        if xi % 10 == 0:
            print(xi)
        for yi, yi_g in enumerate(range(x_pl, nx+x_pl)):
            if xi % 2 == 0 and yi % 2 == 0:
                sx_tmp = calc_sx_xy(dem_g, xx, yy, xx[xi_g, yi_g], yy[xi_g, yi_g], dem_g[xi_g, yi_g], dalpha=dalpha, dist_rad=dist_rad)
            
                # print(len(sx_tmp))
                for ii, sx_i in enumerate(sx_tmp):
                    # print(ii, sx_i)
                    sx_dd[ii][xi, yi] = sx_i
            elif xi % 2 == 0 and yi % 2 != 0:
                for ii in range(len(sx_dd)):
                    sx_dd[ii][xi, yi] = sx_dd[ii][xi-1, yi]
            elif xi % 2 != 0 and yi % 2 == 0:
                for ii in range(len(sx_dd)):
                    sx_dd[ii][xi, yi] = sx_dd[ii][xi, yi-1]
            else:
                for ii in range(len(sx_dd)):
                    sx_dd[ii][xi, yi] = sx_dd[ii][xi-1, yi-1]
    # save sx
    sx_np_out = np.empty((len(sx_dd), nx, nx))
    for ii in range(len(sx_dd)):
        sx_np_out[ii, : , :] = sx_dd[ii]
        
    ds_out = xr.Dataset(coords={'x':np.arange(nx), 'y':np.arange(nx), 'dd':np.arange(0, 360,dalpha)})
    ds_out['sx'] = (('dd', 'x', 'y'), sx_np_out)
    
    ds_out.to_netcdf(os.path.join(path_to_save, savename))
   
   
def cart2pol(x, y):
    # transform cartesian to polar coordinates
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    # transform polar to cartesian coordinates
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def ddff2uv(dd, ff):
    """calculates wind components u and v out of wind direction dd and wind speed ff"""
    # convert dd in radian
    dd_rad = np.deg2rad(dd)
    
    # calculate angle between ff and u
    alfa = (3 * np.pi / 2) - dd_rad
    
    # calculate u and v
    (u, v) = pol2cart(ff, alfa)
    
    return u, v


def uv2ddff(u, v):
    """calculates wind direction and wind speed out of wind components u and v"""
    # transform coordinates 
    (ff, dd_rad) = cart2pol(u,v)
    
    # dd in degree
    dd = np.rad2deg(dd_rad)
    
    # dd in meteorological sense
    dd = 270 - dd
    
    # filter values < 0 and > 360
    dd = np.array(dd)
    
    dd[dd < 0] = dd[dd < 0] + 360
    dd[dd > 360] = dd[dd > 360] - 360
    
    return dd, ff     


def get_slope_angle(dem, dx=30):
    """
    calculate maximum slope angle of dem

    Parameters
    ----------
    dem : ndarray
        Digital elevation model.
    dx : int, optional
        Horizontal grid spacing. The default is 30.

    Returns
    -------
    alpha : ndarray
        maximum slope angle.

    """
    gr_tt = np.gradient(np.abs(dem))
    gr_tt_abs = np.sqrt(gr_tt[0]**2 + gr_tt[1]**2)/dx
    alpha = np.rad2deg(np.arctan(gr_tt_abs))
    
    return alpha


def get_distance(xi, yi, xx, yy):
    """
    calculate distance from one point to all other points in raster
    
    Parameters
    ----------
    xi : int
        x-coordiante of point of interest.
    yi : int
        y-coordinate of point of interest.
    xx : ndarray
        x-coordinates of grid.
    yy : ndarray
        y-coordinates of grid.

    Returns
    -------
    di : ndarray
        distance for every point in grid to x0,y0.

    """
    
    di = np.sqrt((xx - xi)**2 + (yy - yi)**2)
    
    return di


def get_tpi(dem, xx, yy, rad_search=4000, delta_x_fast=1):
    """
    calculate topographic position index based on a defined search radius

    Parameters
    ----------
    dem : ndarray
        Digital elevation model 
    xx : ndarray
        Coordiantes in x
    yy : ndarray
        Coordinates in y.
    rad_search : int, optional
        search radius for tpi calcualtion. The default is 4000.

    Returns
    -------
    tpi : ndarray
        topographic position index.

    """

    tpi = np.nan*np.empty(np.shape(dem))
    
    for ii in range(0, np.shape(tpi)[0], delta_x_fast):
        # print(ii)
        for jj in range(0, np.shape(tpi)[1], delta_x_fast):
            di = get_distance(xx[ii, jj], yy[ii, jj], xx, yy)
            mask_rad = di <= rad_search
            #print(np.sum(mask_rad))
            
            tpi[ii, jj] = dem[ii, jj] - np.mean(dem[mask_rad])
            
    return tpi


def get_tpi_expand(topo, rad_search, dx):
    
    
    tpi = np.empty(np.shape(topo))
    nx = 256
    
    delta_expand = int(rad_search/dx)
    nx_big = nx+2*delta_expand   
    
    topo_expand = np.empty((nx+2*delta_expand, nx+2*delta_expand))
    topo_expand[delta_expand:nx+delta_expand, delta_expand:nx+delta_expand] = topo
    topo_expand[:delta_expand, delta_expand:nx+delta_expand] = topo[nx-delta_expand:, :]
    topo_expand[nx_big-delta_expand:, delta_expand:nx+delta_expand] = topo[:delta_expand, :]
    topo_expand[delta_expand:nx+delta_expand, :delta_expand] = topo[:, nx-delta_expand:]
    topo_expand[delta_expand:nx+delta_expand, nx_big-delta_expand:] = topo[:, :delta_expand]
    topo_expand[:delta_expand, :delta_expand] = topo[nx-delta_expand:, nx-delta_expand:]
    topo_expand[:delta_expand, nx_big-delta_expand:] = topo[nx-delta_expand:, :delta_expand]
    topo_expand[nx_big-delta_expand:, :delta_expand] = topo[:delta_expand, nx-delta_expand:]
    topo_expand[nx_big-delta_expand:, nx_big-delta_expand:] = topo[:delta_expand, :delta_expand]
    
    xbig = np.arange(nx_big)*dx
    ybig = np.arange(nx_big)*dx
    xxbig, yybig = np.meshgrid(xbig,ybig)    
    
    for ii in range(nx):
        for jj in range(nx):
            ii_expand = ii+delta_expand
            jj_expand = jj+delta_expand
            di = get_distance(xxbig[ii_expand, jj_expand], yybig[ii_expand, jj_expand], xxbig, yybig)
            
            mask_rad = di <= rad_search
            
            tpi[ii, jj] = topo[ii, jj] - np.mean(topo_expand[mask_rad])
            
    return tpi
            

def calc_sx_xy(dem, xx, yy, x0, y0, zi, dalpha=30, dist_rad=300):
    """
    calculate maximum upwind slopeangle for single point, for wind direction sectors, based on search radius

    Parameters
    ----------
    dem : ndarray
        Digital elevation model 
    xx : ndarray
        Coordiantes in x
    yy : ndarray
        Coordinates in y.
    x0 : int
        x-coordinate of point of interest.
    y0 : int
        y-coordinate of point of interest.
    zi : int
        height of point of interest.
    dalpha : int, optional
        size of search sector in degree. The default is 30.
    dist_rad : int, optional
        search radius. The default is 300.

    Returns
    -------
    sx : list
        maximum upwind slope angle for each wind sector.

    """

    di = get_distance(x0, y0, xx, yy)
    
    ang, dist_to = uv2ddff(x0-xx, y0-yy)
    
    mask_dist = di <= dist_rad
    
    slopeangle = np.arctan((dem-zi)/di)

    sx = []
    for ang0 in range(0, 360, dalpha):
        
        ang1 = ang0 + dalpha
        
        mask_ang = np.logical_and(ang >= ang0, ang < ang1)
        mask_ang_di = np.logical_and(mask_ang, mask_dist)
        if np.sum(mask_ang_di) > 0:
            sx_tmp = np.nanmax(slopeangle[mask_ang_di])
            sx.append(sx_tmp)
        else:
            ang_mid = np.mean([ang0, ang1])
            delta_ang = ang - ang_mid
            delta_ang_di = np.where(di <= dist_rad, delta_ang, np.nan)
            mask_ang_di = np.abs(delta_ang_di) == np.min(np.abs(delta_ang_di[~np.isnan(delta_ang_di)]))
            sx_tmp = np.nanmax(slopeangle[np.abs(delta_ang_di) == np.min(np.abs(delta_ang_di[~np.isnan(delta_ang_di)]))])
            sx.append(sx_tmp)
    
    return sx

def get_laplacian(dem, xx, yy, dx=30):
    """
    Calculate laplacian of dem

    Parameters
    ----------
    dem : ndarray
        Digital elevation model 
    xx : ndarray
        Coordiantes in x
    yy : ndarray
        Coordinates in y.
    dx : int, optional
        horizontal grid spacing. The default is 30.

    Returns
    -------
    lap : ndarray
        Laplacian of dem.

    """

    index_i = np.arange(1, np.shape(xx)[0]-1)
    index_ip = np.arange(2, np.shape(xx)[0])
    index_im = np.arange(0, np.shape(xx)[0]-2)
    
    lap = np.zeros(np.shape(dem))
    
    lap[index_i[0]:index_i[-1], index_i[0]:index_i[-1]] = (dem[index_ip[0]:index_ip[-1], index_i[0]:index_i[-1]] + 
                                                          dem[index_im[0]:index_im[-1], index_i[0]:index_i[-1]] + 
                                                          dem[index_i[0]:index_i[-1], index_ip[0]:index_ip[-1]] + 
                                                          dem[index_i[0]:index_i[-1], index_im[0]:index_im[-1]] - 
                                                          4*dem[index_i[0]:index_i[-1], index_i[0]:index_i[-1]]) / dx**2  
   
    return lap

def get_aspect(dem):
    """"
    Calculate slope aspect angle
    
    Parameters
    ----------
    dem : ndarray
        Digital elevation model 

    Returns
    -------
    aspect : ndarray
        Aspect slope angle of dem.
    
    """
    gr_tt = np.gradient(np.abs(dem))
    
    aspect = np.rad2deg(np.arctan2(gr_tt[1], gr_tt[0])) + 180

    
    return aspect



