#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:54:44 2024

@author: manuel
"""
import numpy as np
import json
import os


def open_namelist(namelist_name, path_to_namelist):
    
    with open(os.path.join(path_to_namelist, namelist_name), 'r') as fp:
        namelist = json.load(fp)   
    
    return namelist


def get_namelist(savename='', save_list=False, path_to_save=''):
    
    namelist = {
        'model':'unet_devine_circpad_lrelu_5pool',
        
        'learning_rate':0.01,
        'optimizer':'sgd',
        'loss_fn':'mse',
        'loss_huber_delta':0.5,
        'epochs':5,
        'batch_size':8,
        'topo_test':[65, 66, 67, 68, 69, 70, 71, 72],
        'validation_split':1/8,
        'weight_init':'xavier_uni',
        
        'input_layers':['ter', 'uv_in', 'N0', 'cos_dd_asp'],# 'rho0', 'um_10', 'vm_10'],
        # 'input_layers':['ter', 'uv_in', 'N0', 'p0', 'z0', 'rho0', 'T10', 'rh10','T100', 'rh100', 'T1000', 'rh1000','cos_dd_asp'],
        # 'output_layers':['um_10', 'vm_10'],
        'output_layers':['dswe'] ,#'subl_vertint'],# 'phiflux_u_vertint', 'phiflux_v_vertint'],
        # 'output_layers':['phiflux_u_vertint', 'phiflux_v_vertint'],
        # 'normalization':[['minmax_01', 'minmax_11_p0', 'minmax_11_p0', 'minmax_11_p0', 'minmax_11_p0', 'minmax_01'], ['minmax_11_p0']],
        # 'normalization':[['minmax_01', 'minmax_11_p0', 'minmax_11_p0', 'minmax_01', 'minmax_11_p0'], ['minmax_11_p0']],
        'normalization':'meanstd',
        # 'normalization':'minmax_01',
        'batch_norm':False,
        
        
        'data_augmentation':False,
        'factor_augment':4,
        'augment_type':'rot_90',
        
        'n_data':360,
        'nx_wrf':255,
        'nx_data':256,
        
        'plot_epochs_evo':False,
        'reduce_lr':True,
        'use_nn_wind':False,
        'nn_wind_model_number':0,
        'surrounding_model':False,
        'surrounding_model_double':True
        
        
        }
    
    if save_list:
        
        with open(os.path.join(path_to_save, savename), 'w') as nl:
            json.dump(namelist, nl)

    return namelist        



    
    
    
