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

import torch

import snowstorm_helpers.util_funct_meteo
import snowstorm_helpers.snowstorm_config
from snowstorm_helpers.snowstorm_prep_input import get_data_input
from snowstorm_helpers.snowstorm_prep_output import unnorm_data, untrans_data, write_ds_pred
import snowstorm_helpers.helpers




def main(path_to_models, path_to_namelists, path_to_in, type_input, num_model_wind, path_to_dem, region_id, infile1, 
         infile2=None, vars_pred=['wind'], num_model_dswe=94, num_model_subl=95, num_model_phi=68, filter_ter=False,
         type_dem='tdx', interpolate_era=False, single_value_rho0=True, rho0_value=100, single_value_znt=True, znt_value=0.0001,
         i_time_era=0, list_of_synthin=[], make_plot=False, path_to_save='', var_sens='', val_sens='', num_topo=1):

    """
    Main function to run snowstorm with various input for single time instace

    Parameters
    ----------

    path_to_models: str
            path where snowstorm model scripts are located
    path_to_namelists: str
            path where snowstorm namelists are located (default same as path_to_models)
    path_to_in: str
            path where atmospheric input files are located
    type_input: str
            type of atmospheric input (currently: ERA: ERA5 input, WRF: WRF input, SYNTH: synthetic input)
    num_model_wind: int (maybe get rid)
            internal model version number for wind
    path_to_dem: str
            path where input topographic data is located
    region_id: str
            region identifier to run snowstorm for specific region, connects to snowstorm_config.py
    infile1: str
            name of atmospheric input data file
    infile2: str, optional
            name of secondary atmospheric input data file (necessary for ERA5 input: infile1: surface variables, infile2: pressure level variables)
    vars_pred: list, optional
            list of variables to run snowstorm (possible variables: 'wind', 'dswe', 'subl', 'phi'; wind needed for other variables as input)
    num_model_dswe: int, optional
            internal model version number for dswe
    num_model_subl: int, optional
            internal model version number for subl
    num_model_phi: int, optional
            internal model version number for phi
    filter_ter: bool, optional
            flag to apply 1-2-1 filter on input dem
    type_dem: str, optional
            type of input dem (currently: 'tdx': Tandem-X DEM, 'wrfles': same topography as WRF-LES, 'fourierland': synthetic fourier topography)
    interpolate_era: bool, optional
            flag to interpolate ERA 5 field to SNOWstorm grid or use closest neighbour
    single_value_rho0: bool, optional
            flag to use single value for snow density over entire domain (currently: only True works, planned: coupling to snow evolution model)
    rho0_value: int, optional
            value to use as snow density (kg m-3)
    single_value_znt: bool, optional
            flag to use single value for aerodynamic roughness lenght over entire domain (currently: only True works, planned: coupling to snow evolution model)
    znt_value: int, optional
            value to use as aerodynamic roughness length (m)
    i_time_era: int, optional
            for ERA5 input with multi-temporal file: timestep in ERA5 file
    list_of_synthin: list, optional
            for synthetic atmospheric input: list of driving values [ff0, dd0, N0, z0, p0, T0, rh0, znt, rho0]
    
    Returns
    -------
    ds_pred: xr Dataset
            SNOWstorm prediction for single time instance for varibles specified in vars_pred
"""
         
         
             
    # get meteological constants from helping function
    const = util_funct_meteo.constants()
    
    
    # get model setup
    print('### WIND ###')
    print('---prepare model---')
    # get model, specifications and normalization factors
    namelist_wind, normfactors_wind, model_wind = prepare_model(path_to_models, path_to_namelists, 'wind', num_model_wind, eps_model=2000)

    print('---prepare dem---')
    # select and prepare input topography
    if type_dem == 'tdx':
        config_dem = snowstorm_config.get_config_dem()
        dem_glacier_np, dem_glacier_xr = get_dem_tdx(path_to_dem, config_dem, region_id)
        if filter_ter:
            dem_glacier_np = snowstorm_helpers.helpers.filter_121(dem_glacier_np)
    elif type_dem == 'wrfles':
        dem_glacier_np, dem_glacier_xr = get_dem_wrfles(path_to_dem) 
    elif type_dem == 'fourierland':
        dem_glacier_np, dem_glacier_xr = get_dem_fourierland(path_to_dem, num_topo)
            

    print('---prepare input---')
    # select and prepare input (era, wrf, synth)
    config_input = snowstorm_config.get_config_input()
    data_in_wind = get_data_input(dem_glacier_np, dem_glacier_xr, region_id, infile1, infile2, namelist_wind, normfactors_wind,
                                  path_to_in, type_input, type_dem, config_input, const,  i_time_era=i_time_era, interpolate_era=False,
                                  single_value_rho0=single_value_rho0, rho0_value=rho0_value, single_value_znt=single_value_znt, znt_value=znt_value,
                                  list_of_synthin=list_of_synthin)
    
    print('---run model ---')
    # run model
    pred_wind = model_wind(data_in_wind)
    # unnormalize model output
    pred_wind_unnorm = unnorm_data(pred_wind, normfactors_wind, var_pred='wind')
    
    # write to list to later use in output dataset generation
    list_of_predictions = [pred_wind_unnorm]
    
    # if required: snow mass change rate
    if 'dswe' in vars_pred:
        print('### DSWE ###')
        print('---prepare model---')
        # get model, specifications and normalization factors
        namelist_dswe, normfactors_dswe, model_dswe = prepare_model(path_to_models, path_to_namelists, 'dswe', num_model_dswe, eps_model=1000)
        
        print('---prepare input---')
        # prepare input (use same input as for wind + output of wind model)
        data_in_snow = get_data_input_snow(data_in_wind, pred_wind, normfactors_dswe, namelist_dswe)
                
        print('---run model ---')
        # run model
        pred_dswe = model_dswe(data_in_snow)
        
        # unnormalize and back-transform model output 
        pred_dswe_unnorm = unnorm_data(pred_dswe, normfactors_dswe, var_pred='dswe')
        pred_dswe_untrans = untrans_data(pred_dswe_unnorm, normfactors_dswe['offset_trans_dswe'], 'dswe')
        
        # append output to list of predictions
        list_of_predictions.append(pred_dswe_untrans)

    
    # if required: drifting snow sublimation rate
    if 'subl' in vars_pred:
        print('### SUBL ###')
        print('---prepare model---')
        # get model, specifications and normalization factors
        namelist_subl, normfactors_subl, model_subl = prepare_model(path_to_models, path_to_namelists, 'subl', num_model_subl, eps_model=1000)
        
        print('---prepare input---')
        # prepare input (use same input as for wind + output of wind model)
        
        print('---run model ---')
        # run model
        pred_subl = model_subl(data_in_snow)
        
        # unnormalize, back-transform model prediction
        pred_subl_unnorm = unnorm_data(pred_subl, normfactors_subl, var_pred='subl')
        pred_subl_untrans = untrans_data(pred_subl_unnorm, normfactors_subl['offset_trans_subl'], 'subl')

        # append output to list of predictions
        list_of_predictions.append(pred_subl_untrans)
 
    # if required: snow transport rate   
    if 'phi' in vars_pred:
        print('### IST ###')
        print('---prepare model---')
        # get model, specifications and normalization factors
        namelist_phi, normfactors_phi, model_phi = prepare_model(path_to_models, path_to_namelists, 'phi', num_model_phi, eps_model=3000)
        
        print('---prepare input---')
        # prepare input (use same input as for wind + output of wind model)
        
        print('---run model ---')
        # run model
        pred_phi = model_phi(data_in_snow)
        
        # unnormalize model prediction
        pred_phi_unnorm = unnorm_data(pred_phi, normfactors_phi, var_pred='phi')
        pred_phi_untrans = untrans_data(pred_phi_unnorm, None, 'phi')
        
        # append output to list of predictions
        list_of_predictions.append(pred_phi_untrans)

    
    print('---post processing---')
    # write all predictions into single data set
    ds_pred = write_ds_pred(list_of_predictions, dem_glacier_np, dem_glacier_xr,  vars_pred, type_dem)
   
    return ds_pred
    
    




    
 






   
 












def prepare_model(path_to_models, path_to_namelists, var_model, num_model, eps_model=2000):
    """
    Load model and get model specifications
    
    Parameters
    ----------
    path_to_models: str
            path where snowstorm model scripts are located
    path_to_namelists: str
            path where snowstorm namelists are located (default same as path_to_models)
    var_model: str
            model output variable ('wind', 'dswe', 'subl', 'phi')
    num_model: int (maybe get rid)
            internal model version number
    eps_model: int (maybe get rid)
            number of epochs during model training
    
    Returns
    -------
    namelist: dict
            namelist of model specifications
    normfactors: dict
            factors for input and output normalization, back transformation
    model: torch
            pytorch model instance
    """

    #print(eps_model)
    if var_model == 'wind':
        with open(os.path.join(path_to_namelists, 'unet_wind_{}.json'.format(num_model)), 'r') as fp:
            namelist = json.load(fp)  
        with open(os.path.join(path_to_models, 'norm_factors_wind_{}.json'.format(num_model)), 'r') as fp:
            normfactors = json.load(fp) 
            
        model = torch.load(os.path.join(path_to_models, 'unet_uv_{}_e{}.pth'.format(num_model, eps_model)), weights_only=False)
        
    elif var_model == 'dswe':     
        with open(os.path.join(path_to_namelists, 'unet_dswe_{}.json'.format(num_model)), 'r') as fp:
            namelist = json.load(fp)  
        with open(os.path.join(path_to_models, 'norm_factors_dswe_{}.json'.format(num_model)), 'r') as fp:
            normfactors = json.load(fp)  
    
        model = torch.load(os.path.join(path_to_models, 'unet_dswe_{}_e{}.pth'.format(num_model, eps_model)), weights_only=False)

    elif var_model == 'subl':
        with open(os.path.join(path_to_namelists, 'unet_dswe_{}.json'.format(num_model)), 'r') as fp:
            namelist = json.load(fp)  
        with open(os.path.join(path_to_models, 'norm_factors_dswe_{}.json'.format(num_model)), 'r') as fp:
            normfactors = json.load(fp)  
    
        model = torch.load(os.path.join(path_to_models, 'unet_dswe_{}_e{}.pth'.format(num_model, eps_model)), weights_only=False)

    elif var_model == 'phi':   
        with open(os.path.join(path_to_namelists, 'unet_phi_{}.json'.format(num_model)), 'r') as fp:
            namelist = json.load(fp)  
        with open(os.path.join(path_to_models, 'norm_factors_phi_{}.json'.format(num_model)), 'r') as fp:
            normfactors = json.load(fp)  
        
        model = torch.load(os.path.join(path_to_models, 'unet_phi_{}_e{}.pth'.format(num_model, eps_model)), weights_only=False)
    
    return namelist, normfactors, model
    

if __name__ == '__main__':
    main('models_nn', 'nn_wrf_namelists', 'data_era', 'ERA', 62, 'dem_hef', 'hef', 'era5_hef_20212_sfc.nc', infile2='era5_hef_20212_plevel.nc', interpolate_era=False, i_time_era=18)

    
