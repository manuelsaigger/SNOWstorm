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

import snowstorm_helpers.helpers



###################
# INPUT PREPARATION

def get_data_input(dem_glacier_np, dem_glacier_xr, region_id, infile1, infile2, namelist, normfactors, 
                   path_to_in, type_input, type_dem, config_input, const, i_time_era=0, interpolate_era=False,
                   single_value_rho0=True, rho0_value=100, single_value_znt=True, znt_value=0.0001, list_of_synthin=[],
                   usnowstorm=None, vsnowstorm=None, var_out='wind'):
    """
    Prepares input data for SNOWstorm: gets atmospheric input data, normalizes input, writes in form usable for model
    Calls get_data_in_era, get_data_in_wrf, get_data_in_synth
    
    Parameters
    ----------
    dem_glacier_np: ndarray
            numpy array of terrain height
    dem_glacier_xr: xr Dataset
            xarray Dataset of terrain height, with coordinates
    region_id: str
            region identifier to run snowstorm for specific region, connects to snowstorm_config.py
    infile1: str
            name of atmospheric input data file
    infile2: str, optional
            name of secondary atmospheric input data file (necessary for ERA5 input: infile1: surface variables, infile2: pressure level variables)
    namelist: dict
            namelist of model specifications
    normfactors: dict
            factors for input and output normalization, back transformation
    path_to_in: str
            path where atmospheric input files are located
    type_input: str
            type of atmospheric input (currently: ERA: ERA5 input, WRF: WRF input, SYNTH: synthetic input)
    type_dem: str, optional
            type of input dem (currently: 'tdx': Tandem-X DEM, 'wrfles': same topography as WRF-LES, 'fourierland': synthetic fourier topography)
    config_input: dict
           specifications of atmospheric input         
    const: dict
           meteorological standard constants
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
    usnowstorm:   maybe get rid           
            
    Returns
    -------
    data_in_norm: tensor
             input data tensor for model run in normalized space
    """

    
    # dimensions of input data
    chan_in = len(namelist['input_layers'])+1    # number of channels
    nx = namelist['nx_data']                     # number of points in x,y
    
    # create empty tensor to fill up
    data_in = torch.empty((1, chan_in, nx, nx))
    
    # prepare input from ERA and WRF
    if type_input in ['ERA', 'WRF']:
        # load atmospheric input data
        ncfile_input = get_ncfile_input(path_to_in, type_input, infile1, infile2)
        if type_input == 'WRF':
            # pre-load all relevant wrf fields
            uv_wrf = getvar(ncfile_input, 'uvmet')    # wind components in 3D
            ffdd_wrf = getvar(ncfile_input, 'uvmet_wspd_wdir') # wind speed and direction in 3D
            p_wrf = getvar(ncfile_input, 'p')         # pressure in 3D
            th_wrf = getvar(ncfile_input, 'theta')    # potential temperature in 3D
            z_wrf = getvar(ncfile_input, 'z')         # geopotential height in 3D
            ter_wrf = getvar(ncfile_input, 'ter')     # WRF terrain height     
            T2_wrf = getvar(ncfile_input, 'T2')       # 2-m temperature 
            rh2_wrf = getvar(ncfile_input, 'rh2')
                
        # loop over number of input layers, prepare layer, fill up data_in
        ii_layer = 0
        for layer_in in namelist['input_layers']:
            # get input layer for ERA5 input
            if type_input == 'ERA':
                data_in, ii_layer = get_data_in_era(data_in, layer_in, ii_layer , ncfile_input[0], ncfile_input[1], dem_glacier_np, dem_glacier_xr,
                    i_time_era, namelist, const, config_input, region_id, type_dem, usnowstorm=usnowstorm, vsnowstorm=vsnowstorm, single_values_in=None, interpolate_era=interpolate_era,
                    single_value_rho0=single_value_rho0, rho0_value=rho0_value, single_value_znt=single_value_znt, znt_value=znt_value)
        
            # get input layer for WRF input
            elif type_input == 'WRF':
                data_in, ii_layer = get_data_in_wrf(data_in, layer_in, ii_layer, ncfile_input, dem_glacier_np, dem_glacier_xr,
                    namelist, config_input, const, region_id, type_dem, 
                    uv_wrf, ffdd_wrf, p_wrf, th_wrf, z_wrf, ter_wrf, T2_wrf, rh2_wrf, usnowstorm=usnowstorm, vsnowstorm=vsnowstorm,
                    single_value_rho0=single_value_rho0, rho0_value=rho0_value, single_value_znt=single_value_znt, znt_value=znt_value)
            
        # normalize
        data_in_norm = norm_data(data_in, normfactors['norm_factors_in'])
    
    # prepare input for synthetic input    
    elif type_input == 'SYNTH':
        data_in_norm = get_data_in_synth(dem_glacier_np, namelist, normfactors, 
                      ff0=list_of_synthin[0], dd0=list_of_synthin[1], N0=list_of_synthin[2],
                      z0=list_of_synthin[3], p0=list_of_synthin[4], T0=list_of_synthin[5], 
                      rh0=list_of_synthin[6], znt=list_of_synthin[7], rho0=list_of_synthin[8],
                      usnowstorm=usnowstorm, vsnowstorm=vsnowstorm, var_out=var_out)
    
    return data_in_norm

def get_data_input_snow(data_in_wind, pred_wind, normfactors, namelist):
    """
    Prepare input data for model snow modules: take wind input data, add high-resolution wind prediction, normalize
    
    Parameters
    ----------
    data_in_wind: tensor
            input data tensor used for wind prediction (in normalized space)
    pred_wind: list
            list of predicted high-res wind components (in normalized space)
    normfactors: dict
            factors for input and output normalization, back transformation            
    
    Returns
    -------
    data_in_snow_norm: tensor
            input data tensor for snow modules in normalized space
    """
    
    shape_in = data_in_wind.shape
    
    data_in_snow = torch.empty((shape_in[0], shape_in[1]+2, shape_in[2], shape_in[3]))
    
    # number of points in x,y
    nx = namelist['nx_data']
    print(type(data_in_wind))
    print(data_in_wind.shape)
    data_in_snow[:, :-2, :, :] = data_in_wind
    
    #data_in_snow[:, -2, :, :] = torch.reshape(torch.from_numpy(pred_wind[0]), (1, 1, nx, nx))
    #data_in_snow[:, -1, :, :] = torch.reshape(torch.from_numpy(pred_wind[1]), (1, 1, nx, nx))
    data_in_snow[:, -2, :, :] = torch.reshape(pred_wind[:, 0, :, :], (1, 1, nx, nx))
    data_in_snow[:, -1, :, :] = torch.reshape(pred_wind[:, 1, :, :], (1, 1, nx, nx))
    
    
    #data_in_snow_norm = norm_data(data_in_snow, normfactors['norm_factors_in'])
    
    return data_in_snow
    

def get_ncfile_input(path_to_in, type_input, infile1, infile2):
    """
    load atmospheric input data
    
    Parameters
    ----------
    path_to_in: str
            path where atmospheric input files are located
    type_input: str
            type of atmospheric input (currently: ERA: ERA5 input, WRF: WRF input, SYNTH: synthetic input)
    infile1: str
            name of atmospheric input data file
    infile2: str, optional
            name of secondary atmospheric input data file (necessary for ERA5 input: infile1: surface variables, infile2: pressure level variables)
            
    Returns
    -------
    data_out: list of datasets or dataset of atmospheric input data
    
    """
    if type_input == 'ERA':
        ds_era_sfc = xr.open_dataset(os.path.join(path_to_in, infile1))
        ds_era_plev = xr.open_dataset(os.path.join(path_to_in, infile2))
        
        data_out = [ds_era_sfc, ds_era_plev]
    
    elif type_input == 'WRF':
        data_out = Dataset(os.path.join(path_to_in, infile1))
    
    return data_out
    
    
def get_data_in_era(data_in, layer_in, ii_layer, ds_era_sfc, ds_era_pl, dem_np, dem_xr, i_time, namelist, const, config_input, region_id, type_dem,
                usnowstorm=None, vsnowstorm=None, single_values_in=None, interpolate_era=False,
                single_value_rho0=True, rho0_value=100, single_value_znt=True, znt_value=0.0001):
    """            
    Gets input layer from ERA5 input and writes into input data tensor
    
    Parameters
    ----------
    data_in: tensor
            input data tensor to be filled up
    layer_in: str
            name of variable in layer
    ii_layer: int
            number of layer in tensor
    ds_era_sfc: xr Dataset
            Data set of ERA 5 surface variables
    ds_era_pl: xr Dataset
            Dataset of ERA 5 pressure level varibles
    dem_np: ndarray
            numpy array of terrain height
    dem_xr: xr Dataset
            xarray Dataset of terrain height, with coordinates
    i_time: int
            for ERA5 input with multi-temporal file: timestep in ERA5 file
    namelist: dict
            namelist of model specifications
    const: dict
           meteorological standard constants
    config_input: dict
           specifications of atmospheric input        
    region_id: str
            region identifier to run snowstorm for specific region, connects to snowstorm_config.py
    type_dem: str, optional
            type of input dem (currently: 'tdx': Tandem-X DEM, 'wrfles': same topography as WRF-LES, 'fourierland': synthetic fourier topography)
    usnowstorm: maybe get rid
    vsnowstorm:
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
    
    Returns
    -------
    data_in: tensor
            input data tensor with filled up layer
    ii_layer: int
            new layer number in tensor
    
    """
    
    # select time in ERA 5 format
    ti = ds_era_sfc.valid_time[i_time]
    
    # number of points in x,y
    nx = namelist['nx_data']

    print(layer_in)
    # select input variable
    if layer_in == 'ter':        # terrain height
        # set down to have minimum terrain height as 0 m
        ter_min = dem_np - np.min(dem_np)
        tensor_data_in_i = torch.reshape(torch.from_numpy(ter_min), (1, 1, nx, nx))
    
    elif layer_in == 'ter_log':        # terrain height with log-filter
        ter_min = dem_np - np.min(dem_np)
        ter_log = np.log(ter_min + 10)
        tensor_data_in_i = torch.reshape(torch.from_numpy(ter_log), (1, 1, nx, nx))       
    elif layer_in == 'ter_sqrt':       # terrain height with sqrt-filter
        ter_min = dem_np - np.min(dem_np)
        ter_log = np.sqrt(ter_min + 1)
        tensor_data_in_i = torch.reshape(torch.from_numpy(ter_log), (1, 1, nx, nx))       
        
    elif layer_in == 'uv_in':          # coarse-scale wind components
        # uv from era
        if interpolate_era:
            # get components at height and time
            u_era_pl = ds_era_pl.sel(pressure_level=config_input[region_id]['ERA']['plevel_wind']).sel(valid_time=ti).u
            v_era_pl = ds_era_pl.sel(pressure_level=config_input[region_id]['ERA']['plevel_wind']).sel(valid_time=ti).v
            # interpolate to dem grid           
            u_era_interp = interpolate_era_to_dem(u_era_pl, dem_xr, type_dem)
            v_era_interp = interpolate_era_to_dem(v_era_pl, dem_xr, type_dem)
            
            # write to tensor format
            tensor_data_in_i1 = torch.reshape(torch.from_numpy(u_era_interp), (1, 1, nx, nx))
            tensor_data_in_i2 = torch.reshape(torch.from_numpy(v_era_interp), (1, 1, nx, nx))
            
        else:        # no interpolation
            # get components at height, time and location
            u_era_pl = ds_era_pl.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(
                                     pressure_level=config_input[region_id]['ERA']['plevel_wind']).sel(valid_time=ti).u.values
                                     
            v_era_pl = ds_era_pl.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(
                                     pressure_level=config_input[region_id]['ERA']['plevel_wind']).sel(valid_time=ti).v.values

            # write component values in tensor format
            tensor_data_in_i1 = torch.ones((1, 1, nx, nx))*u_era_pl
            tensor_data_in_i2 = torch.ones((1, 1, nx, nx))*v_era_pl

    elif layer_in == 'N0':
        # N0 from era
        if interpolate_era:
            # get temperature and geopotential height at time
            T_era_pl = ds_era_pl.sel(valid_time=ti).t
            z_era_pl = ds_era_pl.sel(valid_time=ti).z 
        else:
            # get temperature and geopotential height at time and location
            T_era_pl = ds_era_pl.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).t
            z_era_pl = ds_era_pl.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).z 
        
        # calculate potential temperature at upper and lower end of layer
        th_era_pllow = util_funct_meteo.calc_theta(T_era_pl.sel(pressure_level=config_input[region_id]['ERA']['plevel_N'][0]).values, config_input[region_id]['ERA']['plevel_N'][0], t_in_celsius=False)
        th_era_plhigh = util_funct_meteo.calc_theta(T_era_pl.sel(pressure_level=config_input[region_id]['ERA']['plevel_N'][1]).values, config_input[region_id]['ERA']['plevel_N'][1], t_in_celsius=False)

        # calculate height difference between pressure levels
        dz_era_pl = (z_era_pl.sel(pressure_level=config_input[region_id]['ERA']['plevel_N'][1]).values - z_era_pl.sel(pressure_level=config_input[region_id]['ERA']['plevel_N'][0]).values) / const['g']['value']

        # calculate dry Brunt-Väisällä-Frequency N of layer
        N_era_pl = np.sqrt(helpers.calc_N2(th_era_plhigh, th_era_pllow, dz_era_pl, const))
       # if th_era_pllow > th_era_plhigh:
       #     N_era_pl = 0 
        
        # interpolate N to grid of dem or fill up with single value
        if interpolate_era:
            N_era_interp = interpolate_era_to_dem(N_era_pl, dem_xr, type_dem, input_xr=False, field_era_xr=T_era_pl)
            tensor_data_in_i = torch.reshape(torch.from_numpy(N_era_interp), (1, 1, nx, nx))
        else:
            tensor_data_in_i = torch.ones((1, 1, nx, nx))*N_era_pl

    
    elif layer_in == 'cos_dd_asp':      # cosine of difference angle between slope aspect and ambient wind direction
        # cos dd asp from era and dem
        if interpolate_era:
            # get wind components at height and time
            u_era = ds_era_pl.sel(pressure_level=config_input[region_id]['ERA']['plevel_wind']).sel(valid_time=ti).u
            v_era = ds_era_pl.sel(pressure_level=config_input[region_id]['ERA']['plevel_wind']).sel(valid_time=ti).v
            u_era_pl = interpolate_era_to_dem(u_era, dem_xr, type_dem)
            v_era_pl = interpolate_era_to_dem(v_era, dem_xr, type_dem)
            
        else:
            # get wind components at height, location and time
            u_era_pl = ds_era_pl.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(
                                     pressure_level=config_input[region_id]['ERA']['plevel_wind']).sel(valid_time=ti).u.values
            v_era_pl = ds_era_pl.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(
                                     pressure_level=config_input[region_id]['ERA']['plevel_wind']).sel(valid_time=ti).v.values
        
        # calculate wind direction and speed from components
        dd_era_pl, ff_era_pl = util_funct_meteo.uv2ddff(u_era_pl, v_era_pl)
        
        # calculate cosine of difference angle
        cosddasp = helpers.calc_cosddasp(dem_np, dd_era_pl)
        
        # put into tensor shape
        tensor_data_in_i = torch.reshape(torch.from_numpy(cosddasp), (1, 1, nx, nx))
    
    elif layer_in == 'p0':        # pressure at ground level reduced to minimum dem terrain height
        
        if interpolate_era:
            # get surface pressure, ERA terrain height, 2m temperature at time and interpolate to grid of dem
            psfc_era = interpolate_era_to_dem(ds_era_sfc.sel(valid_time=ti).sp / 100, dem_xr, type_dem)
            h_era = interpolate_era_to_dem(ds_era_sfc.sel(valid_time=ti).z / const['g']['value'], dem_xr, type_dem)
            T2_era = interpolate_era_to_dem(ds_era_sfc.sel(valid_time=ti).t2m, dem_xr, type_dem)
        else:
            # get surface pressure, ERA terrain height, 2m temperature at time and location
            psfc_era = ds_era_sfc.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).sp.values / 100
            h_era = ds_era_sfc.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).z.values / const['g']['value']

            T2_era = ds_era_sfc.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).t2m.values
        
        
        # reduce pressure from ERA terrain height to minimum dem terrain height
        z0 = np.min(dem_np)        # minimum terrain height
        rho_air = psfc_era*100/(const['Rd']['value']*T2_era)        # air density at ERA ground level (ideal gas law)
        dp = -rho_air * const['g']['value'] * (h_era - z0)        # hypsometric equation: pressure difference between ERA terrain height and minimum terrain height
        psfc_era_z0 = psfc_era - dp/100        # surface pressure reduced to mimimum terrain height
        
        # bring into tensor shape
        if interpolate_era:
            tensor_data_in_i = torch.reshape(torch.from_numpy(psfc_era_z0), (1, 1, nx, nx))
        else:
            tensor_data_in_i = torch.ones((1, 1, nx, nx))*psfc_era_z0
    
    elif layer_in == 'z0':        # minimum terrain height
        z0 = np.min(dem_np)
        
        tensor_data_in_i = torch.ones((1, 1, nx, nx))*z0        
    
    elif layer_in == 'T0':        # 2m temperature reduced to minimum dem terrain height with moist adiabatic lapse rate 
        if interpolate_era:
            # get 2m temperature and ERA terrain height at time and interpolate to grid of dem
            T2_era = interpolate_era_to_dem(ds_era_sfc.sel(valid_time=ti).t2m, dem_xr, type_dem)          
            h_era = interpolate_era_to_dem(ds_era_sfc.sel(valid_time=ti).z / const['g']['value'], dem_xr, type_dem)
        else:
            # get 2m temperature and ERA terrain heigh at time and location
            T2_era = ds_era_sfc.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).t2m.values
            h_era = ds_era_sfc.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).z.values / const['g']['value']
        
        # reduce 2m temperature from ERA terrain height to minimum dem terrain height
        z0 = np.min(dem_np)                 # minimum terrain height
        
        gamma_m = 0.0065                    # moist adiabatic lapse rate 0.0065 K m-1
        dz = h_era - z0                     # height difference between ERA terrain height and minimum dem terrain height
        T2_era_red = T2_era + gamma_m*dz    # 2m temperature reduced to height of minimum dem terrain height
        
        # bring in tensor shape
        if interpolate_era:
            tensor_data_in_i = torch.reshape(torch.from_numpy(T2_era_red), (1, 1, nx, nx))
        else:
            tensor_data_in_i = torch.ones((1, 1, nx, nx))*T2_era_red

    elif layer_in == 'rh0':        # ground level relative humidity
        if interpolate_era:
            # get temperature, dew point, pressure, terrain height at time and interpolate to grid of dem
            T2_era = interpolate_era_to_dem(ds_era_sfc.sel(valid_time=ti).t2m, dem_xr, type_dem)
            Td2_era = interpolate_era_to_dem(ds_era_sfc.sel(valid_time=ti).d2m, dem_xr, type_dem)
            psfc_era = interpolate_era_to_dem(ds_era_sfc.sel(valid_time=ti).sp, dem_xr, type_dem)
            h_era = interpolate_era_to_dem(ds_era_sfc.sel(valid_time=ti).z / const['g']['value'], dem_xr, type_dem)
            
            # calculate relative humidty from temperature, dew point and pressure
            rh_era = util_funct_meteo.calc_rh_from_t_td_p(T2_era, Td2_era, psfc_era, T_unit='kelvin') / 100
            
            # bring in tensor shape
            tensor_data_in_i = torch.reshape(torch.from_numpy(rh_era), (1, 1, nx, nx))
        else:
            # get temperature, dew point, pressure, terrain height at time and location
            T2_era = ds_era_sfc.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).t2m.values
            Td2_era = ds_era_sfc.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).d2m.values
            psfc_era = ds_era_sfc.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).sp.values
            h_era = ds_era_sfc.sel(latitude=config_input[region_id]['ERA']['lat_era']).sel(
                                     longitude=config_input[region_id]['ERA']['lon_era']).sel(valid_time=ti).z.values / const['g']['value']
            
            # calculate relative humidity from temperature, dew point and pressure
            rh_era = util_funct_meteo.calc_rh_from_t_td_p(T2_era, Td2_era, psfc_era, T_unit='kelvin') / 100
            
            # bring in tensor shape
            tensor_data_in_i = torch.ones((1, 1, nx, nx))*rh_era

    elif layer_in == 'rho0':        # snow density: until now only single value over entire domain possible, to be extended
        
        if single_value_rho0:
            tensor_data_in_i = torch.ones((1, 1, nx, nx))*rho0_value
        else:
            print('this should be fixed...')
            pass
    
    elif layer_in == 'znt':        # aerodynamic roughness lenght: until now only single value over entrie domain possible, to be extended
        
        if single_value_znt:
            tensor_data_in_i = torch.ones((1, 1, nx, nx))*znt_value
        else:
            print('this should be fixed...')
            pass
        
    elif layer_in == 'um_10':        # high-resolution u component (output of snowstorm wind)
        tensor_data_in_i = torch.reshape(torch.from_numpy(usnowstorm), (1, 1, nx, nx))
    
    elif layer_in == 'vm_10':        # high-resolution v component (output of snowstorm wind)
        tensor_data_in_i = torch.reshape(torch.from_numpy(vsnowstorm), (1, 1, nx, nx))
    
    
    # fill up input data tensor and add number of layers for next iteration
    if layer_in == 'uv_in':
        data_in[0, ii_layer, 0:nx, 0:nx] = tensor_data_in_i1
        data_in[0, ii_layer+1, 0:nx, 0:nx] = tensor_data_in_i2
        ii_layer += 2
    else:
        data_in[0, ii_layer, 0:nx, 0:nx] = tensor_data_in_i
        ii_layer += 1      
    
    return data_in, ii_layer
    
    
def interpolate_era_to_dem(field_era, dem_xr, type_dem, input_xr=True, field_era_xr=None):
    """"
    Interpolates ERA field to grid of high-resolution dem
    
    Parameters
    ----------
    field_era: xr Dataset or ndarray
            ERA 5 field to be interpolated to dem grid
    dem_xr: xr Dataset
            xarray dataset of terrain height
    type_dem: str, optional
            type of input dem (currently: 'tdx': Tandem-X DEM, 'wrfles': same topography as WRF-LES, 'fourierland': synthetic fourier topography)
    input_xr: bool, optional
            flag if field_era is of type xr Dataset
    field_era_xr: xr Dataset, optional
            xarray dataset of ERA 5 field to extract coordinates in case of field_era of type ndarray
    
    Returns 
    -------
    field_era_interp: ndarray
            numpy array of ERA field interpolated to grid of high-resolution dem
    """
    
    # extract grid coordinates of ERA 5 field
    if input_xr:
        lon_era = field_era.longitude.values
        lat_era = field_era.latitude.values
    else:
        lon_era = field_era_xr.longitude.values
        lat_era = field_era_xr.latitude.values
        
    # create grid of ERA 5 coordinates in shape for interpolation
    lons_era, lats_era = np.meshgrid(lon_era, lat_era)

    erapoints = np.vstack((lons_era.flatten(), lats_era.flatten()))
    
    # extract coordinates of high-resolution dem
    if type_dem == 'tdx':
        lons_dem, lats_dem = np.meshgrid(dem_hef_xr.lon.values, dem_hef_xr.lat.values)
    elif type_dem == 'wrfles':
        lons_dem, lats_dem = dem_hef_xr.lon.values, dem_hef_xr.lat.values
    
    # interpolate ERA 5 field to grid of high-resolution dem
    if input_xr:
        field_era_interp = scipy.interpolate.griddata(erapoints.T, field_era.values.flatten(),
                                                      (lons_dem, lats_dem), method='linear')
    else:
        field_era_interp = scipy.interpolate.griddata(erapoints.T, field_era.flatten(),
                                                      (lons_dem, lats_dem), method='linear')
                                                      
    return field_era_interp
      
    

def get_data_in_wrf(data_in, layer_in, ii_layer, ncfile, dem_np, dem_xr, namelist, config_input, const, region_id, type_dem,
                    uv_wrf, ffdd_wrf, p_wrf, th_wrf, z_wrf, ter_wrf, T2_wrf, rh2_wrf, 
                    usnowstorm=None, vsnowstorm=None, single_value_rho0=True, rho0_value=100, single_value_znt=True, znt_value=0.001):
    """
    Gets input layer from ERA5 input and writes into input data tensor
    
    data_in: tensor
            input data tensor to be filled up
    layer_in: str
            name of variable in layer
    ii_layer: int
            number of layer in tensor
    ncfile: class
            netcdf4 Dataset of WRFoutput field, to be used by wrfpython
    dem_np: ndarray
            numpy array of terrain height
    dem_xr: xr Dataset
            xarray Dataset of terrain height, with coordinates
    namelist: dict
            namelist of model specifications 
    config_input: dict
           specifications of atmospheric input        
    const: dict
           meteorological standard constants
    region_id: str
            region identifier to run snowstorm for specific region, connects to snowstorm_config.py
    type_dem: str, optional
            type of input dem (currently: 'tdx': Tandem-X DEM, 'wrfles': same topography as WRF-LES, 'fourierland': synthetic fourier topography)
    xy_wrf: wrf instance
            
    usnowstorm: maybe get rid
    vsnowstorm:
    single_value_rho0: bool, optional
            flag to use single value for snow density over entire domain (currently: only True works, planned: coupling to snow evolution model)
    rho0_value: int, optional
            value to use as snow density (kg m-3)
    single_value_znt: bool, optional
            flag to use single value for aerodynamic roughness lenght over entire domain (currently: only True works, planned: coupling to snow evolution model)
    znt_value: int, optional
            value to use as aerodynamic roughness length (m)
    
    Returns
    -------
    data_in: tensor
            input data tensor with filled up layer
    ii_layer: int
            new layer number in tensor
    """    
    
    nx = namelist['nx_data']        # dimensions in x, y of input data
            
    if layer_in not in ['znt', 'rho0','um_10', 'vm_10']:        # single value fields treated later, rest extracted by get_field_wrf
        field_wrf = get_field_wrf(ncfile, layer_in, dem_np, const, dem_xr, config_input, region_id, type_dem,
                                  uv_wrf, ffdd_wrf, p_wrf, th_wrf, z_wrf, ter_wrf, T2_wrf, rh2_wrf, )
        
    if layer_in == 'uv_in':        
        
        # extract both wind components
        u_wrf = field_wrf.sel(u_v='u')
        v_wrf = field_wrf.sel(u_v='v')
        # interpolate from WRF grid to dem grid
        u_wrf_interp = interpolate_wrf_to_dem(u_wrf, dem_xr, ncfile, type_dem)
        v_wrf_interp = interpolate_wrf_to_dem(v_wrf, dem_xr, ncfile, type_dem)
            
        # bring into tensor shape
        u_wrf_interp_tensor = torch.reshape(torch.from_numpy(u_wrf_interp), (1, 1, nx, nx))
        v_wrf_interp_tensor = torch.reshape(torch.from_numpy(v_wrf_interp), (1, 1, nx, nx))
            
    elif layer_in in ['ter', 'ter_log', 'ter_sqrt']:        # terrain height fields: no interpolation needed
        field_wrf_interp_tensor = torch.reshape(torch.from_numpy(field_wrf), (1, 1, nx, nx))
      
    elif layer_in == 'um_10':        # high resolution u component (output from snowstorm wind) 
        field_wrf_interp_tensor = torch.reshape(torch.from_numpy(usnowstorm), (1, 1, nx, nx))
    
    elif layer_in == 'vm_10':        # high resolution v component (output from snowstorm wind)
        field_wrf_interp_tensor = torch.reshape(torch.from_numpy(vsnowstorm), (1, 1, nx, nx))
    
    elif layer_in in ['cos_dd_asp', 'z0']:     # minimum terrain height: sinle value, cosine of difference angle: already on dem grid
        field_wrf_interp_tensor = torch.ones((1, 1, nx, nx))*field_wrf 
            
    elif layer_in == 'znt':        # aerodynamic roghness length: single value, no interpolation
        field_wrf_interp_tensor = torch.ones((1, 1, nx, nx))*znt_value
        
    elif layer_in == 'rho0':        # snow density: single value, no interpolation
        field_wrf_interp_tensor = torch.ones((1, 1, nx, nx))*rho0_value
        
    else:        # all other fields: interpolate to dem grid, bring into tensor shape
        field_wrf_interp = interpolate_wrf_to_dem(field_wrf, dem_xr, ncfile, type_dem)
            
        field_wrf_interp_tensor = torch.reshape(torch.from_numpy(field_wrf_interp), (1, 1, nx, nx))
            
    # fill up input data tensor and add number of layers for next iteration
    if layer_in == 'uv_in':
        data_in[0, ii_layer, 0:nx, 0:nx] = u_wrf_interp_tensor
        data_in[0, ii_layer+1, 0:nx, 0:nx] = v_wrf_interp_tensor
        ii_layer += 2
    else:
        data_in[0, ii_layer, 0:nx, 0:nx] = field_wrf_interp_tensor
        ii_layer += 1  
    
    return data_in, ii_layer
 
def get_field_wrf(ncfile, var_in, dem, const, dem_xr, config_input, region_id, type_dem, 
                  uv_wrf, ffdd_wrf, p_wrf, th_wrf, z_wrf, ter_wrf, T2_wrf, rh2_wrf, ):
    """
    extracts field from WRF output file and calculates field if required
    
    Parameters
    ----------
    
    ncfile: class
            netcdf4 Dataset of WRFoutput field, to be used by wrfpython
    var_in: str
            variable name of field to be created
    dem: ndarray
            numpy array of terrain height
    const: dict
           meteorological standard constants
    dem_xr: xr Dataset
            xarray Dataset of terrain height, with coordinates
    config_input: dict
           specifications of atmospheric input  
    region_id: str
            region identifier to run snowstorm for specific region, connects to snowstorm_config.py
    type_dem: str, optional
            type of input dem (currently: 'tdx': Tandem-X DEM, 'wrfles': same topography as WRF-LES, 'fourierland': synthetic fourier topography)
    xy_wrf: wrf instance
            
    Returns
    -------
    
    field_out: ndarray
            numpy array of variable field
    
    """
    
    if var_in == 'ter':        # substract minimum terrain height, so that lowest point is at 0 m
        field_out = dem - np.min(dem)
        
    elif var_in == 'ter_log':        # substract min terrain height and apply log filter (add constant to avoid diverging at 0)
        ter_min = dem - np.min(dem)
        field_out = np.log(ter_min + 10)

    elif var_in == 'ter_sqrt':        # substract min terrain height and apply sqrt filter (add constant to avoid diverging at 0)
        ter_min = dem - np.min(dem)
        field_out = np.sqrt(ter_min + 1)
        
    
    if var_in == 'uv_in':        # get wind components and interpolate to requested pressure level
        
        field_out = interplevel(uv_wrf, p_wrf, config_input[region_id]['WRF']['plevel_wind']*100)

        
    elif var_in == 'N0':        # dry Brunt-Väisällä Frequency of requested layer
                
        # calculate difference in potential temperature between upper and lower end of layer
        dth = interplevel(th_wrf, p_wrf, config_input[region_id]['WRF']['plevel_N'][1]*100) - interplevel(th_wrf, p_wrf, config_input[region_id]['WRF']['plevel_N'][0]*100)
        # calculate difference in height between upper and lower end of layer
        dz = interplevel(z_wrf, p_wrf, config_input[region_id]['WRF']['plevel_N'][1]*100) - interplevel(z_wrf, p_wrf, config_input[region_id]['WRF']['plevel_N'][0]*100)
        
        # calculate average potential temperaure of layer
        theta_m = 0.5*(interplevel(th_wrf, p_wrf, config_input[region_id]['WRF']['plevel_N'][1]*100) + interplevel(th_wrf, p_wrf, config_input[region_id]['WRF']['plevel_N'][0]*100))
        
        # calculate Brunt-Väisällä Frequency 
        field_out = np.sqrt(const['g']['value']/theta_m * (dth/dz))
        
        
    elif var_in == 'cos_dd_asp':        # cosine of difference angle between slope aspect and ambient wind direction
    
        ffdd_level = interplevel(ffdd_wrf, p_wrf, config_input[region_id]['WRF']['plevel_wind']*100)        # interpolate wind to requested pressure level
        
        dd_level = ffdd_level.sel(wspd_wdir='wdir')    # get interpolated wind direction
        
        dd_level_interp = interpolate_wrf_to_dem(dd_level, dem_xr, ncfile, type_dem)        # interpolate wind direction to dem grid
        
        # calculate cosine of difference angle between slope aspect and ambient wind direction
        field_out = helpers.calc_cosddasp(dem, dd_level_interp)        
    
    
    elif var_in == 'p0':        # surface pressure reduced to minimum terrain height
    
        z0_dem = np.min(dem)                   # minimum dem terrain height
        psfc = p_wrf.sel(bottom_top = 0)           # pressure at lowest level
        
        # calcualte air density at surface (ideal gas law)
        rho_air = psfc*100/(const['Rd']['value']*T2_wrf)
        # hypsometric equation: pressure difference between height of wrf terrain and minimum dem terrain height
        dp = -rho_air * const['g']['value'] * (ter_wrf - z0_dem)
        
        psfc_z0 = psfc - dp/100        # surface pressure reduced to minimum terrain height
        
        field_out = psfc_z0/100
    
    
    elif var_in == 'z0':        # minimum terrain height
        field_out = np.min(dem)

       
    elif var_in == 'T0':        # 2m temperature reduced to minimum terrain height using moist adiabatic lapse rate
    
        z0_dem = np.min(dem)                   # minimum dem terrain height
        
        gamma_m = 0.0065                       # moist adiabatic lapse rate (0.0065 K m-1)
        dz = ter_wrf - z0_dem                  # height difference between WRF terrain and minimum terrain height
        field_out = T2_wrf + gamma_m*dz            # 2m temperature reduced to minimum terrain height
        
    elif var_in == 'rh0':        # ground-level relative humidity
        field_out = rh2_wrf / 100
        
    #else: 
    #    print('wrong input layer name ---{}--- you dummy!'.format(var_in))
    
    return field_out
    
    
def interpolate_wrf_to_dem(field_wrf, dem_xr, ncfile, type_dem):
    """
    Interpolates ERA field to grid of high-resolution dem
    
    Parameters
    ----------
    field_wrf: xr Dataarray
            WRF field to be interpolated to dem grid
    dem_xr: xr Dataset
            xarray dataset of terrain height
    ncfile: class
            netcdf4 Dataset of WRFoutput field, to be used by wrfpython
    type_dem: str, optional
            type of input dem (currently: 'tdx': Tandem-X DEM, 'wrfles': same topography as WRF-LES, 'fourierland': synthetic fourier topography)
     
     
    Returns 
    -------
    field_wrf_interp: ndarray
            numpy array of WRF field interpolated to grid of high-resolution dem
    """

    # get coordinates of WRF grid and bring into shape for interpolation
    lons_wrf = field_wrf.XLONG.values.flatten()
    lats_wrf = field_wrf.XLAT.values.flatten()

    points_wrf = np.vstack((lons_wrf, lats_wrf))
        
    # get coordinates of dem and bring into shape for interpolation
    if type_dem == 'tdx':
        lons_dem, lats_dem = np.meshgrid(dem_xr.lon.values, dem_xr.lat.values)
    else:
        lons_dem, lats_dem = dem_xr.lon.values, dem_xr.lat.values
    
    # interpolate WRF field to dem grid
    field_wrf_interp = scipy.interpolate.griddata(points_wrf.T, field_wrf.values.flatten(),
                                                  (lons_dem, lats_dem), method='linear')

    return field_wrf_interp  
    

def get_data_in_synth(dem_glacier, namelist, normfactors_wind, 
                      ff0=10, dd0=270, N0=0.001, z0=3000, p0=700, T0=260, rh0=0.7, znt=0.0001, rho0=200,
                      usnowstorm=None, vsnowstorm=None, var_out='wind'):
    """"
    Write input file for synthetic atmospheric input
    
    Parameters:
    -----------
    
    dem_glacier: ndarray
            numpy array of high-resolution terrain height
    namelist: dict
            namelist of model specifications 
    normfactors_wind: dict
            factors for input and output normalization, back transformation
    ff0: int, optional
            input wind speed
    dd0: int, optional
            input wind direction
    N0: float, optional
            input Brunt-Väisällä Frequency
    z0: int, optional
            input minimum terrain height
    p0: int, optional
            input surface level pressure
    T0: int, optional
            input surface level temperature
    rh0: float, optional
            input surface level relative humidity
    znt: float, optional
            input aerodynamic roughness length 
    rho0: int, optional
            input snow density
    usnowstorm: ndarray
            high-resolution u component, output from snowstorm wind
    vsnowstorm: ndarray
            high-resolution v component, output from snowstorm wind
    var_out: str
            output variable of model ('wind', 'dswe', 'subl', 'phi') 
    
    """
    
    u0, v0 = util_funct_meteo.ddff2uv(dd0, ff0)        # wind components from direction and speed
    
    cosaspdd = helpers.calc_cosddasp(dem_glacier, dd0)         # calculate cosine of difference angle between slope aspect and ambient wind direction
    
    # define dimensions of input data tensor
    chan_in = len(namelist['input_layers'])+1          # number of input channels
    nx = namelist['nx_data']                           # dimension of data in x,y 
    data_in_synth = torch.empty((1, chan_in, nx, nx))  # create empty data tensor to be filled up
    
    if 'ter' in namelist['input_layers']:
        ter_use = dem_glacier - np.min(dem_glacier)
    elif 'ter_sqrt' in namelist['input_layers']:
        ter_use = np.sqrt(dem_glacier - np.min(dem_glacier) + 1)
    
    # fill up data tensor with values
    data_in_synth[0, 0, :, :] = torch.reshape(torch.from_numpy(ter_use), (1, 1, 256, 256))
    data_in_synth[0, 1, :, :] = torch.reshape(torch.ones(1, 1, 256, 256)*u0, (1, 1, 256, 256))
    data_in_synth[0, 2, :, :] = torch.reshape(torch.ones(1, 1, 256, 256)*v0, (1, 1, 256, 256))
    data_in_synth[0, 3, :, :] = torch.reshape(torch.ones(1, 1, 256, 256)*N0, (1, 1, 256, 256))
    data_in_synth[0, 4, :, :] = torch.reshape(torch.from_numpy(cosaspdd), (1, 1, 256, 256))
    data_in_synth[0, 5, :, :] = torch.reshape(torch.ones(1, 1, 256, 256)*z0, (1, 1, 256, 256))
    data_in_synth[0, 6, :, :] = torch.reshape(torch.ones(1, 1, 256, 256)*p0, (1, 1, 256, 256))
    data_in_synth[0, 7, :, :] = torch.reshape(torch.ones(1, 1, 256, 256)*T0, (1, 1, 256, 256)) 
    data_in_synth[0, 8, :, :] = torch.reshape(torch.ones(1, 1, 256, 256)*rh0, (1, 1, 256, 256))
    data_in_synth[0, 9, :, :] = torch.reshape(torch.ones(1, 1, 256, 256)*znt, (1, 1, 256, 256))
    data_in_synth[0, 10, :, :] = torch.reshape(torch.ones(1, 1, 256, 256)*rho0, (1, 1, 256, 256))
    
    # fill up data tensor with high-resolution wind components
    if var_out != 'wind':
        data_in_synth[0, 11, :, :] = torch.reshape(torch.from_numpy(usnowstorm), (1, 1, 256, 256))
        data_in_synth[0, 12, :, :] = torch.reshape(torch.from_numpy(vsnowstorm), (1, 1, 256, 256))
        
    # normalize input data
    data_in_synth_norm = torch.empty_like(data_in_synth)        # create empty data tensor

    for ii in range(0,chan_in):        # loop over tensor, normalize and fill up

        data_in_synth_norm[:, ii, :, :] = (data_in_synth[:, ii, :, :] - normfactors_wind['norm_factors_in'][ii][0]) / (normfactors_wind['norm_factors_in'][ii][1])

    return data_in_synth_norm
  
  
def get_dem_wrfles(path_to_dem):
    # extract terrain from wrf les 
    
    dem_xr = xr.open_dataset(os.path.join(path_to_dem, 'topo_wrfles.nc'))
    
    dem_np = dem_xr.ter.values
    
    return dem_np, dem_xr
    

def get_dem_tdx(path_to_dem, config_dem, region_id):
    # extract terrain from GLO-30 DEM for specified region
    
    dem_xr = xr.open_dataset(os.path.join(path_to_dem, config_dem[region_id]['infile']))
    
    xmid, ymid = config_dem[region_id]['xmid'], config_dem[region_id]['ymid']
    dem_xr_cut = dem_xr.sel(lat=slice(dem_xr.lat[ymid-128], dem_xr.lat[ymid+127])).sel(
        lon=slice(dem_xr.lon[xmid-128], dem_xr.lon[xmid+127]))
 
    dem_np_cut = dem_xr_cut.Band1.values
    
    return dem_np_cut, dem_xr_cut

    
def get_dem_fourierland(path_to_dem, num_topo):
    # extract terrain height of specified fourier land topography
    
    infile_dem = 'topo_{}'.format(num_topo)
    dem_np = np.transpose(np.loadtxt(os.path.join(path_to_dem, infile_dem), skiprows=1, delimiter=','))
    
    return dem_np, None
    

def norm_data(data, norm_factors):
    """
    Normalize data based on given normalization factors
    
    Parameters
    ----------
    
    data: torch tensor
            torch tensor of data to be normalized
    norm_factors: list
            list of normalization factors 
    
    Returns
    -------
    data_norm: torch tensor
            torch tensor of normalized data
    """
    
    # create emtpy tensor to be filled up
    data_norm = torch.empty_like(data)
    
    # loop over tensor fields: substract mean, divide by std, fill up output tensor
    for ii, norm_i in enumerate(norm_factors):
        field_norm = (data[:, ii, :, :] - norm_i[0]) / norm_i[1]
        data_norm[:, ii, :, :] = field_norm
        
    return data_norm

