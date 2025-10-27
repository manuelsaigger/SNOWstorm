#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_config_dem():
    
    config_dem = {
        'hef': {
            'infile' : 'topo_hef_dx50.nc',
            'xmid'   : 1400,
            'ymid'   : 660
            },
        'vernagtferner': {
            'infile' : 'topo_hef_dx50.nc',
            'xmid'   : 1500,
            'ymid'   : 750
            },
         }
    
    return config_dem
    
def get_config_obs():

    config_obs = {
        'STH': {
            'infile' : 'data_sth_20210208.csv',
            'infile_wrf_extract': 'uv_sth_wrf.txt',
            'dt'     : 15,
            'lat'    : 46.798645,
            'lon'    : 10.760647,
            'lat_w'  : 46.798896,
            'lon_w'  : 10.760373,
            'sn_wrf' : 125,
            'we_wrf' : 128
            },
         'IHE': {
            'infile' : '202101-03_ImHinterenEis_FRecDR_calc_TKE.csv',
            'infile_wrf_extract': 'uv_ihe_wrf.txt',
            'dt'     : 30,
            'lat'    : 46.795761,
            'lon'    : 10.783409,
            'sn_wrf' : 115,
            'we_wrf' : 168,
            
            },
          'AWS28': {
            'infile' : '2021_AWS_HEF.csv',
            'infile_wrf_extract': 'uv_aws28_wrf.txt',
            'dt'     : 10,
            'lat'    : 46.79779,
            'lon'    : 10.76967,
            'sn_wrf' : 120,
            'we_wrf' : 146,
            }
        }

    return config_obs


def get_config_input():
    
    config_input = {
        'hef' : {
            'ERA': {
                'plevel_wind' : 600,
                'plevel_N'    : [600,500],
                'lat_era'     : 46.8,
                'lon_era'     : 10.8
                },
            'WRF': {
                'plevel_wind': 600,
                'plevel_N': [600, 500]
                }
            }
    
        }
      
    return config_input
