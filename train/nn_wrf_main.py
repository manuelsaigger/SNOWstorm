#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:51:41 2024

@author: manuel
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchsummary import summary

import os
import json
import time
import sys

import nn_wrf_models
import nn_wrf_helpers
import nn_wrf_namelist
import nn_wrf_train
import nn_wrf_diag




def main(num_namelist_name, hpc, fill_up_wrf=True, update_namelist=False, save_model=False, namelist_add_name='', get_namelist=True, 
         namelist=None, random_namelist=False, save_early_model=False):
    # main function for model training:
    # get model specifications from namelist, build input data, train model, save model
    
    # set seed for different model components
    if namelist_add_name == 'wind_':
        print('wind!')
        torch.manual_seed(0)
    elif namelist_add_name == 'dswe_':
        print('snow!')
        torch.manual_seed(3)
    elif namelist_add_name == 'phi_':
        print('phi!')
        # torch.manual_seed(5)
        
    
    tstart = time.time()
    ## get namelist
    print('---------------')
    print('-- starting process --')
    print('-- retrieving model meta data --')
    path_to_namelist = 'nn_wrf_namelists'
    print(num_namelist_name)
    if update_namelist:
        namelist_name = 'unet_{}{}.json'.format(namelist_add_name, num_namelist_name)
        
        namelist = nn_wrf_namelist.get_namelist(namelist_name, save_list=False, path_to_save=path_to_namelist)
        
    if get_namelist:
        namelist_name = 'unet_{}{}.json'.format(namelist_add_name, num_namelist_name)
        namelist = nn_wrf_namelist.open_namelist(namelist_name, path_to_namelist)
    
    
    print(namelist)
    
    ## prepare data
    # get data 
    print('-- preparing data --')
    if hpc:
        if 'use_erodeponew' in namelist:
            if namelist['use_erodeponew']:
                # path_to_wrf = '../../WRF/sims_training_extended/wrfout_extract_erodeponew'
                path_to_wrf = '../../WRF/sims_training_extended/wrfout_extract_n2'
            else:
                path_to_wrf = '../../WRF/sims_training_extended/wrfout_extract'
        else:
            path_to_wrf = '../../WRF/sims_training_extended/wrfout_extract'
    

    data_in, data_out = get_all_data(namelist, path_to_wrf)
    
    # log transformation for sublimation
    if 'logtrans_subl' in namelist.keys():
        logtrans_subl = namelist['logtrans_subl']
    else:
        logtrans_subl = False
    if logtrans_subl:
        data_out, offset_trans_subl = nn_wrf_helpers.make_logtrans_subl(data_out, namelist)
    
    # log transformation for dswe
    if 'logtrans_dswe' in namelist.keys():
        logtrans_dswe = namelist['logtrans_dswe']
    else:
        logtrans_dswe = False
        offset_trans_dswe = None
    if logtrans_dswe:
        data_out, offset_trans_dswe = nn_wrf_helpers.make_logtrans_dswe(data_out, namelist)
    
    
    # divide in train/test based on topo number
    topo_test = np.array(namelist['topo_test'])
    n_testtopo = len(topo_test)
    loops_topo = int(namelist['n_data'] / 72)
    ii_test = np.empty(int(loops_topo*n_testtopo))
   
    loop_0 = 1
    for i_loop in range(0, loops_topo):
        ii_test[i_loop*n_testtopo:(i_loop+1)*n_testtopo] = topo_test+i_loop*topo_test
        loop_0 += 1
    ii_test -= 1
    ii_all = np.arange(namelist['n_data'])
    ii_train = ii_all[~np.isin(ii_all, ii_test)]
    
    data_in_train = data_in[ii_train, :, :, :]

    data_out_train = data_out[ii_train, :, :, :]
    
    # use nn prediction of wind in training
    if 'use_nn_wind' in namelist.keys():
        use_nn_wind = namelist['use_nn_wind']
    else:
        use_nn_wind = False
    
    if use_nn_wind:

        data_in_train = nn_wrf_helpers.make_prediction_wind(namelist, data_in_train, data_out_train, ii_train, path_to_wrf, prediction_mode='offline')
        
        del data_in, data_out
        
        
    if 'use_nn_dswe' in namelist.keys():
        use_nn_dswe = namelist['use_nn_dswe']
    else:
        use_nn_dswe = False
    
    if use_nn_dswe:
        data_in_train = nn_wrf_helpers.get_prediction_dswe(namelist, data_in_train, ii_train)
        
    
    # data augmetation
    if namelist['data_augmentation']:
        data_in_augment, data_out_augment = nn_wrf_helpers.main_augment(data_in_train, data_out_train, namelist)
    else:
        data_in_augment, data_out_augment = data_in_train, data_out_train
    
    print(data_out_train.mean(), data_out_augment.std())
    print(data_out_augment.mean(), data_out_augment.std())
    
    
    # normalize 
    print('-- normalizing data --')
    if namelist['batch_norm']:
        data_in_norm, data_out_norm = data_in_augment, data_out_augment
        norm_factors_in, norm_factors_out = None, None
    else:
        data_in_norm, data_out_norm, norm_factors_in, norm_factors_out = normalize_all_data(namelist, data_in_augment, data_out_augment)
        
        print(norm_factors_in)
        print(norm_factors_out)
        norm_factors_in_np, norm_factors_out_np = [], []
        for norm_factor_in_i in norm_factors_in:
            norm_factor_in_i0 = np.float64(norm_factor_in_i[0].cpu().detach().numpy())
            norm_factor_in_i1 = np.float64(norm_factor_in_i[1].cpu().detach().numpy())
            norm_factors_in_np.append([norm_factor_in_i0, norm_factor_in_i1])
        for norm_factor_out_i in norm_factors_out:
            norm_factor_out_i0 = np.float64(norm_factor_out_i[0].cpu().detach().numpy())
            norm_factor_out_i1 = np.float64(norm_factor_out_i[1].cpu().detach().numpy())
            norm_factors_out_np.append([norm_factor_out_i0, norm_factor_out_i1])
            
        print(norm_factors_in_np)
        norm_factors = {'norm_factors_in':norm_factors_in_np, 'norm_factors_out':norm_factors_out_np}
        if logtrans_dswe:
            norm_factors['offset_trans_dswe'] = np.float64(offset_trans_dswe)
        if logtrans_subl:
            norm_factors['offset_trans_subl'] = np.float64(offset_trans_subl)
        filename_normfactors = 'norm_factors_{}{}.json'.format(namelist_add_name, num_namelist_name)
        with open(os.path.join('', filename_normfactors), 'w') as f:
            json.dump(norm_factors, f)  
        
    
    
    # ## build model
    print('-- building model --')
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    model = nn_wrf_models.unet_circpad_lrelu_4pool(len(namelist['input_layers'])+1, len(namelist['output_layers'])).to(device)
        
    # ## initialize model
    print('-- initializing weights --')
    if namelist['weight_init'] != None:
        init_weights(model, namelist)
    
        
    # ## train
    # print('---------------')
    tstart_train = time.time()
    dt_preproc = tstart_train - tstart
    dt_m_p = np.floor(dt_preproc / 60)
    dt_s_p = np.floor(dt_preproc - (dt_m_p*60))
    print('----------------------------------------------------------------')
    print('time for preprocessing: {} min {} s '.format(int(dt_m_p), int(dt_s_p)))
    print('----------------------------------------------------------------')   
    
    print('-- start training process --')
    # ii_data_diag = [10, 14, 49, 145]
    # ii_data_diag = [4, 14, 15, 145]
    ii_data_diag = [22, 72, 106, 154] # data set number for evaluation plot
    
    if 'plot_epochs_evo' in namelist.keys():
        plot_epochs_evo = namelist['plot_epochs_evo']
    else:
        plot_epochs_evo = False
    data_in_norm = data_in_norm.to(device)
    data_out_norm = data_out_norm.to(device)
    train_loss, valid_loss, ii_valid = nn_wrf_train.train(data_in_norm, data_out_norm, model, namelist, ii_data_diag, 
                                                norm_factors_in, norm_factors_out, num_namelist_name, hpc, plot_epochs_evo, 
                                                save_early_model=save_early_model, logtrans_subl=logtrans_subl,
                                                logtrans_dswe=logtrans_dswe, offset_trans_dswe=offset_trans_dswe)
    
    
    # ## evaluate
    # print('---------------')
    print('-- make test prediction --')
    if hpc:
        path_to_save = 'plots_diag'
        # path_to_save = ''
    else:
        path_to_save = ''

    pred_unnorm, label_unnorm, namelist, model = nn_wrf_diag.make_diag_main(model, data_in_norm, data_out_norm, train_loss, valid_loss, namelist, 
                    norm_factors_in, norm_factors_out, path_to_save, ii_valid, i_data_diag=ii_data_diag, num_test=num_namelist_name, hpc=hpc, epoch=namelist['epochs'], 
                    plot_epochs_evo=plot_epochs_evo, random_namelist=random_namelist, surr_model=surr_model, surr_model_double=surr_model_double,
                    logtrans_subl=logtrans_subl, logtrans_dswe=logtrans_dswe, offset_trans_dswe=offset_trans_dswe)    
    
    
    ## save model
    if save_model:
        print('-- save model --')
        make_save_model(model, namelist['epochs'], namelist, num_namelist_name)
        
        
    ## visualize
    tend = time.time()
    dt = tend - tstart
    dt_m = np.floor(dt / 60)
    dt_s = np.floor(dt - (dt_m*60))
    print('----------------------------------------------------------------')
    print('Runtime: {} min {} s '.format(int(dt_m), int(dt_s)))
    print('----------------------------------------------------------------')
    print('FINISHED AND HAPPY')
    print('----------------------------------------------------------------')
  
   
    
def make_save_model(model, ti_epoch, namelist, num_namelist_name, addname='', ii_crossvalid=''):
    # build model name and save
     
    if 'dswe' in namelist['output_layers']:
        savename_model = 'unet_dswe_{}_{}{}e{}.pth'.format(num_namelist_name, addname, ii_crossvalid, ti_epoch)
    elif 'subl_vertint' in namelist['output_layers']:
        savename_model = 'unet_dswe_{}_{}{}e{}.pth'.format(num_namelist_name, addname, ii_crossvalid, ti_epoch)
    elif 'phiflux_u_vertint' in namelist['output_layers']:
        savename_model = 'unet_phi_{}_{}{}e{}.pth'.format(num_namelist_name, addname, ii_crossvalid, ti_epoch)
    else:
        savename_model = 'unet_uv_{}_{}{}e{}.pth'.format(num_namelist_name, addname, ii_crossvalid, ti_epoch)
               
    path_to_model = 'models_nn'
    torch.save(model, os.path.join(path_to_model, savename_model))
    


def init_weights(model, namelist):
    # weight initialization with different distribution
    
    if namelist['weight_init'] == 'norm':
        model.apply(init_weights_norm)
        
    elif namelist['weight_init'] == 'zero':
        model.apply(init_weights_zero)
    
    elif namelist['weight_init'] == 'xavier_uni':
        model.apply(init_weights_xavier_uni)
        
    elif namelist['weight_init'] == 'xavier_norm':
        model.apply(init_weights_xavier_norm)
        
    elif namelist['weight_init'] == 'kaiming_uni':
        model.apply(init_weights_kaiming_uni)
        
def init_weights_norm(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight)
        
def init_weights_zero(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.zeros_(m.weight)

def init_weights_xavier_uni(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        
def init_weights_xavier_norm(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

def init_weights_kaiming_uni(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)

    
def get_all_data(namelist, path_to_wrf):
    # get input/output data: loop over all wrf simulations, build data tensor
    
    if 'uv_in' in namelist['input_layers']:
        len_ins = len(namelist['input_layers'])+1
    else:
        len_ins = len(namelist['input_layers'])
    # build empty tensor to fill up
    data_in_wrf = torch.empty((namelist['n_data'], len_ins, namelist['nx_wrf'], namelist['nx_wrf']))
    data_out_wrf = torch.empty((namelist['n_data'], len(namelist['output_layers']), namelist['nx_wrf'], namelist['nx_wrf']))
    
    # loop over all sims, get data, fill up
    for ii_data in range(1, namelist['n_data']+1):
        # print(ii_data)
        if ii_data < 10:
            infile_tmp = 'wrfout_extract_{}.nc'.format(ii_data)
        else:
            infile_tmp = 'wrfout_extract_{}.nc'.format(ii_data)
        
        data_in_wrf, data_out_wrf = nn_wrf_helpers.get_data_wrf(path_to_wrf, infile_tmp, namelist['nx_wrf'],
                                                                namelist['input_layers'], namelist['output_layers'],
                                                                data_in_wrf, data_out_wrf, ii_data-1)
    
    # mirror data to get consistent dimensions
    if namelist['nx_data'] != namelist['nx_wrf']:        
        data_in = nn_wrf_helpers.build_last_row(data_in_wrf, namelist['nx_wrf'])
        data_out = nn_wrf_helpers.build_last_row(data_out_wrf, namelist['nx_wrf'])
    else:
        data_in = data_in_wrf
        data_out = data_out_wrf
    
    return data_in, data_out



def normalize_all_data(namelist, data_in, data_out, calc_new_factors=True, norm_factors_in=None, norm_factors_out=None):
    # data normalization: select strategy, normalize, write out norm factors
    
    data_in_norm = torch.empty_like(data_in)
    data_out_norm = torch.empty_like(data_out)
    if calc_new_factors:
        norm_factors_in = []
        norm_factors_out = []        
    
    if type(namelist['normalization']) == str:
        if namelist['normalization'] == 'minmax_01':
            data_in_norm, norm_factors_in = make_normalization_minmax_01(data_in, data_in_norm, norm_factors_in, calc_new_factors=calc_new_factors)
            data_out_norm, norm_factors_out = make_normalization_minmax_01(data_out, data_out_norm, norm_factors_out, calc_new_factors=calc_new_factors)
            
        elif namelist['normalization'] == 'meanstd':
            data_in_norm, norm_factors_in = make_normalization_meanstd(data_in, data_in_norm, norm_factors_in, calc_new_factors=calc_new_factors)
            data_out_norm, norm_factors_out = make_normalization_meanstd(data_out, data_out_norm, norm_factors_out, calc_new_factors=calc_new_factors)
       
        elif namelist['normalization'] == 'meanvar':
            data_in_norm, norm_factors_in = make_normalization_meanstd(data_in, data_in_norm, norm_factors_in, make_std=False, calc_new_factors=calc_new_factors)
            data_out_norm, norm_factors_out = make_normalization_meanstd(data_out, data_out_norm, norm_factors_out, make_std=False, calc_new_factors=calc_new_factors) 
       
        elif namelist['normalization'] == 'minmax_11':
            pass
        
        else:
            raise RuntimeError('Unknown normalization strategy!')
            
    
    return data_in_norm, data_out_norm, norm_factors_in, norm_factors_out
        


    
def make_normalization_minmax_01(data, data_norm, norm_factors, calc_new_factors=True):
    # normalization by min and max -> all values between 0 and 1
    
    for ii_field in range(data_norm.shape[1]):
        if calc_new_factors:
            field_norm, field_min, field_max = nn_wrf_helpers.normalize_field_minmax(data[:, ii_field, :, :], calc_new_factors=calc_new_factors)
            norm_factors.append([field_min, field_max])
        else:
            field_norm, field_min_tmp, field_max_tmp = nn_wrf_helpers.normalize_field_minmax(data[:, ii_field, :, :], calc_new_factors=calc_new_factors,
                                                                                             field_min=norm_factors[ii_field][0], field_max=norm_factors[ii_field][1])
        
        data_norm[:, ii_field, :, :] = field_norm
        
    
    return data_norm, norm_factors



def make_normalization_meanstd(data, data_norm, norm_factors, make_std=True, calc_new_factors=True):
    # z-score normalization
    
    for ii_field in range(data_norm.shape[1]):
        if calc_new_factors:
            if make_std:
                field_norm, field_mean, field_std = nn_wrf_helpers.normalize_field_meanstd(data[:, ii_field, :, :], calc_new_factors=calc_new_factors)
            else:
                field_norm, field_mean, field_std = nn_wrf_helpers.normalize_field_meanvar(data[:, ii_field, :, :], calc_new_factors=calc_new_factors)
            norm_factors.append([field_mean, field_std])
        
        else:
            if make_std:
                field_norm, field_mean_tmp, field_std_tmp = nn_wrf_helpers.normalize_field_meanstd(data[:, ii_field, :, :], calc_new_factors=calc_new_factors,
                                                                                                   field_mean=norm_factors[ii_field][0], field_std=norm_factors[ii_field][1])
            else:
                field_norm, field_mean_tmp, field_std_tmp = nn_wrf_helpers.normalize_field_meanvar(data[:, ii_field, :, :], calc_new_factors=calc_new_factors,
                                                                                                   field_mean=norm_factors[ii_field][0], field_std=norm_factors[ii_field][1])

        data_norm[:, ii_field, :, :] = field_norm
    
    return data_norm, norm_factors
    
    
if __name__ == "__main__":
    hpc = True
    print('--- STARTING ---')
    if hpc:
        
        num_unet = int(sys.argv[1])
        var_use = sys.argv[2]

        main(num_unet, hpc, save_model=True, namelist_add_name=var_use+'_')
    else:
        main('unet_1.json', hpc, fill_up_wrf=True, update_namelist=True)
    print('---- FINISCHED AND HAPPY -----')  
    
    
    
