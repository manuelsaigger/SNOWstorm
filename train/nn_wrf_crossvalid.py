#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:29:44 2024

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

import nn_wrf_main
import nn_wrf_models
import nn_wrf_helpers
import nn_wrf_namelist
import nn_wrf_train
import nn_wrf_diag


def make_plot(var_model, ii_model):
    
    with open('crossvalid_meas_{}_{}.json'.format(var_model, ii_model), 'r') as meas:
        model_meas = json.load(meas)
        
    plt.style.use('seaborn')
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    fig = plt.figure(figsize=(8,6))
    ax_mae = fig.add_axes([.1, .1, .35, .85])
    ax_bias = fig.add_axes([.6, .1, .35, .85])
       
    
    qs = ['all', 'q1', 'q5', 'q10', 'q100']
    qs_labels = ['all', 'ff < 1', 'ff < 5', 'ff < 10', 'ff > 10']
    ii = 0
    for qi in qs:
        mae_tmp, bias_tmp = [], []
        for cross_i in range(6):
            mae_tmp.append(model_meas['crossvalid_{}'.format(cross_i)][qi]['mae'])
            bias_tmp.append(model_meas['crossvalid_{}'.format(cross_i)][qi]['bias'])
        mae_tmp = np.array(mae_tmp)
        bias_tmp = np.array(bias_tmp)
        
        if qi == 'all':
            color_plot = 'purple'
        elif qi in ['q1', 'q5', 'q10', 'q100']:
            color_plot = 'darkgreen'
        elif qi in ['q5slope', 'q20slope', 'q40slope', 'q100slope']:
            color_plot = 'purple'    
        
        
        ax_mae.plot([ii, ii], [np.min(mae_tmp), np.max(mae_tmp)], '-', color=color_plot, linewidth=3)
        # ax_mae.plot([ii, ii], [np.percentile(mae_tmp, 10), np.percentile(mae_tmp, 90)], '-', color=color_plot, linewidth=2)
        # ax_mae.plot(ii, np.min(mae_tmp), 'k.')
        # ax_mae.plot(ii, np.max(mae_tmp), 'k.')
        ax_mae.plot(ii, np.median(mae_tmp), 'ko')

        
        ax_bias.plot([ii, ii], [np.min(bias_tmp), np.max(bias_tmp)], '-', color=color_plot, linewidth=3)
        ax_bias.plot(ii, np.median(bias_tmp), 'ko')
    
        ii += 1
    
    ax_mae.set_xticks(np.arange(0, 4.1))
    ax_mae.set_xticklabels(qs_labels, fontweight='bold')
    ax_bias.set_xticks(np.arange(0, 4.1))
    ax_bias.set_xticklabels(qs_labels, fontweight='bold')
    ax_bias.axhline(y=0, linestyle='--', color='k')
    
    
    if var_model == 'wind':
        ax_mae.set_ylabel('MAE ff (m s-1)', fontweight='bold')
        ax_bias.set_ylabel('bias ff (m s-1)', fontweight='bold')
        ax_mae.set_ylim([-0.1, 1.7])
        ax_mae.set_yticks(np.arange(0, 1.6, 0.25))
    elif var_model == 'dswe':
        ax_mae.set_ylabel('MAE dswe (kg m-2 s-1)', fontweight='bold')
        ax_bias.set_ylabel('bias dswe (kg m-2 s-1)', fontweight='bold')
        ax_mae.set_ylim([-0.1, 1.1])
        ax_mae.set_yticks(np.arange(0, 1.1, 0.2))
    elif var_model == 'subl':
        ax_mae.set_ylabel('MAE subl (kg m-2 s-1)', fontweight='bold')
        ax_bias.set_ylabel('bias subl (kg m-2 s-1)', fontweight='bold')
        ax_mae.set_ylim([-0.05, 0.45])
        ax_mae.set_yticks(np.arange(0, 0.5, 0.1))
    elif var_model == 'phi':
        ax_mae.set_ylabel('MAE IST (kg m-1 s-1)', fontweight='bold')
        ax_bias.set_ylabel('bias IST (kg m-1 s-1)', fontweight='bold')
        # ax_mae.set_ylim([-0.1, 2)
        # ax_mae.set_yticks(np.arange(0, 1.6, 0.25))
    
    fig.savefig('meas_crossvalid_{}.png'.format(var_model), dpi=300)


def crossvalid_eval_main(var_model, ii_model):
    
    path_to_namelist = 'nn_wrf_namelists'
    path_to_wrf = '../../WRF/sims_training_extended/wrfout_extract_erodeponew'
    path_to_wrf = '../../WRF/sims_training_extended/wrfout_extract_n2'
    if var_model == 'wind':
        namelist_name = 'unet_wind_{}.json'.format(ii_model) 
    elif var_model == 'dswe':
        namelist_name = 'unet_dswe_{}.json'.format(ii_model) 
    elif var_model == 'subl':
        namelist_name = 'unet_dswe_{}.json'.format(ii_model) 
    elif var_model == 'phi':
        namelist_name = 'unet_phi_{}.json'.format(ii_model) 
        
    
    with open(os.path.join(path_to_namelist, namelist_name), 'r') as fp:
        namelist= json.load(fp)
    
    print(namelist['input_layers'])
    # loop over all iicross
    # - get model
    # - prepare valid dataset
    # - make prediction
    # - evaluate
    # - write out
    for ii_crossvalid in range(6):
        
        data_pred_i, data_label_i, data_in_i, data_in_unnorm_i, ii_valid = get_prediction_model(var_model, ii_crossvalid, namelist, path_to_wrf, ii_model)
        
        make_eval_model_whole(data_pred_i, data_label_i, data_in_i, data_in_unnorm_i, var_model, ii_crossvalid, namelist, ii_valid, ii_model)
        
    
    with open('crossvalid_meas_{}_{}.json'.format(var_model, ii_model), 'r') as meas:
        model_meas = json.load(meas)
    
    if var_model == 'wind':
        # var_diag = ['mae_ff', 'mae_uv', 'rmse_ff', 'rmse_uv', 'bias_ff', 'bias_uv','mae_ff_rel', 'mae_uv_rel', 'rmse_ff_rel', 'rmse_uv_rel', 'bias_ff_rel', 'bias_uv_rel', 'r2_ff']
        var_diag = ['mae_ff', 'mae_uv', 'rmse_ff', 'rmse_uv', 'bias_ff', 'bias_uv']#,'mae_ff_rel', 'mae_uv_rel', 'rmse_ff_rel', 'rmse_uv_rel', 'bias_ff_rel', 'bias_uv_rel', 'r2_ff']
        var_diag = ['mae', 'rmse', 'bias']#,'mae_ff_rel', 'mae_uv_rel', 'rmse_ff_rel', 'rmse_uv_rel', 'bias_ff_rel', 'bias_uv_rel', 'r2_ff']
        quant_ff = ['all', 'q1', 'q5', 'q10', 'q100']
    elif var_model == 'dswe':
        var_diag = ['mae_dswe_pos','mae_dswe_neg','rmse_dswe_pos','rmse_dswe_neg','bias_dswe_pos','bias_dswe_neg']
        var_diag = ['mae', 'rmse', 'bias']
        quant_ff = ['all', 'q1', 'q5', 'q10', 'q100']       
    elif var_model == 'subl':
        var_diag = ['mae','rmse','bias']
        quant_ff = ['all', 'q1', 'q5', 'q10', 'q100']       
        
        
    var_diag = ['mae','rmse','bias']
    quant_ff = ['all', 'q1', 'q5', 'q10', 'q100', 'q5slope', 'q20slope', 'q40slope', 'q100slope']       
        
    print(model_meas)
    
    for var_i in var_diag:            
        print(var_i)
        for quant_i in quant_ff:
            meas_i = []
            for i in range(6):
            # print(i)
            # print(model_meas['crossvalid_{}'.format(i)].keys())
                mtmp = model_meas['crossvalid_{}'.format(i)][quant_i][var_i]
                meas_i.append(mtmp)
            meas_i = np.array(meas_i)
            print(quant_i)
            print(meas_i.mean(), meas_i.min(), meas_i.max())
        print('--------')
    
def make_eval_model_whole(data_pred, data_label, data_in, data_in_unnorm, var_model, ii_crossvalid, namelist, ii_valid, ii_model):

     
    path_to_plot_scatter = 'plots_crossvalid_scatter'
    
    if var_model == 'wind':
        
        ff_pred = np.sqrt(data_pred[:, 0, :, :]**2 + data_pred[:, 1, :, :]**2)
        ff_label = np.sqrt(data_label[:, 0, :, :]**2 + data_label[:, 1, :, :]**2)
        delta_ff = ff_pred - ff_label
        delta_ff_rel = delta_ff / ff_label
        
        fig = plt.figure()
        plt.hist(delta_ff.flatten(), bins=100)
        plt.xlabel('ff (m s-1)')
        plt.ylabel('count')
        fig.savefig(os.path.join(path_to_plot_scatter, 'distr_deltaff_{}.png'.format(ii_crossvalid)))
        
        
        
        delta_uv = data_pred - data_label
        delta_uv_rel = delta_uv / data_label 
        print(delta_uv.shape)
        
        q1_ff_label = 1
        q5_ff_label = 5
        q10_ff_label = 10
        
        mask_q1_ff_label = ff_label <= q1_ff_label
        mask_q5_ff_label = np.logical_and(ff_label > q1_ff_label, ff_label <= q5_ff_label)
        mask_q10_ff_label = np.logical_and(ff_label > q5_ff_label, ff_label <= q10_ff_label)
        mask_q100_ff_label = ff_label > q10_ff_label
        mask_all = ff_label < 1000
        
        mask_q5_slope, mask_q20_slope, mask_q40_slope, mask_q100_slope, mask_all, slope_all = get_mask_slope(data_in_unnorm, namelist)
        
        
        fig = plt.figure()
        plt.plot(slope_all.flatten(), delta_ff.flatten(), 'k.', markersize=.05)
        plt.xlabel('slope angle (deg)')
        plt.ylabel('delta ff (m s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dff_slope_{}.png'.format(ii_crossvalid)))
        
        fig = plt.figure()
        plt.plot(ff_pred.flatten(), delta_ff.flatten(), 'k.', markersize=.05)
        plt.xlabel('ff nn (m s-1)')
        plt.ylabel('delta ff (m s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dff_ffnn_{}.png'.format(ii_crossvalid)))
        
        
        pos_cos = int(np.argwhere(np.array(namelist['input_layers']) == 'cos_dd_asp')) + 1
        cosaspdd = np.rad2deg(np.arccos(data_in_unnorm[:, pos_cos, :, :]))
        fig = plt.figure()
        plt.plot(cosaspdd.flatten(), delta_ff.flatten(), 'k.', markersize=.05)
        plt.xlabel('asp - dd (deg)')
        plt.ylabel('delta ff (m s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dff_cosaspdd_{}.png'.format(ii_crossvalid)))
        
        tpi = get_tpi(ii_valid, '/home/titan/gwgk/gwgk007h/WRF/sims_training_extended/files_run', '')
        fig = plt.figure()
        plt.plot(tpi.flatten(), delta_ff.flatten(), 'k.', markersize=.05)
        plt.xlabel('TPI (m)')
        plt.ylabel('delta ff (m s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dff_tpi_{}.png'.format(ii_crossvalid)))
        
        
        
        model_meas, filename_crossvalid_meas = creat_model_meas(var_model, ii_crossvalid, ii_model)   
      
        model_meas = get_errors_var(ff_label, ff_pred, mask_all, model_meas, 'all', ii_crossvalid)
        model_meas = get_errors_var(ff_label, ff_pred, mask_q1_ff_label, model_meas, 'q1', ii_crossvalid)
        model_meas = get_errors_var(ff_label, ff_pred, mask_q5_ff_label, model_meas, 'q5', ii_crossvalid)
        model_meas = get_errors_var(ff_label, ff_pred, mask_q10_ff_label, model_meas, 'q10', ii_crossvalid)
        model_meas = get_errors_var(ff_label, ff_pred, mask_q100_ff_label, model_meas, 'q100', ii_crossvalid)
        
        model_meas = get_errors_var(ff_label, ff_pred, mask_q5_slope, model_meas, 'q5slope', ii_crossvalid)
        model_meas = get_errors_var(ff_label, ff_pred, mask_q20_slope, model_meas, 'q20slope', ii_crossvalid)
        model_meas = get_errors_var(ff_label, ff_pred, mask_q40_slope, model_meas, 'q40slope', ii_crossvalid)
        model_meas = get_errors_var(ff_label, ff_pred, mask_q100_slope, model_meas, 'q100slope', ii_crossvalid)
    
    
        with open(os.path.join('', filename_crossvalid_meas), 'w') as meas:
            json.dump(model_meas, meas)  
    
    elif var_model == 'dswe':
        
        print('---')
        print(np.min(np.sqrt(data_in_unnorm[:, -1, :, :]**2 + data_in_unnorm[:, -2, :, :]**2)))
        print(np.max(np.sqrt(data_in_unnorm[:, -1, :, :]**2 + data_in_unnorm[:, -2, :, :]**2)))
        
        print('---')
        print(data_in_unnorm.shape)
        
        if 'use_nn_wind' in namelist.keys():
            mask_q1_ff, mask_q5_ff, mask_q10_ff, mask_q100_ff, mask_all, ff_pred_all = get_mask_ff(data_in_unnorm, namelist) 

        mask_q5_slope, mask_q20_slope, mask_q40_slope, mask_q100_slope, mask_all, slope_all = get_mask_slope(data_in_unnorm, namelist)
    
        model_meas, filename_crossvalid_meas = creat_model_meas(var_model, ii_crossvalid, ii_model)   

        #############
        fig = plt.figure()
        plt.hist(data_pred.flatten() - data_label.flatten(), bins=100)
        plt.xlabel('delta dswe (kg m-2 s-1)')
        plt.ylabel('count')
        fig.savefig(os.path.join(path_to_plot_scatter, 'distr_ddswe.png'))
        
        
        delta_pred = data_pred.flatten() - data_label.flatten()
        fig = plt.figure()
        plt.plot(slope_all.flatten(), delta_pred, 'k.', markersize=.05)
        plt.xlabel('slope angle (deg)')
        plt.ylabel('delta dswe (kg m-2 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'ddswe_slope_{}.png'.format(ii_crossvalid)))
        
        fig = plt.figure()
        plt.plot(ff_pred_all.flatten(), delta_pred, 'k.', markersize=.05)
        plt.xlabel('ff nn (m s-1)')
        plt.ylabel('delta dswe (kg m-2 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'ddswe_ffnn_{}.png'.format(ii_crossvalid)))
        
        pos_cos = int(np.argwhere(np.array(namelist['input_layers']) == 'cos_dd_asp')) + 1
        print(pos_cos)
        cosaspdd = np.rad2deg(np.arccos(data_in_unnorm[:, pos_cos, :, :]))
        fig = plt.figure()
        plt.plot(cosaspdd.flatten(), delta_pred, 'k.', markersize=.05)
        plt.xlabel('asp - dd (deg)')
        plt.ylabel('delta dswe (kg m-2 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'ddswe_cosaspdd_{}.png'.format(ii_crossvalid)))
        
        tpi = get_tpi(ii_valid, '/home/titan/gwgk/gwgk007h/WRF/sims_training_extended/files_run', '')
        fig = plt.figure()
        plt.plot(tpi.flatten(), delta_pred, 'k.', markersize=.05)
        plt.xlabel('TPI (m)')
        plt.ylabel('delta dswe (kg m-2 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'ddswe_tpi_{}.png'.format(ii_crossvalid)))
        ##############
        
        model_meas = get_errors_var(data_label, data_pred, mask_all, model_meas, 'all', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q1_ff, model_meas, 'q1', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q5_ff, model_meas, 'q5', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q10_ff, model_meas, 'q10', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q100_ff, model_meas, 'q100', ii_crossvalid)
        
        model_meas = get_errors_var(data_label, data_pred, mask_q5_slope, model_meas, 'q5slope', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q20_slope, model_meas, 'q20slope', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q40_slope, model_meas, 'q40slope', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q100_slope, model_meas, 'q100slope', ii_crossvalid)
            
        path_to_crossvalid_meas = ''
        with open(os.path.join(path_to_crossvalid_meas, filename_crossvalid_meas), 'w') as meas:
            json.dump(model_meas, meas)  
        
        
        
        
    elif var_model == 'subl':
        if 'use_nn_wind' in namelist.keys():
            mask_q1_ff, mask_q5_ff, mask_q10_ff, mask_q100_ff, mask_all, ff_pred_all = get_mask_ff(data_in_unnorm, namelist) 
        mask_q5_slope, mask_q20_slope, mask_q40_slope, mask_q100_slope, mask_all, slope_all = get_mask_slope(data_in_unnorm, namelist)

            
        model_meas, filename_crossvalid_meas = creat_model_meas(var_model, ii_crossvalid, ii_model)   
        
        ######################
        fig = plt.figure()
        plt.hist(data_pred.flatten() - data_label.flatten(), bins=100)
        plt.xlabel('delta subl (kg m-2 s-1)')
        plt.ylabel('count')
        fig.savefig('distr_dsubl.png')
        
        
        delta_pred = data_pred.flatten() - data_label.flatten()
        fig = plt.figure()
        plt.plot(slope_all.flatten(), delta_pred, 'k.', markersize=.05)
        plt.xlabel('slope angle (deg)')
        plt.ylabel('delta subl (kg m-2 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dsubl_slope_{}.png'.format(ii_crossvalid)))
        
        fig = plt.figure()
        plt.plot(ff_pred_all.flatten(), delta_pred, 'k.', markersize=.05)
        plt.xlabel('ff nn (m s-1)')
        plt.ylabel('delta subl (kg m-2 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dsubl_ffnn_{}.png'.format(ii_crossvalid)))
        
        pos_cos = int(np.argwhere(np.array(namelist['input_layers']) == 'cos_dd_asp')) +1
        cosaspdd = np.rad2deg(np.arccos(data_in_unnorm[:, pos_cos, :, :]))
        fig = plt.figure()
        plt.plot(cosaspdd.flatten(), delta_pred, 'k.', markersize=.05)
        plt.xlabel('asp - dd (deg)')
        plt.ylabel('delta subl (kg m-2 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dsubl_cosaspdd_{}.png'.format(ii_crossvalid)))
        
        tpi = get_tpi(ii_valid, '/home/titan/gwgk/gwgk007h/WRF/sims_training_extended/files_run', '')
        fig = plt.figure()
        plt.plot(tpi.flatten(), delta_pred, 'k.', markersize=.05)
        plt.xlabel('TPI (m)')
        plt.ylabel('delta subl (kg m-2 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dsubl_tpi_{}.png'.format(ii_crossvalid)))
        #############################
        
        
        model_meas = get_errors_var(data_label, data_pred, mask_all, model_meas, 'all', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q1_ff, model_meas, 'q1', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q5_ff, model_meas, 'q5', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q10_ff, model_meas, 'q10', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q100_ff, model_meas, 'q100', ii_crossvalid)
        
        model_meas = get_errors_var(data_label, data_pred, mask_q5_slope, model_meas, 'q5slope', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q20_slope, model_meas, 'q20slope', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q40_slope, model_meas, 'q40slope', ii_crossvalid)
        model_meas = get_errors_var(data_label, data_pred, mask_q100_slope, model_meas, 'q100slope', ii_crossvalid)
           
        path_to_crossvalid_meas = ''        
        with open(os.path.join(path_to_crossvalid_meas, filename_crossvalid_meas), 'w') as meas:
           json.dump(model_meas, meas)  
        
    elif var_model == 'phi':
        if 'use_nn_wind' in namelist.keys():
            mask_q1_ff, mask_q5_ff, mask_q10_ff, mask_q100_ff, mask_all, ff_pred_all = get_mask_ff(data_in_unnorm, namelist) 
            
        mask_q5_slope, mask_q20_slope, mask_q40_slope, mask_q100_slope, mask_all, slope_all = get_mask_slope(data_in_unnorm, namelist)
        
        phiflux_full_pred = np.sqrt(data_pred[:, 0, :, :]**2 + data_pred[:, 1, :, :]**2)
        phiflux_full_label = np.sqrt(data_label[:, 0, :, :]**2 + data_label[:, 1, :, :]**2)
        
        delta_phiflux = phiflux_full_pred.flatten() - phiflux_full_label.flatten()
        ######################
        fig = plt.figure()
        plt.hist(delta_phiflux, bins=100)
        plt.xlabel('delta ist (kg m-2 s-1)')
        plt.ylabel('count')
        fig.savefig('distr_ist.png')
        
        fig = plt.figure()
        plt.plot(slope_all.flatten(), delta_phiflux, 'k.', markersize=.05)
        plt.xlabel('slope angle (deg)')
        plt.ylabel('delta ist (kg m-1 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dist_slope_{}.png'.format(ii_crossvalid)))
        
        fig = plt.figure()
        plt.plot(ff_pred_all.flatten(), delta_phiflux, 'k.', markersize=.05)
        plt.xlabel('ff nn (m s-1)')
        plt.ylabel('delta ist (kg m-1 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dist_ffnn_{}.png'.format(ii_crossvalid)))
        
        pos_cos = int(np.argwhere(np.array(namelist['input_layers']) == 'cos_dd_asp')) + 1
        cosaspdd = np.rad2deg(np.arccos(data_in_unnorm[:, pos_cos, :, :]))
        fig = plt.figure()
        plt.plot(cosaspdd.flatten(), delta_phiflux, 'k.', markersize=.05)
        plt.xlabel('asp - dd (deg)')
        plt.ylabel('delta ist (kg m-1 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dist_cosaspdd_{}.png'.format(ii_crossvalid)))
        
        tpi = get_tpi(ii_valid, '/home/titan/gwgk/gwgk007h/WRF/sims_training_extended/files_run', '')
        fig = plt.figure()
        plt.plot(tpi.flatten(), delta_phiflux, 'k.', markersize=.05)
        plt.xlabel('TPI (m)')
        plt.ylabel('delta ist (kg m-1 s-1)')
        fig.savefig(os.path.join(path_to_plot_scatter, 'dist_tpi_{}.png'.format(ii_crossvalid)))
        #############################
        
        model_meas, filename_crossvalid_meas = creat_model_meas(var_model, ii_crossvalid, ii_model)   
        model_meas = get_errors_var(phiflux_full_label, phiflux_full_pred, mask_all, model_meas, 'all', ii_crossvalid)
        model_meas = get_errors_var(phiflux_full_label, phiflux_full_pred, mask_q1_ff, model_meas, 'q1', ii_crossvalid)
        model_meas = get_errors_var(phiflux_full_label, phiflux_full_pred, mask_q5_ff, model_meas, 'q5', ii_crossvalid)
        model_meas = get_errors_var(phiflux_full_label, phiflux_full_pred, mask_q10_ff, model_meas, 'q10', ii_crossvalid)
        model_meas = get_errors_var(phiflux_full_label, phiflux_full_pred, mask_q100_ff, model_meas, 'q100', ii_crossvalid)
        
        model_meas = get_errors_var(phiflux_full_label, phiflux_full_pred, mask_q5_slope, model_meas, 'q5slope', ii_crossvalid)
        model_meas = get_errors_var(phiflux_full_label, phiflux_full_pred, mask_q20_slope, model_meas, 'q20slope', ii_crossvalid)
        model_meas = get_errors_var(phiflux_full_label, phiflux_full_pred, mask_q40_slope, model_meas, 'q40slope', ii_crossvalid)
        model_meas = get_errors_var(phiflux_full_label, phiflux_full_pred, mask_q100_slope, model_meas, 'q100slope', ii_crossvalid)

        path_to_crossvalid_meas = ''        
        with open(os.path.join(path_to_crossvalid_meas, filename_crossvalid_meas), 'w') as meas:
            json.dump(model_meas, meas)  
        

    

def creat_model_meas(var_model, ii_crossvalid, ii_model):
    
    path_to_crossvalid_meas = ''
    filename_crossvalid_meas = 'crossvalid_meas_{}_{}.json'.format(var_model, ii_model)
        
    if ii_crossvalid != 0:  
       with open(os.path.join(path_to_crossvalid_meas, filename_crossvalid_meas), 'r') as meas:
            model_meas = json.load(meas)
    else:
        model_meas = {}
    
    return model_meas, filename_crossvalid_meas


def get_mask_slope(data_in, namelist, q5=5, q20=20, q40=40):
    
    ter = data_in[:, 0, :, :]
    
    slope = nn_wrf_helpers.get_slope_angle(ter, dx=50)
    
    mask_q5_slope = slope <= q5
    mask_q20_slope = np.logical_and(slope > q5, slope <= q20)
    mask_q40_slope = np.logical_and(slope > q20, slope <= q40)
    mask_q100_slope = slope > q40
    mask_all = slope < 1000
    
    return mask_q5_slope, mask_q20_slope, mask_q40_slope, mask_q100_slope, mask_all, slope
    



def get_mask_ff(data_in, namelist, q1_ff=1, q5_ff=5, q10_ff=10):
    print(namelist)
    pos_u = int(np.argwhere(np.array(namelist['input_layers']) == 'um_10'))
    pos_v = int(np.argwhere(np.array(namelist['input_layers']) == 'vm_10'))
    print(pos_u, pos_v)
    u_pred = data_in[:, -2, :, :]
    v_pred = data_in[:, -1, :, :]
    ff_pred = np.sqrt(u_pred**2 + v_pred**2)
    print('ff pred: ', ff_pred.max(), ff_pred.min())
            
    mask_q1_ff = ff_pred <= q1_ff
    mask_q5_ff = np.logical_and(ff_pred > q1_ff, ff_pred <= q5_ff)
    mask_q10_ff = np.logical_and(ff_pred > q5_ff, ff_pred <= q10_ff)
    mask_q100_ff = ff_pred > q10_ff
    mask_all = ff_pred < 100
    
    return mask_q1_ff, mask_q5_ff, mask_q10_ff, mask_q100_ff, mask_all, ff_pred


def get_errors_wind(ff_label, ff_pred, delta_uv, mask, model_meas, quant_ff, ii_crossvalid):
    
    
    rmse_ff = np.sqrt(np.nanmean((ff_pred[mask] - ff_label[mask])**2))
    mae_ff = np.nanmean(np.abs(ff_pred[mask] - ff_label[mask]))
    bias_ff = np.nanmean(ff_pred[mask] - ff_label[mask])
         
    # rmse_uv = np.sqrt(np.mean(delta_uv[mask, :, :, :]**2))
    # mae_uv = np.mean(np.abs(delta_uv[mask, :, :, :]))
    # bias_uv = np.mean(delta_uv[mask, :, :, :])
    
    if quant_ff == 'all':
        model_meas['crossvalid_{}'.format(ii_crossvalid)] = {}
    
    model_meas['crossvalid_{}'.format(ii_crossvalid)][quant_ff] = {
            'mae_ff':np.float64(mae_ff),
            # 'mae_uv':np.float64(mae_uv),
            'rmse_ff':np.float64(rmse_ff),
            # 'rmse_uv':np.float64(rmse_uv),
            'bias_ff':np.float64(bias_ff),
            # 'bias_uv':np.float64(bias_uv)
            }
    
    return model_meas
        
    # r2 = []
    # for ii in range(ff_pred.shape[0]):
    #     r2_tmp = np.corrcoef(ff_pred[ii, :, :].flatten(), ff_label[ii, :, :].flatten())[0, 1]
    #     r2.append(r2_tmp)
    # r2 = np.mean(np.array(r2))
        
        
def get_errors_var(data_label, data_pred, mask, model_meas, quant_ff, ii_crossvalid):
    
    
    
    data_pred = np.squeeze(data_pred)
    data_label = np.squeeze(data_label)
    
    rmse_var = np.sqrt(np.nanmean((data_pred[mask] - data_label[mask])**2))
    mae_var = np.sqrt(np.nanmean(np.abs(data_pred[mask] - data_label[mask])))
    bias_var = np.nanmean(data_pred[mask] - data_label[mask])
    
    
    if quant_ff == 'all':
        model_meas['crossvalid_{}'.format(ii_crossvalid)] = {}
    
    model_meas['crossvalid_{}'.format(ii_crossvalid)][quant_ff] = {
            'mae':np.float64(mae_var),
            # 'mae_uv':np.float64(mae_uv),
            'rmse':np.float64(rmse_var),
            # 'rmse_uv':np.float64(rmse_uv),
            'bias':np.float64(bias_var),
            # 'bias_uv':np.float64(bias_uv)
            }
    
    return model_meas

    
        

def get_tpi(ii_valid, path_to_overview, path_to_tpi):
    
    
    num_topo_sim = np.loadtxt(os.path.join(path_to_overview, 'overview_sims'), skiprows=1, usecols=(10))
    
    tpi_alltopo = xr.open_dataset(os.path.join(path_to_tpi, 'tpi_alltopo_rad2000.nc'))
    
    tpi_valid = np.empty((len(ii_valid), 256, 256))
    for ii, i_valid_tmp in enumerate(ii_valid):
        num_topo_tmp = num_topo_sim[i_valid_tmp]
        
        tpi_tmp = tpi_alltopo.sel(i_topo=num_topo_tmp).tpi.values
        
        tpi_valid[ii, :, :] = tpi_tmp
        
    return tpi_valid
    


        
    

    
    
    


def get_prediction_model(var_model, ii_crossvalid, namelist, path_to_wrf, ii_model, kfold=6):
    
    # get model
    if var_model == 'wind':
        savename_model = 'unet_uv_{}_crossvalid{}e2000.pth'.format(ii_model, ii_crossvalid)
    elif var_model == 'dswe':
        savename_model = 'unet_dswe_{}_crossvalid{}e2000.pth'.format(ii_model, ii_crossvalid)
    elif var_model == 'subl':
        savename_model = 'unet_dswe_{}_crossvalid{}e2000.pth'.format(ii_model, ii_crossvalid)
    elif var_model == 'phi':
        savename_model = 'unet_phi_{}_crossvalid{}e2000.pth'.format(ii_model, ii_crossvalid)
    
    path_to_model = 'models_nn'
    model_uv = torch.load(os.path.join(path_to_model, savename_model))
    
    # prepare valid dataset
    # get
    data_in, data_out = nn_wrf_main.get_all_data(namelist, path_to_wrf)
    
    for var_i in range(data_in.shape[1]):
        print(torch.mean(data_in[:, var_i, :, :]))
    print('-------')
    # print(torch.max(data_out))
    # print(torch.min(data_out))
    # print(data_out.shape)
    
    ii_all = np.arange(720)
    if 'use_nn_wind' in namelist.keys():
        use_nn_wind = namelist['use_nn_wind']
    else:
        use_nn_wind = False
    if use_nn_wind:
        data_in = nn_wrf_helpers.make_prediction_wind(namelist, data_in, data_out, ii_all, path_to_wrf, prediction_mode='offline')

    for var_i in range(data_in.shape[1]):
        print(torch.mean(data_in[:, var_i, :, :]))
    print('-------')
    # norm
    data_in_norm, data_out_norm, norm_factors_in, norm_factors_out = nn_wrf_main.normalize_all_data(namelist, data_in, data_out)
    
    # cut
    # i_valid_start = int((360/kfold)*ii_crossvalid)
    # i_valid_end = i_valid_start + int(360/kfold)
    # ii_valid = np.arange(i_valid_start, i_valid_end)   
    #ii_valid =  np.hstack((np.arange(0,10), np.arange(60,70), np.arange(120,130), np.arange(180, 190), np.arange(240, 250), np.arange(300, 310))) + ii_crossvalid*10
    ii_valid =  np.hstack((np.arange(0,10), np.arange(60,70), np.arange(120,130), np.arange(180, 190), np.arange(240, 250), np.arange(300, 310), np.arange(360,370), np.arange(420,430), np.arange(480,490), np.arange(540, 550), np.arange(600, 610), np.arange(660,670))) + ii_crossvalid*10
    print(ii_valid)

   
    
    data_in_valid = data_in_norm[ii_valid, :, :, :]
    data_out_valid_label = data_out_norm[ii_valid, :, :, :]
    data_out_valid_pred = np.empty((len(ii_valid), len(namelist['output_layers']), namelist['nx_data'], namelist['nx_data']))
    data_in_valid_unnorm = data_in[ii_valid, :, :, :]
    
    model_uv.eval()
    print('start predicting')
        
    for ii in range(len(ii_valid)):
        pred_i, ter_i, label_i = nn_wrf_diag.make_prediction_test(ii, model_uv, data_in_valid, data_out_valid_label,
                                                                  namelist, norm_factors_in, norm_factors_out, unnorm_out=True)

        if len(namelist['output_layers']) > 1:
            data_out_valid_pred[ii, 0, :, :] = pred_i[:, 0, :, :].cpu().detach().numpy()
            data_out_valid_pred[ii, 1, :, :] = pred_i[:, 1, :, :].cpu().detach().numpy()
        else:
            data_out_valid_pred[ii, :, :] = pred_i.cpu().detach().numpy()

    
    # return data_out_valid_pred, data_out_valid_label.cpu().detach().numpy(), data_in_valid.cpu().detach().numpy()
    return data_out_valid_pred, data_out[ii_valid, :, :, :].cpu().detach().numpy(), data_in_valid.cpu().detach().numpy(), data_in_valid_unnorm.cpu().detach().numpy(), ii_valid





def crossvalid_train_main(var_model, ii_crossvalid, ii_model, kfold=6):
    
    
    
    path_to_namelist = 'nn_wrf_namelists'
    
    if var_model == 'wind':
        namelist_name = 'unet_wind_{}.json'.format(ii_model) 
        namelist = nn_wrf_namelist.open_namelist(namelist_name, path_to_namelist)
    
    elif var_model == 'dswe':
        namelist_name = 'unet_dswe_{}.json'.format(ii_model) 
        namelist = nn_wrf_namelist.open_namelist(namelist_name, path_to_namelist)
    
    elif var_model == 'subl':
        namelist_name = 'unet_dswe_{}.json'.format(ii_model) 
        namelist = nn_wrf_namelist.open_namelist(namelist_name, path_to_namelist)
    elif var_model == 'phi':
        namelist_name = 'unet_phi_{}.json'.format(ii_model) 
        namelist = nn_wrf_namelist.open_namelist(namelist_name, path_to_namelist)
        
    
    # path_to_wrf = '../../WRF/sims_training_extended/wrfout_extract_erodeponew'
    path_to_wrf = '../../WRF/sims_training_extended/wrfout_extract_n2'
    
    ###################### 
    # prepare data
    # get whole dataset
    data_in, data_out = nn_wrf_main.get_all_data(namelist, path_to_wrf)

    # log transformation for sublimation
    if 'logtrans_subl' in namelist.keys():
        logtrans_subl = namelist['logtrans_subl']
    else:
        logtrans_subl = False
    if logtrans_subl:
        data_out = nn_wrf_helpers.make_logtrans_subl(data_out, namelist)
    
    # log transformation for dswe
    if 'logtrans_dswe' in namelist.keys():
        logtrans_dswe = namelist['logtrans_dswe']
    else:
        logtrans_dswe = False
        offset_trans_dswe = None
    if logtrans_dswe:
        data_out, offset_trans_dswe = nn_wrf_helpers.make_logtrans_dswe(data_out, namelist)


    # make wind prediction
    ii_all = np.arange(namelist['n_data'])
    if 'use_nn_wind' in namelist.keys():
        use_nn_wind = namelist['use_nn_wind']
    else:
        use_nn_wind = False
    if use_nn_wind:
        data_in = nn_wrf_helpers.make_prediction_wind(namelist, data_in, data_out, ii_all, path_to_wrf, prediction_mode='offline')
        
    # normalize
    data_in_norm, data_out_norm, norm_factors_in, norm_factors_out = nn_wrf_main.normalize_all_data(namelist, data_in, data_out)
    
    # cut
    # i_valid_start = int((360/kfold)*ii_crossvalid)
    # i_valid_end = i_valid_start + int(360/kfold)
    # ii_valid = np.arange(i_valid_start, i_valid_end)  
      
    ii_valid =  np.hstack((np.arange(0,10), np.arange(60,70), np.arange(120,130), np.arange(180, 190), np.arange(240, 250), np.arange(300, 310), np.arange(360,370), np.arange(420,430), np.arange(480,490), np.arange(540, 550), np.arange(600, 610), np.arange(660,670))) + ii_crossvalid*10
    ii_train = ii_all[~np.isin(ii_all, ii_valid)]
    print('------')
    print(ii_valid)
    print('------')
    print(ii_train)
    print('------')
    
    data_in_train = data_in_norm[ii_train, :, :, :]
    data_out_train = data_out_norm[ii_train, :, :, :]
    
    # augment
    if namelist['data_augmentation']:
        data_in_augment, data_out_augment = nn_wrf_helpers.main_augment(data_in_train, data_out_train, namelist, crossvalid=True)
    else:
        data_in_augment, data_out_augment = data_in_train, data_out_train
    
    
    #########################
    # build model
    print('-- building model --')
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    model = nn_wrf_main.get_model(namelist, device)
    
    # initialize model
    print('-- initializing weights --')
    if namelist['weight_init'] != None:
        nn_wrf_main.init_weights(model, namelist)


    ###########################
    # train
    ii_data_diag = np.random.choice(ii_train, 4, replace=False)
    
    if 'plot_epochs_evo' in namelist.keys():
        plot_epochs_evo = namelist['plot_epochs_evo']
    else:
        plot_epochs_evo = False
    data_in_augment = data_in_augment.to(device)
    data_out_augment = data_out_augment.to(device)
    train_loss, valid_loss, ii_valid = nn_wrf_train.train(data_in_augment, data_out_augment, model, namelist, ii_data_diag, 
                                                norm_factors_in, norm_factors_out, ii_model, True, False, 
                                                save_early_model=True, surr_model=False, surr_model_double=False, 
                                                crossvalid=True, addname='crossvalid', ii_crossvalid=ii_crossvalid, 
                                                logtrans_subl=logtrans_subl, logtrans_dswe=logtrans_dswe, offset_trans_dswe=offset_trans_dswe)
    
    
    ##########################
    # diagnostic
    
    # print('---------------')
    print('-- make test prediction --')
    path_to_save = 'plots_diag'
    pred_unnorm, label_unnorm, namelist, model = nn_wrf_diag.make_diag_main(model, data_in_augment, data_out_augment, train_loss, valid_loss, namelist, 
                    norm_factors_in, norm_factors_out, path_to_save, ii_valid, i_data_diag=ii_data_diag, num_test=ii_model, hpc=True, epoch=namelist['epochs'], 
                    plot_epochs_evo=False, random_namelist=False, surr_model=False, surr_model_double=False, crossvalid=True, addname='crossvalid', ii_crossvalid=ii_crossvalid, 
                    logtrans_subl=logtrans_subl, logtrans_dswe=logtrans_dswe, offset_trans_dswe=offset_trans_dswe)    
    
    
    ## save model
    print('-- save model --')
    nn_wrf_main.make_save_model(model, namelist['epochs'], namelist, ii_model, addname='crossvalid', ii_crossvalid=ii_crossvalid)



def write_tpi_all_topo():

    path_to_topo = '/home/titan/gwgk/gwgk007h/WRF/sims_training_extended/topo'
    # path_to_topo = '/home/manuel/Documents/MOCCA/Detectivework_in_WRF/env/topo_synth_256_2_smooth'
    tpi_all = np.empty((72, 256, 256))
    
    for i_infile in range(1, 73):
    # for i_infile in range(1,3):
        print(i_infile)
        infile_tmp = 'topo_{}_smooth'.format(i_infile)
        
        topo_tmp = np.loadtxt(os.path.join(path_to_topo, infile_tmp), skiprows=1, delimiter=',')
        
        # fig = plt.figure()
        # plt.pcolormesh(topo_tmp)
        # plt.colorbar()
     
        dx = 50
        x = np.arange(np.shape(topo_tmp)[0])*dx
        y = np.arange(np.shape(topo_tmp)[0])*dx
        xx, yy = np.meshgrid(x,y)     
        
        # tpi_tmp = nn_wrf_helpers.get_tpi(topo_tmp, xx, yy, rad_search=2000)
        rad_search = 500
        tpi_tmp = nn_wrf_helpers.get_tpi_expand(topo_tmp, rad_search, dx)
        # fig = plt.figure()
        # plt.pcolormesh(tpi_tmp, cmap='RdBu', vmin=-50, vmax=50)
        # plt.colorbar()
        
        tpi_all[i_infile-1, :, :] = tpi_tmp
        
    ii_topos = np.arange(1, 73)
    nx = np.arange(256)
    ds_tpi = xr.Dataset(coords={'i_topo':ii_topos,
                                       'nx':nx,
                                        'ny':nx})
    ds_tpi['tpi'] = (('i_topo', 'nx', 'ny'), tpi_all)
        
    ds_tpi.to_netcdf('tpi_alltopo_rad500.nc')



   
        
    
if __name__ == "__main__":
    #ii_crossvalid = int(sys.argv[1])
    #var_model = sys.argv[2]
    #ii_model = int(sys.argv[3])
    
    #crossvalid_train_main(var_model, ii_crossvalid, ii_model)
    
    var_model = sys.argv[1]
    ii_model = sys.argv[2]
    crossvalid_eval_main(var_model, ii_model)
    
    make_plot(var_model, ii_model)
    
    print('---- FINISCHED AND HAPPY -----')  
    
    
    
    




