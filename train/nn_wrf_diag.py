#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:29:15 2024

@author: manuel
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats

import torch
from torch import nn
from torchsummary import summary

import os
import json

import nn_wrf_main
import nn_wrf_models
import nn_wrf_helpers
import nn_wrf_namelist
import nn_wrf_train


        
def make_summary_dswe(ii_summary=[40, 41, 44, 54, 55, 66, 67, 69, 70]):    
    
    path_to_model_meas = ''
    filename_model_meas = 'model_meas_dswe.json'
    with open(os.path.join(path_to_model_meas, filename_model_meas), 'r') as meas:
                model_meas = json.load(meas)
                
    rmse_pos = []
    rmse_neg = []
    bias_pos = []
    bias_neg = []
    por_sign_corr = []
       
    for num_test in ii_summary:
        if num_test <= 10:
            setup_i = 'setup_0{}'.format(num_test)
        else:
            setup_i = 'setup_{}'.format(num_test)
                
        # dswe_pos_i = model_meas[setup_i]['delta_dswe_pos']
        # dswe_neg_i = model_meas[setup_i]['delta_dswe_neg']
        por_corr_i = model_meas[setup_i]['por_sign_correct']
        rmse_pos_i = model_meas[setup_i]['rmse_dswe_pos']
        rmse_neg_i = model_meas[setup_i]['rmse_dswe_neg']
        bias_pos_i = model_meas[setup_i]['bias_dswe_pos']
        bias_neg_i = model_meas[setup_i]['bias_dswe_neg']
                
        # dswe_pos_summary.append(dswe_pos_i)
        # dswe_neg_summary.append(dswe_neg_i)
        por_sign_corr.append(por_corr_i)
        rmse_pos.append(rmse_pos_i)
        rmse_neg.append(rmse_neg_i)
        bias_pos.append(bias_pos_i) 
        bias_neg.append(bias_neg_i)
        
        
    rmse_pos = np.array(rmse_pos)
    rmse_neg = np.array(rmse_neg)
    bias_pos = np.array(bias_pos)
    bias_neg = np.array(bias_neg)
    por_sign_corr = np.array(por_sign_corr)
    
    #### plotting stuff
    
    plt.style.use('seaborn')
    label_size = 14
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    
    ### plot rmse
    fig = plt.figure()
    ax = plt.axes()
    sc = ax.scatter(rmse_pos, rmse_neg, c=por_sign_corr, cmap='RdYlGn', edgecolors='k', vmin=0.4, vmax=0.5)
    cb = plt.colorbar(sc)#label=)
    cb.set_label('Portion correct sign 90', fontsize=label_size)
    
    for ii, ii_run in enumerate(ii_summary):
        ax.text(rmse_pos[ii]+0.0001, rmse_neg[ii]+0.0001, ii_run, ha='left', va='bottom')
    
    ax.set_xlabel('rmse dswe depo (kg m$^{-2}$ h$^{-1}$)', fontsize=label_size)
    ax.set_ylabel('rmse dswe ero (kg m$^{-2}$ h$^{-1}$)', fontsize=label_size)
    
    ax.axhline(y=0, linestyle='--', color='k')
    ax.axvline(x=0, linestyle='--', color='k')


    fig.savefig('summary_snow_rmse.png', dpi=300)
    
    ### plot bias
    fig = plt.figure()
    ax = plt.axes()
    sc = ax.scatter(bias_pos, bias_neg, c=por_sign_corr, cmap='RdYlGn', edgecolors='k', vmin=0.4, vmax=0.5)
    cb = plt.colorbar(sc)#label=)
    cb.set_label('Portion correct sign 90', fontsize=label_size)
    
    for ii, ii_run in enumerate(ii_summary):
        ax.text(bias_pos[ii]+0.0001, bias_neg[ii]+0.0001, ii_run, ha='left', va='bottom')
    
    ax.set_xlabel('bias dswe depo (kg m$^{-2}$ h$^{-1}$)', fontsize=label_size)
    ax.set_ylabel('bias dswe ero (kg m$^{-2}$ h$^{-1}$)', fontsize=label_size)
    
    ax.axhline(y=0, linestyle='--', color='k')
    ax.axvline(x=0, linestyle='--', color='k')


    fig.savefig('summary_snow_bias.png', dpi=300)

def make_summary_subl(ii_summary=[66, 67, 46, 65]):    
    
    path_to_model_meas = ''
    filename_model_meas = 'model_meas_subl.json'
    with open(os.path.join(path_to_model_meas, filename_model_meas), 'r') as meas:
                model_meas = json.load(meas)
                
    pixlsub_neg_summary = []
    por_corr_summary = []
    rmse_summary = []
    bias_summary = []
    
    
    for num_test in ii_summary:
        if num_test <= 10:
            setup_i = 'setup_0{}'.format(num_test)
        else:
            setup_i = 'setup_{}'.format(num_test)
                
        pixlneg_neg_i = model_meas[setup_i]['pixl_subl_neg']
        por_corr_i = model_meas[setup_i]['por_90_correct']
        rmse_i = model_meas[setup_i]['rmse_subl']
        bias_i = model_meas[setup_i]['bias_subl']
               
                
        pixlsub_neg_summary.append(pixlneg_neg_i)
        por_corr_summary.append(por_corr_i)
        rmse_summary.append(rmse_i)
        bias_summary.append(bias_i)
        
        
    pixlsub_neg_summary = np.array(pixlsub_neg_summary)
    por_corr_summary = np.array(por_corr_summary)
    rmse_summary = np.array(rmse_summary)
    bias_summary = np.array(bias_summary)
        
    #### plotting stuff
    
    plt.style.use('seaborn')
    label_size = 14
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    
    
    fig = plt.figure()

    ax = plt.axes()

    for ii, ii_run in enumerate(ii_summary):
        if pixlsub_neg_summary[ii] > 100:
            sc = ax.scatter(rmse_summary[ii], bias_summary[ii], c=por_corr_summary[ii], cmap='RdYlGn_r', edgecolors='r', vmin=0.2, vmax=0.5)
        else:
            sc = ax.scatter(rmse_summary[ii], bias_summary[ii], c=por_corr_summary[ii], cmap='RdYlGn_r', edgecolors='k', vmin=0.2, vmax=0.5)
          
        ax.text(rmse_summary[ii]+0.0001, bias_summary[ii]+0.0001, ii_run, ha='left', va='bottom')
    cb = plt.colorbar(sc)#label=)
    cb.set_label('Portion correct sign 90', fontsize=label_size)


    
    
    ax.set_xlabel('rmse subl (kg m$^{-2}$ h$^{-1}$)', fontsize=label_size)
    ax.set_ylabel('bias subl (kg m$^{-2}$ h$^{-1}$)', fontsize=label_size)
    
    ax.axhline(y=0, linestyle='--', color='k')
    ax.axvline(x=0, linestyle='--', color='k')


    fig.savefig('summary_subl.png', dpi=300)


def make_summary_wind(ii_summary=[0, 30, 31, 32, 36, 37, 38, 39, 40]):
    
    
    path_to_model_meas = ''
    filename_model_meas = 'model_meas_uv.json'
    with open(os.path.join(path_to_model_meas, filename_model_meas), 'r') as meas:
                model_meas = json.load(meas)
                
    bias_summary = []
    rmse_summary = []
    std_summary = []
    std_label = model_meas['setup_00']['std_uv_label']
    
    for num_test in ii_summary:
        if num_test <= 10:
            setup_i = 'setup_0{}'.format(num_test)
        else:
            setup_i = 'setup_{}'.format(num_test)
                
        bias_i = model_meas[setup_i]['bias_uv']
        rmse_i = model_meas[setup_i]['delta_uv']
        std_i = model_meas[setup_i]['std_uv_pred']
                
        bias_summary.append(bias_i)
        rmse_summary.append(rmse_i)
        std_summary.append(std_i)
        
        
    bias_summary = np.array(bias_summary)
    rmse_summary = np.array(rmse_summary)
    std_summary = np.array(std_summary)
    
    
    
    #### plotting stuff
    
    plt.style.use('seaborn')
    label_size = 14
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    
    
    fig = plt.figure()

    ax = plt.axes()

    sc = ax.scatter(rmse_summary, std_summary, c=bias_summary, cmap='RdBu', edgecolors='k', vmin=-0.1, vmax=0.1)
    cb = plt.colorbar(sc)#label=)
    cb.set_label('bias uv (m s$^{-1}$)', fontsize=label_size)

    ax.axhline(y=std_label, linestyle='--', color='k')

    for ii, ii_run in enumerate(ii_summary):
        ax.text(rmse_summary[ii]+0.001, std_summary[ii]+0.001, ii_run, ha='left', va='bottom')
    
    ax.set_xlabel('RMSE uv (m s$^{-1}$)', fontsize=label_size)
    ax.set_ylabel('std uv (m s$^{-1}$)', fontsize=label_size)


    fig.savefig('summary_wind.png', dpi=300)
    
    
    
                

    
        


def make_diag_main(model, data_in, data_out, train_loss, valid_loss, namelist, 
                   norm_factors_in, norm_factors_out, path_to_save, ii_valid, i_data_diag=[0, 1, 2, 3], 
                   num_test=0, hpc=False, epoch=None, plot_epochs_evo=False, random_namelist=False,
                   surr_model=False, surr_model_double=False, crossvalid=False, addname='', ii_crossvalid='',
                   logtrans_subl=False, logtrans_dswe=False, offset_trans_dswe=None):
    
    if 'dswe' in namelist['output_layers']:
        var_diag = 'snow'
    elif 'subl_vertint' in namelist['output_layers']:
        var_diag = 'snow'
    elif 'um_10' in namelist['output_layers']:
        var_diag = 'wind'
    elif 'phiflux_u_vertint' in namelist['output_layers']:
        var_diag = 'phi'
    else: 
        var_diag = None
    
    # make test predictions
    pred_unnorm = []
    ter_unnorm = []
    label_unnorm = []
    
    model.eval()
    
    for ii in i_data_diag:

        pred_i, ter_i, label_i = make_prediction_test(ii, model, data_in, data_out, namelist, norm_factors_in, norm_factors_out, 
                                                      surr_model=surr_model, surr_model_double=surr_model_double, 
                                                      logtrans_subl=logtrans_subl, logtrans_dswe=logtrans_dswe, 
                                                      offset_trans_dswe=offset_trans_dswe)
        pred_unnorm.append(pred_i)
        ter_unnorm.append(ter_i)
        label_unnorm.append(label_i)
    
    fields_diag = {}
    
    for ii, i_valid in enumerate(ii_valid): #range(data_in.shape[0]):
        
        fields_diag = make_prediction_get_delta(ii, i_valid, model, data_in, data_out, namelist, norm_factors_in, norm_factors_out,
                                                fields_diag, var_diag=var_diag, surr_model=surr_model,
                                                surr_model_double=surr_model_double)
          
    config_cmap = config_cmaps()
    
    
    if var_diag == 'wind':
        make_plot_diag_var(pred_unnorm, ter_unnorm, label_unnorm, train_loss, valid_loss, namelist, model, path_to_save,
                          i_data_diag, num_test, hpc, config_cmap['ff'], 'wind', fields_diag, 
                          epoch=epoch, plot_epochs_evo=plot_epochs_evo, surr_model=surr_model, surr_model_double=surr_model_double,
                          crossvalid=crossvalid, addname=addname, ii_crossvalid=ii_crossvalid)
    elif var_diag == 'snow':
        if len(namelist['output_layers']) > 1:
            pred_use = []
            label_use = []
            pos_dswe = int(np.argwhere(np.array(namelist['output_layers']) == 'dswe'))
            for pred_i in pred_unnorm: pred_use.append(pred_i[:, pos_dswe, :, :])
            for label_i in label_unnorm: label_use.append(label_i[:, pos_dswe, :, :])
            
            pred_use_subl = []
            label_use_subl = []
            pos_subl = int(np.argwhere(np.array(namelist['output_layers']) == 'subl_vertint'))
            for pred_i_subl in pred_unnorm: pred_use_subl.append(pred_i_subl[:, pos_subl, :, :])
            for label_i_subl in label_unnorm: label_use_subl.append(label_i_subl [:, pos_subl, :, :])
        
        else:
            pred_use = pred_unnorm
            label_use = label_unnorm
            if 'subl_vertint' in namelist['output_layers']:
                pred_use_subl = pred_unnorm
                label_use_subl = label_unnorm
        
        if 'dswe' in namelist['output_layers']:
            make_plot_diag_var(pred_use, ter_unnorm, label_use, train_loss, valid_loss, namelist, model, path_to_save,
                            i_data_diag, num_test, hpc, config_cmap['dswe'], 'dswe', fields_diag, 
                            epoch=epoch, surr_model=surr_model, surr_model_double=surr_model_double,
                          crossvalid=crossvalid, addname=addname, ii_crossvalid=ii_crossvalid)
        if 'subl_vertint' in namelist['output_layers']:
            make_plot_diag_var(pred_use_subl, ter_unnorm, label_use_subl, train_loss, valid_loss, namelist, model, path_to_save,
                            i_data_diag, num_test, hpc, config_cmap['subl_vertint'], 'subl', fields_diag, 
                            epoch=epoch, surr_model=surr_model, surr_model_double=surr_model_double,
                          crossvalid=crossvalid, addname=addname, ii_crossvalid=ii_crossvalid)
            
    elif var_diag == 'phi':
        make_plot_diag_var(pred_unnorm, ter_unnorm, label_unnorm, train_loss, valid_loss, namelist, model, path_to_save,
                                       i_data_diag, num_test, hpc, config_cmap['phiflux_vertint'], 'phiflux', fields_diag, 
                                       epoch=epoch, surr_model=surr_model, surr_model_double=surr_model_double,
                          crossvalid=crossvalid, addname=addname, ii_crossvalid=ii_crossvalid)
        
    print(model)
    summary(model, (len(namelist['input_layers'])+1, namelist['nx_data'], namelist['nx_data']), device='cpu')
    
    
    return pred_unnorm, label_unnorm, namelist, model


def make_plot_diag_var(pred_unnorm, ter_unnorm, label_unnorm, train_loss, valid_loss, namelist, 
                       model, path_to_save, i_data_diag, num_test, hpc, config_cmap, var_plot, fields_diag, 
                       epoch=None, test_data=False, plot_epochs_evo=False, surr_model=False, surr_model_double=False,
                       crossvalid=False, addname='', ii_crossvalid=''):
    
    adjust = True
        
    label_size = 14
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
  
    plt.style.use('seaborn')
    
    scale_quiver = 25
    width_quiver = 0.007
    scale_quiver_phi = 10
        
    # set up figure
    fig = plt.figure(figsize=(10,8))
    
    yu = .08
    xlp = .5
    dxp = .16
    dyp = .2
    dxcb = .01
    dxmp = .01
    dxmp2 = .05
    dymp = .02
    
    axp4 = fig.add_axes([xlp, yu, dxp, dyp])
    axp3 = fig.add_axes([xlp, yu+dyp+dymp, dxp, dyp])
    axp2 = fig.add_axes([xlp, yu+2*(dyp+dymp), dxp, dyp])
    axp1 = fig.add_axes([xlp, yu+3*(dyp+dymp), dxp, dyp])
    
    dxln = xlp + dxp + dxmp
    axcbp4 = fig.add_axes([dxln, yu, dxcb, dyp])
    axcbp3 = fig.add_axes([dxln, yu+dyp+dymp, dxcb, dyp])
    axcbp2 = fig.add_axes([dxln, yu+2*(dyp+dymp), dxcb, dyp])
    axcbp1 = fig.add_axes([dxln, yu+3*(dyp+dymp), dxcb, dyp])
    
    dxln = dxln + dxmp + dxmp2 + dxcb
    axl4 = fig.add_axes([dxln, yu, dxp, dyp])
    axl3 = fig.add_axes([dxln, yu+dyp+dymp, dxp, dyp])
    axl2 = fig.add_axes([dxln, yu+2*(dyp+dymp), dxp, dyp])
    axl1 = fig.add_axes([dxln, yu+3*(dyp+dymp), dxp, dyp])
    
    dxln = dxln + dxp + dxmp
    axcbl4 = fig.add_axes([dxln, yu, dxcb, dyp])
    axcbl3 = fig.add_axes([dxln, yu+dyp+dymp, dxcb, dyp])
    axcbl2 = fig.add_axes([dxln, yu+2*(dyp+dymp), dxcb, dyp])
    axcbl1 = fig.add_axes([dxln, yu+3*(dyp+dymp), dxcb, dyp])
    
    
    ax_bla = fig.add_axes([.05, .55, .4, .4])
    
    # loss curve
    if not test_data:
        ax_loss = fig.add_axes([.1, yu, 2*dxp+dxmp, 2*dyp+dymp])

        epochs = np.arange(len(train_loss))
        ax_loss.plot(epochs+1, train_loss, '-', linewidth=2, color='darkgreen', label='training')
        ax_loss.plot(epochs+1, valid_loss, '-', linewidth=2, color='purple', label='validation')
        ax_loss.axhline(y=0, linestyle='--', color='k', linewidth=0.5)
    
        ax_loss.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        ax_loss.set_ylabel('Training/Validation Loss', fontsize=14, fontweight='bold')
        if len(epochs) > 10 and len(epochs) < 100:
            epochs_tick = np.arange(0, len(train_loss), 5)
        elif len(epochs) > 100:
            epochs_tick = np.arange(0, len(train_loss), 100)
        elif len(epochs) > 1000:
            epochs_tick = np.arange(0, len(train_loss)+1, 500)
        else: 
            epochs_tick = epochs
        ax_loss.set_xticks(epochs_tick)
        ax_loss.set_ylim([-0.1, 1.1])
        # ax_loss.set_ylim([-0.1, 3.1])
        ax_loss.legend(loc='upper right', frameon=True, facecolor='white')
    
    
    #### test predictions
    #-----
    field_pred, u_q_pred, v_q_pred, x_q, y_q = get_field_plot(pred_unnorm, var_plot, 0, namelist)
    field_label, u_q_label, v_q_label, x_q, y_q = get_field_plot(label_unnorm, var_plot, 0, namelist)
    field_pred, field_label = np.squeeze(field_pred), np.squeeze(field_label)
    
    ter_np = np.squeeze(ter_unnorm[0].cpu().detach().numpy())
    
    levels_cmap_pred, levels_cbar_pred = adjust_cmap(config_cmap, field_pred, adjust=adjust)
    levels_cmap_label, levels_cbar_label = adjust_cmap(config_cmap, field_label, adjust=adjust)
    levels_cmap_pred = levels_cmap_label
    levels_cbar_pred = levels_cbar_label
    
    print(field_pred.shape)
    print(field_label.shape)
    c1 = axp4.pcolormesh(field_pred, cmap=config_cmap['cmap'], vmin=levels_cmap_pred[0], vmax=levels_cmap_pred[-1])
    axp4.contour(ter_np, levels=np.arange(0, 3000, 50), colors='k', linewidths=.5)  
    cb4 = plt.colorbar(c1, cax=axcbp4, label='')
    cb4.set_ticks(levels_cbar_pred)
    c2 = axl4.pcolormesh(field_label, cmap=config_cmap['cmap'], vmin=levels_cmap_label[0], vmax=levels_cmap_label[-1])
    axl4.contour(ter_np, levels=np.arange(0, 3000, 50), colors='k', linewidths=.5) 
    cb4 = plt.colorbar(c2, cax=axcbl4,label=config_cmap['text_cbar'])# label='$\Delta$SWE (kg m$^{-2}$ h$^{-1}$)')
    cb4.set_ticks(levels_cbar_label)
    if var_plot == 'wind':
        axp4.quiver(x_q, y_q, u_q_pred, v_q_pred, angles='uv', units='inches', scale_units='inches', scale=scale_quiver, width=width_quiver)
        axl4.quiver(x_q, y_q, u_q_label, v_q_label, angles='uv', units='inches', scale_units='inches', scale=scale_quiver, width=width_quiver)
    elif var_plot == 'phi':
        axp4.quiver(x_q, y_q, u_q_pred, v_q_pred, angles='uv', units='inches', scale_units='inches', scale=scale_quiver_phi, width=width_quiver)
        axl4.quiver(x_q, y_q, u_q_label, v_q_label, angles='uv', units='inches', scale_units='inches', scale=scale_quiver_phi, width=width_quiver)
    elif var_plot == 'dswe':
        dswe_label_sign = np.sign(field_label)
        dswe_pred_sign = np.sign(field_pred)
        por_sign_correct4 = (np.sum(dswe_label_sign == dswe_pred_sign)) / dswe_label_sign.size
        
    #-----
    field_pred, u_q_pred, v_q_pred, x_q, y_q = get_field_plot(pred_unnorm, var_plot, 1, namelist)
    field_label, u_q_label, v_q_label, x_q, y_q = get_field_plot(label_unnorm, var_plot, 1, namelist)
    field_pred, field_label = np.squeeze(field_pred), np.squeeze(field_label)

    ter_np = np.squeeze(ter_unnorm[1].cpu().detach().numpy())
    
    levels_cmap_pred, levels_cbar_pred = adjust_cmap(config_cmap, field_pred, adjust=adjust)
    levels_cmap_label, levels_cbar_label = adjust_cmap(config_cmap, field_label, adjust=adjust)
    levels_cmap_label = levels_cmap_pred
    levels_cbar_label = levels_cbar_pred

    c1 = axp3.pcolormesh(field_pred, cmap=config_cmap['cmap'], vmin=levels_cmap_pred[0], vmax=levels_cmap_pred[-1])
    axp3.contour(ter_np, levels=np.arange(0, 3000, 50), colors='k', linewidths=.5)  
    cb3 = plt.colorbar(c1, cax=axcbp3, label='')
    cb3.set_ticks(levels_cbar_pred)
    c2 = axl3.pcolormesh(field_label, cmap=config_cmap['cmap'], vmin=levels_cmap_label[0], vmax=levels_cmap_label[-1])
    axl3.contour(ter_np, levels=np.arange(0, 3000, 50), colors='k', linewidths=.5) 
    cb3 = plt.colorbar(c2, cax=axcbl3,label=config_cmap['text_cbar'])# label='$\Delta$SWE (kg m$^{-2}$ h$^{-1}$)')
    cb3.set_ticks(levels_cbar_label)
    if var_plot == 'wind':
        axp3.quiver(x_q, y_q, u_q_pred, v_q_pred, angles='uv', units='inches', scale_units='inches', scale=scale_quiver, width=width_quiver)
        axl3.quiver(x_q, y_q, u_q_label, v_q_label, angles='uv', units='inches', scale_units='inches', scale=scale_quiver, width=width_quiver)
    elif var_plot == 'phi':
        axp3.quiver(x_q, y_q, u_q_pred, v_q_pred, angles='uv', units='inches', scale_units='inches', scale=scale_quiver_phi, width=width_quiver)
        axl3.quiver(x_q, y_q, u_q_label, v_q_label, angles='uv', units='inches', scale_units='inches', scale=scale_quiver_phi, width=width_quiver)
    elif var_plot == 'dswe':
        dswe_label_sign = np.sign(field_label)
        dswe_pred_sign = np.sign(field_pred)
        por_sign_correct3 = (np.sum(dswe_label_sign == dswe_pred_sign)) / dswe_label_sign.size
    
    #-----
    field_pred, u_q_pred, v_q_pred, x_q, y_q = get_field_plot(pred_unnorm, var_plot, 2, namelist)
    field_label, u_q_label, v_q_label, x_q, y_q = get_field_plot(label_unnorm, var_plot, 2, namelist)
    field_pred, field_label = np.squeeze(field_pred), np.squeeze(field_label)

    ter_np = np.squeeze(ter_unnorm[2].cpu().detach().numpy())
    
    levels_cmap_pred, levels_cbar_pred = adjust_cmap(config_cmap, field_pred, adjust=adjust)
    levels_cmap_label, levels_cbar_label = adjust_cmap(config_cmap, field_label, adjust=adjust)
    levels_cmap_pred = levels_cmap_label
    levels_cbar_pred = levels_cbar_label

    c1 = axp2.pcolormesh(field_pred, cmap=config_cmap['cmap'], vmin=levels_cmap_pred[0], vmax=levels_cmap_pred[-1])
    axp2.contour(ter_np, levels=np.arange(0, 3000, 50), colors='k', linewidths=.5)  
    cb2 = plt.colorbar(c1, cax=axcbp2, label='')
    cb2.set_ticks(levels_cbar_pred)
    c2 = axl2.pcolormesh(field_label, cmap=config_cmap['cmap'], vmin=levels_cmap_label[0], vmax=levels_cmap_label[-1])
    axl2.contour(ter_np, levels=np.arange(0, 3000, 50), colors='k', linewidths=.5) 
    cb2 = plt.colorbar(c2, cax=axcbl2,label=config_cmap['text_cbar'])# label='$\Delta$SWE (kg m$^{-2}$ h$^{-1}$)')
    cb2.set_ticks(levels_cbar_label)
    if var_plot == 'wind':
        axp2.quiver(x_q, y_q, u_q_pred, v_q_pred, angles='uv', units='inches', scale_units='inches', scale=scale_quiver, width=width_quiver)
        axl2.quiver(x_q, y_q, u_q_label, v_q_label, angles='uv', units='inches', scale_units='inches', scale=scale_quiver, width=width_quiver)
    elif var_plot == 'phi':
        axp2.quiver(x_q, y_q, u_q_pred, v_q_pred, angles='uv', units='inches', scale_units='inches', scale=scale_quiver_phi, width=width_quiver)
        axl2.quiver(x_q, y_q, u_q_label, v_q_label, angles='uv', units='inches', scale_units='inches', scale=scale_quiver_phi, width=width_quiver)
    elif var_plot == 'dswe':
        dswe_label_sign = np.sign(field_label)
        dswe_pred_sign = np.sign(field_pred)
        por_sign_correct2 = (np.sum(dswe_label_sign == dswe_pred_sign)) / dswe_label_sign.size
    
    #-----
    field_pred, u_q_pred, v_q_pred, x_q, y_q = get_field_plot(pred_unnorm, var_plot, 3, namelist)
    field_label, u_q_label, v_q_label, x_q, y_q = get_field_plot(label_unnorm, var_plot, 3, namelist)
    field_pred, field_label = np.squeeze(field_pred), np.squeeze(field_label)

    ter_np = np.squeeze(ter_unnorm[3].cpu().detach().numpy())
    
    levels_cmap_pred, levels_cbar_pred = adjust_cmap(config_cmap, field_pred, adjust=adjust)
    levels_cmap_label, levels_cbar_label = adjust_cmap(config_cmap, field_label, adjust=adjust)
    levels_cmap_pred = levels_cmap_label
    levels_cbar_pred = levels_cbar_label

    c1 = axp1.pcolormesh(field_pred, cmap=config_cmap['cmap'], vmin=levels_cmap_pred[0], vmax=levels_cmap_pred[-1])
    axp1.contour(ter_np, levels=np.arange(0, 3000, 50), colors='k', linewidths=.5)  
    cb1 = plt.colorbar(c1, cax=axcbp1, label='')
    cb1.set_ticks(levels_cbar_pred)
    c2 = axl1.pcolormesh(field_label, cmap=config_cmap['cmap'], vmin=levels_cmap_label[0], vmax=levels_cmap_label[-1])
    axl1.contour(ter_np, levels=np.arange(0, 3000, 50), colors='k', linewidths=.5) 
    cb1 = plt.colorbar(c2, cax=axcbl1,label=config_cmap['text_cbar'])# label='$\Delta$SWE (kg m$^{-2}$ h$^{-1}$)')
    cb1.set_ticks(levels_cbar_label)
    if var_plot == 'wind':
        axp1.quiver(x_q, y_q, u_q_pred, v_q_pred, angles='uv', units='inches', scale_units='inches', scale=scale_quiver, width=width_quiver)
        axl1.quiver(x_q, y_q, u_q_label, v_q_label, angles='uv', units='inches', scale_units='inches', scale=scale_quiver, width=width_quiver)
    elif var_plot == 'phi':
        axp1.quiver(x_q, y_q, u_q_pred, v_q_pred, angles='uv', units='inches', scale_units='inches', scale=scale_quiver_phi, width=width_quiver)
        axl1.quiver(x_q, y_q, u_q_label, v_q_label, angles='uv', units='inches', scale_units='inches', scale=scale_quiver_phi, width=width_quiver)
    elif var_plot == 'dswe':
        dswe_label_sign = np.sign(field_label)
        dswe_pred_sign = np.sign(field_pred)
        por_sign_correct1 = (np.sum(dswe_label_sign == dswe_pred_sign)) / dswe_label_sign.size
    

    # cosmetics
    axp1.set_title('NN', fontweight='bold', fontsize=14)
    axl1.set_title('WRF', fontweight='bold', fontsize=14)
    
    axp4.set_ylabel('Simulation {}'.format(i_data_diag[0]+1), fontweight='bold')
    axp3.set_ylabel('Simulation {}'.format(i_data_diag[1]+1), fontweight='bold')
    axp2.set_ylabel('Simulation {}'.format(i_data_diag[2]+1), fontweight='bold')
    axp1.set_ylabel('Simulation {}'.format(i_data_diag[3]+1), fontweight='bold')
      
    axp1.set_xticklabels([]), axp1.set_yticklabels([])
    axp2.set_xticklabels([]), axp2.set_yticklabels([])
    axp3.set_xticklabels([]), axp3.set_yticklabels([])
    axp4.set_xticklabels([]), axp4.set_yticklabels([])
    axl1.set_xticklabels([]), axl1.set_yticklabels([])
    axl2.set_xticklabels([]), axl2.set_yticklabels([])
    axl3.set_xticklabels([]), axl3.set_yticklabels([])
    axl4.set_xticklabels([]), axl4.set_yticklabels([])
    
    
    ## blabla panel
    ys = 0.95
    if plot_epochs_evo:
        ax_bla.text(0.05, ys, 'Epoch: {}'.format(epoch), fontsize=14, fontweight='bold')
        ys -= 0.05
    
    ax_bla.text(0.05, ys, 'Model name: '+namelist['model'], transform=ax_bla.transAxes)
    ys -= 0.05
    text_inlayers = 'input layers: '
    for layer_i in namelist['input_layers']: text_inlayers = text_inlayers+layer_i+' '
    ax_bla.text(0.05, ys, text_inlayers, transform=ax_bla.transAxes)
    ys -= 0.05
    text_outlayers = 'output layers: '
    for layer_i in namelist['output_layers']: text_outlayers = text_outlayers+layer_i+' '
    ax_bla.text(0.05, ys, text_outlayers, transform=ax_bla.transAxes)
    ys -= 0.05
    if type(namelist['normalization']) == str:
        ax_bla.text(0.05, ys, 'Normalization: '+namelist['normalization'], transform=ax_bla.transAxes)
        ys -= 0.05
    else:
        text_innorm = 'Input Normalization: '
        for norm_i in namelist['normalization'][0]: text_innorm = text_innorm+norm_i+' '
        ax_bla.text(0.05, ys, text_innorm, transform=ax_bla.transAxes)
        ys -= 0.05
        text_outnorm = 'Output Normalization: '
        for norm_i in namelist['normalization'][1]: text_outnorm = text_outnorm+norm_i+' '
        ax_bla.text(0.05, ys, text_outnorm, transform=ax_bla.transAxes)
        ys -= 0.05
    if namelist['data_augmentation']:
        text_augment = 'Data Augmentation factor: {}'.format(namelist['factor_augment'])
        ax_bla.text(0.05, ys, text_augment, transform=ax_bla.transAxes)
        ys -= 0.05
    ax_bla.text(0.05, ys, 'Optimizer: '+namelist['optimizer'], transform=ax_bla.transAxes)
    ys -= 0.05
    ax_bla.text(0.05, ys, 'Loss Function: '+namelist['loss_fn'], transform=ax_bla.transAxes)
    ys -= 0.05
    ax_bla.text(0.05, ys, 'Batch Size: '+str(namelist['batch_size']), transform=ax_bla.transAxes)
    ys -= 0.05
    ax_bla.text(0.05, ys, 'Epochs: '+str(namelist['epochs']), transform=ax_bla.transAxes)
    ys -= 0.05
    ax_bla.text(0.05, ys, 'Learning Rate: '+str(namelist['learning_rate']), transform=ax_bla.transAxes)
    ys -= 0.05
    if 'reduce_lr' in namelist.keys():
        ax_bla.text(0.05, ys, 'Learning Rate Reduction: {}'.format(namelist['reduce_lr']), transform=ax_bla.transAxes)
        ys -= 0.05
    ax_bla.text(0.05, ys, 'Validation Split: '+str(namelist['validation_split']), transform=ax_bla.transAxes)
    ys -= 0.05
    ax_bla.text(0.05, ys, 'Weight initialization: '+str(namelist['weight_init']), transform=ax_bla.transAxes)
    ys -= 0.05
    
    if var_plot == 'wind':
        # print(fields_diag)
        bias_uv = np.mean(np.array(fields_diag['bias_uv']))
        delta_uv = np.mean(np.array(fields_diag['delta_uv_abs']))
        std_uv_pred = np.mean(np.array(fields_diag['std_uv_pred']))
        std_uv_label = np.mean(np.array(fields_diag['std_uv_label']))
        
        ax_bla.text(0.05, ys, 'Bias uv: {} m s-1'.format(bias_uv), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'D uv: {} m s-1'.format(delta_uv), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'std uv pred: {} m s-1'.format(std_uv_pred), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'std uv label: {} m s-1'.format(std_uv_label), transform=ax_bla.transAxes)
        
        
    
        
        path_to_model_meas = ''
        filename_model_meas = 'model_meas_uv.json'
        if num_test != 0:  
            with open(os.path.join(path_to_model_meas, filename_model_meas), 'r') as meas:
                model_meas = json.load(meas)
        
            bias_uv_0 = model_meas['setup_00']['bias_uv']
            delta_uv_0 = model_meas['setup_00']['delta_uv']
            
            ss_duvm = 1 - delta_uv / delta_uv_0
            ys -= 0.1
            ax_bla.text(0.05, ys, 'Skill Score DUVM: {}'.format(np.round(ss_duvm, decimals=2)), transform=ax_bla.transAxes)
        else:
            model_meas = {}
            
        if num_test < 10:
            setup = 'setup_0{}'.format(num_test)
        else:
            setup = 'setup_{}'.format(num_test)
        if crossvalid:
            setup = 'setup_{}_cross{}'.format(num_test, ii_crossvalid)
        model_meas[setup] = {'delta_uv':np.float64(delta_uv),
                             'bias_uv':np.float64(bias_uv),
                             'std_uv_pred':np.float64(std_uv_pred),
                             'std_uv_label':np.float64(std_uv_label)}
        
        with open(os.path.join(path_to_model_meas, filename_model_meas), 'w') as meas:
            json.dump(model_meas, meas)  
    
    elif var_plot == 'dswe':
        delta_dswe_pos = np.nanmean(np.array(fields_diag['d_dswe_pos']))
        delta_dswe_neg = np.nanmean(np.array(fields_diag['d_dswe_neg']))
        por_sign_correct = np.nanmean(np.array(fields_diag['por_sign_correct']))
        
        rmse_dswe_pos = np.nanmean(np.array(fields_diag['rmse_dswe_pos']))
        rmse_dswe_neg = np.nanmean(np.array(fields_diag['rmse_dswe_neg']))
        mae_dswe_pos = np.nanmean(np.array(fields_diag['mae_dswe_pos']))
        mae_dswe_neg = np.nanmean(np.array(fields_diag['mae_dswe_neg']))
        bias_dswe_pos = np.nanmean(np.array(fields_diag['bias_dswe_pos']))
        bias_dswe_neg = np.nanmean(np.array(fields_diag['bias_dswe_neg']))
        
        ax_bla.text(0.05, ys, 'D DSWE POS: {} kg m-2 h-1'.format(delta_dswe_pos), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'D DSWE NEG: {} kg m-2 h-1'.format(delta_dswe_neg), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'Portion sign correct: {}'.format(por_sign_correct), transform=ax_bla.transAxes)
        ys -= 0.05
        # ax_bla.text(0.05, ys, por_sign_correct1, transform=ax_bla.transAxes)
        # ys -= 0.05
        # ax_bla.text(0.05, ys, por_sign_correct2, transform=ax_bla.transAxes)
        # ys -= 0.05
        # ax_bla.text(0.05, ys, por_sign_correct3, transform=ax_bla.transAxes)
        # ys -= 0.05
        # ax_bla.text(0.05, ys, por_sign_correct4, transform=ax_bla.transAxes)
        ax_bla.text(0.05, ys, 'rmse dswe pos: {}'.format(rmse_dswe_pos), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'rmse dswe neg: {}'.format(rmse_dswe_neg), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'mae dswe pos: {}'.format(mae_dswe_pos), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'mae dswe neg: {}'.format(mae_dswe_neg), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'bias dswe pos: {}'.format(bias_dswe_pos), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'bias dswe neg: {}'.format(bias_dswe_neg), transform=ax_bla.transAxes)
        
        
        
        path_to_model_meas = ''
        filename_model_meas = 'model_meas_dswe.json'
        if num_test != 0:  
            with open(os.path.join(path_to_model_meas, filename_model_meas), 'r') as meas:
                model_meas = json.load(meas)
        else:
            model_meas = {}
            
        if num_test < 10:
            setup = 'setup_0{}'.format(num_test)
        else:
            setup = 'setup_{}'.format(num_test)
        if crossvalid:
            setup = 'setup_{}_cross{}'.format(num_test, ii_crossvalid)
        model_meas[setup] = {'delta_dswe_pos':np.float64(delta_dswe_pos),
                             'delta_dswe_neg':np.float64(delta_dswe_neg),
                             'rmse_dswe_pos':np.float64(rmse_dswe_pos),
                             'rmse_dswe_neg':np.float64(rmse_dswe_neg),
                             'mae_dswe_pos':np.float64(mae_dswe_pos),
                             'mae_dswe_neg':np.float64(mae_dswe_neg),
                             'bias_dswe_pos':np.float64(bias_dswe_pos),
                             'bias_dswe_neg':np.float64(bias_dswe_neg),
                             'por_sign_correct':np.float64(por_sign_correct)}
        
        with open(os.path.join(path_to_model_meas, filename_model_meas), 'w') as meas:
            json.dump(model_meas, meas)  
            
    elif var_plot == 'subl':
        
        por_90_correct = np.nanmean(np.array(fields_diag['por_90_correct']))
        pixl_subl_neg = np.nanmean(np.array(fields_diag['pixl_subl_neg_list']))
        rmse_subl = np.nanmean(np.array(fields_diag['rmse_subl']))
        mae_subl = np.nanmean(np.array(fields_diag['mae_subl']))
        bias_subl = np.nanmean(np.array(fields_diag['bias_subl']))
        
        ax_bla.text(0.05, ys, 'Portion 90 correct: {}'.format(por_90_correct), transform=ax_bla.transAxes)
        ys -= 0.05      
        ax_bla.text(0.05, ys, 'Number of negative pixels: {}'.format(pixl_subl_neg), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'rmse subl {} kg m-2 h-1'.format(rmse_subl), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'mae subl {} kg m-2 h-1'.format(mae_subl), transform=ax_bla.transAxes)
        ys -= 0.05
        ax_bla.text(0.05, ys, 'bias subl {} kg m-2 h-1'.format(bias_subl), transform=ax_bla.transAxes)
        ys -= 0.05
        if len(namelist['output_layers']) > 1:
            delta_mass_cons = np.nanmean(np.array(fields_diag['delta_mass_cons']))
            ax_bla.text(0.05, ys, 'D mass cons : {}'.format(delta_mass_cons), transform=ax_bla.transAxes)
            ys -= 0.05
        
        
       
        
        path_to_model_meas = ''
        filename_model_meas = 'model_meas_subl.json'
        if os.path.exists(os.path.join(path_to_model_meas, filename_model_meas)):  
            with open(os.path.join(path_to_model_meas, filename_model_meas), 'r') as meas:
                model_meas = json.load(meas)
        else:
            model_meas = {}
            
        if num_test < 10:
            setup = 'setup_0{}'.format(num_test)
        else:
            setup = 'setup_{}'.format(num_test)
        if crossvalid:
            setup = 'setup_{}_cross{}'.format(num_test, ii_crossvalid)
            
        if len(namelist['output_layers']) > 1:
            model_meas[setup] = {'por_90_correct':np.float64(por_90_correct),
                             'pixl_subl_neg':np.float64(pixl_subl_neg),
                             'delta_mass_cons':np.float64(delta_mass_cons),
                             'rmse_subl':np.float64(rmse_subl),
                             'mae_subl':np.float64(mae_subl),
                             'bias_subl':np.float64(bias_subl)}
        else:
            model_meas[setup] = {'por_90_correct':np.float64(por_90_correct),
                             'pixl_subl_neg':np.float64(pixl_subl_neg),
                             'rmse_subl':np.float64(rmse_subl),
                             'mae_subl':np.float64(mae_subl),
                             'bias_subl':np.float64(bias_subl)}
        
        with open(os.path.join(path_to_model_meas, filename_model_meas), 'w') as meas:
            json.dump(model_meas, meas)  
                
    
        
    ax_bla.set_facecolor('white')
    ax_bla.set_xticks([])
    ax_bla.set_yticks([])
    
 
    savename = 'plot_diag_{}_{}{}{}.png'.format(var_plot, num_test, addname, ii_crossvalid)
          
    fig.savefig(os.path.join(path_to_save, savename),dpi=500)
    
def get_field_plot(field_in, var_plot, ii_list, namelist):
    
    print(field_in[0].shape)
    
    if var_plot == 'wind':
        u_np = np.squeeze(field_in[ii_list][0, 0, :, :].cpu().detach().numpy())
        v_np = np.squeeze(field_in[ii_list][0, 1, :, :].cpu().detach().numpy())
        
        dd_pn, field_out = nn_wrf_helpers.uv2ddff(u_np, v_np)
       
        xx, yy = np.meshgrid(np.arange(namelist['nx_data']), np.arange(namelist['nx_data']))
        dx_qv = 20
        u_q, v_q = u_np[::dx_qv, ::dx_qv], v_np[::dx_qv, ::dx_qv]
        x_q, y_q = xx[::dx_qv, ::dx_qv], yy[::dx_qv, ::dx_qv]
    
    elif var_plot == 'dswe':
        field_out = np.squeeze(field_in[ii_list].cpu().detach().numpy())
        u_q, v_q, x_q, y_q = None, None, None, None
    
    elif var_plot == 'subl':
        field_out = - np.squeeze(field_in[ii_list].cpu().detach().numpy()) 
        u_q, v_q, x_q, y_q = None, None, None, None
    
    elif var_plot == 'phiflux':
        field_out = np.squeeze(np.sqrt(field_in[ii_list][0, 0, :, :].cpu().detach().numpy()**2 +
                                 field_in[ii_list][0, 1, :, :].cpu().detach().numpy()**2))
        
        
        uphi_np = np.squeeze(field_in[ii_list][0, 0, :, :].cpu().detach().numpy())
        vphi_np = np.squeeze(field_in[ii_list][0, 1, :, :].cpu().detach().numpy())
               
        xx, yy = np.meshgrid(np.arange(namelist['nx_data']), np.arange(namelist['nx_data']))
        dx_qv = 20
        u_q, v_q = uphi_np[::dx_qv, ::dx_qv], vphi_np[::dx_qv, ::dx_qv]
        x_q, y_q = xx[::dx_qv, ::dx_qv], yy[::dx_qv, ::dx_qv]

    
    return field_out, u_q, v_q, x_q, y_q
    



def make_prediction_get_delta(ii, i_valid, model, data_in, data_out, namelist, norm_factors_in, norm_factors_out, fields_diag,
                              make_single_prediction=True, var_diag='wind', surr_model=False, surr_model_double=False):
    
    # print(ii)
    # print(var_diag)
    # preparation stuff: get input data in right shape
    chan_in = data_in.shape[1]
    chan_out = data_out.shape[1]
    nx_in, nx_out = namelist['nx_data'], namelist['nx_data']

    if surr_model:
        nx_in = 2*namelist['nx_data']
    if surr_model_double:
        nx_in, nx_out = 2*namelist['nx_data'], 2*namelist['nx_data']
        
    
    xi, yi = data_in[i_valid, :, :, :], data_out[i_valid, :, :, :]
    
    if make_single_prediction:
        xi = torch.reshape(xi, (1, chan_in, nx_in, nx_in))
        yi = torch.reshape(yi, (1, chan_out, nx_out, nx_out))
    
    # make prediction
    pred_norm = model(xi)
    ter_norm = xi[:, 0, :, :]
    
    if chan_out == 1:
        label_norm = yi[:, 0, :, :]
    else:
        label_norm = yi
        
    # denormalization stuff
    if chan_out == 1:
        if type(namelist['normalization']) == str:
            if namelist['normalization'] == 'minmax_01':
                pred_unnorm = nn_wrf_helpers.unnormalize_field_minmax(pred_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                ter_unnorm = nn_wrf_helpers.unnormalize_field_minmax(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                label_unnorm = nn_wrf_helpers.unnormalize_field_minmax(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
            elif namelist['normalization'] == 'meanstd':
                pred_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(pred_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                ter_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                label_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
            elif namelist['normalization'] == 'meanvar':
                pred_unnorm = nn_wrf_helpers.unnormalize_field_meanvar(pred_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                ter_unnorm = nn_wrf_helpers.unnormalize_field_meanvar(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                label_unnorm = nn_wrf_helpers.unnormalize_field_meanvar(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
        
        else:
            if namelist['normalization'][0][0] == 'minmax_01':
                ter_unnorm = nn_wrf_helpers.unnormalize_field_minmax(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
            elif namelist['normalization'][0][0] == 'meanstd':
                ter_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
            if namelist['normalization'][0][0] == 'minmax_11':
                ter_unnorm = nn_wrf_helpers.unnormalize_field_minmax_11(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
            
            if namelist['normalization'][1][0] == 'minmax_01':
                pred_unnorm = nn_wrf_helpers.unnormalize_field_minmax(pred_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                label_unnorm = nn_wrf_helpers.unnormalize_field_minmax(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
            elif namelist['normalization'][1][0] == 'meanstd':
                pred_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(pred_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                label_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
            elif namelist['normalization'][1][0] == 'minmax_11_p0':
                pred_unnorm = nn_wrf_helpers.unnormalize_field_minmax_11_p0(pred_norm, norm_factors_in[0][0])
                label_unnorm = nn_wrf_helpers.unnormalize_field_minmax_11_p0(label_norm, norm_factors_out[0][0])        
    else:
        # for now: only meanstd for all fields
        pred_unnorm = torch.empty_like(pred_norm)
        label_unnorm = torch.empty_like(label_norm)
        for chan_out_i in range(chan_out):
            pred_unnorm_tmp = nn_wrf_helpers.unnormalize_field_meanstd(pred_norm[:, chan_out_i, :, :], norm_factors_out[chan_out_i][0], norm_factors_out[chan_out_i][1])
            pred_unnorm[:, chan_out_i, :, :] = pred_unnorm_tmp
            label_unnorm_tmp = nn_wrf_helpers.unnormalize_field_meanstd(label_norm[:, chan_out_i, :, :], norm_factors_out[chan_out_i][0], norm_factors_out[chan_out_i][1])
            label_unnorm[:, chan_out_i, :, :] = label_unnorm_tmp
        ter_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])

        
    # diagnostic stuff
    # wind: pixel-wise deviation, bias, spread
    if var_diag == 'wind':
        delta_u = pred_unnorm[:, 0, :, :] - label_unnorm[:, 0, :, :]
        delta_v = pred_unnorm[:, 1, :, :] - label_unnorm[:, 1, :, :]
   
        bias_u_m = torch.mean(delta_u).cpu().detach().numpy()
        bias_v_m = torch.mean(delta_v).cpu().detach().numpy()
        bias_uv_out = np.mean((bias_u_m, bias_v_m))
        
        delta_u_abs_m = torch.mean(torch.abs(delta_u)).cpu().detach().numpy()
        delta_v_abs_m = torch.mean(torch.abs(delta_v)).cpu().detach().numpy()
        delta_uv_abs_out = np.mean((delta_u_abs_m, delta_v_abs_m))
        
        std_u_pred = torch.std(pred_unnorm[:, 0, :, :]).cpu().detach().numpy()
        std_v_pred = torch.std(pred_unnorm[:, 1, :, :]).cpu().detach().numpy()
        std_u_label = torch.std(label_unnorm[:, 0, :, :]).cpu().detach().numpy()
        std_v_label = torch.std(label_unnorm[:, 1, :, :]).cpu().detach().numpy()
        
        std_uv_pred = np.mean((std_u_pred, std_v_pred))
        std_uv_label = np.mean((std_u_label, std_v_label))
        
        # store in out dictionary
        if ii == 0:
            # print('hello! round zero!')
            bias_uv_list = []
            delta_uv_abs_list = []
            std_uv_pred_list = []
            std_uv_label_list = []
        else:
            # print(fields_diag)
            bias_uv_list = fields_diag['bias_uv']
            delta_uv_abs_list = fields_diag['delta_uv_abs']
            std_uv_pred_list = fields_diag['std_uv_pred']
            std_uv_label_list = fields_diag['std_uv_label']
        
        bias_uv_list.append(bias_uv_out)
        delta_uv_abs_list.append(delta_uv_abs_out)
        std_uv_pred_list.append(std_uv_pred)
        std_uv_label_list.append(std_uv_label)
        
        fields_diag['bias_uv'] = bias_uv_list
        fields_diag['delta_uv_abs'] = delta_uv_abs_list
        fields_diag['std_uv_pred'] = std_uv_pred_list
        fields_diag['std_uv_label'] = std_uv_label_list      
        
        # print(fields_diag)
        
        # idee: rmse u, v -> mean, std u, v (label, pred) -> mean => quasi-taylor
        
        
    # dswe: pixel-wise deviation in ero and depo, position of max ero nad depo areas
    elif var_diag == 'snow': 
        
        if 'dswe' in namelist['output_layers']:
            if chan_out == 1:
                dswe_label = label_unnorm.cpu().detach().numpy().squeeze()
                dswe_pred = pred_unnorm.cpu().detach().numpy().squeeze()
            else:
                dswe_label = label_unnorm[:, 0, :, :].cpu().detach().numpy().squeeze()
                dswe_pred = pred_unnorm[:, 0, :, :].cpu().detach().numpy().squeeze()
            
            delta_dswe_pos = np.nanmean(dswe_pred[dswe_pred > 0]) - np.nanmean(dswe_label[dswe_label > 0])
            delta_dswe_neg = np.nanmean(dswe_pred[dswe_pred < 0]) - np.nanmean(dswe_label[dswe_label < 0])
        
            # idee: stärkste ero/dep gebiete: 10/90 perc. -> masken -> anteil überlappen -> stärkste Gebiete richtig platziert?
            dswe_label_pos90 = np.nanpercentile(dswe_label[dswe_label > 0], 90)
            dswe_pred_pos90 = np.nanpercentile(dswe_pred[dswe_pred > 0], 90)
        
            dswe_label_neg10 = np.nanpercentile(dswe_label[dswe_label < 0], 10)
            dswe_pred_neg10 = np.nanpercentile(dswe_pred[dswe_pred < 0], 10)
        
            mask_pos90_label = dswe_label >= dswe_label_pos90
            mask_pos90_pred = dswe_pred >= dswe_pred_pos90

            mask_neg10_label = dswe_label <= dswe_label_neg10
            mask_neg10_pred = dswe_pred <= dswe_pred_neg10 
        
            por_sign_correct_pos = np.sum(np.logical_and(mask_pos90_label, mask_pos90_pred)) / np.sum(mask_pos90_label)
            por_sign_correct_neg = np.sum(np.logical_and(mask_neg10_label, mask_neg10_pred)) / np.sum(mask_neg10_label)
          
            por_sign_correct = np.mean((por_sign_correct_pos, por_sign_correct_neg))
            
            
            # mae/rmse/bias pos/neg regions in label
            mask_pos_label = dswe_label >= 0
            mask_neg_label = dswe_label < 0
            # print(dswe_pred.shape)
            # print(dswe_label.shape)
            # print(dswe_pred[mask_pos_label].shape)
            # print(dswe_label[mask_pos_label].shape)
            
            rmse_dswe_pos = np.sqrt(np.nanmean((dswe_pred[mask_pos_label].flatten() - dswe_label[mask_pos_label].flatten())**2))
            rmse_dswe_neg = np.sqrt(np.nanmean((dswe_pred[mask_neg_label].flatten() - dswe_label[mask_neg_label].flatten())**2))
            mae_dswe_pos = np.nanmean(np.abs(dswe_pred[mask_pos_label].flatten() - dswe_label[mask_pos_label].flatten()))
            mae_dswe_neg = np.nanmean(np.abs(dswe_pred[mask_neg_label].flatten() - dswe_label[mask_neg_label].flatten()))
            bias_dswe_pos = np.nanmean(dswe_pred[mask_pos_label].flatten() - dswe_label[mask_pos_label].flatten())
            bias_dswe_neg = np.nanmean(dswe_pred[mask_neg_label].flatten() - dswe_label[mask_neg_label].flatten())
        
            # store in out dictionary
            if ii == 0:
                d_dswe_pos_list = []
                d_dswe_neg_list = []
                por_sign_correct_list = []
                rmse_dswe_pos_list = []
                rmse_dswe_neg_list = []
                mae_dswe_pos_list = []
                mae_dswe_neg_list = []
                bias_dswe_pos_list = []
                bias_dswe_neg_list = []
            else:
                d_dswe_pos_list = fields_diag['d_dswe_pos']
                d_dswe_neg_list = fields_diag['d_dswe_neg']
                por_sign_correct_list = fields_diag['por_sign_correct']
                rmse_dswe_pos_list = fields_diag['rmse_dswe_pos']
                rmse_dswe_neg_list = fields_diag['rmse_dswe_neg']
                mae_dswe_pos_list = fields_diag['mae_dswe_pos']
                mae_dswe_neg_list = fields_diag['mae_dswe_neg']
                bias_dswe_pos_list = fields_diag['bias_dswe_pos']
                bias_dswe_neg_list = fields_diag['bias_dswe_neg']
        
            d_dswe_pos_list.append(delta_dswe_pos)
            d_dswe_neg_list.append(delta_dswe_neg)
            por_sign_correct_list.append(por_sign_correct)
            rmse_dswe_pos_list.append(rmse_dswe_pos)
            rmse_dswe_neg_list.append(rmse_dswe_neg)
            mae_dswe_pos_list.append(mae_dswe_pos)
            mae_dswe_neg_list.append(mae_dswe_neg)
            bias_dswe_pos_list.append(bias_dswe_pos)
            bias_dswe_neg_list.append(bias_dswe_neg)
        
            fields_diag['d_dswe_pos'] = d_dswe_pos_list
            fields_diag['d_dswe_neg'] = d_dswe_neg_list
            fields_diag['por_sign_correct'] = por_sign_correct_list
            fields_diag['rmse_dswe_pos'] = rmse_dswe_pos_list
            fields_diag['rmse_dswe_neg'] = rmse_dswe_neg_list
            fields_diag['mae_dswe_pos'] = mae_dswe_pos_list
            fields_diag['mae_dswe_neg'] = mae_dswe_neg_list
            fields_diag['bias_dswe_pos'] = bias_dswe_pos_list
            fields_diag['bias_dswe_neg'] = bias_dswe_neg_list
       
    
    # subl: pixel-wise deviation, position of max area, any wrong sign?, mass conservation
        if 'subl_vertint' in namelist['output_layers']:

            if chan_out == 1:
                subl_label = label_unnorm.cpu().detach().numpy()
                subl_pred = pred_unnorm.cpu().detach().numpy()
            else:
                subl_pred = pred_unnorm[:, 1, :, :].cpu().detach().numpy()
                subl_label = label_unnorm[:, 1, :, :].cpu().detach().numpy()
        
            # pixelwise deviation
            bias_subl = np.nanmean(subl_pred - subl_label)
        
            # position of max area
            subl_label_pos90 = np.nanpercentile(subl_label, 90)
            subl_pred_pos90 = np.nanpercentile(subl_pred, 90)
        
            mask_90_label = subl_label >= subl_label_pos90
            mask_90_pred = subl_pred >= subl_pred_pos90
        
            por_90_correct = np.sum(np.logical_and(mask_90_label, mask_90_pred)) / np.sum(mask_90_label)
        
            # any wrong sign?
            pixl_subl_neg = np.sum(subl_pred > 0)
            
            
            # rmse/mae/bias
            rmse_subl = np.sqrt(np.nanmean((subl_pred - subl_label)**2))
            mae_subl = np.nanmean(np.abs(subl_pred - subl_label))
        
            if chan_out == 1:
                delta_mass_cons = None
            else:
                # mass conservation? relative to total eroded mass
                delta_mass_cons = (np.sum(dswe_pred) + np.sum(subl_pred)) / np.sum(dswe_pred[dswe_pred < 0])
            
            # store in out dictionary
            if ii == 0:
                por_90_correct_list = []
                pixl_subl_neg_list = []
                delta_mass_cons_list = []
                rmse_subl_list = []
                mae_subl_list = []
                bias_subl_list = []
            else:
                por_90_correct_list = fields_diag['por_90_correct']
                pixl_subl_neg_list = fields_diag['pixl_subl_neg_list']
                delta_mass_cons_list = fields_diag['delta_mass_cons']
                rmse_subl_list = fields_diag['rmse_subl']
                mae_subl_list = fields_diag['mae_subl']
                bias_subl_list = fields_diag['bias_subl']
                
            por_90_correct_list.append(por_90_correct)
            pixl_subl_neg_list.append(pixl_subl_neg)
            delta_mass_cons_list.append(delta_mass_cons)
            rmse_subl_list.append(rmse_subl)
            mae_subl_list.append(mae_subl)
            bias_subl_list.append(bias_subl)
            
            fields_diag['por_90_correct'] = por_90_correct_list
            fields_diag['pixl_subl_neg_list'] = pixl_subl_neg_list
            fields_diag['delta_mass_cons'] = delta_mass_cons_list
            fields_diag['rmse_subl'] = rmse_subl_list
            fields_diag['mae_subl'] = mae_subl_list 
            fields_diag['bias_subl'] = bias_subl_list
    
            
    elif var_diag == 'phi':
        fields_diag = None
                
        # pixel wise, por 90 correct?
    return fields_diag
    


def make_prediction_test(ii, model, data_in, data_out, namelist, norm_factors_in, norm_factors_out, 
                         make_single_prediction=True, unnorm_out=True, surr_model=False, surr_model_double=False, logtrans_subl=False, 
                         logtrans_dswe=False, offset_trans_dswe=None):
    
    chan_in = data_in.shape[1]
    chan_out = data_out.shape[1]
    nx_in, nx_out = namelist['nx_data'], namelist['nx_data']
    
    if surr_model:
        nx_in = 2*namelist['nx_data']
    if surr_model_double:
        nx_in, nx_out = 2*namelist['nx_data'], 2*namelist['nx_data']
        
    i_mid1 = int(namelist['nx_data']/2)
    i_mid2 = i_mid1+namelist['nx_data']     
    
    
    if namelist['batch_norm']:
        xi, yi, norm_factors_in, norm_factors_out = nn_wrf_main.normalize_all_data(namelist, torch.reshape(data_in[ii, :, :, :], (1, chan_in, nx_in, nx_in)),
                                                                                   torch.reshape(data_out[ii, :, :, :], (1, chan_out, nx_out, nx_out)))
    else:
        xi, yi = data_in[ii, :, :, :], data_out[ii, :, :, :]
        
    if make_single_prediction:
        xi = torch.reshape(xi, (1, chan_in, nx_in, nx_in))
        yi = torch.reshape(yi, (1, chan_out, nx_out, nx_out))
    
    pred_norm = model(xi)
    ter_norm = xi[:, 0, :, :]
    
    if chan_out == 1:
        label_norm = yi[:, 0, :, :]
    else:
        label_norm = yi
    
    
    if unnorm_out:
        if chan_out == 1:
            if type(namelist['normalization']) == str:
                if namelist['normalization'] == 'minmax_01':
                    pred_unnorm = nn_wrf_helpers.unnormalize_field_minmax(pred_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                    ter_unnorm = nn_wrf_helpers.unnormalize_field_minmax(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                    label_unnorm = nn_wrf_helpers.unnormalize_field_minmax(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                elif namelist['normalization'] == 'meanstd':
                    pred_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(pred_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                    ter_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                    label_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                elif namelist['normalization'] == 'meanvar':
                    pred_unnorm = nn_wrf_helpers.unnormalize_field_meanvar(pred_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                    ter_unnorm = nn_wrf_helpers.unnormalize_field_meanvar(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                    label_unnorm = nn_wrf_helpers.unnormalize_field_meanvar(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
        
            else:
                if namelist['normalization'][0][0] == 'minmax_01':
                    ter_unnorm = nn_wrf_helpers.unnormalize_field_minmax(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                elif namelist['normalization'][0][0] == 'meanstd':
                    ter_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                if namelist['normalization'][0][0] == 'minmax_11':
                    ter_unnorm = nn_wrf_helpers.unnormalize_field_minmax_11(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])
            
                if namelist['normalization'][1][0] == 'minmax_01':
                    pred_unnorm = nn_wrf_helpers.unnormalize_field_minmax(pred_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                    label_unnorm = nn_wrf_helpers.unnormalize_field_minmax(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                elif namelist['normalization'][1][0] == 'meanstd':
                    pred_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(pred_norm, norm_factors_in[0][0], norm_factors_in[0][1])
                    label_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(label_norm, norm_factors_out[0][0], norm_factors_out[0][1])
                elif namelist['normalization'][1][0] == 'minmax_11_p0':
                    pred_unnorm = nn_wrf_helpers.unnormalize_field_minmax_11_p0(pred_norm, norm_factors_in[0][0])
                    label_unnorm = nn_wrf_helpers.unnormalize_field_minmax_11_p0(label_norm, norm_factors_out[0][0])        
        else:
        # for now: only meanstd for all fields
            pred_unnorm = torch.empty_like(pred_norm)
            label_unnorm = torch.empty_like(label_norm)
            for chan_out_i in range(chan_out):
                pred_unnorm_tmp = nn_wrf_helpers.unnormalize_field_meanstd(pred_norm[:, chan_out_i, :, :], norm_factors_out[chan_out_i][0], norm_factors_out[chan_out_i][1])
                pred_unnorm[:, chan_out_i, :, :] = pred_unnorm_tmp
                label_unnorm_tmp = nn_wrf_helpers.unnormalize_field_meanstd(label_norm[:, chan_out_i, :, :], norm_factors_out[chan_out_i][0], norm_factors_out[chan_out_i][1])
                label_unnorm[:, chan_out_i, :, :] = label_unnorm_tmp
            ter_unnorm = nn_wrf_helpers.unnormalize_field_meanstd(ter_norm, norm_factors_in[0][0], norm_factors_in[0][1])

        
        
        if logtrans_subl:
            label_unnorm = nn_wrf_helpers.unmake_logtrans_subl(label_unnorm, namelist)
            pred_unnorm = nn_wrf_helpers.unmake_logtrans_subl(pred_unnorm, namelist)
            
        if logtrans_dswe:
            label_unnorm = nn_wrf_helpers.unmake_logtrans_dswe(label_unnorm, namelist, offset_trans_dswe)
            pred_unnorm = nn_wrf_helpers.unmake_logtrans_dswe(pred_unnorm, namelist, offset_trans_dswe)

        if 'ter_log' in namelist['input_layers']:
            ter_unnorm = torch.exp(ter_unnorm) - 10
        # print(label_unnorm.shape)
        # print(pred_unnorm.shape)
        # print(ter_unnorm.shape)
        
        return pred_unnorm, ter_unnorm, label_unnorm
        
    
    
        return pred_norm, ter_norm, label_norm
    
    
def config_cmaps():
    
    config = {
        'phiflux_vertint':{
            'cmap':mpl.colors.ListedColormap(("white","#F0F2DC","#EAF2D7","#E2F1D2","#D8F0CD","#CBEDCA","#BDEAC6","#ADE5C4",
                                                       "#9CDFC2","#8AD8C0","#77CFBE","#63C6BC","#4EBBB9","#39AEB6","#23A1B2","#0A92AC","#0082A6","#00709E","#135E95","#21498C","#2D3184")),
            'level_min':0,
            'level_max':0.01,
            'n_cmap':21,
            'extend_cmap':'max',
            'n_cbar':5,
            'text_cbar':'$\Phi uv_{vint}$ (kg m$^{-1}$ s$^{-1}$)',
            'factor_adjust':10,
            'perc_thres':95
            },
        'subl_vertint':{
            'cmap':mpl.colors.ListedColormap(("white","#F7FDDA","#EFF9D5","#E5F4CD","#DAEDC4","#CDE6BA","#BEDEAF","#AED6A3",
                                        "#9BCD96","#85C489","#6CBA7C","#4BB06F","#00A663","#009B57","#00904D","#008544",
                                        "#007643","#00663F","#00563A","#004632","#00362A")),
            'level_min':0,
            'level_max':0.02,
            'n_cmap':21,
            'extend_cmap':'max',
            'n_cbar':5,
            'text_cbar':'$\Delta m_{subl, vint}$ (kg m$^{-2}$ h$^{-1}$)',
            'factor_adjust':10,
            'perc_thres':95
            },
        'dswe':{
            'cmap':'RdBu',
            'level_min':-0.1,
            'level_max':0.1,
            'n_cmap':21,
            'extend_cmap':'both',
            'n_cbar':5,
            'text_cbar':'$\Delta$SWE (kg m$^{-2}$ h$^{-1}$)',
            'factor_adjust':10,
            'perc_thres':95
            },
        'ff':{
            'cmap':'plasma_r',
            'level_min':0,
            'level_max':5,
            'n_cmap':21,
            'extend_cmap':'max',
            'n_cbar':5,
            'text_cbar':'ff (m s$^{-1}$)',
            'factor_adjust':2,
            'perc_thres':99
            }
        }
    
    return config

def adjust_cmap(config_cmap, field, adjust=True):
    
    perc_thres = config_cmap['perc_thres']
    
    print(np.nanpercentile(field, perc_thres))
    if adjust:
        if config_cmap['extend_cmap'] == 'max':
            if np.nanpercentile(field, perc_thres) < config_cmap['level_max']:
                levels_cmap = np.linspace(config_cmap['level_min'], config_cmap['level_max'], config_cmap['n_cmap'])
                levels_cbar = np.linspace(config_cmap['level_min'], config_cmap['level_max'], config_cmap['n_cbar'])
            
            else:
                 level_new_max = config_cmap['level_max'] * config_cmap['factor_adjust']
                 while np.nanpercentile(field, perc_thres) > level_new_max:
                     print(level_new_max)
                     level_new_max *= config_cmap['factor_adjust']
                 levels_cmap = np.linspace(config_cmap['level_min'], level_new_max, config_cmap['n_cmap'])
                 levels_cbar = np.linspace(config_cmap['level_min'], level_new_max, config_cmap['n_cbar'])
        
        
        elif config_cmap['extend_cmap'] == 'both':
            if np.nanpercentile(np.abs(field), perc_thres) < config_cmap['level_max']:
                levels_cmap = np.linspace(config_cmap['level_min'], config_cmap['level_max'], config_cmap['n_cmap'])
                levels_cbar = np.linspace(config_cmap['level_min'], config_cmap['level_max'], config_cmap['n_cbar'])
                
            else:
                level_new_max = config_cmap['level_max'] * config_cmap['factor_adjust']
                while np.nanpercentile(np.abs(field), perc_thres) > level_new_max:
                    level_new_max *= config_cmap['factor_adjust']
                
                levels_cmap = np.linspace(-level_new_max, level_new_max, config_cmap['n_cmap'])
                levels_cbar = np.linspace(-level_new_max, level_new_max, config_cmap['n_cbar'])
                
        
        elif config_cmap['extend_cmap'] == 'min':
            if np.nanpercentile(field, 100-perc_thres) > config_cmap['level_min']:
                levels_cmap = np.linspace(config_cmap['level_min'], config_cmap['level_max'], config_cmap['n_cmap'])
                levels_cbar = np.linspace(config_cmap['level_min'], config_cmap['level_max'], config_cmap['n_cbar'])
            
            else:
                 level_new_min = config_cmap['level_min'] * config_cmap['factor_adjust']
                 print(level_new_min)
                 while np.nanpercentile(field, 100-perc_thres) <= level_new_min:
                     print(level_new_min)
                     level_new_min *= config_cmap['factor_adjust']
                 levels_cmap = np.linspace(level_new_min, config_cmap['level_max'], config_cmap['n_cmap'])
                 levels_cbar = np.linspace(level_new_min, config_cmap['level_max'], config_cmap['n_cbar'])
    else:
        levels_cmap = np.linspace(config_cmap['level_min'], config_cmap['level_max'], config_cmap['n_cmap'])
        levels_cbar = np.linspace(config_cmap['level_min'], config_cmap['level_max'], config_cmap['n_cbar'])
        
        
    return levels_cmap, levels_cbar


