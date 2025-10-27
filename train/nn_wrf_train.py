#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:38:42 2024

@author: manuel
"""

import numpy as np
import xarray as xr

import torch
from torch import nn

import os
import json

import nn_wrf_main
import nn_wrf_diag
import nn_wrf_helpers

def train(data_in, data_out, model, namelist, ii_data_diag, norm_factors_in, norm_factors_out, num_namelist_name, hpc, plot_epochs_evo, 
          save_early_model=False, surr_model=False, surr_model_double=False, crossvalid=False, addname='', ii_crossvalid='', 
          logtrans_subl=False, logtrans_dswe=False, offset_trans_dswe=None):
    # main function for training
    
    # set up hyperparameters
    # get loss function and optimizer from namelist
    loss_mc = False
    loss_sublneg = False
    if namelist['loss_fn'] == 'mse':
        loss_fn = nn.MSELoss()
    elif namelist['loss_fn'] == 'mae':
        loss_fn = nn.L1Loss()
    elif namelist['loss_fn'] == 'huber':
        loss_fn = nn.HuberLoss(delta=namelist['loss_huber_delta'])
    elif namelist['loss_fn'] == 'mse_mc':
        loss_mc = True
        loss_fn = mse_mc_loss()
    elif namelist['loss_fn'] == 'mse_norm':
        loss_fn = mse_norm()
    elif namelist['loss_fn'] == 'mae_norm':
        loss_fn = mae_norm()
    elif namelist['loss_fn'] == 'mse_sublneg':
        loss_sublneg = True
        loss_fn = mse_sublneg_loss()
    else:
        raise RuntimeError('Unknown loss function!')
    
    # optimizer
    if namelist['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=namelist['learning_rate'])
    elif namelist['optimizer'] == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=namelist['learning_rate'])
    elif namelist['optimizer'] == 'sgd':
        if 'optim_moment' in namelist.keys():
            optim_moment = namelist['optim_moment']
        else:
            optim_moment = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=namelist['learning_rate'], momentum=optim_moment)
    else:
        raise RuntimeError('Unknown optimizer!')
    
    # learning rate reduction
    if 'reduce_lr' in namelist.keys():
        reduce_lr = namelist['reduce_lr']
        lr = namelist['learning_rate']
        if 'factor_reduce' in namelist.keys():
            factor_reduce = namelist['factor_reduce']
        else:
            factor_reduce = 0.5
    else:
        reduce_lr = False
        
    # online augmentation
    augmentation_online = False
    if 'augmentation_online' in namelist.keys():
        augmentation_online = namelist['augmentation_online']
    
    
    # get dimensions for data sets
    ii_data_train = np.arange(data_in.shape[0]) # indices for training samples
    chan_in = data_in.shape[1]                  # number of input channels
    chan_out = data_out.shape[1]                # number of output channels
    nx_in, nx_out = namelist['nx_data'], namelist['nx_data'] # dimensions in x and y
    
    
    ## prepare train and validation data
    ii_data_train_exclude = ii_data_train[~np.isin(ii_data_train, ii_data_diag)] # pre-defined sims for diagnostic are default in validation data
    np.random.seed(1)
    np.random.shuffle(ii_data_train_exclude)
    ii_divide = int(len(ii_data_train_exclude)*namelist['validation_split']) - len(ii_data_diag) + 1
    
    ii_valid = np.concatenate((ii_data_train_exclude[0:ii_divide], ii_data_diag))
    print('Id Valiation Dataset: {}'.format(ii_valid))  
    ii_train = ii_data_train_exclude[ii_divide:]
    

    
    
    # training loop
    train_loss, valid_loss = [], []
    for t_epoch in range(namelist['epochs']):
        
        if namelist['epochs'] <= 10:
            n_print = 1
        else:
            n_print = 50
        if t_epoch % n_print == 0:
            print('----- Epoch {} / {} -----'.format(t_epoch+1, namelist['epochs']))
            
        if reduce_lr: # reduce learning rate every 100 epochs
            if t_epoch >= 100 and t_epoch % 100 == 0:
                lr *= factor_reduce
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            
        
        np.random.shuffle(ii_train)
        data_in_train, data_out_train = data_in[ii_train, :, :, :], data_out[ii_train, :, :, :]
        data_in_valid, data_out_valid = data_in[ii_valid, :, :, :], data_out[ii_valid, :, :, :]
        n_train, n_valid = len(ii_train), len(ii_valid)   
        # print(n_train)
        # print(data_in_train.shape)
        
        if augmentation_online:
            data_in_train, data_out_train = nn_wrf_helpers.main_augment(data_in_train, data_out_train, namelist,
                                                                        augmentation_online=augmentation_online)
            data_in_valid, data_out_valid = nn_wrf_helpers.main_augment(data_in_valid, data_out_valid, namelist,
                                                                        augmentation_online=augmentation_online)      
        
        
            
        train_loss_tmp = train_loop(data_in_train, data_out_train, model, loss_fn, optimizer, namelist, n_train, 
                                    chan_in, chan_out, nx_in, nx_out, norm_factors_out, loss_mc, loss_sublneg, 
                                    logtrans_subl=logtrans_subl, logtrans_dswe=logtrans_dswe, offset_trans_dswe=offset_trans_dswe)
        valid_loss_tmp = valid_loop(data_in_valid, data_out_valid, model, loss_fn, optimizer, namelist, n_valid, 
                                    chan_in, chan_out, nx_in, nx_out, norm_factors_out, loss_mc, loss_sublneg, 
                                    logtrans_subl=logtrans_subl, logtrans_dswe=logtrans_dswe, offset_trans_dswe=offset_trans_dswe)
    
        train_loss.append(train_loss_tmp)
        valid_loss.append(valid_loss_tmp)
        
            
        if save_early_model:
            if t_epoch % 500 == 0 and t_epoch > 0:
                nn_wrf_main.make_save_model(model, t_epoch, namelist, num_namelist_name,
                                            addname=addname, ii_crossvalid=ii_crossvalid)
        
        
        
    return train_loss, valid_loss, ii_valid
        
        

def train_loop(data_in, data_out, model, loss_fn, optimizer, namelist, n_data, 
               chan_in, chan_out, nx_in, nx_out, norm_factors_out, loss_mc, loss_sublneg, logtrans_subl=False, logtrans_dswe=False, offset_trans_dswe=None):
    # single training epoch: build batch, loop over batches, make prediction, calculate loss, backpropagation, optimize
    
    model.train() # with optimization
        
    # get number of batch loops
    n_batchloops = ((n_data-1) // namelist['batch_size']) + 1
    
    # loop over batches
    ii_data = np.arange(n_data)
    ii_floor = 0    
    train_loss = 0
    for i_batch in range(n_batchloops):
        # build input data
        if ii_floor+namelist['batch_size'] < n_data:
            ii_use = ii_data[ii_floor:ii_floor+namelist['batch_size']]
        else:
            ii_use = ii_data[ii_floor:]
        ii_floor += namelist['batch_size']
        
        if namelist['batch_norm']:
            xi, yi, norm_factors_in, norm_factors_out = nn_wrf_main.normalize_all_data(namelist, data_in[ii_use, :, :, :],
                                                                                       data_out[ii_use, :, :, :])
            if len(ii_use) <= 1:
                xi = torch.reshape(xi, (1, chan_in, nx_in, nx_in))
                yi = torch.reshape(yi, (1, chan_in, nx_out, nx_out))
                
        else:
            # print(ii_use)
            if len(ii_use) > 1:
                xi = data_in[ii_use, :, :, :]
                yi = data_out[ii_use, :, :, :]
            else:
                xi = torch.reshape(data_in[ii_use, :, :, :], (1, chan_in, nx_in, nx_in))
                yi = torch.reshape(data_out[ii_use, :, :, :], (1, chan_out, nx_out, nx_out))
        
        # make model prediction
        pred_tmp = model(xi)
        
        # calculate loss
        if loss_mc:
            area_full = (namelist['nx_wrf']*50)**2
            loss_tmp = loss_fn(pred_tmp, yi, norm_factors_out[0][0], norm_factors_out[0][1], norm_factors_out[1][0], norm_factors_out[1][1], 
                               area_full, namelist['alpha_loss_mc'], logtrans_subl=logtrans_subl, logtrans_dswe=logtrans_dswe, offset_trans_dswe=offset_trans_dswe)
        elif loss_sublneg:
            loss_tmp = loss_fn(pred_tmp, yi, norm_factors_out[0][1], namelist['alpha_loss_sublneg'])
        else:
            loss_tmp = loss_fn(pred_tmp, yi)
        
        train_loss += loss_tmp.item()
        
        
        # backpropagation
        loss_tmp.backward(retain_graph=True)
        
        # optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss /= n_batchloops

    return train_loss


def valid_loop(data_in, data_out, model, loss_fn, optimizer, namelist, n_data, 
               chan_in, chan_out, nx_in, nx_out, norm_factors_out, loss_mc, loss_sublneg, logtrans_subl=False, logtrans_dswe=False, offset_trans_dswe=None):
    # single validation epoch: loop over validation set, make prediction, calculate loss, write out
    
    model.eval() # no optimization
    
    # loop over validation data
    valid_loss = 0
    for ii_use in range(n_data):
        if namelist['batch_norm']:
            xi, yi, norm_factors_in, norm_factors_out = nn_wrf_main.normalize_all_data(namelist,
                                                                                       torch.reshape(data_in[ii_use, :, :, :], (1, chan_in, nx_in, nx_in)),
                                                                                       torch.reshape(data_out[ii_use, :, :, :], (1, chan_out, nx_out, nx_out)))
            xi = torch.reshape(xi, (1, chan_in, nx_in, nx_in))
            yi = torch.reshape(yi, (1, chan_out, nx_out, nx_out))
            
        else:
            xi = torch.reshape(data_in[ii_use, :, :, :], (1, chan_in, nx_in, nx_in))
            yi = torch.reshape(data_out[ii_use, :, :, :], (1, chan_out, nx_out, nx_out))        
       
        # make model prediction
        pred_tmp = model(xi)
        
        # calculate loss
        if loss_mc:
            area_full = (namelist['nx_wrf']*50)**2
            loss_tmp = loss_fn(pred_tmp, yi, norm_factors_out[0][0], norm_factors_out[0][1], norm_factors_out[1][0], norm_factors_out[1][1], 
                               area_full, namelist['alpha_loss_mc'], logtrans_subl=logtrans_subl, logtrans_dswe=logtrans_dswe, offset_trans_dswe=offset_trans_dswe)
        elif loss_sublneg:
            loss_tmp = loss_fn(pred_tmp, yi, norm_factors_out[0][1], namelist['alpha_loss_sublneg'])
        else:
            loss_tmp = loss_fn(pred_tmp, yi)
        
        valid_loss += loss_tmp.item()

    valid_loss /= n_data

    return valid_loss    
    
        
    
class mse_mc_loss(nn.Module):
    # selfmade loss function: mse with penalty term for violation of mass conservation
    def __init__(self):
        super(mse_mc_loss, self).__init__()
    
    def forward(self, inputs, targets, norm_factor_1_1, norm_factor_1_2, norm_factor_2_1, norm_factor_2_2, area_full, alpha, 
                logtrans_subl=False, logtrans_dswe=False, offset_trans_dswe=None):
        
        mse = torch.mean((inputs - targets)**2)
        
        if logtrans_subl:
            subl_log_unnorm = inputs[:, 1, :, :] * norm_factor_2_1 + norm_factor_2_2
            subl_unnorm = -(torch.exp(subl_log_unnorm) - 0.0001)
        else:
            subl_unnorm = inputs[:, 1, :, :] * norm_factor_2_1 + norm_factor_2_2
        
        if logtrans_dswe:
            dswe_log_unnorm = inputs[:, 0, :, :] * norm_factor_1_1 + norm_factor_1_2
            dswe_unnorm = torch.exp(dswe_log_unnorm) - offset_trans_dswe
        else:
            dswe_unnorm = inputs[:, 0, :, :] * norm_factor_1_1 + norm_factor_1_2

                   
        dmc = (torch.sum(dswe_unnorm) - torch.sum(subl_unnorm)) / area_full
        loss = mse + alpha * dmc
        
        return loss
        

class mse_sublneg_loss(nn.Module):
    # selfmade loss function: mse with penalty term for negative sublimation rates
    
    def __init__(self):
        super(mse_sublneg_loss, self).__init__()
    
    def forward(self, inputs, targets, norm_factor_1_2, alpha):
        
        mse = torch.mean((inputs - targets)**2)
        
        penalty_neg = torch.sum(inputs > norm_factor_1_2) 
               
        
        loss = mse + alpha * penalty_neg
        
        return loss
    
class mse_norm(nn.Module):
    
    def __init__(self):
        super(mse_norm, self).__init__()
    
    def forward(self, inputs, targets):
        
        dif_input_targ = inputs - targets
        
        norm_factor = torch.amax(targets, (1, 2), keepdim=True)
        
        # norm_factor = torch.max(torch.max(targets, -1).values, -1).values
        
        loss = torch.mean(torch.div(dif_input_targ, norm_factor)**2)
        
        return loss
    
class mae_norm(nn.Module):
    
    def __init__(self):
        super(mae_norm, self).__init__()
    
    def forward(self, inputs, targets):
        
        dif_input_targ = inputs - targets
        
        norm_factor = torch.amax(targets, (1, 2), keepdim=True)
        
        # norm_factor = torch.max(torch.max(targets, -1).values, -1).values
        
        loss = torch.mean(torch.abs(torch.div(dif_input_targ, norm_factor)))
        
        return loss
        
    
        
        
        
        
    
    
    
    
    
    
    
    
    
        
        
        
    
