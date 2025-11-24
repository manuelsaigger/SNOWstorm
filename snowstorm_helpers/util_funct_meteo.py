#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:59:36 2020

@author: manuel
"""

import numpy as np

def constants():
    
    # all taken from Wallace+Hobbs (2006)
    constants = {
        'g':{'value':9.81,
               'description':'gravity acceleration',
               'unit':'m s-2'},
        
        'r_earth':{'value':6.37e3,
               'description':'Radius of the Earth',
               'unit':'km'},
        
        'sigma':{'value':5.67e-8,
               'description':'Stefan-Bolzmann Constant',
               'unit':'J s-1 m-2 K-4'},
        
        'rho_0':{'value':1.25,
               'description':'typical density of air at sea level',
               'unit':'kg m-3'},
        
        'Rd':{'value':287,
               'description':'Gas constant for dry air',
               'unit':'J K-1 kg-1'},
       
        'cp':{'value':1004,
               'description':'specific heat of air (const pressure)',
               'unit':'J kg-1 K-1'},
        
        'cv':{'value':717,
               'description':'specific heat of air (const volume)',
               'unit':'m s-2'},
        
        'gamma_d':{'value':9.8e-3,
               'description':'Dry adiabatic lapse rate (g/cp)',
               'unit':'K m-1'},
        
        'K':{'value':2.40e-2,
               'description':'Thermal conductivity at 0 °C (independent of pressure)',
               'unit':'J m-1 s-1 K-1'},
        
        'rho_water':{'value':1e3,
               'description':'Density of liquid water at 0 °C',
               'unit':'kg m-3'},
        
        'rho_ice':{'value':0.917e3,
               'description':'Density of ice at 0 °C',
               'unit':'kg m-3'},
        
        'Rv':{'value':461,
               'description':'Gas constant for water vapor',
               'unit':'J K-1 kg-1'},
        
        'epsilon':{'value':0.622,
               'description':'Molecular weight ratio of H2O to dry air',
               'unit':'-'},
        
        'cw':{'value':4218,
               'description':'specific heat of water at 0°C',
               'unit':'J kg-1 K-1'},
        
        'ci':{'value':2106,
               'description':'specific heat of ice at 0°C',
               'unit':'J kg-1 K-1'},
        
        'cp_water':{'value':1952,
               'description':'specific heat of water vapor (const pressure)',
               'unit':'J kg-1 K-1'},
        
        'cv_water':{'value':1463,
               'description':'specific heat of water vapor (const volume)',
               'unit':'J kg-1 K-1'},
        
        'Lv':{'value':2.50e6,
               'description':'Latent heat of vaporization at 0 °C',
               'unit':'J kg-1'},
        
        'Lv100':{'value':2.25e6,
               'description':'Latent heat of vaporization at 100 °C',
               'unit':'J kg-1'},
        
        'Ls':{'value':2.50e6,
               'description':'Latent heat of sublimation (H2O)',
               'unit':'J kg-1'},
        
        'Lm':{'value':3.34e5,
               'description':'Latent heat of fusion (H2O)',
               'unit':'J kg-1'},
        
        'kappa':{'value':0.4,
                 'description':'von-Karman constant',
                 'unit':''},
        
        'T0':{'value':273.15,
              'description':'0°C in K',
              'unit':'K'}
        }
    
    return constants




def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
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

def get_components_along_perp(u, v, lats_along, lons_along, direction='along'):
    
    vec_along = np.array([lons_along[1]- lons_along[0], lats_along[1] - lats_along[0]])
    
    vec_norm_along = vec_along / np.linalg.norm(vec_along)
    # print('-----')
    # print(direction)
    # print(lats_along)
    # print(lons_along)
    # print(vec_along)
    # print(vec_norm_along)
    
    vec_norm_along_3 = list(vec_norm_along)
    vec_norm_along_3.append(0)
    
    vec_norm_perp_3 = np.cross(vec_norm_along_3, [0, 0, -1])
    vec_norm_perp = np.array(vec_norm_perp_3[:2])
    
    ff_along = np.zeros(np.shape(u))
    ff_perp = np.zeros(np.shape(u))
    
    for ii in range(np.shape(u)[0]):
        for jj in range(np.shape(u)[1]):
            ff_along[ii, jj] = np.dot(np.array([u[ii, jj],v[ii, jj]]), vec_norm_along)
            ff_perp[ii, jj] = np.dot(np.array([u[ii, jj], v[ii, jj]]), vec_norm_perp)
    
   # print(ff_along)
    #print(ff_perp)
    if direction == 'along':
        ff_out = ff_along
    elif direction == 'perp':
        ff_out = ff_perp
    
    return ff_out

def calc_theta(t, p, t_in_celsius=True, p_unit='hpa'):
    # calculates potential temperature out of temperature and pressure, theata in same unit as t
    
    const = constants()
    
    if t_in_celsius:
        # t_K = t + constant´constants['T0']['value']
        t_k = t + const['T0']['value']
    # else:
    t_K = t
        
    if p_unit == 'hpa':
        p0 = 1000
    elif p_unit == 'pa':
        p0 = 100000
    else:
        raise ValueError('Invalid pressure unit!')
    
    
    theta = t_K * (p0/p)**(const['Rd']['value']/const['cp']['value'])
    
    if t_in_celsius:
        theta -= const['T0']['value']
    

    return theta
    
    
    
def calc_t(theta, p, theta_in_kelvin=True, p_unit='hpa'):
    # calcualtes temperature out of potential temperature and pressure, t in same unit as theta
    const = constants()
    
    if not theta_in_kelvin:
        th_K = theta + const['T0']['value']
    else:
        th_K = theta
        
    if p_unit == 'hpa':
        p0 = 1000
    elif p_unit == 'pa':
        p0 = 100000
    else:
        raise ValueError('Invalid pressure unit!')
    
   
    T = th_K / ((p0/p)**(const['Rd']['value']/const['cp']['value']))
    
    if not theta_in_kelvin:
        T -= const['T0']['value']
        
    return T

def calc_rh_td_from_t_p_qv(T, p, qv, T_unit='celsius', p_unit='hpa', qv_unit='g_kg'):

    const = constants()

    if T_unit == 'celsius':
        T_K = T + const['T0']['value']
    else:
        T_K = T
    
    if p_unit == 'hpa':
        es0 = 6.11
    elif p_unit == 'pa':
        es0 = 611
    else:
        raise ValueError('Invalid pressure unit!')
        
        
    if qv_unit == 'q_kg':
        qv_kkgg = qv / 1000
    else:
        qv_kkgg = qv
        
        
    
    e = qv_kkgg / (qv_kkgg + const['epsilon']['value']) * p
    
    es = es0 * np.exp(const['Lv']['value']/const['Rv']['value'] * (1/const['T0']['value'] - 1/T_K))
    
    rh = e / es * 100
    
    Td = (-(np.log(e/es0)*(const['Rv']['value']/const['Lv']['value']) - (1/const['T0']['value'])))**(-1)
    
    if T_unit == 'celsius':
        Td -= const['T0']['value']
    
    
    return rh, Td

def calc_rh_from_t_td_p(T, Td, p, T_unit='celsius', p_unit='h_pa'):
    
    const = constants()
    
    if T_unit == 'celsius':
        T_K = T + const['T0']['value']
        Td_K = Td + const['T0']['value']
    else:
        T_K = T
        Td_K = Td
    
    if p_unit == 'h_pa':
        es0 = 6.11
    elif p_unit == 'pa':
        es0 = 611
    else:
        raise ValueError('Invalid pressure unit!')
        
    
    
    es = es0 * np.exp(const['Lv']['value']/const['Rv']['value'] * (1/const['T0']['value'] - 1/T_K))
    e = es0 * np.exp(const['Lv']['value']/const['Rv']['value'] * (1/const['T0']['value'] - 1/Td_K))

    rh = e / es * 100
    
    return rh
    
    


def calc_qv(T, p, rh, T_unit='celsius', p_unit='hpa', qv_unit='g_kg'):
    
    const = constants()
    
    if T_unit == 'celsius':
        T_K = T + const['T0']['value']
    else:
        T_K = T
    
    if p_unit == 'hpa':
        es0 = 6.11
    elif p_unit == 'pa':
        es0 = 611
    
    
    es = es0 * np.exp(const['Lv']['value']/const['Rv']['value'] * (1/const['T0']['value'] - 1/T_K))
    
    e = es / (rh/100)
    
    qv = (e*const['epsilon']['value']) / (p - e)
    
    if qv_unit == 'g_kg':
        qv_out = qv * 1000
    else:
        qv_out = qv
    
    return qv_out

def latlon2dist(lat1, lon1, lat2, lon2, return_course=False):
    """calculates the distance and course between two coordinates in decimal-degree-format, 
    approximations only valid for rather short distances 
    lat/lon1: first coordinates pair (if movement: original position)
    lat/lon2: second coordinates pair (if movement: new position)"""
    const = constants()
    
    d_lat = lat2 - lat1
    d_y = 2 * const['r_earth']['value'] * np.pi * (d_lat / 360)
    
    d_lon = lon2 - lon1
    r_eff = np.cos(np.deg2rad(lat1)) * const['r_earth']['value']
    d_x = 2 * r_eff * np.pi * (d_lon / 360)
    
    (course, dist) = uv2ddff(d_x, d_y)
    
    if return_course:
        return course, dist
    else:
        return dist
    
def calc_theta_il_1(Th, p, qv, ql, qs):
    const = constants()
           
    T = calc_t(Th, p)
    
    Th_il = Th * np.exp(-((const['Lv']['value']*ql)/(const['cp']['value']*T)) - ((const['Ls']['value']*qs)/(const['cp']['value']*T)))
    
    return Th_il

def calc_theta_il_2(Th, p, qv, ql, qs):
    const = constants()
    
    T = calc_t(Th, p)
    
    cp = 1004
    cpv = 1885
    
    R = 287
    Rv = 461
    
    qt = qv + ql + qs
    # qt = ql + qs
    
    gamma = (Rv * qt) / (cp + cpv*qt)
    xi = (R + Rv*qt) / (cp + cpv*qt)
    eps = R/Rv
    
    Lv = 2.501e6
    Ls = 2.834e6    
    
    p0 = 1000
    
    Th_il = (T*(p0/p)**xi) * ((1 - ((ql + qs)/(eps+qt)))**xi) * ((1 - ((ql + qs)/qt))**(-gamma)) * np.exp((-Lv * ql - Ls*qs)/((cp+cpv*qt)*T))
    
    return Th_il

    
def calc_theta_e(Th, p, qv):
    
    const = constants()
    
    T = calc_t(Th, p)
    
    
    th_e = Th * np.exp((const['Lv']['value'] * qv) / (const['cp']['value'] * T))
    
    return th_e
    
    
def calc_theta_x(th, p, qv, ql, qs):
    
    const = constants()
    
    T = calc_t(th, p)
    
    qv0 = qv[0,:]
    ql0 = ql[0,:]
    qs0 = qs[0,:]
    
    dqv = qv - qv0
    dql = ql - ql0
    dqs = qs - ql0

    th_x = th * np.exp(-((const['Lv']['value']*dql)/(const['cp']['value']*T)) - 
                       ((const['Ls']['value']*dqs)/(const['cp']['value']*T)) + 
                       ((const['Ls']['value']*dqv)/(const['cp']['value']*T)))
    
    # qt = qv + dql + dqs
    # dqt = qt - qt[0,:]
    
    # th_x = th_il * np.exp(-(Lv*dqt)/(cp*T))
    # th_x = th * np.exp(-(Lv * dqv)/(cp*T))
    return th_x

def calc_dT_dia_adia_dry(Th, p, dt=1):
    
    p0 = 1000
    kappa = 0.286
    
    index_i = np.arange(1, np.shape(Th)[0])
    index_im = np.arange(0, np.shape(Th)[0]-1)
    
    # Th = np.fliplr(Th)
    # p = np.fliplr(p)
    
    T = calc_t(Th, p)
    
    dT_dt = np.zeros(np.shape(T))
    # dT_dt[index_i, :] = -(T[index_i, :] - T[index_im, :])
    dT_dt[index_i, :] = T[index_i, :] - T[index_im, :]
    
    
    dT_dia = np.zeros(np.shape(T))
    # dT_dia[index_i, :] = -((Th[index_i, :] - Th[index_im, :]) / dt) * ((2 * p0) / (p[index_i, :] + p[index_im, :]))**(-kappa)
    dT_dia[index_i, :] = -((Th[index_i, :] - Th[index_im, :]) / dt * ((2 * p0) / (p[index_i, :] + p[index_im, :]))**(-kappa))
    
    
    # dT_adia_res = -(dT_dt - dT_dia)
    dT_adia_res = dT_dt - dT_dia
    
    
    dT_adia = np.zeros(np.shape(T))
    # dT_adia[index_i, :] = -kappa * ((T[index_i, :] + T[index_im, :]) / (p[index_i, :] + p[index_im, :])) * ((p[index_i, :] - p[index_im, :]) / dt)
    dT_adia[index_i, :] = -(kappa * ((T[index_i, :] + T[index_im, :]) / (p[index_i, :] + p[index_im, :])) * ((p[index_i, :] - p[index_im, :]) / dt))


    return dT_dt, dT_dia, dT_adia, dT_adia_res 

def calc_dT_cum_dia_adia_dry(Th, p, dt=1):
    
    dT_dt, dT_dia, dT_adia, dT_adia_res = calc_dT_dia_adia_dry(Th, p, dt=dt)
    
    # dT_cum = -np.cumsum(dT_dt, axis=0)
    # dT_dia_cum = -np.cumsum(dT_dia, axis=0)
    
    # dT_adia_res_cum = -np.cumsum(dT_adia_res)
    
    # dT_adia_cum = -np.cumsum(dT_adia, axis=0)

    dT_cum = np.cumsum(dT_dt, axis=0)
    dT_dia_cum = np.cumsum(dT_dia, axis=0)
    
    dT_adia_res_cum = np.cumsum(dT_adia_res)
    
    dT_adia_cum = np.cumsum(dT_adia, axis=0)
    
    return dT_cum, dT_dia_cum, dT_adia_cum, dT_adia_res_cum     
       
def calc_dT_dia_adia_virt(Th, p, qv, dt=1):
    
    p0 = 1000
    kappa = 0.286
    
    index_i = np.arange(1, np.shape(Th)[0])
    index_im = np.arange(0, np.shape(Th)[0]-1)
    
    T = calc_t(Th, p)
    Tv = calc_Tv(T, p, qv)
    
    Th_v = calc_theta(T, p, t_in_celsius=False)
    
    
    dTv_dt = np.zeros(np.shape(Tv))
    dTv_dt[index_i, :] = -(Tv[index_i, :] - Tv[index_im, :])
    
    
    dTv_dia = np.zeros(np.shape(Tv))
    dTv_dia[index_i, :] = -((Th_v[index_i, :] - Th_v[index_im, :]) / dt) * ((2 * p0) / (p[index_i, :] + p[index_im, :]))**(-kappa)
    
    
    dTv_adia_res = -(dTv_dt - dTv_dia)
    
    
    dTv_adia = np.zeros(np.shape(Tv))
    dTv_adia[index_i, :] = -kappa * ((Tv[index_i, :] + Tv[index_im, :]) / (p[index_i, :] + p[index_im, :])) * ((p[index_i, :] - p[index_im, :]) / dt)


    return dTv_dt, dTv_dia, dTv_adia, dTv_adia_res 

def calc_dT_cum_dia_adia_virt(Th, p, qv, dt=1):
    
    dTv_dt, dTv_dia, dTv_adia, dTv_adia_res = calc_dT_dia_adia_virt(Th, p, qv, dt=dt)
    
    dTv_cum = -np.cumsum(dTv_dt, axis=0)
    dTv_dia_cum = -np.cumsum(dTv_dia, axis=0)
    
    dTv_adia_res_cum = -np.cumsum(dTv_adia_res)
    
    dTv_adia_cum = -np.cumsum(dTv_adia, axis=0)

    return dTv_cum, dTv_dia_cum, dTv_adia_cum, dTv_adia_res_cum 
    
    
    
    
    
def calc_Tv(T, p, qv):
    
    const = constants()
    
    e = (qv / (qv + const['epsilon']['value'])) * p
    
    # Tv = T * (qv + epsilon) / (1 - (e/p) * (1-(Rd/Rv)))
    Tv = T / (1 - e/p * (1-const['epsilon']['value']))

    return Tv    
    
def calc_nondim_mnth(hm, z, Th, u):
    
    dz = hm - z
    # dz = hm
    
    g = 9.81
    
    N2 = np.nan * np.zeros(np.shape(Th))
    
    index_i = np.arange(1, len(N2)-1)
    index_im = np.arange(0, len(N2)-2)
    index_ip = np.arange(2, len(N2))
    
    N2[index_i] = g / Th[index_i] * ((Th[index_ip] - Th[index_im]) / (z[index_ip] - z[index_im]))
    N = np.sqrt(N2)
    
    epsilon = dz*N/u
    
    return epsilon
    
        
