# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:10:20 2022

@author: seongjoon kang
"""
import numpy as np

def get_los_prob(scenario:str, elev_angle:list()):
    # tabel 6.6.1-1 of 3gpp 38.811
    LOS_prob_table = {'dense_urban': [28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82, 98.1], 
                          'urban': [24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2]
                          , 'rural':[78.2, 86.9, 91.9, 92.9, 93.5, 94, 94.9, 95.2, 99.8]}
    elev_indices = np.array(elev_angle/10-1, dtype = int)
    return np.array(LOS_prob_table[scenario])[elev_indices]

def get_delay_scaling_factor(scenario:str, link_state:list(), elev_angle:list()=None):
    # return delay scaling factors corresponding to link states
    r_tau = np.zeros(len(link_state))
    if scenario == 'rural':
        r_tau[link_state==1] = 3.8
        r_tau[link_state==2] = 1.7
    return r_tau

def get_XPR_params(scenario:str, link_state:list(),elev_angle_:list()=None):
    mu_XPR = np.zeros(len(link_state))
    sigma_XPR = np.zeros(len(link_state))
    if scenario == 'rural':
        mu_XPR[link_state==1] =12
        mu_XPR[link_state==2] =7
        
        sigma_XPR[link_state==1] = 4
        sigma_XPR[link_state==2] = 3
    return mu_XPR, sigma_XPR

def get_per_cluster_shadowing_std(scenrio:str,  link_state:list(), elev_angle:list()=None):
    cluster_std = 3
    return np.ones_like(link_state)*cluster_std

def get_n_cluser_ray (scenario:str,link_state:list(), elev_angle:list()):
    #return number of clusters corresponding to link sates
    n_clusters, n_rays = np.zeros(len(link_state)), np.zeros(len(link_state))
    if scenario == 'rural':
        n_clusters[link_state == 1] =2
        n_clusters[link_state == 2] =3
        n_rays = np.repeat([20], len(link_state))
        
    return np.array(n_clusters,dtype = int), np.array(n_rays, dtype = int)
    
def get_scaling_factors_for_phi_theta():
    # the following scailing factors are related to the total number of clusters from 2
    ########## 2      3      4      5       6     7       8     9      10
    C_phi = [ 0.501, 0.680, 0.779, 0.860, 1.018, 1.090, 1.123, 1.146, 1.190, 1.211, 1.226, 1.273, 1.289]
    C_theta = [0.430, 0.594, 0.697, 0.889, 0.957, 1.031, 1.104, 1.1088, 1.184, 1.178]
    return C_phi, C_theta

def get_cluster_spread(scenario:str, link_state:int, elev_angle:int):
    elv_idx = elev_angle/10 -1
    if scenario == 'rural':
        if link_state == 1: #if LOS
            table = np.array([
                [np.nan,	np.nan,	np.nan,	np.nan,	np.nan,	np.nan,	np.nan,	np.nan,	np.nan],
                [0.39,	0.31,	0.29,	0.37,	0.61,	0.9,	1.43,	2.87,	5.48],
                [10.81,	8.09,	13.7,	20.05,	24.51,	26.35,	31.84,	36.62,	36.77],
                [1.94,	1.83,	2.28,	2.93,	2.84,	3.17,	3.88,	4.17,	4.29]
                ])
        else: # NLOS CASE
            table = np.array([
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan,np.nan],
                [0.03,	0.05,	0.07,	0.1,	0.15,	0.22,	0.5,	1.04,	2.11],
                [18.16,	26.82,	21.99,	22.86,	25.93,	27.79,	28.5,	37.53,	29.23],
                [2.32,	7.34,	8.28,	8.76,	9.68,	9.94,	8.9,	13.74,	12.16],
                ])
            
    values = table[:,elv_idx]
    return {'C_DS':values[0], 'C_ASD':values[1], 'C_ASA':values[2],'C_ZSA':values[3]}

def get_shadowing_params(scenario:str, elev_angle:list()):
    elev_angle = np.array(elev_angle)
    # table 6.6.2-1,2,3 of 3gpp 38.811
    ######################### sigma_los, sigma_nlos, CL,sigma_los, sigma_nlos, CL ###### 
    shadowing_fading_CL_dese_urban =np.array( [[3.5, 15.5, 34.3], 
                                            [3.4, 13.9, 30.9],
                                            [2.9, 12.4, 29.0],
                                            [3.0, 11.7, 27.7],
                                            [3.1, 10.6, 26.8],
                                            [2.7, 10.5, 26.2],
                                            [2.5, 10.1, 25.8],
                                            [2.3, 9.2, 25.5],
                                            [1.2, 9.2, 25.5]])
    shadowing_fading_CL_urban = np.copy(shadowing_fading_CL_dese_urban)
    shadowing_fading_CL_urban[:,0] = 4
    shadowing_fading_CL_urban[:,1] = 6
    
    shadowing_fading_CL_rural = np.array([ [1.79, 8.93, 19.52],
                                        [1.14, 9.08, 18.17],
                                        [1.14, 8.78, 18.42],
                                        [0.92, 10.25, 18.28],
                                        [1.42, 10.56, 18.63],
                                        [1.56, 10.74, 17.68],
                                        [0.85, 10.17, 16.50],
                                        [0.72, 11.52, 16.30],
                                        [0.72, 11.52, 16.30]])
    
    shadowing_fading_CL_table = {'rural': shadowing_fading_CL_rural, 
                                 'urban':shadowing_fading_CL_urban,
                                 'dense_urban':shadowing_fading_CL_dese_urban}
    el_idx = np.array(elev_angle/10-1, dtype = int)
    return shadowing_fading_CL_table[scenario][el_idx]

def get_correlation_spread(scenario:str, link_state:list(), elev_angle:list()):
    link_state = np.array(link_state)
    elev_angle = np.array(elev_angle)
    ## 3gpp 38.811, table 6.7.2 - 7a
    #### 10  20  30   40  50 60 70 80 90
    if scenario == 'rural':
        correlation_matrix_rural_LOS = np.array([
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [-0.5,  -0.5,  -0.5,  -0.5,  -0.5, -0.5, -0.5, -0.5, -0.5],
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [0.01, 0.01, 0.01, 0.01, 0.01,0.01,0.01,0.01,0.01],
            [-0.17, -0.17, -0.17, -0.17, -0.17,-0.17, -0.17,-0.17,-0.17],
            [0,  0,  0,  0,  0, 0, 0, 0, 0],
            [-0.02, -0.02, -0.02, -0.02, -0.02,-0.02, -0.02,-0.02,-0.02],
            [-0.05, -0.05, -0.05, -0.05, -0.05,-0.05, -0.05,-0.05,-0.05],
            [0.27, 0.27, 0.27, 0.27, 0.27,0.27, 0.27,0.27,0.27],
            [0.73, 0.73, 0.73, 0.73, 0.73,0.73,0.73,0.73, 0.73],
            [-0.14, -0.14,-0.14, -0.14,-0.14, -0.14,-0.14, -0.14,-0.14],
            [-0.20, -0.20,-0.20, -0.20,-0.20, -0.20,-0.20,-0.20, -0.20],
            [0.24,  0.24,  0.24,  0.24, 0.24,  0.24, 0.24, 0.24, 0.24],
            [-0.07, -0.07, -0.07, -0.07, -0.07,-0.07, -0.07,-0.07,-0.07]])
        
        spread_value_LOS = np.array([
            [-9.55,	-8.68,	-8.46,	-8.36,	-8.29,	-8.26,	-8.22,	-8.2,	-8.19],
            [0.66,	0.44,	0.28,	0.19,	0.14,	0.1,	0.1,	0.05,	0.06],
            [-3.42,	-3,	   -2.86,	-2.78,	-2.7,	-2.66,	-2.53,	-2.21,	-1.78],
            [0.89,	0.63,	0.52,	0.45,	0.42,	0.41,	0.42,	0.5,	0.91],
            [-9.45,	-4.45,	-2.39,	-1.28,	-0.99,	-1.05,	-0.9,	-0.89,	-0.81],
            [7.83,	6.86,	5.14,	3.44,	2.59,	2.42,	1.78,	1.65,	1.26],
            [-4.2,	-2.31,	-0.28,	-0.38,	-0.38,	-0.46,	-0.49,	-0.53,	-0.46],
            [6.3,	5.04,	0.81,	1.16,	0.82,	0.67,	1,	1.18,	0.91,],
            [-6.03,	-4.31,	-2.57,	-2.59,	-2.59,	-2.65,	-2.69,	-2.65,	-2.65,],
            [5.19,	4.18,	0.61,	0.79,	0.65,	0.52,	0.78,	1.01,	0.71,],
            [24.72,	12.31,	8.05,	6.21,	5.04,	4.42,	3.92,	3.65,	3.59],
            [5.07,	5.75,	5.46,	5.23,	3.95,	3.75,	2.56,	1.77,	1.77]
            ])
        
        correlation_matrix_rural_NLOS = np.array([
            [0.32,	0.19,	0.23,	0.25,	0.15,	0.08,	0.13,	0.15,	0.64],
            [0.3,	0.32,	0.32,	0.4,	0.45,	0.39,	0.51,	0.27,	0.05],
            [0.02,	0,	0,	0.01,	0.02,	0.02,	0.04,	0.01,	0.06],
            [0.45,	0.52,	0.54,	0.53,	0.55,	0.56,	0.56,	0.58,	0.47],
            [-0.36,	-0.39,	-0.41,	-0.37,	-0.4,	-0.41,	-0.4,	-0.46,	-0.3],
            [0.45,	0.12,	0.07,	0.22,	0.16,	0.14,	0.2,	-0.04,	-0.11],
            [np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan],
            [np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan],
            [np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan],
            [np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan],
            [-0.06,	-0.04,	-0.04,	-0.05,	-0.06,	-0.07,	-0.11,	-0.05,	-0.1],
            [-0.07,	-0.17,	-0.19,	-0.17,	-0.19,	-0.2,	-0.19,	-0.23,	-0.13],
            [np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan],
            [np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan],
            [0.58,	0.67,	0.65,	0.73,	0.79,	0.81,	0.79,	0.7,	0.42],
            [0.06,	0.03,	0,	    -0.09,	-0.2,	-0.22,	-0.32,	-0.41,	-0.35],
            [0.6,	0.41,	0.37,	0.32,	0.19,	0.16,	0.2,	0.15,	0.28],
            [0.21,	-0.02,	-0.09,	-0.1,	-0.12,	-0.11,	-0.1,	-0.14,	-0.25],
            [0.33,	0.35,	0.31,	0.37,	0.46,	0.44,	0.49,	0.27,	0.07],
            [0.1,	0.21,	0.22,	0.07,	-0.04,	-0.12,	-0.29,	-0.26,	-0.36],
            [0.01,	-0.02,	-0.12,	-0.21,	-0.27,	-0.27,	-0.38,	-0.35,	-0.36]])
        
        spread_value_NLOS = np.array([
            [-9.01,	-8.37,	-8.05,	-7.92,	-7.92,	-7.96,	-7.91,	-7.79,	-7.74],
            [1.59,	0.95,	0.92,	0.92,	0.87,	0.87,	0.82,	0.86,	0.81],
            [-2.9,	-2.5,	-2.12,	-1.99,	-1.9,	-1.85,	-1.69,	-1.46,	-1.32],
            [1.34,	1.18,	1.08,	1.06,	1.05,	1.06,	1.14,	1.16,	1.3],
            [-3.33,	-0.74,	0.08,	0.32,	0.53,	0.33,	0.55,	0.45,	0.4],
            [6.22,	4.22,	3.02,	2.45,	1.63,	2.08,	1.58,	2.01,	2.19],
            [-0.88,	-0.07,	0.75,	0.72,	0.95,	0.97,	1.1,	0.97,	1.35],
            [3.26,	3.29,	1.92,	1.92,	1.45,	1.62,	1.43,	1.88,	0.62],
            [-4.92,	-4.06,	-2.33,	-2.24,	-2.24,	-2.22,	-2.19,	-2.41,	-2.45],
            [3.96,	4.07,	1.7,	2.01,	2,	   1.82,	1.66,	2.58,	2.52],
            [np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan],
            [np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan]
            ])
        
        el_idx = np.array(elev_angle/10-1, dtype = int)
        el_idx_los = el_idx[link_state==1]
        el_idx_nlos = el_idx[link_state==2]
        los_idx = (link_state ==1)
        nlos_idx = (link_state ==2)
        correlation_matrix = np.zeros(shape=[21, len(link_state)])
        correlation_matrix[:,los_idx] = correlation_matrix_rural_LOS[:,el_idx_los]
        correlation_matrix[:,nlos_idx] = correlation_matrix_rural_NLOS[:,el_idx_nlos]
        
        spread_matrix = np.zeros(shape=[12, len(link_state)])
        spread_matrix[:,los_idx] = spread_value_LOS[:,el_idx_los]
        spread_matrix[:,nlos_idx] = spread_value_NLOS[:,el_idx_nlos]
        
    return correlation_matrix, spread_matrix

