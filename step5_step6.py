# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 18:23:46 2022

@author: seongjoon kang
"""

import numpy as np
from three_gpp_tables_s_band import get_delay_scaling_factor
from three_gpp_tables_s_band import get_n_cluser_ray
from three_gpp_tables_s_band import get_per_cluster_shadowing_std

def step5_and_step6(scenario:str, DS:list(), K:list(),  elev_angle:list(), link_state:list()):    
    # return multi-tap delay corresponding to each cluster
    
    link_state = np.array(link_state)
    elev_angle = np.array(elev_angle)
    #delays are drawn randomly defined in Table 7.5-6
    
    # r_tau values are the same across all the elevation angles
    r_tau = get_delay_scaling_factor(scenario,link_state, elev_angle)
    n_cluster, n_ray = get_n_cluser_ray(scenario,link_state, elev_angle)
    
    cluster_shadowing_std= get_per_cluster_shadowing_std(scenario, link_state, elev_angle)
    # step 5
    tau_n_list = []
    for i,link_state_i in enumerate(link_state):
        tau_n = np.zeros(n_cluster[i])
        X_n = np.random.uniform(low=0,high=1, size = n_cluster[i]);
        tau_n = -1*r_tau[i]*DS[i]*np.log(X_n)
        tau_n = np.sort(tau_n - min(tau_n)) # 3gpp, 38.901, 7.5-2
        #print(tau_n)
        if link_state_i == 1: # if LOS
            # in case of LOS condition, additonal scailing of delays is requried to compensate
            # for the effect of LOS peak addition to the delay spread
            K_dB_i = 10*np.log10(K[i])
            C_tau = 0.7705 - 0.0433*K_dB_i + 0.0002*K_dB_i**2 + 0.000017*K_dB_i**3 # 7.5-3
            tau_n = tau_n/C_tau
            
        tau_n_list.append(tau_n)
    # step 6, Generate cluster powers
    # cluster powers are cacluated assuming a single slop exponential delay profile
    P_n_list = []
    for i,link_state_i in enumerate(link_state):    
        Z_n = np.random.normal(0, 3)   
        P_n_prime = np.exp(-tau_n_list[i]*(r_tau[i]-1)/(r_tau[i]*DS[i]))*10**(-Z_n/10)    
        #normalize the cluster power so that the sum of all cluster power is equal to one
        P_n = P_n_prime/np.sum(P_n_prime)
        if link_state_i ==1: #if LOS
            P_n[0] += K[i]/(K[i]+1)
            P_n[1:] *= 1/(K[i]+1)
        P_n_list.append(P_n)  
    
    # remove cluster with less than -25 dB power compared to the maximum cluster power
    #################### 
    return tau_n_list, P_n_list
