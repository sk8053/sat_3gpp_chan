# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:24:55 2022

@author: seongjoon kang
"""

import numpy as np
from get_antenna_field_patterns import get_ue_antenna_field_pattern, get_sate_antenna_field_pattern
from three_gpp_tables_s_band import get_cluster_spread

def step10_step11(angle_data:list(), P_n_list_without_K:list(), tau_n_list:list(), C_DS_list:list(),
                  link_state:list(), K_list:list(), LOS_angle_list:list(),  sat_ant_radius:float, f_c:float,
                  slant_angle:float = 45):
    # step 10
    # draw random initial phase {theta_theta, theta_phi, phi_theta, phi_phi} ={TT, TP, PT,PP}
    # shape should be (n_ray, n_cluster)
    
    # step 11, For N-2 weakest clusters, channel coeffient of one cluster is 
    # sum of those of all the rays according to 38.901, 7.5.22
    # For first two strongest clusters, 
    # rays are spread in delay to three sub-clusters
    
    tau_n_list_new = [] # list containing all the cluster delays for each link
    H_list = [] # list containing all channel coefficient per each cluster
    for i in range (len(angle_data)):
        
        rot_angle = {'alpha':20,'beta':50,'gamma':60}
        
        
        n_rays, n_clusters = angle_data[i]['AOA_cluster_rays'].shape
        theta_ZOA, phi_AOA = angle_data[i]['ZOA_cluster_rays'], angle_data[i]['AOA_cluster_rays']
        theta_ZOD, phi_AOD = angle_data[i]['ZOD_cluster_rays'], angle_data[i]['AOD_cluster_rays']
        K = angle_data[i]['XPR']
        
        # step 10
        TT = np.random.uniform(low=-np.pi, high = np.pi, size=(n_rays, n_clusters))
        TP = np.random.uniform(low=-np.pi, high = np.pi, size=(n_rays, n_clusters))
        PT = np.random.uniform(low=-np.pi, high = np.pi, size=(n_rays, n_clusters))
        PP = np.random.uniform(low=-np.pi, high = np.pi, size=(n_rays, n_clusters))
        # step 11
        H_nlos = np.zeros(shape=[n_rays, n_clusters])
        
        tau_n_cluster = tau_n_list[i]
        
        # cluster delay spread of link i
        c_ds = C_DS_list[i]

        tau_n_cluster_new = []
        H_n_cluster = []
        for n in range(n_clusters):
            # return-shape is (2,n_rays)
            F_vec_ue_list = get_ue_antenna_field_pattern(theta_ZOA[:,n], phi_AOA[:,n], rot_angle, slant_angle)
            F_vec_sat_list = get_sate_antenna_field_pattern(theta_ZOD[:,n], phi_AOD[:,n], sat_ant_radius, f_c)
            P_n = P_n_list_without_K[i][n]
            
            for k in range(n_rays):
                F_ue= F_vec_ue_list[:,k]
                F_sat = F_vec_sat_list[:,k]
                coupling_M = np.array([[np.exp(1j*TT[k,n]),np.sqrt(1/K[k,n])*np.exp(1j*TP[k,n])],
                                      [np.sqrt(1/K[k,n])*np.exp(1j*PT[k,n]), np.exp(1j*PP[k,n])]])
                
             
                H_nlos[k,n] = np.sqrt(P_n/n_rays)*F_ue.T.dot(coupling_M).dot(F_sat)
             
            
            if n == 0 or n==1:
                # For first two strongest clusters, 
                # rays are spread in delay to three sub-clusters
                tau_n_cluster_new.append(tau_n_cluster[n])
                H_n_cluster.append(10/20*np.sum(H_nlos[[0,1,2,3,4,5,6,7,18,19], n]))
                
                tau_n_cluster_new.append(tau_n_cluster[n] + 1.28*c_ds*1e-9)
                H_n_cluster.append(6/20*np.sum(H_nlos[[8,9,10,11,16,17], n]))
                
                tau_n_cluster_new.append(tau_n_cluster[n] + 2.56*c_ds*1e-9)
                H_n_cluster.append(4/20*np.sum(H_nlos[[12, 13,14,15], n]))
            else:
                #For N-2 weakest clusters, channel coeffient of one cluster is 
                # sum of those of all the rays according to 38.901, 7.5.22
                tau_n_cluster_new.append(tau_n_cluster[n])
                H_n_cluster.append(np.sum(H_nlos[:,n]))
                
        if link_state[i] ==1:
            LOS_angle = LOS_angle_list[i]
            LOS_ZOA = LOS_angle['ZOA']
            LOS_AOA = LOS_angle['AOA']
            
            LOS_ZOD = LOS_angle['ZOD']
            LOS_AOD = LOS_angle['AOD']
            
            F_ue = get_ue_antenna_field_pattern([LOS_ZOA], [LOS_AOA], rot_angle, slant_angle)
            F_sat = get_sate_antenna_field_pattern([LOS_ZOD], [LOS_AOD], sat_ant_radius, f_c)
            
            coupling_M = np.array([[1, 0],
                                  [0, -1]])
            H_los = F_ue.dot(coupling_M).dot(F_sat)
            K = K_list[i]
            H_nlos[0] = np.sqrt(K/(K+1))*H_los + np.sqrt(1/(K+1))*H_nlos[0]
            
        tau_n_list_new.append(tau_n_cluster)
        H_list.append(H_nlos)
        
    return H_list, tau_n_list
                
