# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 23:47:54 2022

@author: seongjoon kang
"""
import numpy as np
from three_gpp_tables_s_band import get_scaling_factors_for_phi_theta
from three_gpp_tables_s_band import get_cluster_spread
from three_gpp_tables_s_band import get_correlation_spread
from three_gpp_tables_s_band import get_XPR_params

def step7_step8_step9(scenario, ASA_list:list(), ASD_list:list(), ZSA_list:list(), ZSD_list:list(), 
          K_list:list(), elev_angle_list:list(),
          P_n_list:list(), link_state:list(), LOS_angles:dict()):
    
    # ray offset angle given in 3GPP 38.901, table 7.5-3
    ray_offset_angles =[0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715,
                        0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 
                        1.5195, -1.5195, 2.1551, -2.1551]
    
    # Create dictionary to save angles for all clusters and rays per each link
    angle_data = [{'AOA_cluster_rays':None, 'AOD_cluster_rays':None, 
                   'ZOA_cluster_rays':None, 'ZOD_cluster_rays':None,
                   'XPR':None} for i in range(len(link_state))]
    # Generate arrival angeles and departure angles for both azimuth and elevation
    C_phi, C_theta = get_scaling_factors_for_phi_theta()
    
    mu_XPR, sigma_XPR = get_XPR_params(scenario, link_state, elev_angle_list)
 
    
    # for all links that have several clusters
    for i, link_state_i in enumerate(link_state):
        n_cluster = len(P_n_list[i])
        # get power distribution across all clusters
        P_clusters = P_n_list[i] # shape = (n_cluster, )
        
        ############## generate azimuth angle for all clusters and rays ##########################
        # find the scaling factor corresponding to total number of clusters
        C_phi_nlos = C_phi[n_cluster] 
        
        if link_state_i ==1:
            C_phi = C_phi_nlos * (1.1035 - 0.028*K_list[i] - 0.002*K_list[i]**2 + 0.0001*K_list[i]**3)
        else:
            C_phi = C_phi_nlos
            
        phi_clusters_AOA_prime = 2*(ASA_list[i]/1.4)*np.sqrt(-np.log(P_clusters/max(P_clusters)))/C_phi
        phi_clusters_AOD_prime = 2*(ASD_list[i]/1.4)*np.sqrt(-np.log(P_clusters/max(P_clusters)))/C_phi
        
        # Assign positive or negative sign to the angles by multiplying with 
        # a random variable X_n ~unif(-1,1)
        # add component Y_n ~ N(0, (ASA/7)^2) to introduce random variation
        
        # 1) AOA
        X_clusters = np.random.uniform(-1,1,n_cluster)
        Y_clusters = np.random.normal(0, ASA_list[i]/7, n_cluster)
        LOS_AOA_phi = LOS_angles[i]['AOA']
        # get arrival azimuth angle of all clusters, shape = (n_cluster, 1)
        phi_AOA_clusters = X_clusters * phi_clusters_AOA_prime + Y_clusters + LOS_AOA_phi
        
        if link_state[i] ==1:
            phi_AOA_clusters = X_clusters * phi_clusters_AOA_prime + Y_clusters - \
                (X_clusters[0]*phi_clusters_AOA_prime[0] + Y_clusters[0]-LOS_AOA_phi)
        # 2) AOD
        X_clusters = np.random.uniform(-1,1,n_cluster)
        Y_clusters = np.random.normal(0, ASD_list[i]/7, n_cluster)
        LOS_AOD_phi = LOS_angles[i]['AOD']
        # get arrival azimuth angle of all clusters, shape = (n_cluster, 1)
        phi_AOD_clusters = X_clusters * phi_clusters_AOD_prime + Y_clusters + LOS_AOD_phi
        
        if link_state[i] ==1:
            phi_AOD_clusters = X_clusters * phi_clusters_AOD_prime + Y_clusters - \
                (X_clusters[0]*phi_clusters_AOD_prime[0] + Y_clusters[0]-LOS_AOD_phi)
        
        
        # finally add offset angles from 3gpp 38.901 table 7.5-3 to cluster angles
        # the cluster-spread values fro 3gpp 38.811 tables
        cluster_spreads = get_cluster_spread(scenario, link_state[i], elev_angle_list[i])
        c_ASA = cluster_spreads['C_ASA']
        c_ASD = cluster_spreads['C_ASD']
        #phi_AOA_cluster_rays shape = (n_ray, n_cluster)
        phi_AOA_cluster_rays = phi_AOA_clusters + c_ASA*ray_offset_angles[:,None] # (n_ray, n_cluster)
        phi_AOD_cluster_rays = phi_AOD_clusters + c_ASD*ray_offset_angles[:,None] #(n_ray, n_cluster)
        
        phi_AOA_cluster_rays[phi_AOA_cluster_rays>360] -= 360
        phi_AOA_cluster_rays[phi_AOA_cluster_rays<0] += 360
        
        phi_AOD_cluster_rays[phi_AOD_cluster_rays>360] -= 360
        phi_AOD_cluster_rays[phi_AOD_cluster_rays<0] += 360
        
        ############## generate zenith angle for all clusters and rays ##########################
        # zenith angles are determined by applying the inverse Laplacian function
        # with input parameter P_n and RMS angle spread ZSA
        # find the scaling factor corresponding to total number of clusters
        C_theta_nlos = C_theta[n_cluster] 
        if link_state_i ==1:
            C_theta = C_theta_nlos * (1.3086 - 0.0339*K_list[i] - 0.0077*K_list[i]**2 + 0.0002*K_list[i]**3)
        else:
            C_theta = C_theta_nlos
            
        theta_clusters_ZOA_prime = - ZSA_list[i]*np.log(P_clusters/max(P_clusters))/C_theta
        theta_clusters_ZOD_prime = - ZSD_list[i]*np.log(P_clusters/max(P_clusters))/C_theta
        
        # 1) ZOA
        X_clusters = np.random.uniform(-1,1,n_cluster)
        Y_clusters = np.random.normal(0, ZSA_list[i]/7, n_cluster)
        LOS_ZOA_theta = LOS_angles[i]['ZOA']
        
        # get arrival zenith angle of all clusters, shape = (n_cluster, 1)
        theta_ZOA_clusters = X_clusters * theta_clusters_ZOA_prime + Y_clusters + LOS_ZOA_theta
        
        if link_state[i] == 1: # LOS
            theta_ZOA_clusters = X_clusters * theta_clusters_ZOA_prime + Y_clusters - \
                (X_clusters[0]*theta_clusters_ZOA_prime[0] + Y_clusters[0]-LOS_ZOA_theta)
        
        # 2) ZOD
        X_clusters = np.random.uniform(-1,1,n_cluster)
        Y_clusters = np.random.normal(0, ZSD_list[i]/7, n_cluster)
        LOS_ZOD_theta = LOS_angles[i]['ZOD']
        
        # get arrival zenith angle of all clusters, shape = (n_cluster, 1)
        mu_offset_ZOD = 0;
        theta_ZOD_clusters = X_clusters * theta_clusters_ZOD_prime + Y_clusters \
        + LOS_ZOD_theta + mu_offset_ZOD
        
        if link_state[i] == 1: # LOS
            theta_ZOD_clusters = X_clusters * theta_clusters_ZOD_prime + Y_clusters - \
                (X_clusters[0]*theta_clusters_ZOD_prime[0] + Y_clusters[0]-LOS_ZOD_theta)
        
        
        c_ZSA = cluster_spreads['C_ZSA']
        # we don't need the departure zenith angle spread per cluster
        #c_ZSD = cluster_spreads['C_ZSD'] 
        
        #theta_ZOA_cluster_rays shape = (n_ray, n_cluster)
        theta_ZOA_cluster_rays = theta_ZOA_clusters + c_ZSA*ray_offset_angles[:,None] # (n_ray, n_cluster)
         
        
        _, spread_values= get_correlation_spread(scenario, link_state[i], elev_angle_list[i])
        mu_lg_ZSD = spread_values[-4]
        # 38.901, 7.5-20
        theta_ZOD_cluster_rays = theta_ZOD_clusters + (3/8)*(10**(mu_lg_ZSD))*ray_offset_angles[:,None] #(n_ray, n_cluster)
        
        theta_ZOA_cluster_rays[theta_ZOA_cluster_rays>360] -=360
        theta_ZOA_cluster_rays[theta_ZOA_cluster_rays<0] +=360
        theta_ZOD_cluster_rays[theta_ZOD_cluster_rays>360] -=360
        theta_ZOD_cluster_rays[theta_ZOD_cluster_rays<0] +=360
        
        theta_ZOA_cluster_rays[theta_ZOA_cluster_rays>180] = 360 - theta_ZOA_cluster_rays[theta_ZOA_cluster_rays>180]
        theta_ZOD_cluster_rays[theta_ZOD_cluster_rays>180] = 360 - theta_ZOD_cluster_rays[theta_ZOD_cluster_rays>180] 
        
        # step 8 
        # couple randomly AOD, AOA, ZOD, and ZOA within a cluster
        phi_AOA_cluster_rays = np.random.Generator.shuffle(phi_AOA_cluster_rays, axis = 0)
        phi_AOD_cluster_rays = np.random.Generator.shuffle(phi_AOD_cluster_rays, axis =0)
        theta_ZOA_cluster_rays = np.random.Generator.shuffle(theta_ZOA_cluster_rays, axis = 0)
        theta_ZOD_cluster_rays = np.random.Generator.shuffle(theta_ZOD_cluster_rays, axis = 0)
        
        # save all angles, AOA, AOD, ZOA, and ZOD (shape=[n_rays, n_cluster]) 
        # corresponding to each link
        angle_data[i]['AOA_cluster_rays'] = phi_AOA_cluster_rays
        angle_data[i]['AOD_cluster_rays'] = phi_AOD_cluster_rays
        angle_data[i]['ZOA_cluster_rays'] = theta_ZOA_cluster_rays
        angle_data[i]['ZOD_cluster_rays'] = theta_ZOD_cluster_rays
        
        # step 9 
        # generate the cross polarization ratios (XPR) k for each ray m for each cluster n
        n_ray_i, n_cluster_i = theta_ZOA_cluster_rays.shape
        mu_XPR_i = mu_XPR[i]
        sigma_XPR_i = sigma_XPR[i]
        
        X = np.random.normal(mu_XPR_i, sigma_XPR_i, (n_ray_i, n_cluster_i))
        K = 10**(0.1*X)
        angle_data[i]['XPR'] = K
        
    return angle_data
        
        
        
        
        
        
        
        
        