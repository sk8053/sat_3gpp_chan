# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 22:31:02 2022

@author: seongjoon kang
"""
import numpy as np

def compute_LOS_angle(Tx_loc:list(), Rx_loc:list()):
    dist_vec = Rx_loc - Tx_loc
  
    R_3d = np.sqrt(dist_vec[:,0]**2 + dist_vec[:,1]**2 + dist_vec[:,2]**2)
    #R_2d = np.sqrt(dist_vec[:,0]**2 + dist_vec[:,1]**2)
    # azimuth angle of departure
    LOS_AOD_phi = np.arctan2(dist_vec[:,1], dist_vec[:,0])*180/np.pi
    LOS_AOD_phi[LOS_AOD_phi<0] += 360
    LOS_AOD_phi[LOS_AOD_phi>360] -=360
    # zenith angle of departure
    LOS_ZOD_theta = np.arccos(dist_vec[:,2]/R_3d)*180/np.pi
    # azimuth angle of arrival
    LOS_AOA_phi = LOS_AOD_phi
    LOS_AOA_phi[LOS_AOA_phi<0] += 360
    LOS_AOA_phi[LOS_AOA_phi>360] -=360
    # zenith angle of arrival
    LOS_ZOA_theta = 180-LOS_ZOD_theta
    
    return LOS_AOD_phi, LOS_ZOD_theta, LOS_AOA_phi, LOS_ZOA_theta


def step1():
    # choose one of the scenarios 
    scenario = 'rural'
    
    # b) Give number of sat and UE
    n_sat = 1
    n_UE = 10
    
    sat_location = np.array([20,30, 6000])
    UE_location_xy = np.random.uniform(low = -200, high = 200, size = (n_UE,2))
    UE_z = np.repeat([0], n_UE)
    UE_location = np.column_stack((UE_location_xy, UE_z))
    
    LOS_AOD, LOS_ZOD, LOS_AOA, LOS_ZOA = compute_LOS_angle(sat_location, UE_location)
    
    LOS_angles = [{'AOD':LOS_AOD[i],'AOA':LOS_AOA[i],'ZOD':LOS_ZOD[i], 'ZOA':LOS_ZOA[i]} \
    for i in range(n_UE)]
        
    return LOS_angles
