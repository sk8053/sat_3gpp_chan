# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:37:36 2022

@author: seongjoon Kang
"""

import numpy as np

from three_gpp_tables_s_band import get_correlation_spread, get_shadowing_params

def find_near_ref_angle(angle):
    # find reference elevation angle that is nearest to original one
    angle = np.array(angle)
    angles2 = np.zeros_like(angle)
    angles2[angle<10] = 10
    angles2[angle>=10] = np.round(angle[angle>=10]/10)*10
    return angles2

def create_corr_matrix(T:dict(), link_state:int):
    # return one correlation matrix corresponding one link state
    C = np.array([
        [T['SF_SF'], T['SF_K'], T['SF_DS'], T['SF_ASD'], T['SF_ASA'], T['SF_ZSD'], T['SF_ZSA']],
         [T['K_SF'], T['K_K'], T['K_DS'], T['K_ASD'], T['K_ASA'], T['K_ZSD'], T['K_ZSA']],
         [T['DS_SF'], T['DS_K'], T['DS_DS'], T['DS_ASD'], T['DS_ASA'], T['DS_ZSD'], T['DS_ZSA']],
         [T['ASD_SF'], T['ASD_K'], T['ASD_DS'], T['ASD_ASD'], T['ASD_ASA'], T['ASD_ZSD'], T['ASD_ZSA']],
         [T['ASA_SF'], T['ASA_K'], T['ASA_DS'], T['ASA_ASD'], T['ASA_ASA'], T['ASA_ZSD'], T['ASA_ZSA']],
         [T['ZSD_SF'], T['ZSD_K'], T['ZSD_DS'], T['ZSD_ASD'], T['ZSD_ASA'], T['ZSD_ZSD'], T['ZSD_ZSA']],
         [T['ZSA_SF'], T['ZSA_K'], T['ZSA_DS'], T['ZSA_ASD'], T['ZSA_ASA'], T['ZSA_ZSD'], T['ZSA_ZSA']]
        ])

    if link_state == 2: # if link state is NLOS , remove K
        C = np.delete(C, 1 , axis = 0)
        C = np.delete(C, 1, axis = 1)   
    return np.linalg.cholesky(C)    

def step4(scenario:str, link_state:list(),elev_angle:list(),):
    # Step 4: Generate large scale parameters, e.g. delay spread (DS), 
    # angular spreaDS (ASA, ASD, ZSA, ZSD), Ricean K factor (K) 
    # and shadow fading (SF) taKing into account 
    # cross correlation according to Table 7.5-6 
    params_set={'SF','K','DS', 'ASD', 'ASA', 'ZSD','ZSA'}
    T = dict()
    for p1 in params_set:
        for p2 in  params_set:
            if p1 == p2:    
                T[p1+'_'+p2] =1
            else:
                T[p1+'_'+p2] =np.nan
    # write data from 3gpp table
    elev_angle = find_near_ref_angle(elev_angle)
    correlation_matrix, spread_matrix = get_correlation_spread(scenario, link_state, elev_angle)
    shadow_CL_table = get_shadowing_params(scenario, elev_angle)
    
    mu_SF = 0;
    SF_list, K_list, DS_list = [],[],[]
    ASA_list, ASD_list = [], []
    ZSA_list, ZSD_list = [], []
    for i, link_state_i in enumerate(link_state):
      
        corr_matrix_i = correlation_matrix[:,i]
        mu_DS, sig_DS, mu_ASD, sig_ASD, mu_ASA, sig_ASA, \
        mu_ZSA, sig_ZSA, mu_ZSD, sig_ZSD, mu_K, sig_K  = spread_matrix[:,i]
        sig_SF_LOS, sig_SF_NLOS,_ = shadow_CL_table[i]
 
        # NOTE 8 from 3GPP 38.811 6.7.2
        # For satellite (GEO/LEO), the departure angle spreads are zeros, i.e., 
        # mu_asd and mu_zsd = -inf, -inf
        mu_ASD, mu_ZSD = -1*np.inf, -1*np.inf
        
        keys = ['ASD_DS', 'ASA_DS', 'ASA_SF','ASD_SF','DS_SF', 'ASD_ASA', 'ASD_K','ASA_K',
                'DS_K','SF_K', 'ZSD_SF', 'ZSA_SF', 'ZSD_K','ZSA_K', 'ZSD_DS','ZSA_DS',
                'ZSD_ASD','ZSA_ASD', 'ZSD_ASA','ZSA_ASA', 'ZSD_ZSA']
        for j, key in enumerate(keys):
            s = key.split('_')
            key2 = s[-1]+'_'+s[0]
            T[key] = corr_matrix_i[j]
            T[key2] = T[key]
        # create cross-correlation matrix with cholesky decomposition
        sqrt_C = create_corr_matrix(T, link_state_i)
        
        if link_state_i == 1:
            sf_k_ds_asd_asa_zsd_zsa = sqrt_C.dot(np.random.normal(0,1,size = (7,1)))
            
            # Note that SF and K are in dB-scale, and others are in linear scale
            DS = 10**(mu_DS + sig_DS*sf_k_ds_asd_asa_zsd_zsa[2])
            K = 10**(0.1*(mu_K + sig_K*sf_k_ds_asd_asa_zsd_zsa[1])) # dB -> linear
            SF = mu_SF + sig_SF_LOS *sf_k_ds_asd_asa_zsd_zsa[0] # keep dB-scale         
            ASD = 10**(mu_ASD + sig_ASD*sf_k_ds_asd_asa_zsd_zsa[3])
            ASA = 10**(mu_ASA + sig_ASA*sf_k_ds_asd_asa_zsd_zsa[4])
            ZSD = 10**(mu_ZSD + sig_ZSD*sf_k_ds_asd_asa_zsd_zsa[5])
            ZSA = 10**(mu_ZSA + sig_ZSA*sf_k_ds_asd_asa_zsd_zsa[6])
            
        else:
            sf_ds_asd_asa_zsd_zsa = sqrt_C.dot(np.random.normal(0,1,size = (6,1)))
            DS = 10**(mu_DS + sig_DS*sf_ds_asd_asa_zsd_zsa[1]) # dB -> linear
            SF = mu_SF + sig_SF_NLOS*sf_ds_asd_asa_zsd_zsa[0] # keep dB-scale         
            ASD = 10**(mu_ASD + sig_ASD*sf_ds_asd_asa_zsd_zsa[2])
            ASA = 10**(mu_ASA + sig_ASA*sf_ds_asd_asa_zsd_zsa[3])
            ZSD = 10**(mu_ZSD + sig_ZSD*sf_ds_asd_asa_zsd_zsa[4])
            ZSA = 10**(mu_ZSA + sig_ZSA*sf_ds_asd_asa_zsd_zsa[5])
            K = np.nan;
        
        # limit random RMS azimuth arrival and azimuth departure spread values 
        # to 104 degree
        ASA=  min(ASA, 104)
        ASD = min(ASD, 104)
        # in the same way, limit zenith angles to 52
        ZSA = min(ZSA, 52)
        ZSD = min(ZSD, 52)
        
        SF_list.append(SF)
        K_list.append(K)
        DS_list.append(DS)
        ASD_list.append(ASD)
        ASA_list.append(ASA)
        ZSA_list.append(ZSA)
        ZSD_list.append(ZSD)
        
    return SF_list,K_list,  DS_list, ASD_list, ASA_list, ZSD_list, ZSA_list
                                                            