# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 10:29:00 2022

@author: seongjoon kang
"""
import numpy as np
from three_gpp_tables_s_band import get_los_prob, get_shadowing_params

def find_near_ref_angle(angle):
    # find reference elevation angle that is nearest to original one
    angle = np.array(angle)
    angles2 = np.zeros_like(angle)
    angles2[angle<10] = 10
    angles2[angle>=10] = np.round(angle[angle>=10]/10)*10
    return angles2

def get_link_state(elev_angle, scenario = 'rural'):
    # return link states for each elevation angle
    elev_angle = find_near_ref_angle(elev_angle)
    
    return get_los_prob(scenario, elev_angle)

def get_path_loss(height:int, elev_angle:list(), f_c:int, link_state:list(), scenario:str = 'rural'):
    # return path loss values corresponding to elevation angles and linnk states
    R_E = 6371e3 # [m]
    elev_angle = np.array(elev_angle)
    link_state = np.array (link_state)
    d = np.sqrt(R_E**2 * np.sin(np.deg2rad(elev_angle))**2 + height**2 + 2*height*R_E)- R_E*np.sin(np.deg2rad(elev_angle))
    # f_c in Ghz, d in m
    fspl = 32.45 + 20*np.log10(f_c/1e9) + 20*np.log10(d)
    # find ref elevation angle
    elev_angle = find_near_ref_angle(elev_angle)
    shadow_CL_table = get_shadowing_params(scenario, elev_angle)
    
    # take LOS sigma values
    sigma_los = shadow_CL_table[:,0][link_state==1]
    # NLOS sigma and CL 
    sigma_nlos = shadow_CL_table[:,1][link_state==2]
    CL = shadow_CL_table[:,2][link_state==2]

    # compute basic pathloss
    PL_b = np.zeros_like(fspl)
    PL_b[link_state==1] = fspl[link_state==1] \
    + sigma_los*np.random.normal(0,1, np.sum(link_state==1))
    PL_b[link_state==2] = fspl[link_state==2]  \
    + sigma_nlos*np.random.normal(0,1, np.sum(link_state==2)) + CL
        
    ## attenuation due to atmophere following ITU-R P.676
    # https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-13-202208-I!!PDF-E.pdf
    T = 288.15 # unit is K
    p = 1013.25 # hPa
    ro = 7.5 # 7.5g / m^3
    e = 9.98 # hPa, additonal pressuer due to water vapor
    ro_gas = 1 # assume no attenuation due to oxigen
    # read coefficient from 
    # https://www.itu.int/dms_pub/itu-r/oth/11/01/R11010000020001TXTM.txt
    if f_c < 10e9:
        # take values corresponding to 2GHz
        a = -2.37
        b = 2.71e-2
        c = -2.53e-4
        d = -2.83e-4
    else:
        # take values corresponding to 28GHz
        a = -2.43
        b = 2.8e-2
        c = -6.168e-4
        d = -9.517e-4
        
    h_0 = a +b*T + c*p + d*ro    
    A_zenith = (ro_gas * h_0)
    PL_a = A_zenith/np.sin(np.deg2rad(elev_angle))
    PL_a = 10*np.log10(PL_a)
    
    Path_loss = PL_b + PL_a
    # we will ignore the attenuation 
    #due to either ionospheric or trospheric scintillation loss
    return Path_loss

