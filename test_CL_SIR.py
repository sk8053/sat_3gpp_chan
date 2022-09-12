# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 13:20:40 2022

@author: seongjoon kang
"""

import numpy as np
import pandas as pd
from sat_three_gpp_chan_model import sat_three_gpp_channel_model
#__reset()__
from drop_ue_in_hex_cell import create_hexagonal_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

f_c = 2e9
sat_height = 600e3 #[m]
scenario = 'rural'
# adjacent beam spacing
HPBW = 4.4127 # [deg], set 1 LEO 600, S band
ABS = np.sqrt(3)*np.sin(np.deg2rad(HPBW)/2)
EIRP_density = 34 # dBW/MHz
sat_max_ant_gain = 30 # dBi, if we use 34dBw/MHz of EIRP density, this value becomes 0dBi
sat_ant_diameter = 2 # [m], note that element gains are related to radius of satellite antenna
beam_diameter = np.tan(np.deg2rad(HPBW)/2)*sat_height*2 # [m], beam diameter = height*sin(HPBW/2)*2*cos(30) = height *ABS
beam_radius = beam_diameter/2

n_ue_per_cell = 100
ue_location, center_point_list,_ = create_hexagonal_grid(5, beam_radius, 
                                                         n_ue_per_cell=n_ue_per_cell)

ue_location = np.array(ue_location).reshape(-1,3)
center_points_hex_cells_xyz = np.array(center_point_list)

sat_location = np.array([[0,0, sat_height]])
sat_beam_directions = center_points_hex_cells_xyz - sat_location

CL_list = np.array([])

CL_total = []
for i in tqdm(range(sat_beam_directions.shape[0])):
    
    a = sat_three_gpp_channel_model(scenario, sat_location, 
                        ue_location,sat_beam_directions[i], f_c)
    a.force_LOS = True
    sat_los_gain =  a.step1(return_sat_gain=True)
    a.step2_and_step3()
    sat_los_gain = 10*np.log10(sat_los_gain)
    
    CL_all_ = sat_los_gain - a.path_loss + sat_max_ant_gain 
    
    CL_all = -(10*np.log10(10**(0.1*CL_all_)))
    CL_all = CL_all.reshape(-1, n_ue_per_cell)
    ind = np.argmin(CL_all, axis = 0)
    
    CL_list = np.append(CL_list, CL_all[i,:])
    
    
    CL_all_linear = 10**(-0.1*CL_all)
    CL_total.append(CL_all_linear)

CL_list_new = np.max(CL_total, axis =0)    
CL_table = pd.read_csv('CL_case9.csv')
plt.figure()
for key in CL_table.keys():
    data = CL_table[key]
    plt.plot(np.sort(data), np.linspace(0,1,len(data)), label = key)
plt.plot(np.sort(CL_list), np.linspace(0,1,len(CL_list)), 'k', label = 'Simulator')
plt.xlabel('Coupling Loss (dB)')
plt.ylabel ('CDF')
plt.grid()
plt.legend()

CL_total = np.array(CL_total) # 61*61*10
Rx_power = np.max(CL_total, axis =0)
interference = np.sum(CL_total, axis=0) - Rx_power

SIR = 10*np.log10(Rx_power[:19]/interference[:19])
SIR = SIR.reshape(-1)

SIR_table = pd.read_csv('SIR_case9.csv')
plt.figure()
for key in SIR_table.keys():
    data = SIR_table[key]
    plt.plot(np.sort(data), np.linspace(0,1,len(data)), label = key)

plt.plot(np.sort(SIR), np.linspace(0,1,len(SIR)), 'k', label = 'Simulator')
plt.grid()
plt.xlabel('SIR (dB)')
plt.ylabel('CDF')
plt.legend()




