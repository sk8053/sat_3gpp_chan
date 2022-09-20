# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:21:59 2022

@author: seongjoon kang
"""
import numpy as np
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
KT = -174
Nf = 7
BW = 10e6

Noise_power_db = KT + Nf + 10*np.log10(BW)
noise_power = 10**(0.1*Noise_power_db)

ABS = np.sqrt(3)*np.sin(np.deg2rad(HPBW)/2)
EIRP_density = 34 # dBW/MHz
Tx_power = EIRP_density+30+10*np.log(10) # 74 dBm
Tx_power_lin = 10**(0.1*Tx_power)

sat_max_ant_gain = 30 # dBi, if we use 34dBw/MHz of EIRP density, this value becomes 0dBi
sat_ant_diameter = 2 # [m], note that element gains are related to radius of satellite antenna
beam_diameter = np.tan(np.deg2rad(HPBW)/2)*sat_height*2 # [m], beam diameter = height*sin(HPBW/2)*2*cos(30) = height *ABS
beam_radius = beam_diameter/2

n_ue_per_cell = 10
ue_location, center_point_list,_ = create_hexagonal_grid(5, beam_radius, 
                                                         n_ue_per_cell=n_ue_per_cell)

ue_location = np.array(ue_location).reshape(-1,3)
center_points_hex_cells_xyz = np.array(center_point_list)

sat_location = np.array([[0,0, sat_height]])

sat_beam_directions = center_points_hex_cells_xyz - sat_location

CL_list = np.array([])

CL_total = []

subcarrier_spacing = 15e3 # sub-carrier spacing
total_n_carrier = np.floor(BW/subcarrier_spacing)
#f_list = np.arange(-330,330)*15e3 # 10MHz

f_list = np.arange(-np.floor(total_n_carrier/2),
                   np.floor(total_n_carrier/2))*15e3 + f_c # 10MHz
chan_instant_list = []

for i in tqdm(range(sat_beam_directions.shape[0])):    
    a = sat_three_gpp_channel_model(scenario, sat_location, 
                        ue_location,sat_beam_directions[i], f_c)
    chan_instant_list.append(a)
    sat_ant_gain =  a.step1(return_sat_gain=True)
    a.step2_and_step3()
    sat_ant_gain = 10*np.log10(sat_ant_gain)
    
    CL_all_ = sat_ant_gain - a.path_loss + sat_max_ant_gain 
    CL_all_ = CL_all_.reshape(-1, n_ue_per_cell)
    CL_list = np.append(CL_list, -CL_all_[i,:])
    
    CL_all_linear = 10**(0.1*CL_all_)
    CL_total.append(CL_all_linear)

associ_indices = np.argmax(CL_total, axis = 0)
associ_indices = associ_indices.reshape(-1)

H_list = []
delay_list = []
for i in tqdm(range(len(chan_instant_list))):
    chan = chan_instant_list[i] 
    # channel tap gain and delays including clusters
    chan.step4()
    chan.step5_and_step6()
    chan.step7_step8_step9()
    H, delay = chan.step10_step11()
    H_list.append(H)
    delay_list.append(delay)

    
h_45_list = np.zeros(shape=[len(H[45]),len(chan_instant_list), len(f_list)],dtype = complex)
h_n_45_list = np.zeros(shape=[len(H[45]),len(chan_instant_list), len(f_list)], dtype = complex)

# (n_link, n_beams, n_frequencies)
for link_idx in range(len(H[45])): 
# for each link, compute SINR per sub-carrier
    
    # compute chanel-frequency response corresponding to each subcarrier
    associated_beam_idx = associ_indices[link_idx]
    power_list = []
    for beam_idx in range(len(chan_instant_list)):
        H = H_list[beam_idx]
        delay = delay_list[beam_idx]
        h_45 = 0
        h_n_45 =0
        for cluster_idx in range(len(H[45][link_idx])):
            h_45 = h_45 + H[45][link_idx][cluster_idx]*np.exp(-1j*2*np.pi*f_list*np.array(delay[45][link_idx][cluster_idx]))
            h_n_45 = h_n_45 + H[-45][link_idx][cluster_idx]*np.exp(-1j*2*np.pi*f_list*np.array(delay[-45][link_idx][cluster_idx]))
        
        h_45_list[link_idx, beam_idx,:] = h_45
        h_n_45_list[link_idx, beam_idx,:] = h_n_45

SINR_avg_list = []    
for link_idx in tqdm(range(len(H[45]))):
    serv_beam_idx = associ_indices[link_idx]
    #serv_beam_idx = int(np.floor(link_idx/10))
   # print(serv_beam_idx)
    itf_f =0
    rx_power_f = 0
    
    for f_idx in range(len(f_list)):
        h_serv1 = h_45_list[link_idx,serv_beam_idx,f_idx] 
        h_serv2 = h_n_45_list[link_idx,serv_beam_idx,f_idx]
        w = [h_serv1, h_serv2]
        w = w/np.linalg.norm(w)
        rx_power_f += np.linalg.norm([h_serv1, h_serv2])**2
        itf = 0
        for beam_idx in range(len(chan_instant_list)):
            if beam_idx != serv_beam_idx:
                h_itf1 = h_45_list[link_idx,beam_idx,f_idx] 
                h_itf2 = h_n_45_list[link_idx,beam_idx,f_idx] 
                itf += abs(np.dot(np.conj(w), [h_itf1, h_itf2]))**2                
        itf_f += itf
        
    rx_power_avg = rx_power_f/len(f_list)
    itf_power_avg = itf_f/len(f_list)
    
    SINR_avg_list.append(10*np.log10(Tx_power_lin*rx_power_avg
                                     /(Tx_power_lin*itf_power_avg+noise_power)))
    
    if link_idx ==122 or link_idx ==123:
        #print(rx_power_avg, itf_power_avg)
        print(link_idx, h_serv1)
        print(link_idx, h_serv2)

# outside tiers are used only for wrap-around mechanism
# we only consider UEs deployed in 19 center cells, inside 3 tiers
plt.plot(np.sort(SINR_avg_list[:19*n_ue_per_cell]), 
         np.linspace(0,1,len(SINR_avg_list[:19*n_ue_per_cell])))
plt.xlim([-30, 15])
plt.grid()