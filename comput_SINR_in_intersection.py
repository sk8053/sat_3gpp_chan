# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:06:45 2022

@author: seongjoon kang
"""
import numpy as np
from drop_ue_in_hex_cell import create_hexagonal_grid
from sat_three_gpp_chan_model import sat_three_gpp_channel_model
import matplotlib.pyplot as plt
from tqdm import tqdm

R_E = 6371e3 # [m] radius of Earth
HPBW = 4.4217 #[deg] half power beam width 
sate_height = 600e3 #[m] satellite height for LEO
theta = 0*np.pi/180 # [rad] longtitude
n_ue_per_cell = 6 # number of UE per cell

alpha_min = 30*np.pi/180 # [rad] minimum elevation angle in radius

N_s = 30 # number of satellite per orbit
N_o = 17 # number of orbit 
y = 2*np.pi*R_E/N_s # vertical width between satellites
x = np.pi*R_E*np.cos(theta)/N_o # horizontal width between satellites

# radius of footprint satesfying minimum elevation angle=30
R_footprint = R_E *( np.pi/2 - np.arcsin(R_E*np.cos(alpha_min)/(R_E+sate_height)) - alpha_min)
ABS = np.sqrt(3) *np.sin(np.deg2rad(HPBW)/2)*sate_height # adjacent beamwidth
beam_radius = ABS/(2*np.cos(np.pi/6)) # beam radius 
n_tier = np.ceil(R_footprint/ABS) +1 # number of tier to cover the whole area of footprint
n_tier = int(n_tier)

# suppose we create one footprint in the center (0,0)
ue_xy,center_points,hex_xy = create_hexagonal_grid(n_tier, beam_radius ,n_ue_per_cell,center_cell_xy=[0,0] )

# get the beam indices in intersection area
ind_in_intersec = []

for n in range(n_tier-1):
    indices_nth_layer = np.arange(3*n*(n+1)+1,3*(n+2)*(n+1)+1)
    ind_in_intersec = np.append(ind_in_intersec,indices_nth_layer[np.arange(9-n,n-10)])
ind_in_intersec = np.array(ind_in_intersec, dtype = int)

ue_inter_sec = []
# extract the locations of UE and center points in the intersection
ue_inter_sec = np.array(ue_xy)[ind_in_intersec]

ue_inter_sec_new = ue_inter_sec[0]
for i in range(1,ue_inter_sec.shape[0]):
    ue_inter_sec_new = np.append(ue_inter_sec_new, ue_inter_sec[i], axis =0)

center_points = np.array(center_points)[ind_in_intersec]
#center_points = np.array(center_points)



##################################################################################3

f_c = 2e9 # carrier frequency
scenario = 'rural'
EIRP_density = 34 # dBW/MHz
sat_max_ant_gain = 30 # [dBi], if we use 34dBw/MHz of EIRP density, this value becomes 0dBi
sat_ant_diameter = 2 # [m], note that element gains are related to radius of satellite antenna

KT = -174 # [dBm/Hz]
Nf = 7 
BW = 10e6 # 10MHz

Noise_power_db = KT + Nf + 10*np.log10(BW) # noise pwer in dB
noise_power = 10**(0.1*Noise_power_db) # noise power in linear

EIRP_density = 34 # dBW/MHz
Tx_power = EIRP_density+30+10*np.log10(BW/1e6) # 74 dBm when maximum antenna gain is 0dBi
Tx_power_lin = 10**(0.1*Tx_power)

subcarrier_spacing = 15e3 # sub-carrier spacing
total_n_carrier = np.floor(BW/subcarrier_spacing)
freq_list = np.arange(-np.floor(total_n_carrier/2),
                   np.floor(total_n_carrier/2))*15e3 # 10MHz
# 4 satellite locations
sat_locations = np.array([[0,0,sate_height],[0,y,sate_height]
                          ,[x,y/2,sate_height],[-x,y/2,sate_height]])

H_45_dict = {}
H_n_45_dict = {}
delay_dict = {}
associ_ind_dict = {}

for j in range(2):
    # determine the satellite and compute channel 
    sat_beam_directions = center_points - sat_locations[j]
    
    H_list = []
    delay_list = []
    for i in tqdm(range(sat_beam_directions.shape[0])):    
        a = sat_three_gpp_channel_model(scenario, np.array([sat_locations[j]]), 
                            ue_inter_sec_new,sat_beam_directions[i], f_c)
        H, delay = a.run()
        H_list.append(H)
        delay_list.append(delay)
        
    # shape = (number of link, number of beam direction, number of frequency)
    h_45_list = np.zeros(shape=[len(H[45]),sat_beam_directions.shape[0], len(freq_list)],dtype = complex)
    h_n_45_list = np.zeros(shape=[len(H[45]),sat_beam_directions.shape[0], len(freq_list)], dtype = complex)
    
    # (n_link, n_beams, n_frequencies)
    for link_idx in range(len(H[45])): 
    # for each link, compute SINR per sub-carrier
        h_45, h_n_45 = 0,0
        # compute chanel-frequency response corresponding to each subcarrier
        #associated_beam_idx = associ_indices[link_idx]
        power_list = []
        for beam_idx in range(sat_beam_directions.shape[0]):
            H = H_list[beam_idx]
            delay = delay_list[beam_idx]
            for cluster_idx in range(len(H[45][link_idx])):
                h_45 = h_45 + H[45][link_idx][cluster_idx]*np.exp(-1j*2*np.pi*freq_list*np.array(delay[45][link_idx][cluster_idx]))
                h_n_45 = h_n_45 + H[-45][link_idx][cluster_idx]*np.exp(-1j*2*np.pi*freq_list*np.array(delay[-45][link_idx][cluster_idx]))
            
            h_45_list[link_idx, beam_idx,:] = h_45
            h_n_45_list[link_idx, beam_idx,:] = h_n_45
    
    H_45_dict[j] = h_45_list
    H_n_45_dict[j] = h_n_45_list
    delay_dict[j] = delay_list
    
########################################################################################################
############################## SINR Computation ########################################################

vec_list = [1,-1,1j, -1j]
precoder_list = []
for i in range(4):
    for j in range(4):
        if vec_list[i] != vec_list[j]:
            precoder_list.append([vec_list[i], vec_list[j]])

SINR_avg_list = []
for iter_ in range(3):
    for link_idx in tqdm(range(ue_inter_sec_new.shape[0])):
        #serv_beam_idx = associ_ind_dict[0][link_idx]
        serv_beam_idx = int(np.floor(link_idx/n_ue_per_cell))
        itf_f =0 
        rx_power_f = 0
        #H_serv = []
        
        # suppose 10 cells are served simultaneously by two satellites
        serving_ind = np.random.choice(np.arange(ue_inter_sec.shape[0]),10)
        
        for f_idx in range(len(freq_list)):
            h_serv_45_1 = H_45_dict[0][link_idx,serv_beam_idx,f_idx] 
            h_serv_n_45_1 = H_n_45_dict[0][link_idx,serv_beam_idx,f_idx]
            
            h_serv_45_2 = H_45_dict[1][link_idx,serv_beam_idx,f_idx] 
            h_serv_n_45_2 = H_n_45_dict[1][link_idx,serv_beam_idx,f_idx]      
            
            H_serv = np.array([[h_serv_45_1, h_serv_45_2],[h_serv_n_45_1, h_serv_n_45_2]])
            #w_p = np.array([1,1]) # precoder for one stream
            prg_idx = f_idx%4
            w_p = precoder_list[prg_idx%12]
            
            h_serv = H_serv.dot(w_p)
            
            h_itf_mat = np.zeros((2,2),dtype = complex)
            for beam_idx in serving_ind:
                if beam_idx != serv_beam_idx:
                    
                    h_itf_45_1 = H_45_dict[0][link_idx,beam_idx,f_idx] 
                    h_itf_n_45_1 = H_n_45_dict[0][link_idx,beam_idx,f_idx]
                    
                    h_itf_45_2 = H_45_dict[1][link_idx,beam_idx,f_idx] 
                    h_itf_n_45_2 = H_n_45_dict[1][link_idx,beam_idx,f_idx]
                    H_itf = np.array([[h_itf_45_1, h_itf_45_2],[h_itf_n_45_1, h_itf_n_45_2]])
                    
                    prg_idx = f_idx%4
                    w_p = precoder_list[prg_idx%12]
                    
                    h_itf = H_itf.dot(w_p)
                    h_itf_mat += np.outer(h_itf, np.conjugate(h_itf).T)
            
            #w_c = h_serv/np.linalg.norm(h_serv) # matched filter combiner 
            A= np.linalg.inv(np.eye(2)*noise_power + Tx_power_lin*h_itf_mat)
            w_c = np.conjugate(h_serv).dot(A)
            w_c = w_c/np.linalg.norm(w_c)
            
            rx_power_f += np.abs(np.conjugate(w_c).dot(h_serv))**2
            itf = 0
            
            
            
            for beam_idx in serving_ind:
                if beam_idx != serv_beam_idx:
                    
                    h_itf_45_1 = H_45_dict[0][link_idx,beam_idx,f_idx] 
                    h_itf_n_45_1 = H_n_45_dict[0][link_idx,beam_idx,f_idx]
                    
                    h_itf_45_2 = H_45_dict[1][link_idx,beam_idx,f_idx] 
                    h_itf_n_45_2 = H_n_45_dict[1][link_idx,beam_idx,f_idx]
                    H_itf = np.array([[h_itf_45_1, h_itf_45_2],[h_itf_n_45_1, h_itf_n_45_2]])
                    
                    prg_idx = f_idx%4
                    w_p = precoder_list[prg_idx%12]
                    
                    h_itf = H_itf.dot(w_p)
                    itf += np.abs(np.conjugate(w_c).dot(h_itf))**2
                    
            itf_f += itf
            
        rx_power_avg = rx_power_f/len(freq_list)
        itf_power_avg = itf_f/len(freq_list)
        
        SINR_avg_list.append(10*np.log10(Tx_power_lin*rx_power_avg
                                         /(Tx_power_lin*itf_power_avg+noise_power)))
plt.plot(np.sort(SINR_avg_list), np.linspace(0,1,len(SINR_avg_list)))