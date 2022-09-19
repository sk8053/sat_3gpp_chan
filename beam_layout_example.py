# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:39:09 2022

@author: seongjoon kang
"""
import numpy as np
from drop_ue_in_hex_cell import create_hexagonal_grid
import matplotlib.pyplot as plt

R_E = 6371e3 # radius of Earth
HPBW = 4.4217 #[deg] half power beam width 
sate_height = 600e3 # satellite height for LEO
theta = 30*np.pi/180 # longtitude
n_ue_per_cell = 10 # number of UE per cell
alpha_min = 30*np.pi/180 # minimum elevation angle in radius

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

ue_inter_sec = np.array(ue_xy)[ind_in_intersec]
center_points = np.array(center_points)[ind_in_intersec]
sat_locations = np.array([[0,0,sate_height],[0,y,sate_height]
                          ,[x,y/2,sate_height],[-x,y/2,sate_height]])


enable_plot = False
if enable_plot is True:
    hex_xy = np.array(hex_xy)
    plt.figure(figsize = (12,10))
    plt.scatter(0,0,c='r')
    plt.scatter(0,y, c='r')
    plt.scatter(x,y/2,c='r')
    plt.scatter(-x,y/2, c='r')
    for i in range(hex_xy.shape[0]):
        
        if i in ind_in_intersec:
            plt.scatter(ue_xy[i][:,0],ue_xy[i][:,1], c= 'r',s=1)
            #plt.plot(hex_xy[i][:,0], hex_xy[i][:,1], 'k')
            
             
        plt.plot(hex_xy[i][:,0], hex_xy[i][:,1], 'k')
        plt.plot(hex_xy[i][:,0], hex_xy[i][:,1]+y, 'k')
       
        #plt.plot(hex_xy[i][:,0], hex_xy[i][:,1]-y, 'k')
        plt.plot(hex_xy[i][:,0]+x, hex_xy[i][:,1]+y/2, 'k')
        #plt.plot(hex_xy[i][:,0]+x, hex_xy[i][:,1]-y/2, 'k')
        
        plt.plot(hex_xy[i][:,0]-x, hex_xy[i][:,1]+y/2, 'k')
        #plt.plot(hex_xy[i][:,0]-x, hex_xy[i][:,1]-y/2, 'k')
        
    plt.grid()
    plt.savefig('plots/show_overlapped_area.jpg')
    #plt.savefig('plots/show_overall_deployment.jpg')
    
