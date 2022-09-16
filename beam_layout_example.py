# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:39:09 2022

@author: seongjoon kang
"""
import numpy as np
from drop_ue_in_hex_cell import create_hexagonal_grid
import matplotlib.pyplot as plt

R_E = 6371e3
theta = np.deg2rad(0)
y = 2*np.pi*R_E/30
x = np.pi*R_E*np.cos(theta)/17

_,center_points,hex_xy = create_hexagonal_grid(23, 23e3,0,center_cell_xy=[0,0] )

indice_list = []
for n in range(23):
    indice_list.append(np.arange(3*n*(n+1)+1,3*(n+2)*(n+1)+1))
indice_list_new = []
for i in np.arange(10,23):
    indice_list_new = np.append(indice_list_new, indice_list[i][np.arange(9-i,i-10)])
    
hex_xy = np.array(hex_xy)
plt.figure(figsize = (12,10))
for i in range(hex_xy.shape[0]):
        
    if i in indice_list_new:
        plt.scatter(center_points[i][0], center_points[i][1], c= 'r')
       # plt.scatter(center_points[i][0], center_points[i][1]+y, c= 'r')
       
    
    plt.plot(hex_xy[i][:,0], hex_xy[i][:,1], 'k')
    plt.plot(hex_xy[i][:,0], hex_xy[i][:,1]+y, 'k')
   
    #plt.plot(hex_xy[i][:,0], hex_xy[i][:,1]-y, 'k')
    #plt.plot(hex_xy[i][:,0]+x, hex_xy[i][:,1]+y/2, 'k')
    #plt.plot(hex_xy[i][:,0]+x, hex_xy[i][:,1]-y/2, 'k')
    
    #plt.plot(hex_xy[i][:,0]-x, hex_xy[i][:,1]+y/2, 'k')
    #plt.plot(hex_xy[i][:,0]-x, hex_xy[i][:,1]-y/2, 'k')
    
plt.grid()
plt.savefig('plots/show_overlapped_area.jpg')
#plt.savefig('plots/show_overall_deployment.jpg')

