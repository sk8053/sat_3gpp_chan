# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 23:00:01 2022

@author: seongjoon kang
"""
import numpy as np
from drop_ue_in_hex_cell import create_hexagonal_grid
import matplotlib.pyplot as plt

def Rx(theta):
    return np.array([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
def Ry(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
    return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

_,_, hex_xy = create_hexagonal_grid(13, 25e3 )

hex_xy = np.array(hex_xy)
#plt.plot(hex_xy[0][:,0],hex_xy[0][:,1])
R_E = 6371*1e3
sat_height = 600*1e3
for i in range(hex_xy.shape[0]):
    phi = np.arctan2(hex_xy[i][:,1], hex_xy[i][:,0])
    
    R_2D = np.sqrt(hex_xy[i][:,1]**2 + hex_xy[i][:,0]**2)
    R_3D = np.sqrt(hex_xy[i][:,1]**2 + hex_xy[i][:,0]**2+ np.array([R_E]*len(hex_xy[i]))**2)
    
    theta1 = np.arctan2(R_2D,R_E)
    theta2 = np.pi/2 - np.arctan2(sat_height, R_2D) + np.arctan2(R_2D,R_E)
    theta3 = np.pi- np.arcsin(R_3D*np.sin(theta2)/R_E)
    
    theta = np.pi - theta2 - theta3 + theta1
    xyz_new = []
    
    for phi_0, theta_0 in zip(phi,theta):
        xyz = np.array([0,0,R_E/np.cos(theta_0)]).dot(Rx(theta_0)).dot(Rz(phi_0))
        xyz_new.append(xyz)
    xyz_new = np.array(xyz_new)
    
    plt.plot(xyz_new[:,0], xyz_new[:,1])
    