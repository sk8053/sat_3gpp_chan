# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:47:27 2022

@author: gangs
"""

import numpy as np
import PIL
import matplotlib.pyplot as plt
Earth_R = 6371e3

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

def plot_orbital(rot_x:float = 80):
    sat_height = 600e3
    theta = np.linspace(0,2*np.pi, 200)
    x, y = Earth_R*np.cos(theta), Earth_R*np.sin(theta)
    z = [sat_height]*len(x)
    xyz = np.column_stack((x,y,z))
    
    xyz = xyz.dot(Rx(np.deg2rad(rot_x)))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot3D(xyz[:,0],xyz[:,1],xyz[:,2])
    #ax.view_init(30,10)
    
                     
def plot_Earth(rot_zenith:int = 20, rot_azimuth:int= 60,
               orbit_tilt_angle:float = 80, sat_height:float = 1200e3):
    
    bm = PIL.Image.open('blue_marble.jpg')
    bm = np.array(bm)/256.
    
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    x = Earth_R*np.outer(np.cos(lons), np.cos(lats)).T
    y = Earth_R*np.outer(np.sin(lons), np.cos(lats)).T
    z = Earth_R*np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    ax.plot_surface(x, y, z, rstride=2, cstride=2, facecolors = bm)

    

    theta = np.linspace(-np.pi,np.pi, 200)
    x, y = (Earth_R+sat_height)*np.cos(theta), (Earth_R+sat_height)*np.sin(theta)
    z = [0]*len(x)
    xyz = np.column_stack((x,y,z))
    
    xyz = xyz.dot(Rx(np.deg2rad(orbit_tilt_angle)))

    ax.plot3D(xyz[:,0],xyz[:,1],xyz[:,2])
    ax.view_init(rot_zenith, rot_azimuth)
    
    plt.show()
    
plot_Earth(orbit_tilt_angle= 87)