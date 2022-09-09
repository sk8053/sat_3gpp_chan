# -*- coding: utf-8 -*-
"""
Created on Fri Sep  10 10:32:18 2022

@author: seongjoon kang

"""
import matplotlib.pyplot as plt
import numpy as np



def drop_uniform_in_hexagon(R:float, center_point:list = [0,0]):
    
    x_len = 2*R
    y_len = np.sqrt(3)*R
    
    u_x = np.random.uniform(0,1,50)*x_len
    u_y = np.random.uniform(0,1,50) *y_len
    u_xy = np.column_stack((u_x,u_y))
    
    x = u_xy[:,0]
    y = u_xy[:,1]
    
    u_xy[y/(np.sqrt(3)*R/2) + x/(R/2)<1] = np.nan
    u_xy[y< np.sqrt(3)*x -  3*R*np.sqrt(3)/2 ] = np.nan
    
    u_xy[y> np.sqrt(3)*x +  R*np.sqrt(3)/2 ] = np.nan
    u_xy[y> -np.sqrt(3)*x +  R*(5*np.sqrt(3)/2) ] = np.nan
    
    u_xy = u_xy[~np.isnan(u_xy[:,0])]

    hexagon_x = np.array ([0, R/2, 3*R/2, 2*R, 3*R/2, R/2,0])
    hexagon_y = np.array([np.sqrt(3)*R/2, 0, 0, np.sqrt(3)*R/2,
                          np.sqrt(3)*R
                    ,np.sqrt(3)*R,
                 np.sqrt(3)*R/2])
    
    u_xy[:,0] -= R
    u_xy[:,1] -= (1+np.sqrt(3))*R/2
    
    u_xy[:,0] += center_point[0]
    u_xy[:,1] += center_point[1]
    
    hexagon_x -=R
    hexagon_y -= (1+np.sqrt(3))*R/2
    
    hexagon_x += center_point[0]
    hexagon_y += center_point[1]
    
    hexagon_xy = np.column_stack((hexagon_x, hexagon_y))
    return u_xy, hexagon_xy



def create_hexagonal_grid(n_tier:int, R:float):
    colors = ['k','b','r','g','m','c','y','tab:gray']
    xy_list = []
    hex_xy_list = []
    center_point_list = []
    
    for n in range(n_tier):
        R_n = np.sqrt(3)*R*n
        c_ = colors[n]
        if n==0:
            x, y = 0,0
            xy, hex_xy = drop_uniform_in_hexagon(R, center_point=[x,y])
            
            plt.scatter(xy[:,0], xy[:,1], s= 5, c= c_)
            plt.plot(hex_xy[:,0], hex_xy[:,1], c_)
            
            xy_list.append(xy)
            hex_xy_list.append(hex_xy)
            center_point_list.append([x,y])
            
        elif n>=1:
            theta = np.pi/6 + 5*np.pi/3
            prev_x, prev_y = np.cos(theta)*R_n, np.sin(theta)*R_n  
            for j  in range(6):
                theta = np.pi/6 + j*np.pi/3
                x, y = np.cos(theta)*R_n, np.sin(theta)*R_n
                xy, hex_xy = drop_uniform_in_hexagon(R, center_point=[x,y])
                
                plt.scatter(xy[:,0], xy[:,1], s =5, c= c_)
                plt.plot(hex_xy[:,0], hex_xy[:,1], c_)
                
                xy_list.append(xy)
                hex_xy_list.append(hex_xy)
                center_point_list.append([x,y])
                
                for m in range(1,n):
                    x_new = (m*x + (n-m)*prev_x)/n
                    y_new = (m*y + (n-m)*prev_y)/n
                    xy, hex_xy = drop_uniform_in_hexagon(R, center_point=[x_new,y_new])
                    
                    plt.scatter(xy[:,0], xy[:,1],s = 5,c=c_)
                    plt.plot(hex_xy[:,0], hex_xy[:,1], c_)
                    
                    xy_list.append(xy)
                    hex_xy_list.append(hex_xy)
                    center_point_list.append([x_new,y_new])
                    
                prev_x, prev_y = x, y
                
    return xy_list, center_point_list, hex_xy_list,
                    
        
            
    