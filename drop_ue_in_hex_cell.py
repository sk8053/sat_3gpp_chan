# -*- coding: utf-8 -*-
"""
Created on Fri Sep  10 10:32:18 2022

@author: seongjoon kang

"""
import matplotlib.pyplot as plt
import numpy as np



def drop_uniform_in_hexagon(R:float, n_points:int =10, center_point:list = [0,0]):
    """
    Parameters
    ----------
    R : float
        radius of hexagonal cell.
    n_points: int
        number of users randomly dropped inside hexagonal cell
    center_point : list, optional
        center location of hexagonal cell. The default is [0,0].

    Returns
    -------
    u_xy : array
        ue locations in the hexagonal cell
    hexagon_xy : array
        locations of vertices of hexagon

    """
 
    
    x_len = 2*R
    y_len = np.sqrt(3)*R
    
    # generate three times more points to consider removing some points 
    # that are placed outside the hexagonal cell
    u_x = np.random.uniform(0,1,n_points*2)*x_len
    u_y = np.random.uniform(0,1,n_points*2) *y_len
    u_xyz = np.column_stack((u_x,u_y, [0]*len(u_x)))
    
    x = u_xyz[:,0]
    y = u_xyz[:,1]
    
    u_xyz[y/(np.sqrt(3)*R/2) + x/(R/2)<1] = np.nan
    u_xyz[y< np.sqrt(3)*x -  3*R*np.sqrt(3)/2 ] = np.nan
    
    u_xyz[y> np.sqrt(3)*x +  R*np.sqrt(3)/2 ] = np.nan
    u_xyz[y> -np.sqrt(3)*x +  R*(5*np.sqrt(3)/2) ] = np.nan
    
    u_xyz = u_xyz[~np.isnan(u_xyz[:,0])]

    hexagon_x = np.array ([0, R/2, 3*R/2, 2*R, 3*R/2, R/2,0])
    hexagon_y = np.array([np.sqrt(3)*R/2, 0, 0, np.sqrt(3)*R/2,
                          np.sqrt(3)*R
                    ,np.sqrt(3)*R,
                 np.sqrt(3)*R/2])
    
    u_xyz[:,0] -= R
    u_xyz[:,1] -= R 
    
    u_xyz[:,0] += center_point[0]
    u_xyz[:,1] += center_point[1]
    
    hexagon_x -=R
    hexagon_y -= R 
    
    hexagon_x += center_point[0]
    hexagon_y += center_point[1]
    
    hexagon_xy = np.column_stack((hexagon_x, hexagon_y))
    return u_xyz[:n_points], hexagon_xy



def create_hexagonal_grid(n_tier:int, R:float, n_ue_per_cell:int = 10,
                          center_cell_xy:list()=[0,0], enable_plot:bool = False):
    """
    Parameters
    ----------
    n_tier : int
       number tier from center
    R : float
       radius of one hexagonal cell
    n_ue_per_cell: int
        number of users per hexagonal cell. The default is 10
    center_cell_xy : list(), optional
       2D location of center cell among hexagonal grids. The default is [0,0].
    enable_plot : bool, optional
      decide if plotting or not. The default is False.

    Returns
    -------
    xy_list : array
       locations of all UEs dropped in hexagonal grids.
    center_point_list : TYPE
        center location of each cell
    hex_xy_list : array
        locations of hexagonal vertices
    """
    
    
    #colors = ['k','b','r','g','m','c','y',
    #          'tab:gray', 'tab:blue', 'darkblue','red', 'b','g','c']
    colors = ['k']*n_tier
    xyz_list = []
    hex_xy_list = []
    center_point_list = []
    
    for n in range(n_tier):
        R_n = np.sqrt(3)*R*n
        c_ = colors[n]
        if n==0:
            x, y = 0 + center_cell_xy[0],0+center_cell_xy[1]
            ue_xyz, hex_xy = drop_uniform_in_hexagon(R, n_ue_per_cell, center_point=[x,y])
            
            if enable_plot is True:
                plt.scatter(ue_xyz[:,0], ue_xyz[:,1], s= 5, c= c_)
                plt.plot(hex_xy[:,0], hex_xy[:,1], c_)
                plt.scatter(x,y, c = 'k', s= 8)
                #plt.text(x, y, n+1,c = c_)
            
            xyz_list.append(ue_xyz)
            hex_xy_list.append(hex_xy)
            center_point_list.append([x,y,0])
            
        elif n>=1:
            theta = np.pi/2 
            prev_x, prev_y = np.cos(theta)*R_n, np.sin(theta)*R_n  
            for j  in range(6):
                theta = np.pi/2 + (j+1)*np.pi/3 
                x, y = center_cell_xy[0]+ np.cos(theta)*R_n, center_cell_xy[1]+np.sin(theta)*R_n
                ue_xyz, hex_xy = drop_uniform_in_hexagon(R,n_ue_per_cell, center_point=[x,y])
                
                if enable_plot is True:
                    plt.scatter(ue_xyz[:,0], ue_xyz[:,1], s =5, c= c_)
                    plt.plot(hex_xy[:,0], hex_xy[:,1], c_)
                    plt.scatter(x,y, c = 'k', s= 8)
                    #plt.text(x, y, n+1,c =c_)
                
                
                for m in range(1,n):
                    x_new = (m*x + (n-m)*prev_x)/n + center_cell_xy[0]
                    y_new = (m*y + (n-m)*prev_y)/n + center_cell_xy[1]
                    ue_xyz_new, hex_xy_new = drop_uniform_in_hexagon(R,n_ue_per_cell, center_point=[x_new,y_new])
                   
                    if enable_plot is True:
                        plt.scatter(ue_xyz_new[:,0], ue_xyz_new[:,1],s = 5,c=c_)
                        plt.plot(hex_xy_new[:,0], hex_xy_new[:,1], c_)
                        plt.scatter(x_new,y_new, c = 'k', s =8)
                        #plt.text(x_new, y_new, n+1, c=c_)
                    
                    xyz_list.append(ue_xyz_new)
                    hex_xy_list.append(hex_xy_new)
                    center_point_list.append([x_new,y_new,0])
                
                xyz_list.append(ue_xyz)
                hex_xy_list.append(hex_xy)
                center_point_list.append([x,y,0])
                    
                prev_x, prev_y = x, y
                
    return xyz_list, center_point_list, hex_xy_list
                    
        
            
    