# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:29:27 2022

@author: seongjoon kang
"""
import numpy as np
import scipy.special as sp
import scipy.constants as sc

# in the following functions, all the angles are in radians
# all input angles are radian, and all return angles are radian 

def GCS_LCS_conversion(rot_angle:dict(), theta:list(), phi:list()):
    #GCS to LCS conversion function following 3GPP 38.901, 7.1-7, 7.1-8, and 7.1-15
    # set rotation angles of UE
    # alpha: bearing angle, rotation about z axis
    # beta: downtilt angle, rotation about y axis
    # gamma: slant angle, rotation about x axis
    
    # theta, phi are radians
    # alpha, beta, gamma are all radians
    alpha, beta, gamma = rot_angle['alpha'], rot_angle['beta'], rot_angle['gamma']
    
    
    theta_prime = np.arccos(np.cos(beta)*np.cos(gamma)*np.cos(theta) + 
                            (np.sin(beta)*np.cos(gamma)*np.cos(phi-alpha) - 
                             np.sin(gamma)*np.sin(phi-alpha))*np.sin(theta)) #7.1-7
    a_jb = (np.cos(beta)*np.sin(theta)*np.cos(phi-alpha) - np.sin(beta)*np.cos(theta))\
        +1j*(np.cos(beta)*np.sin(gamma)*np.cos(theta) 
             + (np.sin(beta)*np.sin(gamma)*np.cos(phi-alpha) 
                + np.cos(gamma)*np.sin(phi-alpha))*np.sin(theta))
    phi_prime = np.angle(a_jb) # 7.1-8
    
    a_jb2 = (np.sin(gamma)*np.cos(theta)*np.sin(phi-alpha) + np.cos(gamma)*
             (np.cos(beta)*np.sin(theta) - np.sin(beta)*np.cos(theta)*np.cos(phi-alpha)))\
        + 1j*(np.sin(gamma)*np.cos(phi-alpha) + np.sin(beta)*np.cos(gamma)*np.sin(phi-alpha))
    Psi = np.angle(a_jb2) # 7.1-15
    
    # return local angles and rotation angle
    local_angle = {'theta_prime':theta_prime, 'phi_prime':phi_prime}
    # return angles are all radians
    return local_angle, Psi

def sat_antenna_power_pattern(theta:list(), radius:float, f_c:float=2e9):
    # antenna element gain by 3GPP 38.811, 6.4.1
    
    # theta, phi are radians
    theta = np.squeeze(theta)
    k = 2*np.pi*f_c/sc.speed_of_light
    
    g = np.zeros(len(theta))
    g[theta==0] = 1
    y = sp.jv(1, k*radius*np.sin(theta[theta !=0]))
    x = k*radius*np.sin(theta[theta !=0])
    g[theta !=0 ] = 4 *np.abs(y/x)**2
    # return power, i.e., element gain of one antenna

    return g

def ue_antenna_power_pattern(theta:list(), phi:list(), ant_pattern:str = "quasi-isotropic"):
    # we assume that UE element pattern is quasi-Isotropic according to 3GPP 38.811, 6.4.2
    # 3GPP 38.811,6.10.1, Note 4
    # Quasi isotropic refers to dipole antenna which is omini-directional in one plane
    # let's use very simple one, sin(theta)^2,
    # which is omini-directional across all azimuth angles, phi
    
    if ant_pattern == 'isotropic':
        g =  [1]*len(theta)
    elif ant_pattern == 'quasi-isotropic':
        g = np.sin(theta)**2
       
    return g

def sph2cart(r:list(), theta:list(), phi:list()):
     # all input angles are radian
     return np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])

def get_sat_antenna_field_pattern(theta:list(), phi:list(),
                                   beam_direction:list(), radius:float, f_c:float
                                   , return_gain:bool = False):
    # theta, phi are radians
    # beam_direction: [x,y,z], boresight direction of satellite antenna
    xyz_array = sph2cart(1, theta, phi) # (3, number of thetas)
    beam_direction = beam_direction/np.linalg.norm(beam_direction)
    # get angle between departure ray and boresight direction of satelliet antenna
    theta = np.arccos(xyz_array.T.dot(beam_direction))
    g = sat_antenna_power_pattern(theta, radius, f_c)
    # relationship between power pattern and field pattern 
    # g = |F_theta|^2 + |F_phi|^2 # 3GPP 38.901, 7.3.2
    # assume circular polarization 
    F_theta = np.sqrt(g/2)
    F_phi = 1j*np.sqrt(g/2)
    
    F_vec = np.vstack((F_theta, F_phi))
    
    if return_gain is False:
        return F_vec
    else:
        return F_vec, g

def get_ue_antenna_field_pattern(theta:list(), phi:list(), rot_angle:dict()
                                 , slant_angle:float, ant_pattern:str = 'quasi-isotropic'):
    # theta, phi are radians
    
    # first get local angle in the coordinate system of UE
    local_angle, Psi_prime = GCS_LCS_conversion(rot_angle, theta, phi)
    theta_prime, phi_prime = local_angle['theta_prime'], local_angle['phi_prime']
    
    # do one more converstion to consider two-cross polarization antenna 
    # with a certain slant angle
    rot_angle2 = dict()
    rot_angle2['alpha'] =0
    rot_angle2['beta'] = slant_angle
    rot_angle2['gamma'] = 0
    local_angle2, Psi_prime2 = GCS_LCS_conversion(rot_angle2, theta_prime, phi_prime)
    
    theta_two_prime, phi_two_prime = local_angle2['theta_prime'], local_angle2['phi_prime']
    
    # get power pattern with the obtained local angles
    g = ue_antenna_power_pattern(theta_two_prime, phi_two_prime, ant_pattern=ant_pattern)
    # relationship between power pattern and field pattern 
    # g = |F_theta|^2 + |F_phi|^2 # 3GPP 38.901, 7.3.2
    
    # in LCS2, we assume purely vertical polarization, those F_phi_two_prime =0
    F_phi_two_prime = np.repeat([0], len(theta))
    F_theta_two_prime = np.sqrt(g)
    F_vec_prime2 = np.vstack((F_theta_two_prime, F_phi_two_prime))
    
    F_vec_list = np.zeros((2,len(theta)))
    for i in range(len(theta)):
        # go back from LCS2 -> LCS1 -> GCS using Psi_prime2 and Psi_prime
        # considering 3GPP 38.901, 7.3-3, 7.1-11
        # LCS2 -> LCS1
        F_vec_prime_i = np.array([[np.cos(Psi_prime2[i]), -np.sin(Psi_prime2[i])],
                                [np.sin(Psi_prime2[i]), np.cos(Psi_prime2[i])]]).dot(F_vec_prime2[:,i])
        
        # LCS1 -> GCS
        
        F_vec = np.array([[np.cos(Psi_prime[i]), -np.sin(Psi_prime[i])],
                            [np.sin(Psi_prime[i]), np.cos(Psi_prime[i])]]).dot(F_vec_prime_i)
        
        F_vec_list[:,i] = F_vec
        
    return F_vec_list