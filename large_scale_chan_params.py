# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:32:18 2022

@author: seongjoon kang
"""
import numpy as np
import scipy.constants as sc
import three_gpp_tables_s_band as th_gpp_T
from get_antenna_field_patterns import get_ue_antenna_field_pattern, get_sat_antenna_field_pattern


class sat_three_gpp_channel_model(object):

    def __init__(self, scenario: str, sat_location: list(), ue_location: list(),
                 sat_beam_direction: list(), f_c: float, ue_ant_pattern: str = 'quasi-isotropic'):

        # set scenarios and locations of satellite and ue

        self.scenario = scenario
        self.sat_location = sat_location
        self.sat_beam_direction = sat_beam_direction
        self.ue_location = ue_location
        self.sat_ant_radius = 1
        self.f_c = f_c
        self.ue_ant_pattern = ue_ant_pattern
        self.r_velocity = np.repeat([[1, 1, 1]], len(self.ue_location), axis=0)
        self.ant_slant_angle_list = [45, -45]
        self.pol_type_list = ['RHCP', 'LHCP']
        self.time = 0
        self.force_LOS = False

    def compute_LOS_angle(self, Tx_loc: list(), Rx_loc: list()):
        dist_vec = Rx_loc - Tx_loc

        R_3d = np.sqrt(dist_vec[:, 0] ** 2 + dist_vec[:, 1] ** 2 + dist_vec[:, 2] ** 2)
        # R_2d = np.sqrt(dist_vec[:,0]**2 + dist_vec[:,1]**2)
        # azimuth angle of departure
        LOS_AOD_phi = np.arctan2(dist_vec[:, 1], dist_vec[:, 0]) * 180 / np.pi
        # LOS_AOD_phi[LOS_AOD_phi<0] += 360
        # LOS_AOD_phi[LOS_AOD_phi>360] -=360
        # zenith angle of departure
        LOS_ZOD_theta = np.arccos(dist_vec[:, 2] / R_3d) * 180 / np.pi
        # azimuth angle of arrival
        LOS_AOA_phi = LOS_AOD_phi
        # LOS_AOA_phi[LOS_AOA_phi<0] += 360
        # LOS_AOA_phi[LOS_AOA_phi>360] -=360
        # zenith angle of arrival
        LOS_ZOA_theta = 180 - LOS_ZOD_theta

        return LOS_AOD_phi, LOS_ZOD_theta, LOS_AOA_phi, LOS_ZOA_theta

    def step1(self, return_sat_gain=False):

        # b) Give number of sat and UE
        # n_sat = 1
        n_UE = len(self.ue_location)

        # sat_location = np.array([20,30, 6000])
        # UE_location_xy = np.random.uniform(low = -200, high = 200, size = (n_UE,2))
        # UE_z = np.repeat([0], n_UE)
        # UE_location = np.column_stack((UE_location_xy, UE_z))

        # the unit of LOS angles is degree
        LOS_AOD, LOS_ZOD, LOS_AOA, LOS_ZOA = \
            self.compute_LOS_angle(self.sat_location, self.ue_location)

        sat_height = self.sat_location[0][2]

        R_E = 6371e3
        d = np.linalg.norm(self.ue_location - self.sat_location, axis=1)

        self.elev_angle = np.arcsin((sat_height ** 2 + 2 * sat_height * R_E - d ** 2) / (2 * R_E * d))  # [rad]
        self.elev_angle = np.rad2deg(self.elev_angle)  # [deg]
        self.elev_angle_ref = self.find_near_ref_angle(self.elev_angle)

        self.LOS_angles = [{'AOD': LOS_AOD[i], 'AOA': LOS_AOA[i], 'ZOD': LOS_ZOD[i], 'ZOA': LOS_ZOA[i]} \
                           for i in range(n_UE)]
        if return_sat_gain is True:
            LOS_AOD = np.array(LOS_AOD)
            LOS_ZOD = np.array(LOS_ZOD)
            _, sat_ant_gain_los = get_sat_antenna_field_pattern(np.array([LOS_ZOD * np.pi / 180]),
                                                                np.array([LOS_AOD * np.pi / 180]),
                                                                self.sat_beam_direction,
                                                                self.sat_ant_radius,
                                                                self.f_c,
                                                                return_gain=True)

            return sat_ant_gain_los

    def find_near_ref_angle(self, angle: list()):
        # find reference elevation angle that is nearest to original one
        angle = np.array(angle)
        angles2 = np.zeros_like(angle)
        angles2[angle < 10] = 10
        angles2[angle >= 10] = np.round(angle[angle >= 10] / 10) * 10
        return angles2

    def get_link_state(self, elev_angle: list(), scenario='rural'):
        # return link states for each elevation angle
        elev_angle_ref = self.find_near_ref_angle(elev_angle)
        if self.force_LOS is False:
            return th_gpp_T.get_los_prob(scenario, elev_angle_ref)
        else:
            return np.repeat([100], len(elev_angle_ref))

    def step2_and_step3(self):
        # return path loss values corresponding to elevation angles and linnk states
        R_E = 6371e3  # [m]

        height = self.sat_location[:, 2]
        los_prob = self.get_link_state(self.elev_angle, self.scenario) / 100
        uniform_rand_v = np.random.uniform(0, 1, len(los_prob))
        link_state = np.ones_like(los_prob)

        for i, link_state_i in enumerate(link_state):
            los_prob_i = los_prob[i]
            uniform_rand_v_i = uniform_rand_v[i]
            if uniform_rand_v_i > los_prob_i:  #
                link_state[i] = 2

        # d = np.sqrt(R_E**2 * np.sin(np.deg2rad(self.elev_angle))**2 + height**2 + 2*height*R_E)- R_E*np.sin(np.deg2rad(self.elev_angle))
        distance = np.linalg.norm(self.ue_location - self.sat_location, axis=1)  # [m]
        # f_c in Ghz, d in m
        fspl = 32.45 + 20 * np.log10(self.f_c / 1e9) + 20 * np.log10(distance)
        # find ref elevation angle
        shadow_CL_table = th_gpp_T.get_shadowing_params(self.scenario, self.elev_angle_ref)

        # take LOS sigma values
        sigma_los = shadow_CL_table[:, 0][link_state == 1]
        # NLOS sigma and CL
        sigma_nlos = shadow_CL_table[:, 1][link_state == 2]
        CL = shadow_CL_table[:, 2][link_state == 2]

        # compute basic pathloss
        PL_b = np.zeros_like(fspl)
        PL_b[link_state == 1] = fspl[link_state == 1] \
                                + np.random.normal(0, sigma_los, np.sum(link_state == 1))
        PL_b[link_state == 2] = fspl[link_state == 2] \
                                + np.random.normal(0, sigma_nlos, np.sum(link_state == 2)) + CL

        ## attenuation due to atmophere following ITU-R P.676
        # https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-13-202208-I!!PDF-E.pdf
        T = 288.15  # unit is K
        p = 1013.25  # hPa
        ro = 7.5  # 7.5g / m^3
        e = 9.98  # hPa, additonal pressuer due to water vapor
        # specific gaseous attenuation attributable to oxygen, in dB/km
        # read coefficient from
        # https://www.itu.int/dms_pub/itu-r/oth/11/01/R11010000020001TXTM.txt
        if self.f_c < 10e9:
            # take values corresponding to 2GHz
            a = -2.37
            b = 2.71e-2
            c = -2.53e-4
            d = -2.83e-4

            # Rec. ITU R  P.676 13, FIGURE 1
            ro_gas = 0.0069  # [dB/Km]  assume approximated value

        else:
            # take values corresponding to 28GHz
            a = -2.43
            b = 2.8e-2
            c = -6.168e-4
            d = -9.517e-4

            # Rec. ITU R  P.676 13, FIGURE 1
            ro_gas = 0.1104  # [dB/Km]  assume approximated value

        h_0 = a + b * T + c * p + d * ro

        A_zenith = (ro_gas * h_0)
        PL_a = A_zenith / np.sin(np.deg2rad(self.elev_angle))
        # PL_a = 10*np.log10(PL_a)
        path_loss = PL_b + PL_a
        # we will ignore the attenuation
        # due to either ionospheric or trospheric scintillation loss
        self.path_loss = path_loss
        self.link_state = link_state

    def create_corr_matrix(self, T: dict(), link_state: int):
        # return one correlation matrix corresponding one link state
        C = np.array([
            [T['SF_SF'], T['SF_K'], T['SF_DS'], T['SF_ASD'], T['SF_ASA'], T['SF_ZSD'], T['SF_ZSA']],
            [T['K_SF'], T['K_K'], T['K_DS'], T['K_ASD'], T['K_ASA'], T['K_ZSD'], T['K_ZSA']],
            [T['DS_SF'], T['DS_K'], T['DS_DS'], T['DS_ASD'], T['DS_ASA'], T['DS_ZSD'], T['DS_ZSA']],
            [T['ASD_SF'], T['ASD_K'], T['ASD_DS'], T['ASD_ASD'], T['ASD_ASA'], T['ASD_ZSD'], T['ASD_ZSA']],
            [T['ASA_SF'], T['ASA_K'], T['ASA_DS'], T['ASA_ASD'], T['ASA_ASA'], T['ASA_ZSD'], T['ASA_ZSA']],
            [T['ZSD_SF'], T['ZSD_K'], T['ZSD_DS'], T['ZSD_ASD'], T['ZSD_ASA'], T['ZSD_ZSD'], T['ZSD_ZSA']],
            [T['ZSA_SF'], T['ZSA_K'], T['ZSA_DS'], T['ZSA_ASD'], T['ZSA_ASA'], T['ZSA_ZSD'], T['ZSA_ZSA']]
        ])

        if link_state == 2:  # if link state is NLOS , remove K
            C = np.delete(C, 1, axis=0)
            C = np.delete(C, 1, axis=1)
        return np.linalg.cholesky(C)

    def step4(self):
        # Step 4: Generate large scale parameters, e.g. delay spread (DS),
        # angular spreaDS (ASA, ASD, ZSA, ZSD), Ricean K factor (K)
        # and shadow fading (SF) taKing into account
        # cross correlation according to Table 7.5-6
        params_set = {'SF', 'K', 'DS', 'ASD', 'ASA', 'ZSD', 'ZSA'}
        T = dict()
        for p1 in params_set:
            for p2 in params_set:
                if p1 == p2:
                    T[p1 + '_' + p2] = 1
                else:
                    T[p1 + '_' + p2] = np.nan
        # write data from 3gpp table

        correlation_matrix, spread_matrix = th_gpp_T.get_correlation_spread(self.scenario, self.link_state,
                                                                            self.elev_angle_ref)
        shadow_CL_table = th_gpp_T.get_shadowing_params(self.scenario, self.elev_angle_ref)

        mu_SF = 0;

        self.SF_list, self.K_list, self.DS_list = [], [], []
        self.ASA_list, self.ASD_list = [], []
        self.ZSA_list, self.ZSD_list = [], []

        for i, link_state_i in enumerate(self.link_state):
            corr_matrix_i = correlation_matrix[:, i]
            mu_DS, sig_DS, mu_ASD, sig_ASD, mu_ASA, sig_ASA, \
                mu_ZSA, sig_ZSA, mu_ZSD, sig_ZSD, mu_K, sig_K = spread_matrix[:, i]
            sig_SF_LOS, sig_SF_NLOS, _ = shadow_CL_table[i]

            # NOTE 8 from 3GPP 38.811 6.7.2
            # For satellite (GEO/LEO), the departure angle spreads are zeros, i.e.,
            # mu_asd and mu_zsd = -inf, -inf
            mu_ASD, mu_ZSD = -1 * np.inf, -1 * np.inf

            keys = ['ASD_DS', 'ASA_DS', 'ASA_SF', 'ASD_SF', 'DS_SF', 'ASD_ASA', 'ASD_K', 'ASA_K',
                    'DS_K', 'SF_K', 'ZSD_SF', 'ZSA_SF', 'ZSD_K', 'ZSA_K', 'ZSD_DS', 'ZSA_DS',
                    'ZSD_ASD', 'ZSA_ASD', 'ZSD_ASA', 'ZSA_ASA', 'ZSD_ZSA']
            for j, key in enumerate(keys):
                s = key.split('_')
                key2 = s[-1] + '_' + s[0]
                T[key] = corr_matrix_i[j]
                T[key2] = T[key]
            # create cross-correlation matrix with cholesky decomposition
            sqrt_C = self.create_corr_matrix(T, link_state_i)

            if link_state_i == 1:
                sf_k_ds_asd_asa_zsd_zsa = sqrt_C.dot(np.random.normal(0, 1, size=(7, 1)))

                # Note that SF and K are in dB-scale, and others are in linear scale
                DS = 10 ** (mu_DS + sig_DS * sf_k_ds_asd_asa_zsd_zsa[2])
                K = 10 ** (0.1 * (mu_K + sig_K * sf_k_ds_asd_asa_zsd_zsa[1]))  # dB -> linear

                SF = mu_SF + sig_SF_LOS * sf_k_ds_asd_asa_zsd_zsa[0]  # keep dB-scale
                ASD = 10 ** (mu_ASD + sig_ASD * sf_k_ds_asd_asa_zsd_zsa[3])
                ASA = 10 ** (mu_ASA + sig_ASA * sf_k_ds_asd_asa_zsd_zsa[4])
                ZSD = 10 ** (mu_ZSD + sig_ZSD * sf_k_ds_asd_asa_zsd_zsa[5])
                ZSA = 10 ** (mu_ZSA + sig_ZSA * sf_k_ds_asd_asa_zsd_zsa[6])

            else:
                sf_ds_asd_asa_zsd_zsa = sqrt_C.dot(np.random.normal(0, 1, size=(6, 1)))
                DS = 10 ** (mu_DS + sig_DS * sf_ds_asd_asa_zsd_zsa[1])  # dB -> linear
                SF = mu_SF + sig_SF_NLOS * sf_ds_asd_asa_zsd_zsa[0]  # keep dB-scale
                ASD = 10 ** (mu_ASD + sig_ASD * sf_ds_asd_asa_zsd_zsa[2])
                ASA = 10 ** (mu_ASA + sig_ASA * sf_ds_asd_asa_zsd_zsa[3])
                ZSD = 10 ** (mu_ZSD + sig_ZSD * sf_ds_asd_asa_zsd_zsa[4])
                ZSA = 10 ** (mu_ZSA + sig_ZSA * sf_ds_asd_asa_zsd_zsa[5])
                K = np.nan;

            # limit random RMS azimuth arrival and azimuth departure spread values
            # to 104 degree
            ASA = min(ASA, 104)
            ASD = min(ASD, 104)
            # in the same way, limit zenith angles to 52
            ZSA = min(ZSA, 52)
            ZSD = min(ZSD, 52)

            self.SF_list.append(SF)
            self.K_list.append(K)
            self.DS_list.append(DS)
            self.ASD_list.append(ASD)
            self.ASA_list.append(ASA)
            self.ZSA_list.append(ZSA)
            self.ZSD_list.append(ZSD)

    def step5_and_step6(self):
        # return multi-tap delay corresponding to each cluster

        # delays are drawn randomly defined in Table 7.5-6

        # r_tau values are the same across all the elevation angles

        r_tau = th_gpp_T.get_delay_scaling_factor(self.scenario, self.link_state, self.elev_angle_ref)
        n_cluster, n_ray = th_gpp_T.get_n_cluser_ray(self.scenario, self.link_state, self.elev_angle_ref)

        cluster_shadowing_std = th_gpp_T.get_per_cluster_shadowing_std(self.scenario, self.link_state,
                                                                       self.elev_angle_ref)
        # step 5
        self.tau_n_list = []
        for i, link_state_i in enumerate(self.link_state):
            tau_n = np.zeros(n_cluster[i])

            X_n = np.random.uniform(low=0, high=1, size=n_cluster[i]);
            tau_n = -1 * r_tau[i] * self.DS_list[i] * np.log(X_n)
            tau_n = np.sort(tau_n - min(tau_n))  # 3gpp, 38.901, 7.5-2

            if link_state_i == 1:  # if LOS
                # in case of LOS condition, additonal scailing of delays is requried to compensate
                # for the effect of LOS peak addition to the delay spread
                K_dB_i = 10 * np.log10(self.K_list[i])
                C_tau = 0.7705 - 0.0433 * K_dB_i + 0.0002 * K_dB_i ** 2 + 0.000017 * K_dB_i ** 3  # 7.5-3
                tau_n = tau_n / C_tau

            self.tau_n_list.append(tau_n)

        # step 6, Generate cluster powers
        # cluster powers are cacluated assuming a single slop exponential delay profile
        self.P_n_list = []
        self.P_n_list_without_K = []
        for i, link_state_i in enumerate(self.link_state):
            Z_n = np.random.normal(0, 3)
            P_n_prime = np.exp(-self.tau_n_list[i] * (r_tau[i] - 1) / (r_tau[i] * self.DS_list[i])) * 10 ** (-Z_n / 10)
            # normalize the cluster power so that the sum of all cluster power is equal to one
            P_n = P_n_prime / np.sum(P_n_prime)
            self.P_n_list_without_K.append(list(P_n))

            if link_state_i == 1:  # if LOS
                P_n *= 1/(self.K_list[i]+1)
                P_n[0] += self.K_list[i] / (self.K_list[i] + 1)

            self.P_n_list.append(list(P_n))

            # remove cluster with less than -25 dB power compared to the maximum cluster power
        ####################
        ##############################################3

    def step7_step8_step9(self):

        # ray offset angle given in 3GPP 38.901, table 7.5-3
        ray_offset_angles = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715,
                             0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481,
                             1.5195, -1.5195, 2.1551, -2.1551]

        # Create dictionary to save angles for all clusters and rays per each link
        self.angle_data = [{'AOA_cluster_rays': None, 'AOD_cluster_rays': None,
                            'ZOA_cluster_rays': None, 'ZOD_cluster_rays': None,
                            'XPR': None} for i in range(len(self.link_state))]
        # Generate arrival angeles and departure angles for both azimuth and elevation
        C_phi_list, C_theta_list = th_gpp_T.get_scaling_factors_for_phi_theta()

        mu_XPR, sigma_XPR = th_gpp_T.get_XPR_params(self.scenario, self.link_state, self.elev_angle_ref)
        # the cluster-spread values fro 3gpp 38.811 tables
        cluster_spreads = th_gpp_T.get_cluster_spread(self.scenario, self.link_state, self.elev_angle_ref)
        C_ASA_list = cluster_spreads['C_ASA']
        C_ASD_list = cluster_spreads['C_ASD']
        C_ZSA_list = cluster_spreads['C_ZSA']
        self.C_DS_list = cluster_spreads['C_DS']

        # for all links that have several clusters
        for i, link_state_i in enumerate(self.link_state):
            n_cluster = len(self.P_n_list[i])
            # get power distribution across all clusters
            P_clusters = self.P_n_list[i]  # shape = (n_cluster, )

            ############## generate azimuth angle for all clusters and rays ##########################
            # find the scaling factor corresponding to total number of clusters
            C_phi_nlos = C_phi_list[n_cluster]

            if link_state_i == 1:
                C_phi = C_phi_nlos * (1.1035 - 0.028 * self.K_list[i]
                                      - 0.002 * self.K_list[i] ** 2 + 0.0001 * self.K_list[i] ** 3)
            else:
                C_phi = C_phi_nlos

            phi_clusters_AOA_prime = 2 * (self.ASA_list[i] / 1.4) * np.sqrt(
                -np.log(P_clusters / max(P_clusters))) / C_phi
            phi_clusters_AOD_prime = 2 * (self.ASD_list[i] / 1.4) * np.sqrt(
                -np.log(P_clusters / max(P_clusters))) / C_phi
            # Assign positive or negative sign to the angles by multiplying with
            # a random variable X_n ~unif(-1,1)
            # add component Y_n ~ N(0, (ASA/7)^2) to introduce random variation

            # 1) AOA
            X_clusters = np.random.uniform(-1, 1, n_cluster)
            Y_clusters = np.random.normal(0, self.ASA_list[i] / 7, n_cluster)
            LOS_AOA_phi = self.LOS_angles[i]['AOA']
            # get arrival azimuth angle of all clusters, shape = (n_cluster, 1)
            phi_AOA_clusters = X_clusters * phi_clusters_AOA_prime + Y_clusters + LOS_AOA_phi

            if link_state_i == 1:
                phi_AOA_clusters = X_clusters * phi_clusters_AOA_prime + Y_clusters - \
                                   (X_clusters[0] * phi_clusters_AOA_prime[0] + Y_clusters[0] - LOS_AOA_phi)
            # 2) AOD
            X_clusters = np.random.uniform(-1, 1, n_cluster)
            Y_clusters = np.random.normal(0, self.ASD_list[i] / 7, n_cluster)
            LOS_AOD_phi = self.LOS_angles[i]['AOD']
            # get arrival azimuth angle of all clusters, shape = (n_cluster, 1)
            phi_AOD_clusters = X_clusters * phi_clusters_AOD_prime + Y_clusters + LOS_AOD_phi
            if link_state_i == 1:
                phi_AOD_clusters = X_clusters * phi_clusters_AOD_prime + Y_clusters - \
                                   (X_clusters[0] * phi_clusters_AOD_prime[0] + Y_clusters[0] - LOS_AOD_phi)

            # finally add offset angles from 3gpp 38.901 table 7.5-3 to cluster angles

            # phi_AOA_cluster_rays shape = (n_ray, n_cluster)
          
            phi_AOA_cluster_rays = phi_AOA_clusters + C_ASA_list[i] * np.array(ray_offset_angles)[:,
                                                                      None]  # (n_ray, n_cluster)
            phi_AOD_cluster_rays = phi_AOD_clusters + C_ASD_list[i] * np.array(ray_offset_angles)[:,
                                                                      None]  # (n_ray, n_cluster)

            phi_AOA_cluster_rays[phi_AOA_cluster_rays > 360] -= 360
            phi_AOA_cluster_rays[phi_AOA_cluster_rays < 0] += 360

            phi_AOD_cluster_rays[phi_AOD_cluster_rays > 360] -= 360
            phi_AOD_cluster_rays[phi_AOD_cluster_rays < 0] += 360

            ############## generate zenith angle for all clusters and rays ##########################
            # zenith angles are determined by applying the inverse Laplacian function
            # with input parameter P_n and RMS angle spread ZSA
            # find the scaling factor corresponding to total number of clusters
            C_theta_nlos = C_theta_list[n_cluster]
            if link_state_i == 1:
                C_theta = C_theta_nlos * (1.3086 - 0.0339 * self.K_list[i] - 0.0077 * self.K_list[i] ** 2
                                          + 0.0002 * self.K_list[i] ** 3)
            else:
                C_theta = C_theta_nlos

            theta_clusters_ZOA_prime = - self.ZSA_list[i] * np.log(P_clusters / max(P_clusters)) / C_theta
            theta_clusters_ZOD_prime = - self.ZSD_list[i] * np.log(P_clusters / max(P_clusters)) / C_theta

            # 1) ZOA
            X_clusters = np.random.uniform(-1, 1, n_cluster)
            Y_clusters = np.random.normal(0, self.ZSA_list[i] / 7, n_cluster)
            LOS_ZOA_theta = self.LOS_angles[i]['ZOA']

            # get arrival zenith angle of all clusters, shape = (n_cluster, 1)
            theta_ZOA_clusters = X_clusters * theta_clusters_ZOA_prime + Y_clusters + LOS_ZOA_theta

            if link_state_i == 1:  # LOS
                theta_ZOA_clusters = X_clusters * theta_clusters_ZOA_prime + Y_clusters - \
                                     (X_clusters[0] * theta_clusters_ZOA_prime[0] + Y_clusters[0] - LOS_ZOA_theta)

            # 2) ZOD
            X_clusters = np.random.uniform(-1, 1, n_cluster)
            Y_clusters = np.random.normal(0, self.ZSD_list[i] / 7, n_cluster)
            LOS_ZOD_theta = self.LOS_angles[i]['ZOD']

            # get arrival zenith angle of all clusters, shape = (n_cluster, 1)
            mu_offset_ZOD = 0;
            theta_ZOD_clusters = X_clusters * theta_clusters_ZOD_prime + Y_clusters \
                                 + LOS_ZOD_theta + mu_offset_ZOD

            if link_state_i == 1:  # LOS
                theta_ZOD_clusters = X_clusters * theta_clusters_ZOD_prime + Y_clusters - \
                                     (X_clusters[0] * theta_clusters_ZOD_prime[0] + Y_clusters[0] - LOS_ZOD_theta)

            c_ZSA = C_ZSA_list[i]

            # when satellites transmit data to Earth
            # we don't need the departure zenith angle spread per cluster
            # c_ZSD = cluster_spreads['C_ZSD']

            # theta_ZOA_cluster_rays shape = (n_ray, n_cluster)
            theta_ZOA_cluster_rays = theta_ZOA_clusters + c_ZSA * np.array(ray_offset_angles)[:,
                                                                  None]  # (n_ray, n_cluster)

            _, spread_values = th_gpp_T.get_correlation_spread(self.scenario, [link_state_i], [self.elev_angle_ref[i]])
            mu_lg_ZSD = spread_values[-4]
            # 38.901, 7.5-20
            theta_ZOD_cluster_rays = theta_ZOD_clusters + (3 / 8) * (10 ** (mu_lg_ZSD)) * np.array(ray_offset_angles)[:,
                                                                                          None]  # (n_ray, n_cluster)

            theta_ZOA_cluster_rays[theta_ZOA_cluster_rays > 360] -= 360
            theta_ZOA_cluster_rays[theta_ZOA_cluster_rays < 0] += 360
            theta_ZOD_cluster_rays[theta_ZOD_cluster_rays > 360] -= 360
            theta_ZOD_cluster_rays[theta_ZOD_cluster_rays < 0] += 360

            theta_ZOA_cluster_rays[theta_ZOA_cluster_rays > 180] = 360 - theta_ZOA_cluster_rays[
                theta_ZOA_cluster_rays > 180]
            theta_ZOD_cluster_rays[theta_ZOD_cluster_rays > 180] = 360 - theta_ZOD_cluster_rays[
                theta_ZOD_cluster_rays > 180]

            # step 8
            # couple randomly AOD, AOA, ZOD, and ZOA within a cluster
            np.random.shuffle(phi_AOA_cluster_rays)
            np.random.shuffle(phi_AOD_cluster_rays)
            np.random.shuffle(theta_ZOA_cluster_rays)
            np.random.shuffle(theta_ZOD_cluster_rays)

            # save all angles, AOA, AOD, ZOA, and ZOD (shape=[n_rays, n_cluster])
            # corresponding to each link
            self.angle_data[i]['AOA_cluster_rays'] = phi_AOA_cluster_rays
            self.angle_data[i]['AOD_cluster_rays'] = phi_AOD_cluster_rays
            self.angle_data[i]['ZOA_cluster_rays'] = theta_ZOA_cluster_rays
            self.angle_data[i]['ZOD_cluster_rays'] = theta_ZOD_cluster_rays

            # save angles for all clusters
            self.angle_data[i]['AOA_cluster'] = phi_AOA_clusters
            self.angle_data[i]['AOD_cluster'] = phi_AOD_clusters
            self.angle_data[i]['ZOA_cluster'] = theta_ZOA_clusters
            self.angle_data[i]['ZOD_cluster'] = theta_ZOD_clusters
            # step 9
            # generate the cross polarization ratios (XPR) k for each ray m for each cluster n
            n_ray_i, n_cluster_i = theta_ZOA_cluster_rays.shape
            mu_XPR_i = mu_XPR[i]
            sigma_XPR_i = sigma_XPR[i]

            X = np.random.normal(mu_XPR_i, sigma_XPR_i, (n_ray_i, n_cluster_i))
            K = 10 ** (0.1 * X)
            self.angle_data[i]['XPR'] = K

    def sph2cart(self, r: float, theta: float, phi: float):
        # all input angles are radian
        return np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])

    def step10_step11(self):
        # step 10
        # draw random initial phase {theta_theta, theta_phi, phi_theta, phi_phi} ={TT, TP, PT,PP}
        # shape should be (n_ray, n_cluster)

        # step 11, For N-2 weakest clusters, channel coeffient of one cluster is
        # sum of those of all the rays according to 38.901, 7.5.22
        # For first two strongest clusters,
        # rays are spread in delay to three sub-clusters

        # final return values
        # channels and delays corresponding to each X-pol antenna
        H_per_crossed_ant = {}
        delay_per_crossed_ant = {}
        # self.los_sat_ant_gain = {} # this variable is used for only verification

        # get channel for each antenna from X-pole antennas
        #for slant_angle in zip(self.ant_slant_angle_list, self.pol_type_list):
        delay_n_list = []  # list containing all the cluster delays for each link
        H_list = []  # list containing all channel coefficient per each cluster
        for i in range(len(self.link_state)):

            # assume each UE rotates randomly
            #ue_rot_angle = {'alpha': np.random.uniform(0, 360, 1) * np.pi / 180
            #    , 'beta': np.random.uniform(0, 360, 1) * np.pi / 180,
            #                'gamma': np.random.uniform(0, 360, 1) * np.pi / 180}

            n_rays, n_clusters = self.angle_data[i]['AOA_cluster_rays'].shape

            theta_ZOA, phi_AOA = self.angle_data[i]['ZOA_cluster_rays'], self.angle_data[i]['AOA_cluster_rays']
            theta_ZOD, phi_AOD = self.angle_data[i]['ZOD_cluster_rays'], self.angle_data[i]['AOD_cluster_rays']

            #kappa = self.angle_data[i]['XPR']

            # step 10
            #TT = np.random.uniform(low=-np.pi, high=np.pi, size=(n_rays, n_clusters))
            #TP = np.random.uniform(low=-np.pi, high=np.pi, size=(n_rays, n_clusters))
            #PT = np.random.uniform(low=-np.pi, high=np.pi, size=(n_rays, n_clusters))
            #PP = np.random.uniform(low=-np.pi, high=np.pi, size=(n_rays, n_clusters))

            # step 11
            H_nlos = np.zeros(shape=[n_rays, n_clusters], dtype=complex)
            print(n_rays)
            tau_n_cluster = self.tau_n_list[i]  # delay list for all clusters

            # cluster delay spread of link i
            c_ds = self.C_DS_list[i]
            if np.isnan(c_ds):
                c_ds = 3.91  # cluster spread is N/A, 3.9ns is used by 3GPP 38.901, step 10
            tau_n_cluster_new = []
            H_n_cluster = []
            for n in range(n_clusters):
                # return-shape is (2,n_rays)

                P_n = self.P_n_list_without_K[i][n]

                for k in range(n_rays):

                    H_nlos[k, n] = np.sqrt(P_n / n_rays)

                if n == 0 or n == 1:
                    # For first two strongest clusters,
                    # rays are spread in delay to three sub-clusters

                    # sub cluster 1
                    # n'th column means n'th cluster
                    tau_n_cluster_new.append(tau_n_cluster[n])
                    H_n_cluster.append(np.sum(H_nlos[[0, 1, 2, 3, 4, 5, 6, 7, 18, 19], n]))

                    # sub cluster 2
                    tau_n_cluster_new.append(tau_n_cluster[n] + 1.28 * c_ds * 1e-9)
                    H_n_cluster.append(np.sum(H_nlos[[8, 9, 10, 11, 16, 17], n]))

                    # sub cluster 3
                    tau_n_cluster_new.append(tau_n_cluster[n] + 2.56 * c_ds * 1e-9)
                    H_n_cluster.append(np.sum(H_nlos[[12, 13, 14, 15], n]))
                else:
                    # For N-2 weakest clusters, channel coeffient of one cluster is
                    # sum of those of all the rays according to 38.901, 7.5.22
                    tau_n_cluster_new.append(tau_n_cluster[n])
                    H_n_cluster.append(np.sum(H_nlos[:, n]))

            if self.link_state[i] == 1:

                H_los = 1
                K = np.squeeze(self.K_list[i])
                # 38.901, 7.5-30
                H_n_cluster[0] = np.sqrt(K / (K + 1)) * H_los + np.sqrt(1 / (K + 1)) * H_n_cluster[0]

            # step 12: apply pathloss and shadowing for the channel coefficients
            self.path_loss[i] += np.random.normal(0, abs(self.SF_list[i]))
            path_loss_lin_i = 10 ** (-0.05 * self.path_loss[i])
            H_list.append(path_loss_lin_i * np.array(H_n_cluster))

            delay_n_list.append(tau_n_cluster_new)

        return H_list, delay_n_list

    def run(self):
        # run from step 1 to step 11
        self.step1()
        self.step2_and_step3()
        self.step4()
        self.step5_and_step6()
        self.step7_step8_step9()
        H, delay = self.step10_step11()

        return H, delay


if __name__ == "__main__":
    n_sat = 1
    n_UE = 2

    sat_location = np.array([[20, 30, 600e3]])
    ue_location_xy = np.random.uniform(low=-2e5, high=2e5, size=(n_UE, 2))
    ue_z = np.repeat([0], n_UE)

    ue_location = np.column_stack((ue_location_xy, ue_z))

    f_c = 2e9
    scenario = 'rural'
    sat_beam_direction = [0, 0, 0] - sat_location[0]
    a = sat_three_gpp_channel_model(scenario, sat_location, ue_location, sat_beam_direction, f_c)
    H, delay = a.run()
    print(len(H[0]))
    #a.run()
    print(a.link_state)
    print(a.P_n_list[0])
    print(a.tau_n_list[0])
    print(a.path_loss[0] - 10*np.log10(a.P_n_list[0][0]),a.path_loss[0] - 10*np.log10(a.P_n_list[0][1]))
    print(a.angle_data[0]['AOA_cluster'])
    print(a.angle_data[0]['AOD_cluster'])
    print(a.angle_data[0]['ZOA_cluster'])
    print(a.angle_data[0]['ZOD_cluster'])




