import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math
import os
import sys

import astropy.units as u
import astropy.cosmology as acosmo

from params import p_cosmology

#Volumetric rates per transient

def R_SFR(z, R0=1):
    '''
    From Madau & Dickinson 2014
    '''
    
    # return 0.0068*0.0015*(1 + z)**2.7/(1 + ((1 + z)/2.9)**5.6) * (u.Mpc ** -3 * u.yr ** -1)
    return R0 * (1 + z)**2.7/(1 + ((1 + z)/2.9)**5.6) 

def R_TDE(z, R0=8e-7, eta=-2):
    '''
    Computes the volumetirc rate following Sun++2015
    '''
    term_1 = (1 + z) ** (0.2 * eta)
    term_2 = ((1 + z) / 1.43) ** (-3.2 * eta)
    term_3 = ((1 + z) / 2.66) ** (-7.0 * eta)
    f_TDE = (term_1 + term_2 + term_3) ** (1 / eta)

    return R0 * f_TDE 

def R_sGRB_gaus(z, R0=1.0, eta=-2):
    
    '''
    From Sun++2015. Estiamted empirically from a MC simulation and taking a gaussian time-delay profile to 
    account from the inspiral time from Virgili++2011. Model is normalized to be  1 at redshift z=0, so we multiply with the sGRB (or KN) 
    rate  in the local universe
    '''

    term_1 = (1 + z) **  (5 * eta)
    term_2 = ((1 + z) / 0.17) ** (0.87 * eta)
    term_3 = ((1 + z) / 4.12) ** (-8 * eta)
    term_4 = ((1 + z) / 4.05) **  (-20.5  * eta)

    f_sGRB = (term_1 + term_2 + term_3 + term_4) ** (1 / eta)

    return R0 * f_sGRB

def R_sGRB_lognorm(z, R0=1, eta=-2):
    
    '''
    From Sun++2015. Estiamted empirically from a MC simulation and taking a log-normal time-delay profile to 
    account from the inspiral time from Wanderman & Piran 2014. Model is normalized to be  1 at redshift z=0, so we multiply with the sGRB (or KN) 
    rate  in the local universe
    '''

    term_1 = (1 + z) **  (5.7 * eta)
    term_2 = ((1 + z) / 0.36) ** (1.3 * eta)
    term_3 = ((1 + z) / 3.3) ** (-9.5 * eta)
    term_4 = ((1 + z) / 3.3) **  (-24.5 * eta)

    f_sGRB = (term_1 + term_2 + term_3 + term_4) ** (1 / eta)

    return R0 * f_sGRB

def R_sGRB_powerlaw(z, R0=1, eta=-2):
    
    '''
    From Sun++2015. Estiamted empirically from a MC simulation and taking a power-law time-delay profile to 
    account from the inspiral time from Wanderman & Piran 2014. Model is normalized to be  1 at redshift z=0, so we multiply with the sGRB (or KN) 
    rate  in the local universe

    According to Sun++2015 this one is disfavoured by  the data although its not completely  ruled out
    '''

    term_1 = (1 + z) **  (1.9 * eta)
    term_2 = ((1 + z) / 2.5) ** (-1.2 * eta)
    term_3 = ((1 + z) / 3.8) ** (-4.4 * eta)
    term_4 = ((1 + z) / 7.7) **  (-11 * eta)

    f_sGRB = (term_1 + term_2 + term_3 + term_4) ** (1 / eta)

    return R0 * f_sGRB

def plot_rates(unit='Gpc', print_r0=False, all_kn=False):
    '''
    Prints all rates in this page.
    unit: str, Mpc for rates in Mpc^-3 yr^-1, or Gpc for Gpc^-3 yr^-1
    '''
    R0 = p_cosmology['R0']
    for key in R0.keys():
        if unit == 'Mpc':
            R0[key] = R0[key].to(u.Mpc**-3 * u.yr**-1)
        elif unit == 'Gpc':
            R0[key] = R0[key].to(u.Gpc**-3 * u.yr**-1)
        # print(f"R0 {key} = {np.format_float_scientific(R0[key].value)} [Mpc^-3 yr^-1]")
        if print_r0:
            print(f"R0 {key} = {np.format_float_scientific(R0[key].value)} [{unit}^-3 yr^-1]")

    z_range = np.linspace(0, 5, p_cosmology['N_int_steps'])
    plt.plot(z_range, R_SFR(z_range, R0=R0['SN Ia'].value), label='SN Ia', c='green')
    plt.plot(z_range, R_SFR(z_range, R0=R0['SN CC'].value), label='SN CC', c='blue')
    plt.plot(z_range, R_TDE(z_range, R0=R0['TDE'].value), label='TDE', c='orange')
    plt.plot(z_range, R_sGRB_gaus(z_range, R0=R0['KN'].value), label='KN Gaussian', c='purple')
    if all_kn:
        plt.plot(z_range, R_sGRB_lognorm(z_range, R0=R0['KN'].value), label='KN log-normal', c='purple', linestyle='dashed')
        plt.plot(z_range, R_sGRB_powerlaw(z_range, R0=R0['KN'].value), label='KN powerlaw', c='purple', linestyle='dotted')

    plt.yscale('log')
    plt.xlabel('z', fontsize=14)
    if unit == 'Mpc':
        plt.ylabel(r'$R_k(z) \;[Mpc^{-3}\,yr^{-1}]$', fontsize=14)
    elif unit == 'Gpc':
        plt.ylabel(r'$R_k(z) \;[Gpc^{-3}\,yr^{-1}]$', fontsize=14)
    else:
        raise AttributeError('Wrong unit specification. It can be Gpc or Mpc. Its Gpc by default. ')
    plt.legend(fontsize=14)
    plt.title('Volumetric rates per transient type k', fontsize=15)
    plt.show()

    return None

### Functions for estimating the neutrino luminosity from the diffuse flux

def diffuse_nu_spl(E, gamma=-2.52, phi_0=1.80):
    '''
    Best fit for the measured IceCube diffuse neutrino flux using the single-powerlaw.

    Params
    ------
    E: array, GeV. Energy of the neutrinos
    gamma: float. Poweerlaw index
    phi_0: float, powerlaw normalization in [10^-18 / GeV / cm^2 / s / sr]

    Returns
    -------
    phi_nu: arrray. Neutrino diffuse flux dN/dEdAdt in [ N / GeV / cm^2 / s / sr]
    '''
    E_0 = 1e5 #if E in GeV
    phi_nu = phi_0 * 1e-18 * (E / E_0) ** gamma
    return phi_nu

def  get_phi_0_k(R_z, E_nu, lambda_k, T_k, R0=1, p=p_cosmology, z_max=6, gamma_k=-2.52):
    '''
    Estimate the intrinsic source luminosity from a contribution of the total diffuse neutrino flux

    Params:
    ------
    R_z: func. Volumetric  rate evolution chosen for a specific source  class.
    E_nu: float, GeV. Reconstructed energy of the  neutrino as publshed in the alert.
    lambda_k: float. Fractional  ontribution of the diffuse neutrino flux. lambda_k=0.1 means we assume that all sources of type k 
        contribute  10%  of the observed diffuse flux at an energy  of E_nu
    T_k: float, duration of transient k in days
    R0:  float. Volumetric rate  of  transient oftypee k at z ~ 0  in Mpc^-3 yr^-1
    gamma_k: float. Powerlaw index for thee  population flux. Assumed  to be same as SPL fit for IC diffuse flux.
    

    '''
    if not isinstance(T_k, u.quantity.Quantity):
        T_k = T_k * u.d #we add  the days units. 
    
    
    #get comologicalmodel
    cosmo = acosmo.LambdaCDM(H0=p['h0']* u.km / u.s / u.Mpc, 
                         Om0= p['omega_M'],
                        Ode0=p['omega_L'])#, Tcmb0=2.725 * u.K, 
    
    #ingtegration limit
    z_range = np.linspace(p['z_min'], z_max, p['N_int_steps'])
    

    
    diff_nu_flux = diffuse_nu_spl(E_nu) * u.GeV ** -1 * u.s ** -1 * u.sr ** -1 * u.cm ** -2

    

    #integral
    D_L = cosmo.luminosity_distance(z_range).to(u.cm)
    dVdz =  cosmo.differential_comoving_volume(z_range)
    R_k =  R_z(z_range, R0=R0).to(u.s ** -1 * u.Mpc ** -3)
    y = (1 + z_range) ** (2 - gamma_k) * T_k.to('s') / (4 * np.pi *  D_L ** 2) * R_k * dVdz  
    
    E_range = np.logspace(4, 7, 1000)
    phi_0_k_E_range = lambda_k * diff_nu_flux / scipy.integrate.trapezoid(y, z_range) * (E_range / 1e5) ** gamma_k

    
    phi_0_k = scipy.interpolate.interp1d(E_range, phi_0_k_E_range)(E_nu)

    return phi_0_k



if __name__ == '__main__':

    #This plots the rates staed in this file
    R0 = p_cosmology['R0']
    for key in R0.keys():
        #we change to the desired units
        # R0[key] = R0[key].to(u.Mpc**-3 * u.yr**-1)
        R0[key] = R0[key].to(u.Gpc**-3 * u.yr**-1)
        # print(f"R0 {key} = {np.format_float_scientific(R0[key].value)} [Mpc^-3 yr^-1]")
        print(f"R0 {key} = {np.format_float_scientific(R0[key].value)} [Gpc^-3 yr^-1]")

    z_range = np.linspace(0, 5, p_cosmology['N_int_steps'])
    plt.plot(z_range, R_CC(z_range, R0=R0['SN CC'].value), label='CC', c='blue')
    plt.plot(z_range, R_TDE(z_range, R0=R0['TDE'].value), label='TDE', c='orange')
    plt.plot(z_range, R_sGRB_gaus(z_range, R0=R0['KN'].value), label='KN Gaussian', c='purple')
    plt.plot(z_range, R_sGRB_lognorm(z_range, R0=R0['KN'].value), label='KN log-normal', c='purple', linestyle='dashed')
    plt.plot(z_range, R_sGRB_powerlaw(z_range, R0=R0['KN'].value), label='KN powerlaw', c='purple', linestyle='dotted')

    plt.yscale('log')
    plt.xlabel('z', fontsize=14)
    plt.ylabel(r'$R(z) \;[Gpc^{-3}\,yr^{-1}]$', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()