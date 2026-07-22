import sys
import os


import numpy as np
import scipy
import astropy.units as u
import astropy.cosmology as acosmo

from transient_rates import R_SFR, R_TDE, R_sGRB_gaus
from params import p_cosmology



def get_N_k(R_k, R0, p=p_cosmology, z_max=5):
    '''
    Computes the total number of transients up to a maximum redshift given a transient rate and a local rate. 
    '''

    #get comologicalmodel

    #esnsure the correct units for R0:
    R0 = R0.to(u.Mpc ** -3 * u.yr**-1)

    
    cosmo = acosmo.LambdaCDM(H0=p['h0']* u.km / u.s / u.Mpc, 
                         Om0= p['omega_M'],
                        Ode0=p['omega_L'])#, Tcmb0=2.725 * u.K,
    
    z_range = np.linspace(p['z_min'], z_max, p['N_int_steps'])

    dVdz =  cosmo.differential_comoving_volume(z_range)

    y = R_k(z_range, R0=R0) / (1 + z_range) * dVdz * 4 * np.pi * u.sr
    N_k = scipy.integrate.trapezoid(y, z_range)

    return N_k

def generate_random_transients(transient_type, N_tot=None, p=p_cosmology, time=1*u.yr, z_max=5, **kwargs):
    '''  
    Generates N_total number of transients given a volumetric rate R_z and local density R0. If N_t is not speficied, its computed
    from the total integrated volumetric rate.

    Params:
    ------
    transient_type: str. It can be any of the following: 
            - SN Ia
            - SN Ibc
            - SN II
            - TDE 
            - KN 
    N_tot: int. Total number of transients to be simulated. If None, then we 
            copmute the total number of transients  up to a maximum redshift.
    p: dict. Parameter dictionary with the comsology parameters for the simulation. 
            Defined in paramams.py, this also includes the local transient rate R0.
    time: astropy units objet. N_tot is a quantity per year, so this specifies the time 
            window in which we are looking at this. 
    z_max: float, maximum redshift to integrate to for obtaining  N_tot. Only used if 
            N_tot is not given

    **kwargs: Extra key-word arguments to pass for the get_N_k function. Only used if 
            N_tot is not defined
    
    Returns
    -------
    transients:  dict of arrays. Contains the simulated transients, where for each parameter 
            we have an array of size N_tot. 
    '''
    rate_funtions = {
        'SN Ia': R_SFR,
        'SN Ibc': R_SFR,
        'SN IIn': R_SFR,
        'TDE': R_TDE,
        'KN': R_sGRB_gaus
    }
    #We select the transient rate function and local rate we will use
    try: 
        R_z = rate_funtions[transient_type] 
        R0 = p['R0'][transient_type]
    except ValueError as error:
        print('Invalid transient type inputed. Choose any from: transient_type: str. It can be any of the following: SN Ia, SN Ibc, SN II, TDE, KN')

    
    time = time.to(u.yr)

    if not N_tot:
        N_tot = get_N_k(R_z, R0=R0, p=p, z_max=z_max, **kwargs) * time 
    
    sample_size = int(N_tot)
    z_grid = np.linspace(p_cosmology['z_min'], z_max, 1000)
    rates_k = R_z(z_grid, R0.to(u.Mpc ** -3 * u.yr ** -1))

    p_k = rates_k / np.sum(rates_k)
    redshift_sample = np.random.choice(z_grid, p=p_k, size=sample_size)

    ra = np.random.uniform(0, 360, size=sample_size)
    dec = np.random.uniform(-90, 90, size=sample_size)

    transients = {
        'z': redshift_sample,
        'z_max': z_max,
        'ra': ra,
        'dec': dec,
        'time_window': time
    }
    return transients