from filter_limits import lsst_bands, limit_mag_dict
# from filter_limits import limit_flux_dict


#we will build the lightcurve generating functions here
import sys
import os

import skysurvey
import sncosmo
import params
from params import p_cosmology
from transient_rates import plot_rates, R_SFR, R_TDE, R_sGRB_gaus
from plotting_functions import plot_population
from random_population import generate_random_transients

import astropy.units as u
from astropy.time import Time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

opsim_path = "baseline_v5.3.5_10yrs.db"

def visualize_nights(opsim_path = opsim_path, start_night = None, stop_night = None, vmin=0,
            vmax_quantile=0.95,
            cmap="coolwarm"):
    """
    Visualise individual nights. Creates n-row subplots in one column.

    """
    if start_night is None:
        raise ValueError("Start night input required.")
    if stop_night is None:
        stop_night = start_night + 1
     
    nights = [skysurvey.LSST.from_opsim(opsim_path,sql_where=
                                        f"night >= {n} AND night < {n+1}")
                                        for n in range(start_night, stop_night)
                                        ]
    n_nights = len(nights)
    fig = plt.figure(figsize=(4, 2 * n_nights)) #initialise size of entire figure

    for i, (night, night_number) in enumerate(zip(nights, range(start_night, stop_night))):

        coverage = night.get_fieldcoverage(incl_zeros=False)

        night.show(
            vmin=0,
            vmax=coverage.quantile(vmax_quantile),
            cmap="coolwarm",
            title=f"Night {night_number}",
            sub=(n_nights, 1, i + 1),
        )
        ax = plt.gca()
        ax.set_title(
            f"Night {night_number}",
            fontsize=8,
        )

    plt.show()


# old set of five functions

def five_sigma_detection(dataset, index = 0, return_obs_id = False):
    """
    For one index --> one dataset
    return_obs_id (bool): If set  to True, return the list of observation_id
    
    """
    detection_rows = dataset.get_data(index = index, detection = True) #returns the rows
    target_data = dataset.data.loc[index]
    if len(detection_rows) > 0:
        obs_idxs = detection_rows.index # a list
        return True, obs_idxs if return_obs_id else True
    return False

def five_sigma_detection_by_band(dataset, index = 0):
    """
    Checks if one target (as part of N_tot) fulfills Detection Criterion 2
    Det. Cri. 2: Five sigma detection (flux/fluxerr, in-built into)
    
    Input:
        - target_data: use dset.data.loc[index] to specify SNIa target
    """
    if dataset is None:
        raise ValueError('dataset instance input required.')

    valid_bands = get_valid_bands(snia_models.template.sncosmo_model, redshift = snia_models.data.loc[index], bands = lsst_bands)
    row = {band: 0 if band in valid_bands else -1 for band in limit_mag_dict}

    detection_rows = dataset.get_data(index = index, detection = True) #returns the rows
    target_data = dataset.data.loc[index]
    unique_bands = get_unique_bands(target_data)

    for band in unique_bands:
        band_data = detection_rows[band]
        if len(band_data) > 0:
            row[band] = 1
    return row

def flux_limit_detection(target_data = None, return_index = False, band = None):
    """
    Additional filter apart from simply 5 sigma. Checks if flux > theoretical band limiting flux

    Input:
        - target_data: use dset.data.loc[index] to specify SNIa target
    """
    if target_data is None:
            raise ValueError('target_data input required.')
    cond_fivesigma = five_sigma_detection(target_data = target_data, return_index=False)
    if cond_fivesigma == False:
        return False
    else:
        # extract fluxes from band
        flux_lookup = {band: limit_mag_dict[band]['flux'] for band in limit_mag_dict}
        target_data['band_limitflux'] = target_data['band'].map(flux_lookup)
        if target_data['flux'] >= target_data['band_limitflux']:
            obs_idxs = target_data[target_data['flux'] >= target_data['band_limitflux']].index
            return True, obs_idxs if return_index else True
        return False

def flux_limit_detection_by_band(target_data = None, return_index = False, band = None):
    if target_data is None:
        raise ValueError('target_data input required.')
    
    valid_bands = get_valid_bands(snia_models.template.sncosmo_model, redshift = snia_models.data.loc[index], bands = lsst_bands)
    row = {band: 0 if band in valid_bands else -1 for band in limit_mag_dict}
    
    cond_fivesigma = five_sigma_detection(target_data = target_data, return_index=False)
    if cond_fivesigma == False:
        return False
    else:
        # extract fluxes from band
        flux_lookup = {band: limit_mag_dict[band]['flux'] for band in limit_mag_dict}
        target_data['band_limitflux'] = target_data['band'].map(flux_lookup)
        detection_rows = dataset.get_data(index = index, detection = True)

        if detection_rows['flux'] >= detection_rows['band_limitflux']:
            for band in valid_bands:
                band_data = target_data['band']
                obs_idxs = target_data[target_data['flux'] >= target_data['band_limitflux']].index
            return True, obs_idxs if return_index else True
        return False