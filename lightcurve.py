#we will build the lightcurve generating functions here
import sys
import os

import skysurvey
import sncosmo
import params
from params import p_cosmology
from transient_rates import R_SFR, R_TDE, R_sGRB_gaus
from transient_rates import plot_rates
from plotting_functions import plot_population
from random_population import generate_random_transients

import astropy.units as u
from astropy.time import Time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

p_cosmology = {
    'c': 299792.458, #km/s
    'h0': 67.4, #km/s/Mpc
    'omega_M': 0.315, 
    'omega_K': 0,
    'omega_L': 0.685,
    'z_min': 1e-4, # minimum redshift for integrating
    'N_int_steps': 100,
    'R0': {
        'SN Ia':  2.3e4 * u.Gpc ** -3 *  u.yr ** -1, #BTS ZTF
        'SN CC': 10.1e4  * u.Gpc ** -3 *   u.yr  ** -1, #BTS ZTF
        'SN Ibc': 10.1e4  * u.Gpc ** -3 *   u.yr  ** -1, #BTS ZTF
        'SN II': 10.1e4  * u.Gpc ** -3 *   u.yr  ** -1, #BTS ZTF
        'TDE': 8e-7 * u.Mpc **  -3  * u.yr ** -1, #?  sun et al maybe
        'KN': 59.3 * u.Gpc ** -3 * u.yr ** -1, #BTS ZTF, upper limit
    }
}

limit_mag_dict = {
                "lsstu": {"mag": 23.8, "color": "tab:purple"},
              "lsstg": {"mag": 24.5, "color": "tab:green"},
              "lsstr": {"mag": 24.0, "color": "tab:orange"},
              "lssti": {"mag": 23.4, "color": "tab:red"},
              "lsstz": {"mag": 22.7, "color": "tab:brown"},
              "lssty": {"mag": 22.0, "color": '0.5'},
              }


lsst_colors = ["tab:purple", "tab:green", "tab:orange", "tab:red", "tab:brown", "0.5"]

def generate_snia_lightcurve(redshift = None, bands = None, p = p_cosmology, N_tot = 1, plot_curve = False, return_values = False, in_mag = False, return_models = False):
    """
    Generate lightcurve for a given input redshift.

    All parameters are generated independently, following which z is manually set
    Code to be modified in the future for more than one event.

    N_tot: Number of snia targets to simulate. Use N_tot = 1 for now.
    plot_curve: Plots the lightcurve using skysurvey.show_lightcurve()
    return_values: Returns y valeus (flux or mag) as a 2D array, with each layer as different filters
    
    """
    if redshift is None:
        raise ValueError("Redshift input required")

    if redshift < p_cosmology['z_min'] or redshift > 5.0:
        raise ValueError("Redshift input out of bounds. Valid range: {} to {}".format(p_cosmology['z_min'], 5.0))

    def fixed_z(size, z = redshift):
        return np.full(size, z)

    def get_valid_bands(model, bands):
        valid = []
        for band in bands:
            bp = sncosmo.get_bandpass(band)
            # redshift shifts the model wavelength range
            if bp.minwave() / (1 + redshift) >= model.minwave() and bp.maxwave() / (1 + redshift) <= model.maxwave():
                valid.append(band)
            else:
                print(f"Skipping {band}: outside model range.")

        return valid

    custom_model = {
        "redshift": {
            "func": fixed_z,
            "kwargs": {"size": N_tot, "z": redshift},
            "as": "z",
        }
    }

    # make the snia.target.core instance using default model
    snia = skysurvey.SNeIa()
    # update the snia.target.core.Target.model instance.
    snia.update_model(**custom_model) #unpacking is IMPORTANT
    new_snia = snia.from_draw(N_tot, model = custom_model)
    # new_snia.data["z"] = redshift #brute-force changing of lc dataframe.
    # you can use indexing e.g. .data["z"][0] if you have more than one

    t0 = new_snia.data["t0"][0] #check why u need 0 again. because of index 0?
    phase_start = new_snia.template.get().mintime()
    phase_stop = min(new_snia.template.get().maxtime(), 200.0)
    times = np.linspace(phase_start, phase_stop, 500) + t0

    # Step 2: Filter out those filter bands which lie outside of SN model range
    valid_bands = get_valid_bands(new_snia.template.sncosmo_model, bands)
    # Step 3: Use .get_lightcurve(). Note: this has in_mag = FALSE by default
    
    lc = new_snia.get_lightcurve(band = valid_bands, phase_range = (phase_start, phase_stop), times=times,index=0,zp=25,in_mag=in_mag)
    if plot_curve:
        fig, ax = plt.subplots(figsize=(6, 4))

        for i in range(len(valid_bands)):
            ax.plot(times, lc[i], label=valid_bands[i], color=lsst_colors[i])

        for band, v in limit_mag_dict.items():
            ax.axhline(v["mag"], color=v["color"], linestyle="--", label=f"{band} limit")

        if in_mag:
            ax.invert_yaxis()
        ax.set_xlabel('MJD')
        ax.set_ylabel('mag')
        ax.legend(fontsize=8)
  
        plt.show()
    
    snia_targets = new_snia #recall you can do .from_draw(size, model ,...) then use .get_lightcurve(index = ...)

    if not return_values:
        if return_models:
            return snia_targets
        return None

    y_vals = np.column_stack([times] + list(lc))
    value_key = "mag" if in_mag else "flux (phot/s/cm^2)"
    lc_by_band = {
        band: np.asarray(lc[i])
        for i, band in enumerate(valid_bands)
    }
    result_dict = {
        "times": times,
        "lc": lc,
        "lc_by_band": lc_by_band,
        value_key: y_vals,
    }

    if not return_models:
        return result_dict

    return snia_targets, result_dict

def make_population(dict):
    """
    Returns a dictionary with the key parameters:
        z, z_max, ra, dec, time_window
    """
    population_dict = {}
    # ...
    return population_dict

def make_visibility_row(target, bands, limit_mag_dict):
    """
    Has to complement the values from y_vals.
    """
    redshift = target.data["z"][0]
    row = {}
    target = generate_snia_lightcurve(redshift = None, bands = bands, p = p_cosmology, N_tot = 1, plot_curve = False, return_values = False, in_mag = True, return_models = True)
    for i, band in enumerate(bands):
        mag_vals = np.asarray(lc[i])   # works for tuple/list/ndarray
        visible = int(np.any(mag_vals < limit_mag_dict[band]["mag"]))
        row[band] = visible
    return row

target_ids = [0]   # add more event IDs later if needed
rows = []


"""
def determine_visibility(target, times = None, model_data):
    
    Determine using lightcurve input.
    You will use all bands available inside the selected target (model)
    

    for target_id in target_ids:
        lc = target.get_lightcurve(
            band = bands,
            times= times,
            index=target_id,
            zp=25,
            in_mag=True,
        )
        rows.append(make_visibility_row(lc, bands, limit_mag_dict))

    sn_visibility = pd.DataFrame(rows, index=target_ids)
    sn_visibility.index.name = "target_id"
    print(sn_visibility)

    sn_visibility_dict = sn_visibility.to_dict()
    sn_visibility_dict

    return 0
"""

