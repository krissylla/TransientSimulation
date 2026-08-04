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

#generate_random_transients give dict of z, zmax, ra, dec, timewindow

# put these in params.py when you don't need for reference anymore

lsst_bands = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]
limit_mag_dict = {
                "lsstu": {"mag": 23.8, "color": "tab:purple"},
              "lsstg": {"mag": 24.5, "color": "tab:green"},
              "lsstr": {"mag": 24.0, "color": "tab:orange"},
              "lssti": {"mag": 23.4, "color": "tab:red"},
              "lsstz": {"mag": 22.7, "color": "tab:brown"},
              "lssty": {"mag": 22.0, "color": '0.5'},
              }

snia_lightcurve_params = ["redshift", "x1", "c", "t0", "magabs", "ra", "dec"]
lsst_colors = ["tab:purple", "tab:green", "tab:orange", "tab:red", "tab:brown", "0.5"]

def fixed_param(size, value = None):
    """
    Fix the parameter value to be used in the generated Transient model.
    """
    if value is None:
        raise ValueError("Input value required")
    return np.full(size, value)

def fixed_radec(size, ra = None, dec = None):
    """
    Specificially for 'ra' and 'dec': generate list ['ra', 'dec']
    """
    if ra is None:
        raise ValueError("Input ra required")
    if dec is None:
        raise ValueError("Input dec required")
    return np.full(size, ra), np.full(size, dec)

def get_valid_bands(model, redshift, bands):

    valid_bands = []
    for band in bands:
        bp = sncosmo.get_bandpass(band)
        # redshift shifts the model wavelength range
        if bp.minwave() / (1 + redshift) >= model.minwave() and bp.maxwave() / (1 + redshift) <= model.maxwave():
            valid_bands.append(band)
        else:
            print(f"Skipping {band}: outside model range.")

    return valid_bands

def generate_custom_model(redshift=None, x1=None, c=None, t0=None, magabs=None,
                          ra=None, dec=None, N_tot=1):

    # if custom_model.keys() not in snia_lightcurve_params...?
    
    """
    Generate dictionary of custom model.
    
    """
    custom_model = {}

    for param_name, value in zip(
        ["redshift", "x1", "c", "t0", "magabs"],
        [redshift, x1, c, t0, magabs],
    ):
        if value is None:
            continue
        if param_name == "redshift":
            new_param_name = "z"
        else:
            new_param_name = param_name
        custom_model[param_name] = {
            "func": fixed_param,
            "kwargs": {"size": N_tot, "value": value},
            "as": new_param_name,
        }

    if (ra is None) ^ (dec is None):
        raise ValueError("Input of ra and dec required together; else leave both as None")

    if ra is not None and dec is not None:
        custom_model["radec"] = {
            "func": fixed_radec,
            "kwargs": {"size": N_tot, "ra": ra, "dec": dec},
            "as": ["ra", "dec"],
        }
    return custom_model

def generate_snia_dict(redshift=None, x1=None, c=None, t0=None, magabs=None,
                          ra=None, dec=None):
    """
    Makes a dict.
    Unlike Cris' generate_random_transients() function, this function lets you build a dict.
    You do not need to fix every parameter. All unfixed params will be randomised.

    ------------------------------------
    Inputs:
        - params (float?)
    
    Returns:
        - a dictionary with all the input parameter names and their values (dictionary)
    """

    snia_dict = {}

    for param_name, value in zip(
        ["redshift", "x1", "c", "t0", "magabs", "ra", "dec"],
        [redshift, x1, c, t0, magabs, ra, dec],
    ):
        if value is None:
            continue
        snia_dict[param_name] = value

    if (ra is None) ^ (dec is None):
            raise ValueError("Input of ra and dec required together; else leave both as None")
 
    return snia_dict
    
    

def generate_ordered_parameter_list(redshift=None, x1=None, c=None, t0=None, magabs=None,
                                    ra=None, dec=None, input_dict=False, params=None):
    """
    Return a parameter list in the order expected by generate_snia_lightcurve().

    The returned list can be unpacked into generate_snia_lightcurve() as:
        params = generate_ordered_parameter_list(...)
        generate_snia_lightcurve(*params, bands=bands)

    When input_dict=True, the `params` dictionary is used to fill values.
    Any None values are preserved so the lightcurve builder can randomize
    the corresponding parameters.
    """
    if input_dict:
        if params is None:
            raise ValueError("params dict is required when input_dict=True")
        if not isinstance(params, dict):
            raise TypeError("params must be a dict when input_dict=True")
        redshift = params.get("redshift", redshift)
        x1 = params.get("x1", x1)
        c = params.get("c", c)
        t0 = params.get("t0", t0)
        magabs = params.get("magabs", magabs)
        ra = params.get("ra", ra)
        dec = params.get("dec", dec)

    return [redshift, x1, c, t0, magabs, ra, dec]


def generate_snia_lightcurve(redshift = None, x1 = None, c = None, t0 = None, magabs = None, ra = None, dec = None,
                              bands = None, p = p_cosmology, N_tot = 1, plot_curve = False, return_values = False,
                              in_mag = False, return_models = False):
    """
    Generate lightcurve for a given input redshift.

    Every other parameter (x1, c, t0, magabs (--> magobs), and radec are randomised.
    To customise all parameters, see generate_custom_snia()

    All parameters are generated independently, following which z is manually set
    Code to be modified in the future for more than one event.

    N_tot: Number of snia targets to simulate. Use N_tot = 1 for now AND if all params are fixed.
    plot_curve: Plots the lightcurve using skysurvey.show_lightcurve()
    return_models: Returns the snia.target.core.Target instance. Recommended to use False if you have already generated the model.
    return_values: Returns y values (flux or mag) as a 2D array, with each layer as different filters
    
    """
    if redshift < p_cosmology['z_min'] or redshift > 5.0:
        # we set p_cosmology['z_min'] = 1e-4.
        raise ValueError("Redshift input out of bounds. Valid range: {} to {}".format(p_cosmology['z_min'], 5.0))

    if redshift is None and x1 is None and c is None and t0 is None and magabs is None and ra is None and dec is None and N_tot > 1:
        raise ValueError("If N_tot > 1, at least one parameter must be fixed in order to generate unique targets.")

    if N_tot < 1:
        raise ValueError("N_tot must be a positive integer.")

    if isinstance(N_tot, int) is False:
        raise TypeError("N_tot must be an integer.")

    custom_model = generate_custom_model(redshift=redshift, x1=x1, c=c, t0=t0, magabs=magabs,
                          ra=ra, dec=dec, N_tot=1)

    print(custom_model, type(custom_model))

    # make the snia.target.core instance using default model
    snia = skysurvey.SNeIa() #this line should be flexible for future models

    # update the snia.target.core.Target.model instance.
    snia.update_model(**custom_model) #unpacking is IMPORTANT. But I don't think this line does anything since we use from_draw() again
    new_snia = snia.from_draw(size = N_tot, model = custom_model, zmin = p_cosmology['z_min'], zmax = 5.0, nyears = 1, cosmology = None, rate = p_cosmology["R0"]["SN Ia"].value)

    # Step 2: get times range
    t0 = new_snia.data["t0"][0] #check why u need 0 again. because of index 0?
    phase_start = new_snia.template.get().mintime()
    phase_stop = min(new_snia.template.get().maxtime(), 200.0)
    times = np.linspace(phase_start, phase_stop, 500) + t0

    # Step 3: Filter out those filter bands which lie outside of SN model range
    valid_bands = get_valid_bands(model = new_snia.template.sncosmo_model, redshift = redshift, bands = bands)

    # Step 4: Use .get_lightcurve(). Note: this has in_mag = FALSE by default
    print('valid bands are:', valid_bands)
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
        ax.set_title('SNIa z={}, x1={}, c={}, t0={}, magabs={}, radec = [{}, {}]'.format(redshift, x1, c, t0, magabs, ra, dec), fontsize=7)
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
    lightcurves_data = {
        "times": times,
        # "lc": lc,
        "lc_by_band": lc_by_band,
        "units": "mag" if in_mag else "flux (phot/s/cm^2)",
        # value_key: y_vals,
    }

    if not return_models:
        return lightcurves_data

    return snia_targets, lightcurves_data


def make_visibility_row(lc, valid_bands, limit_mag_dict = limit_mag_dict):
    #initialise every band as -1 meaning not available.
    row = {band: -1 for band in limit_mag_dict}

    for band, mag_vals in zip(valid_bands, lc):
        row[band] = int(
            np.any(np.asarray(mag_vals) < limit_mag_dict[band]["mag"])
        ) #returns 0 (falls within filter wavelength range but too faint), 1 if visible

    return row


def determine_visibility(lightcurves_data, return_dict = False):

    lc = lightcurves_data["lc"]
    valid_bands = list(lightcurves_data["lc_by_band"])

    row = make_visibility_row(lc, valid_bands)

    if return_dict:
        return row
    
    return pd.DataFrame([row])