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
from skysurvey.tools.utils import random_radec


import astropy.units as u
from astropy.time import Time
import astropy.cosmology as acosmo

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

cosmo = acosmo.LambdaCDM(H0=p_cosmology['h0']* u.km / u.s / u.Mpc, 
                         Om0=p_cosmology['omega_M'],
                        Ode0=p_cosmology['omega_L'])

#generate_random_transients give dict of z, zmax, ra, dec, timewindow

# put these in params.py when you don't need for reference anymore

lsst_bands = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]
# probably add zp in param
limit_mag_dict = {'lsstu': {'mag': 23.8, 'flux': 301.995, 'color': 'tab:purple'},
            'lsstg': {'mag': 24.5, 'flux': 158.489, 'color': 'tab:green'},
            'lsstr': {'mag': 24.0, 'flux': 251.189, 'color': 'tab:orange'},
            'lssti': {'mag': 23.4, 'flux': 436.516, 'color': 'tab:red'},
            'lsstz': {'mag': 22.7, 'flux': 831.764, 'color': 'tab:brown'},
            'lssty': {'mag': 22.0, 'flux': 1584.893, 'color': '0.5'}}

snia_lightcurve_params = ["redshift", "x1", "c", "t0", "magabs", "ra", "dec"]
# note: skysurvey uses "z" to  refer to redshift.

lsst_colors = ["tab:purple", "tab:green", "tab:orange", "tab:red", "tab:brown", "0.5"]

def fixed_param(size, value = None):
    """ Fix the parameter value to be used in the generated Transient model.
    
    Parameters
    ----------
    size : int
        Number of targets (equivalent to N_tot)
    value : int, float
        A flag used to print the columns to the console (default is False)

    Returns
    -------
    ndarray:
        an array of length (size), each with a given value.
    
    Raises
    ------
        ValueError: if value is None

    See Also
    --------
    :func:`fixed_radec` : Returns two separate arrays, one for ra and one for dec

    """

    if value is None:
        raise ValueError("Input value required")
    return np.full(size, value)

def fixed_radec(size, ra = None, dec = None):
    """ Specificially for 'ra' and 'dec': generate list ['ra', 'dec'].
    The function is essentially identical to fixed_param. Refer to its documentation.

    See Also
    --------
    :func:`fixed_param` : The identical function, but for other other params that are not radec
    """
    if ra is None:
        raise ValueError("Input ra required")
    if dec is None:
        raise ValueError("Input dec required")
    
    return np.full(size, ra), np.full(size, dec)

# lsst in skysurvey uses zp 30
def flux_to_mag(flux, zp=30):
    return zp - 2.5*np.log10(flux)

def mag_to_flux(mag, zp=30):
    return 10**(-0.4*(mag-zp))

def get_valid_bands(model, redshift, bands, verbose = False):
    """ This function is not used because DataSet.get_ndetections() is similar.
    
    It can be useful if you want to distiguish sources that are not detected because:
    - outside of all six filter bandwidths due to lightcurve being redshifted too much (around ~1.0)
    - too faint to exceed 5 sigma detection limit.
    Hence, if using this function, when you write a function that assigns detectability to each band:
    You can initialise a list where all filters are initialised at -1,
    and after analysing the detectability, you might get, for example:
    For a given source:
        e.g. {'lsstu': -1, 'lsstg': 0, 'lsstr': 1, 'lssti': 1, 'lsstz': 0, 'lssty': 0}
        where -1 is invalid bands / undefined detection, 0 is not detectable, and 1 is detected.

    The salt2 model simulates lightcurves from 2000 to 9200 \AA, so after computing the redshifted wavelengths,
    this function checks if a SNIa lightcurve falls within the bandwidth of each LSST filter.

    Parameters
    ----------
    model : `sncosmo.models.Model`
        Cosmology model used. (Default is sncosmo_model)
    redshift: int, float
        Redshift of target
    bands: list of strings
        e.g. ['lsstr', 'lssti']. You can use `from lightcurve import lsst_bands`
    verbose : int, float
        A flag used to print the columns to the console (default is False)

    Returns
    -------
    list:
        List of valid bands

    """

    valid_bands = []
    for band in bands:
        bp = sncosmo.get_bandpass(band)
        # redshift shifts the model wavelength range
        if bp.minwave() / (1 + redshift) >= model.minwave() and bp.maxwave() / (1 + redshift) <= model.maxwave():
            valid_bands.append(band)
        else:
            if verbose:
                print(f"Skipping {band}: outside model range.")

    return valid_bands


# Cris' code --> to replace the default .draw_redshift()
def sample_redshift_from_Rz(N, R_z, z_max=1.0, p_cosmology=p_cosmology):
    '''
    Parameters
    ----------
    N: int
        Total number of redshifts to sample 
    R_z: func
        Volumetric rate evolution function that takes as arguments R_z(z, R0).
        Since we are normalizing the distirbution to the nearby universe, R0 can be set to 1. 
    z_max: int
        Maximum redshift to sample from. z_mibn is defined in the parameter dictionary p_cosmology
    p_cosmology: dict
        Cosmological parameters used for integration. We just take z_min from this
    
    '''
    
    # z_grid = np.linspace(p_cosmology['z_min'], z_max, 100000)
    z_edges = np.linspace(p_cosmology["z_min"], z_max + 1e-6, 100000)
    z_grid = 0.5 * (z_edges[:-1] + z_edges[1:])
    shell_volumes = np.diff(cosmo.comoving_volume(z_edges)).to(u.Mpc**3)
    rates_k = R_z(z_grid) / (1 + z_grid) * shell_volumes #multiply by comoving vol of shell

    p_k = rates_k / np.sum(rates_k)
    redshift_sample = np.random.choice(z_grid, p=p_k, size=N)
    return redshift_sample

def generate_custom_model(redshift=None, x1=None, c=None, t0=None, magabs=None,
                          ra=None, dec=None, N_tot=1):

    """Generate dictionary of custom model.

    This function is created to be an in-built method in generate_snia_lightcurve

    We want our generated SNIa population to be able to follow certain distributions.
    This function customises the selection function for each parameter.
    We initialising a new default redshift selection; sample_redshift_from_Rz(). Note: skysurvey uses its own selection function

    Parameters
    ------
    SNIa lightcurve params [redshift, x1, c, t0, magabs, ra, dec]

    N_tot: int
        number of targets to generate. (Default is 1)2

    Returns
    -------
    custom_model: dict
        Comprising all params to update the snia.target.core.Target.model instance with.
    
    See Also
    ---------
    :func: generate_snia_dict()
    :func: generate_ordered_parameter_dict()
    :func: generate_snia_lightcurve()

    """
    custom_model = {}
    if redshift is None:
        custom_model["redshift"] = {
            "func": sample_redshift_from_Rz,
            "kwargs": {"N": N_tot, "R_z": R_SFR, "z_max": 1.0, "p_cosmology": p_cosmology},
            "as": "z",
        }

    if (ra is None) and (dec is None):
        custom_model["radec"] = {
            'as': ['ra', 'dec'],
           'func': random_radec,
           'kwargs': {'dec_range':[-90, 90]}, #put this in params.py later
        }


    for param_name, value in zip(
        ["redshift", "x1", "c", "t0", "magabs"],
        [redshift, x1, c, t0, magabs],
    ):
        if value is None:
            continue
        if param_name == "redshift":
            new_param_name = "z"
        else:
            new_param_name = param_name # name unchanged
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
    """Main function 1 of 3 in generating your custom SNIa population.
    Makes a dict of SNIa parameters to simulate.

    Unlike Cris' generate_random_transients() function, this function lets you build a dict.
    You do not need to fix every parameter. All unfixed params will be randomised.

    IMPORTANT: It would be useful to add ra_range and dec_range as input parameters.
    ------------------------------------
    
    Parameters
    -----------
    SNIa lightcurve params [redshift, x1, c, t0, magabs, ra, dec]
    
    Returns
    -------
    snia_dict: dict
        A dictionary with all the input parameter names and their values (dictionary)

    See Also
    ---------
    :func: generate_custom_model()
    :func: generate_ordered_parameter_dict()
    :func: generate_snia_lightcurve()
    

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
    """Main function 2 of 3 in generating your custom SNIa population.
    
    Reorders the parameters in the parameter dict to a specified order, and then
    returns a parameter list in the order expected by generate_snia_lightcurve().

    The returned list can be unpacked into generate_snia_lightcurve() as:
        params = generate_ordered_parameter_list(...)
        generate_snia_lightcurve(*params, bands=bands)

    When input_dict=True, the `params` dictionary is used to fill values.
    Any None values are preserved so the lightcurve builder can randomize
    the corresponding parameters.

    See Also
    ---------
    :func: generate_custom_model()
    :func: generate_snia_dict()
    :func: generate_snia_lightcurve()

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

def plot_lightcurve(lightcurve_data, observed_data = None, times=None, bands=None, in_mag=False, plot_limits = False, verbose = False):
    """ Plots lightcurve in different bands, optionally with observed data from lsst observations.

    Parameters
    ----------
    lightcurve_data: nested dict
        comprises 'times' and 'lc_by_band' -- which is maded up of a dict of 'band' and mag/flux values 
    observed_data : lsst observations Dataset(), optional
        This is what you get from dset.from_targets_and_survey(). It uses the dataset columns.
    times: list or arraylike
        SNIa times
    bands: list of str
        Bands you want to plot.
    in_mag: bool
        Whether to plot in magnitude or fluxes. Recommended to use False.
    plot_limits: bool
        Whether to plot the theoretical mag or flux needed from the VRO documentation (which is what our limit_mag_dict used)
        Recommended to keep as False as the results are not particularly qualitative.
    verbose: bool
    
    Returns
    --------
    fig: matplotlib.figure.Figure() object
        Plot of lightcurve (optionally with observed datapoints) in different bands

    Examples:
    ---------
    model = data_models[8]
    plot_lightcurve(model, times=model['times'], in_mag = False)
    
    or 

    fig = plot_lightcurve(data_models[max_index], observed_data = dset.get_target_lightcurve(index=max_index), times = data_models[max_index]['times'], plot_limits = False)
    
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    params = lightcurve_data["params"]
    lc_by_band = lightcurve_data["lc_by_band"]

    # Plot all valid bands by default
    if bands is None:
        bands = lc_by_band.keys()

    for band in bands:

        if band not in lc_by_band:
            if verbose:
                print(f"Skipping {band}: no light curve available.")
            continue

        ax.plot(
            times,
            lc_by_band[band],
            label=band,
            color=limit_mag_dict[band]["color"],
        )

    if plot_limits:
        if in_mag:
            for band in bands:
                if band in limit_mag_dict:
                    ax.axhline(
                        limit_mag_dict[band]["mag"],
                        color=limit_mag_dict[band]["color"],
                        linestyle="--",
                        label=f"{band} limit",
                    )
            ax.invert_yaxis()

        else:
            for band in bands:
                if band in limit_mag_dict:
                    ax.axhline(
                        limit_mag_dict[band]["flux"],
                        color=limit_mag_dict[band]["color"],
                        linestyle="--",
                        label=f"{band} limit",
                    )
    ###
    if observed_data is not None:

        for band in bands:

            obs_band = observed_data[
                observed_data["band"] == band
            ]

            if len(obs_band) == 0:
                continue

            # Convert observations to requested zero point
            coef = 10 ** (
                -(obs_band["zp"].to_numpy() - 30) / 2.5
            )

            flux = obs_band["flux"].to_numpy() * coef
            fluxerr = obs_band["fluxerr"].to_numpy() * coef

            obs_times = obs_band["mjd"].to_numpy()

            if in_mag:

                # Only positive fluxes can be converted to magnitudes
                valid = flux > 0

                mags = 30 - 2.5 * np.log10(flux[valid])

                magerr = (
                    2.5 / np.log(10)
                    * fluxerr[valid]
                    / flux[valid]
                )

                ax.errorbar(
                    obs_times[valid],
                    mags,
                    yerr=magerr,
                    fmt="o",
                    color=limit_mag_dict[band]["color"],
                    markersize=4,
                    capsize=2,
                )

            else:

                ax.errorbar(
                    obs_times,
                    flux,
                    yerr=fluxerr,
                    fmt="o",
                    color=limit_mag_dict[band]["color"],
                    markersize=4,
                    capsize=2,
                )
    ###
    ax.set_xlabel("MJD")
    ax.set_ylabel("Magnitude" if in_mag else "Flux (phot/s/cm²)")
    ax.set_title(
        "z={z:.3f}, x1={x1:.2f}, c={c:.2f}, "
        "t0={t0:.1f}, magabs={magabs:.2f}, "
        "ra={ra:.2f}, dec={dec:.2f}".format(**params),
        fontsize=7,
    )
    
    ax.legend()

    return fig



def generate_snia_lightcurve(redshift = None, x1 = None, c = None, t0 = None, magabs = None, ra = None, dec = None,
                              bands = None, p = p_cosmology, tstart = None, tstop = None, N_tot = 1, zp = 30, zpsys = 'ab', plot_curve = False, return_values = False,
                              in_mag = False, return_models = False, verbose = False):
    """Main function 3 of 3 in generating your custom SNIa population.

    This is our golden function to generate the population. I call it the lightcurve since we are analysing
    detectability of each source from their expected lightcurve.

    Generate lightcurve for a given input redshift.

    Every other parameter (x1, c, t0, magabs (--> magobs), and radec are randomised.
    To customise all parameters, see generate_custom_snia()

    All parameters are generated independently, following which z is manually set
    Code to be modified in the future for more than one event.

    Parameters
    ----------
    SNIa lightcurve parameters
        To be unpacked from the SNIa parameter dict
    N_tot: int
        Number of snia targets to simulate (default is 1)
    plot_curve: bool
        Plots the lightcurve using skysurvey.show_lightcurve()
    zp: zero-point magnitude
        (scales the magnitude to zero. (For LSST in skysurvey, default is 30)
    return_models: bool
        If True, returns the snia.target.core.Target instance. Recommended to use False if you have already generated the model.
    return_values: bool
        If True, returns y values (flux or mag) as a 2D array, with each layer as different filters

    Returns
    -------
    snia_targets: skysurvey.target.snia.SNeIa
        Collection of all targets.

    lc_data_all_targets: dict
        lightcurve data dict ('times' and 'lc_by_band' and 'params') for each index. Very large file, so best to call using an index.
        For given index, you can do:
            lc_data_all_targets[index]['times']
            lc_data_all_targets[index]['lc_by_band']['lsstu']
            lc_data_all_targets[index][s'params']
        In the project, I named lc_data_all_targets as data_models


    Raises
    ------
    ValueError: If redshift is empty or N_tot takes a non-integer value
    ...

    Examples
    --------
    Recommeded to use snia_models, data_models = generate_snia_lightcurve(...)
    You can then see each SNIa target data using.
        snia_models.data.loc[0] or snia_models.data.iloc[0]
        data_models[0]
    
    """
    if redshift is not None:
        if redshift < p_cosmology['z_min'] or redshift > 1.0:
            # we set p_cosmology['z_min'] = 1e-4.
            raise ValueError("Redshift input out of bounds. Valid range: {} to {}".format(p_cosmology['z_min'], 1.0))

    if redshift is None and x1 is None and c is None and t0 is None and magabs is None and ra is None and dec is None and N_tot > 1:
        print('All params input as None. Using default sampling functions, but with p_cosmology.')
        # raise ValueError("If N_tot > 1, at least one parameter must be fixed in order to generate unique targets.")

    if N_tot < 1:
        raise ValueError("N_tot must be a positive integer.")

    if isinstance(N_tot, int) is False:
        raise TypeError("N_tot must be an integer.")

    custom_model = generate_custom_model(redshift=redshift, x1=x1, c=c, t0=t0, magabs=magabs,
                          ra=ra, dec=dec, N_tot=N_tot)
    if verbose:
        print(custom_model, type(custom_model))

    # make the snia.target.core instance using default model
    snia = skysurvey.SNeIa() #this line should be flexible for future models

    # update the snia.target.core.Target.model instance.
    snia.update_model(**custom_model) #unpacking is IMPORTANT. But I don't think this line does anything since we use from_draw() again
    new_snia = snia.from_draw(size = N_tot, model = custom_model, zmin = p_cosmology['z_min'], zmax = 1.0, cosmology = None, tstart = tstart, tstop = tstop, rate = p_cosmology["R0"]["SN Ia"].value)
    print('CUSTOM MODEL IS', custom_model)
    # Step 2: get times range for each model new_snia[i]
    phase_start = new_snia.template.get().mintime() #phases should be const
    phase_stop = min(new_snia.template.get().maxtime(), 200.0)
    data = new_snia.data #returns df of all N SNIa targets
    snia_times_series = new_snia.data["t0"]

    lc_data_all_targets = []

    for target_idx in new_snia.data.index:
        t0 = snia_times_series[target_idx] #for index i
        times = np.linspace(phase_start, phase_stop, 500) + t0

        # Step 3: Filter out those filter bands which lie outside of SN model range, using redshift
        valid_bands = get_valid_bands(model = new_snia.template.sncosmo_model, redshift = data["z"][target_idx], bands = bands, verbose = verbose)

        # Step 4: Use .get_lightcurve(). Note: this has in_mag = FALSE by default
        if verbose:
            print(f"Target {target_idx}: valid bands = {valid_bands}")
        if not valid_bands:
            lightcurves_data = {
                "times": times,
                "lc_by_band": {},
                "units": "mag" if in_mag else "flux (phot/s/cm^2)",
                "params": {
                    "z": data.loc[target_idx, "z"],
                    "x1": data.loc[target_idx, "x1"],
                    "c": data.loc[target_idx, "c"],
                    "t0": data.loc[target_idx, "t0"],
                    "magabs": data.loc[target_idx, "magabs"],
                    "ra": data.loc[target_idx, "ra"],
                    "dec": data.loc[target_idx, "dec"],
                },
            }
            lc_data_all_targets.append(lightcurves_data)
            continue
        lc = new_snia.get_lightcurve(band = valid_bands, phase_range = (phase_start, phase_stop), times=times, index=target_idx, zp=zp, zpsys=zpsys, in_mag=in_mag)

        lc_by_band = {band: np.asarray(lc[i]) for i, band in enumerate(valid_bands)}

        lightcurves_data = {
            "times": times,
            # "lc": lc,
            "lc_by_band": lc_by_band,
            "units": "mag" if in_mag else "flux (phot/s/cm^2)",
            "params": {
                "z": data.loc[target_idx, "z"],
                "x1": data.loc[target_idx, "x1"],
                "c": data.loc[target_idx, "c"],
                "t0": data.loc[target_idx, "t0"],
                "magabs": data.loc[target_idx, "magabs"],
                "ra": data.loc[target_idx, "ra"],
                "dec": data.loc[target_idx, "dec"],
            },
        }

        if plot_curve:
            plot_lightcurve(lightcurve_data = lightcurves_data, times = times, bands=valid_bands, in_mag=in_mag, verbose = verbose)
        
        snia_targets = new_snia #recall you can do .from_draw(size, model , ...) then use .get_lightcurve(index = ...)
        lc_data_all_targets.append(lightcurves_data)

    if not return_values:
        if return_models:
            return snia_targets
        return None
    
    if not return_models:
        return lc_data_all_targets

    return snia_targets, lc_data_all_targets


def make_visibility_row(lc_by_band, limit_mag_dict = limit_mag_dict, in_mag = True):
    """This function is not used.
    
    You can use this if you want a one-row series to show visiblity for one target
    e.g. 0, 0, 0, 1, 1, 1

    Returns
    -------
    row: dict
        Can be converted into pandas Series to stack

    """
    #if preferred, initialise every band as -1 meaning detectability is invalid due to redshifted lighcurve outside of a given band.
    #i choose 0 for now to prevent confusion
    row = {band: 0 for band in limit_mag_dict}
    
    for band, y_vals in lc_by_band.items():
        if in_mag:
            row[band] = int(
                np.any(np.asarray(y_vals) < limit_mag_dict[band]["mag"])
            ) #returns 0 (falls within filter wavelength range but too faint), 1 if visible
        else:
            row[band] = int(
                np.any(np.asarray(y_vals) > limit_mag_dict[band]["flux"])
            )

    return row


def determine_visibility(lightcurves_data, return_dict = False, in_mag = True):
    """ This function is not used.

    This assumes that you are using the theoretical lc of 500 datapoints.

    Parameters
    ----------
    lightcurves_data: nested dict
        lightcurve data generated from generate_snia_lightcurve()

    Returns
    -------
        if return_dict == False: pd.Dataframe (default)
        if return_dict == True: dict
    """

    lc_by_band = lightcurves_data["lc_by_band"]
    row = make_visibility_row(lc_by_band, limit_mag_dict = limit_mag_dict, in_mag = in_mag)

    if return_dict:
        return row
    
    return pd.DataFrame([row])

# make visibility using data from obs = dset.get_target_lightcurve(index = ...)
# and then obs[obs["band"] == "filter_name")]

