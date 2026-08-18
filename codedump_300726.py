# custom_model = {"redshift": {"func": fixed_z, "kwargs": {"size": 1}},
#  'x1': {'func': nicolas2021},
#  'c': {'func': np.random.uniform,
#        'kwargs': {'low': -1, 'high': 3}},
#  't0': {'func': np.random.uniform,
#         'kwargs': {'low': 56000, 'high': 56200}},
#  'magabs': {'func': tripp1998,
#             'kwargs': {'x1': '@x1',
#                        'c': '@c',
#                        'mabs': -19.3,
#                        'sigmaint': 0.1,
#                        'alpha': -0.15,
#                        'beta': 3.8}},
#  'magobs': {'func': 'magabs_to_magobs',
#             'kwargs': {'z': '@z', 'magabs': '@magabs'}},
# #  'x0': {'func': 'magobs_to_amplitude',
# #         'kwargs': {'magobs': '@magobs', 'param_name': 'x0'}},
#  'radec': {'func': random_radec,
#            'kwargs': {},
#            'as': ['ra', 'dec']}}

#030826 Don't generate lc again. Make it only inspect array
def make_visibility_row(target = None, band = None, limit_mag_dict = None):
    """
    Makes a row to add to the dataframe of visibility:

    Note: Has to complement the values from lc[band]
    Three inputs are needed:
        - Target: the snia model of interest. recall that target.data gives a pandas.df
        - bands = all 6 lsst filters used
        - limit_mag_dict: collection of filters and the relevant parameters
    """
    if bands == None:
        raise ValueError("Input bands are missing. Examples are ['lsstg', 'lsstr', ...]")
    redshift = target.data["z"][0]
    row = {}
    target_lc = generate_snia_lightcurve(redshift = redshift, bands = bands, p = p_cosmology, N_tot = 1, plot_curve = False, return_values = False, in_mag = True, return_models = True)
    valid_bands = get_valid_bands()
    for i, band in enumerate(bands):
        if band in valid_bands:
            mag_vals = np.asarray(target_lc[i])   # works for tuple/list/ndarray
            # example: = int(np.any ([21.5, 21.9, 22.3] < 22.0))
            visible = int(np.any(mag_vals < limit_mag_dict[band]["mag"]))
            row[band] = visible # e.g. row[band] becomes "lsstg"
        else:
            row[band] = 222 #test value to know it is working
    return row

    
def determine_visibility(targets, bands):


    rows = []

    for idx in targets.data.index:
        target = targets[idx]
        target_data = target.data
        # compute valid bands
        valid_bands = get_valid_bands(target.template.sncosmo_model, target_data["z"][0], bands)
        # compute light curve
        _, lc = generate_snia_lightcurve(*target_data.values(), bands = valid_bands, p = p_cosmology, N_tot = 1, plot_curve = False, return_models = False, return_values = True, in_mag = True)
        # compute visibility row
        rows.append(make_visibility_row(lc, valid_bands))

    return pd.DataFrame(rows, index=targets.data.index)


def determine_visibility(target, lc, valid_bands, return_df = False):
    """
    Determine visibility of ONE target using lightcurve input.
    Build a dataframe which naturally have NaN for bands where SN is not visible
    This dataframe becomes a dict.

    Input:
        - target: snia target instance using generate_snia_lightcurve()
        - lc: lightcurve of the target using generate_snia_lightcurve()
        - valid_bands: use lightcurves_data["lc_by_band"].keys() to get the valid bands for the target

    limit_mag_dict should be the default list of 6 bands and their selected bands.

    TO DO: if a filter does not appear, how to handle it??

    Returns:
        - ONE row (if return_df = False)
        dataframe (if return_df = True) of visibility for the target.
    """
    row = []

    t0 = target.data["t0"][0] # I believe 0 gives the value itself?
    phase_start = target.template.get().mintime()
    phase_stop = min(target.template.get().maxtime(), 200.0)
    times = np.linspace(phase_start, phase_stop, 500) + t0

    row.append(make_visibility_row(lc, valid_bands, limit_mag_dict = limit_mag_dict))

    # make df first, then convert to dict using pd.Dataframe.to_dict()

    snia_visibility = pd.DataFrame(row)
    if return_df:
        return snia_visibility
    else:
        return snia_visibility.to_dict()

# almost worked.. but I noticed it re-computes everything..:

def determine_visibility(targets, bands=lsst_bands):
    """
    Assuming a constant transient type.
    skysurvey stores multiple targets of the same transient type generated with .from_draw with indexes.
    
    """
    rows = []
    # extract the first target to get the phase range
    phase_start = targets[0].template.get().mintime()
    phase_stop = min(targets[0].template.get().maxtime(), 200)
    phases = np.linspace(phase_start, phase_stop, 500)

    for target in targets:
        z = target.data["z"][0]
        valid_bands = get_valid_bands(target.template.sncosmo_model, z, bands)
        if not valid_bands:
            rows.append(make_visibility_row([], []))
            continue
        t0 = target.data["t0"][0]
        print(f"Target {target} has t0 = {t0}")
        times = phases + t0

        lc = target.get_lightcurve(band=valid_bands,times=times,index=0,in_mag=True,)
        rows.append(make_visibility_row(lc, valid_bands))

    return pd.DataFrame(rows, index=targets.data.index)

def plot_lightcurve(lightcurve_data, bands = bands, in_mag=False):
    """
    for ONE given lightcurve_data (nested dictionary of times and lc_by_band)

    Input:
        - lightcurve_data (dict): nested dict object generated using generate_snia_lightcurve()
            - if N_tot > 1: use lightcurve_data[idx].
        - bands: you should pass bands = valid_bands (see generate_snia_lightcurve())
        - in_mag (bool): if True, plot y axis in magnitude (inverted y axis). If False, plot in flux (phot/s/cm^2)
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    times = lightcurve_data["times"]
    lc_by_band = lightcurve_data["lc_by_band"]

    for band, y_vals in zip(bands, lc_by_band.values()):
        ax.plot(times, y_vals,label=band,color=limit_mag_dict[band]["color"],)
    
    if in_mag:
        for band, v in limit_mag_dict.items():
                ax.axhline(v["mag"], color=v["color"], linestyle="--", label=f"{band} limit")
        ax.invert_yaxis()
    ax.set_xlabel('MJD')
    ax.set_ylabel("mag" if in_mag else "Flux (phot/s/cm²)")
    ax.set_title('SNIa z={}, x1={}, c={}, t0={}, magabs={}, radec = [{}, {}]'.format(redshift, x1, c, t0, magabs, ra, dec), fontsize=7)
    ax.legend(fontsize=8)

    plt.show()

# to inspect the unique filters:
import sqlite3
import pandas as pd

conn = sqlite3.connect(opsim_path)

pd.read_sql("SELECT DISTINCT filter FROM observations",conn)

def five_sigma_detection(target_data = None, return_index = False):
    """
    Checks if one target (as part of N_tot) fulfills Detection Criterion 2
    Det. Cri. 2: Five sigma detection (flux/fluxerr, in-built into)
    
    Input:
        - target_data: use dset.data.loc[index] to specify SNIa target
    """
    if target_data is None:
        raise ValueError('target_data input required.')
    
    # first match each observation (all in same fieldid)
    target_data['SNR'] = target_data['flux']/target_data['fluxerr']\
    if len(.)
    obs_idx
    # if sum(target_data['SNR'] >= 5.0) > 0:
    #     obs_idxs = target_data[target_data['SNR'] >= 5.0].index
    #     return True, obs_idxs if return_index else True
    
    return False

def make_lsst_visibility_row(band_data):
    row = {band: -1 for band in limit_mag_dict}
    flux = band_data['flux']
    row = int(np.any(np.asarray(flux) < limit_mag_dict[band]["mag"]))
    return row

def five_sigma_detection_by_band(target_data = None, return_index = False):
    """
    Checks if one target (as part of N_tot) fulfills Detection Criterion 2
    Det. Cri. 2: Five sigma detection (flux/fluxerr, in-built into)
    
    Input:
        - target_data: use dset.data.loc[index] to specify SNIa target
    """
    if target_data is None:
        raise ValueError('target_data input required.')
    
    # first match each observation (all in same fieldid)

    unique_bands = target_data['band'].unique()
    target_data['SNR'] = target_data['flux']/target_data['fluxerr']

    for band in unique_bands:
        band_data = get_band_data(target_data = target_data, band = band)
        
    if sum(band_data['SNR'] >= 5.0) > 0:
        obs_idxs = target_data[target_data['SNR'] >= 5.0].index
        return True, obs_idxs if return_index else True

    
    return False
