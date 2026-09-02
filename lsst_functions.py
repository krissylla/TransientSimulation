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

from lightcurve import get_valid_bands, generate_snia_lightcurve, determine_visibility

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
            cmap=cmap,
            title=f"Night {night_number}",
            sub=(n_nights, 1, i + 1),
        )
        ax = plt.gca()
        ax.set_title(
            f"Night {night_number}",
            fontsize=8,
        )

    plt.show()


def set_selection_criteria(total_points, n_filters, min_points_per_filter):
    """
    Create a dict of selection criteria of a given source which is flexible.
    We have our own estimated set of selection criteria.
    
        - total_points: minimum total number of n_detections neede to be visible
        - n_filters: minimum number of filters the source needs to be visible in
        - min_pts_per_filter: the first n_filtersneed to each have > min_pts_per_filter.
                            If True, extra filters with less are counted as visible too

        e.g. you need minimum 10 points, and in 3 filters you need at least 3 points --> put (10, 3, 3)
    """
    for value in [total_points,n_filters,min_points_per_filter]:
        if value is not None and not isinstance(value, int):
            raise TypeError("Input parameters must be either None or int.")

    return {"total_points": total_points,"n_filters": n_filters,"min_points_per_filter": min_points_per_filter,}
    

# functions for total sources detected

def exist_in_multiband_check(indiv_data, selection_criteria = None):
    """
    We only say a source is detected if there is minimum n_detections (from .get_ndetection)
    and two filters
    Input:
        - indiv_data (pd.Series): dataset.get_data(index=index,detection=True).loc[index] for chosen index
        - selection_criteria (dict): contains total_pts, min_filters, min_pts_per_filter 
    """

    if selection_criteria is None:
        return True

    total_points = selection_criteria["total_points"]
    n_filters = selection_criteria["n_filters"]
    min_points = selection_criteria["min_points_per_filter"]

    # Minimum total number of detections
    if total_points is not None and indiv_data.sum() < total_points:
        return False

    # Sort by number of detections
    indiv_sorted = indiv_data.sort_values(ascending=False)

    # At least n_filters must satisfy min_points
    if n_filters is not None:

        if len(indiv_sorted) < n_filters:
            return False

        top_filters = indiv_sorted.iloc[:n_filters]

        if min_points is not None:
            if (top_filters < min_points).any():
                return False

    return True


def five_sigma_detection_multiband(dataset, targets, index=0, selection_criteria = None):
    """
    Not splitting by band. As long as it passes the selection criteria (with exist_in_multiband_check)
    valid_bands is probably not needed due to .get_ndetections() being used, but good to be safe.
    """
    z = targets.data.loc[index, "z"]

    valid_bands = get_valid_bands(model=targets.template.sncosmo_model,redshift=z,bands=lsst_bands,)
 
    detection_rows = dataset.get_data(index=index,detection=True) #auto 5 sigma

    if len(detection_rows) == 0:
        return False

    indiv_data = detection_rows["band"].value_counts()

    if not exist_in_multiband_check(indiv_data, selection_criteria = selection_criteria):
        return False
    
    return True

def detectability_sources(snia_param_list = None, opsim = None, N_tot = None, redshift_list = None, selection_criteria = set_selection_criteria(total_points=5,n_filters=2,min_points_per_filter=2)):
    """

    For a given redshift list, loop through each redshift. Sees if there are any available sources for 

    Input:
        - snia_param_list (list): parameter listed redshift, x1, c, t0, magabs, ra, dec
    Output:
        -theoretical and actual number of sources detected
    """
    from astropy.time import Time
    
    if N_tot == None:
        raise ValueError('N_tot must be a non-zero value!')

    all_summaries = []

    if opsim is None:
        raise TypeError('opsim cannot be empty! Choose a fixed lsst param')
    lsst = opsim

    tmin, tmax = lsst.date_range
    print(f"Start: {Time(tmin, format='mjd').iso}")
    print(f"End:   {Time(tmax, format='mjd').iso}")
    print(f"Duration: {(tmax - tmin)/365.25:.1f} years")

    for redshift in redshift_list:
        snia_param_list[0] = redshift
        print(snia_param_list)
        snia_models, data_models = generate_snia_lightcurve(*snia_param_list, bands=lsst_bands, in_mag=False,
        N_tot=N_tot, zp=30, tstart = tmin, tstop = tmax,plot_curve=False,
        return_values=True, return_models=True, verbose=False
        )
        dset = skysurvey.DataSet.from_targets_and_survey(snia_models, lsst, progress_bar=True, discard_bands=True)
        dset_target_indices = dset.data.index.get_level_values(0).unique().values

        detected = []
        detected_theoretical = []

        for index in snia_models.data.index:
        
            if index in dset_target_indices:
                is_detected = five_sigma_detection_multiband(
                    dataset=dset,
                    targets=snia_models,
                    index=index,
                    selection_criteria=selection_criteria
                )
            else:
                is_detected = False

            detected.append(is_detected)

            theoretical = determine_visibility(data_models[index],return_dict=True,in_mag=False) #returns a dict

            # At least 2 theoretically visible bands
            is_detected_theoretical = (sum(theoretical.values()) >= 2)

            detected_theoretical.append(is_detected_theoretical)

        n_detected = sum(detected)
        n_detected_theoretical = sum(detected_theoretical)
    
        summary = {
            "z": redshift,
            "n_detected": n_detected,
            "fraction_detected": n_detected / N_tot,
            "n_detected_theoretical": n_detected_theoretical,
            "fraction_detected_theoretical": n_detected_theoretical / N_tot,
            "x1": snia_models.data["x1"].iloc[0],
            "c": snia_models.data["c"].iloc[0],
            "t0": snia_models.data["t0"].iloc[0],
            "magabs": snia_models.data["magabs"].iloc[0],
            "ra": snia_models.data["ra"].iloc[0],
            "dec": snia_models.data["dec"].iloc[0],
        }

        all_summaries.append(summary)
    
    final_summary = pd.DataFrame(all_summaries)
    
    return final_summary


def get_total_alerts_from_survey(snia_param_list = None, opsim = None, N_tot = None, selection_criteria = set_selection_criteria(total_points=5,n_filters=2,min_points_per_filter=2)):
    """
    For a given survey input, generate expected alerts/detections.

    ***This is the MAIN script to be run through bash.***

    Parameters
    ----------
    size : int, optional
        Number of target you want to sample. If None, 1 is assumed.
        Ignored if `nyears` is given. By default None.
            
    Returns
    ----------

    int:
        Total number of alerts expected across all observations of lsst.

    Notes
    ----------
    This function can be combined with:
        z_bins = np.arange(0, 1.01, 0.02)
        detected["z_bin"] = pd.cut(detected["z"],bins=z_bins,right=False)
    which returns
        1. The number of sources detectable per redshift bin
        2. The number of alerts per redshift bin

    """

    if snia_param_list is None:
        raise ValueError('Parameter list required! If none, an input with all None values is required.')

    #useful to keep this snippet in all lsst_functions here.

    if opsim is None:
        raise TypeError('opsim cannot be empty! Choose a fixed lsst param')
    lsst = opsim

    tmin, tmax = lsst.date_range
    print(f"Start: {Time(tmin, format='mjd').iso}")
    print(f"End:   {Time(tmax, format='mjd').iso}")
    print(f"Duration: {(tmax - tmin)/365.25:.1f} years")

    tmin_SN = tmin - 100 #we want some t0 to be before the survey starts
    snia_models, data_models = generate_snia_lightcurve(*snia_param_list, bands=lsst_bands, in_mag=False,
        N_tot=N_tot, zp=30, tstart = tmin_SN, tstop = tmax,plot_curve=False,
        return_values=True, return_models=True, verbose=False
        )
        
    dset = skysurvey.DataSet.from_targets_and_survey(snia_models, lsst, progress_bar=True, discard_bands=True)
    dset_target_indices = dset.data.index.get_level_values(0).unique().values

    detected = [] #
    is_detected = False #initialise bool

    for index in snia_models.data.index:
        total_detection_points = 0
        is_detected = False

        if index in dset_target_indices:
            is_detected = five_sigma_detection_multiband(dataset=dset,targets=snia_models,index=index,selection_criteria=selection_criteria)
        else:
            is_detected = False
            total_detection_points = 0

        if is_detected:
            total_detection_points = get_number_detections_allbands(dataset=dset,targets=snia_models,index=index,selection_criteria=selection_criteria)
        # you need to append in order
        detected.append({
            "target": index,
            "z": snia_models.data.loc[index, "z"],
            "x1": snia_models.data.loc[index, "x1"],
            "c": snia_models.data.loc[index, "c"],
            "t0": snia_models.data.loc[index, "t0"],
            'magabs': snia_models.data.loc[index, "magabs"],
            'magobs': snia_models.data.loc[index, "magobs"],
            "ra": snia_models.data.loc[index, "ra"],
            "dec": snia_models.data.loc[index, "dec"],
            "detected": is_detected,
            "n_detections": total_detection_points,

        })
    detected = pd.DataFrame(detected)

    # to parquet here??
    # make function return parquet? I think you should do it externally in the submit .py script.

    return detected # you need to convert to parquet later.





def get_total_alerts_from_survey_by_band(snia_param_list = None, opsim = None, N_tot = None, selection_criteria = set_selection_criteria(total_points=5,n_filters=2,min_points_per_filter=2)):
    """
    For a given survey input, generate expected alerts/detections.

    ***This is the MAIN script to be run through bash.***

    Parameters
    ----------
    size : int, optional
        Number of target you want to sample. If None, 1 is assumed.
        Ignored if `nyears` is given. By default None.
            
    Returns
    ----------

    int:
        Total number of alerts expected across all observations of lsst.

    Notes
    ----------
    This function can be combined with:
        z_bins = np.arange(0, 1.01, 0.02)
        detected["z_bin"] = pd.cut(detected["z"],bins=z_bins,right=False)
    which returns
        1. The number of sources detectable per redshift bin
        2. The number of alerts per redshift bin

    """

    if snia_param_list is None:
        raise ValueError('Parameter list required! If none, an input with all None values is required.')

    #useful to keep this snippet in all lsst_functions here.

    if opsim is None:
        raise TypeError('opsim cannot be empty! Choose a fixed lsst param')
    lsst = opsim

    tmin, tmax = lsst.date_range
    print(f"Start: {Time(tmin, format='mjd').iso}")
    print(f"End:   {Time(tmax, format='mjd').iso}")
    print(f"Duration: {(tmax - tmin)/365.25:.1f} years")

    tmin_SN = tmin - 100 #we want some t0 to be before the survey starts
    snia_models, data_models = generate_snia_lightcurve(*snia_param_list, bands=lsst_bands, in_mag=False,
        N_tot=N_tot, zp=30, tstart = tmin_SN, tstop = tmax,plot_curve=False,
        return_values=True, return_models=True, verbose=False
        )
        
    dset = skysurvey.DataSet.from_targets_and_survey(snia_models, lsst, progress_bar=True, discard_bands=True)
    dset_target_indices = dset.data.index.get_level_values(0).unique().values

    detected = [] #
    is_detected = False #initialise bool

    for index in snia_models.data.index:
        total_detection_points = 0
        is_detected = False

        if index in dset_target_indices:
            is_detected = five_sigma_detection_multiband(dataset=dset,targets=snia_models,index=index,selection_criteria=selection_criteria)
        else:
            is_detected = False
            total_detection_points = 0

        if is_detected:
            total_detection_points = get_number_detections_allbands(dataset=dset,targets=snia_models,index=index,selection_criteria=selection_criteria)

        # you need to append in order
        detected.append({
            "target": index,
            "z": snia_models.data.loc[index, "z"],
            "x1": snia_models.data.loc[index, "x1"],
            "c": snia_models.data.loc[index, "c"],
            "t0": snia_models.data.loc[index, "t0"],
            'magabs': snia_models.data.loc[index, "magabs"],
            'magobs': snia_models.data.loc[index, "magobs"],
            "ra": snia_models.data.loc[index, "ra"],
            "dec": snia_models.data.loc[index, "dec"],
            "detected": is_detected,
            "n_detections": total_detection_points,
            "lsstu"

        })
    detected = pd.DataFrame(detected)

    # to parquet here??
    # make function return parquet? I think you should do it externally in the submit .py script.

    return detected # you need to convert to parquet later.

def add_zbin_column(dataframe, zbin_max = 1.0, zbin_size = 0.2):
    """
    
    Parameters
    -----------
    detected_df (pandas.DataFrame):
        dataframe containing all sourcess and their params + detectability + n_detections.
        It needs to have a column called 'detected' which is True or False
    
    zbin_max (int or float):
        maximum range of the redshifts to 
    
    Returns
    -----------
    detected_models (pandas.DataFrame):
        dataframe with the z_bin column included. This informs each target the redshift bin to be added in for histogram plotting.
    """

    if zbin_size < 0.001:
        raise ValueError('Redshift increment is too small! Accepted values are >= 0.001')
    
    detected_models = dataframe[dataframe['detected'] == True] # == True is optional
    # params = ['z', 'x1', 'c', 't0', 'magabs', 'magobs', 'ra', 'dec']
    z_bins = np.arange(0, zbin_max + 0.001, zbin_size)
    detected_models["z_bin"] = pd.cut(detected_models["z"],bins=z_bins,right=False)

    return detected_models #with one additional column

def get_nsources_nalerts_per_zbin(detected_models):

    """
    With the dataframe of only all detected SNIa models.

    Returns
    -----------
    n_sources (pandas.Series):
        total number of detected targets per redshift bin

    n_alerts (pandas.Series):
        total number of alerts expected over entire lsst observation per redshift bin
    """

    if 'z_bin' not in detected_models.columns:
        raise AttributeError('"z_bin" column missing from dataframe! See add_zbin_column()')
    detected_sources = detected_models[detected_models["detected"]]
    n_sources = (detected_sources.groupby("z_bin", observed=True).size())
    n_alerts = (detected_sources.groupby("z_bin", observed=True)["n_detections"].sum())
    return n_sources, n_alerts

























def get_number_detections_allbands(dataset, targets, index=0, selection_criteria = None):
    """
    Combined n_detections across all bands for one target.
	This target still needs to pass the selection criterion
    """
    detection_rows = dataset.get_data(index=index,detection=True)
    indiv_data = detection_rows["band"].value_counts()

    # still need to pass the selection criteria
    # one of the initial studies is to understand avg n_detections per source
    if not exist_in_multiband_check(indiv_data,selection_criteria=selection_criteria):
        return 0 # set zero for 

    return len(detection_rows)



def n_detections_sources(snia_param_list = None, opsim = None, N_tot = None, redshift_list = None, selection_criteria = set_selection_criteria(total_points=5,n_filters=2,min_points_per_filter=2)):
    from astropy.time import Time
    all_summaries = []
    
    if opsim is None:
        raise TypeError('opsim cannot be empty! Choose a fixed lsst param')
    
    lsst = opsim

    tmin, tmax = lsst.date_range #automatically considers length of observation
    print(f"Start: {Time(tmin, format='mjd').iso}")
    print(f"End:   {Time(tmax, format='mjd').iso}")
    print(f"Duration: {(tmax - tmin)/365.25:.1f} years")

    for redshift in redshift_list:
        snia_param_list[0] = redshift
        print(snia_param_list) 
        snia_models, data_models = generate_snia_lightcurve(
                *snia_param_list,
                bands=lsst_bands,
                in_mag=False,
                N_tot=N_tot,
                tstart = tmin - 100,
                tstop = tmax,
                zp=30,
                plot_curve=False,
                return_values=True,
                return_models=True,
                verbose=False
            )
    
        dset = skysurvey.DataSet.from_targets_and_survey(snia_models,lsst,progress_bar=True,discard_bands=True)
    
        total_results = 0
        unfiltered_detections = 0
    
        dset_target_indices = dset.data.index.get_level_values(0).unique().values
    
        for index in snia_models.data.index:
            if index in dset_target_indices:
                total_result = get_number_detections_allbands(dataset=dset,targets=snia_models,index=index,selection_criteria=selection_criteria)
                total_results += total_result
            else:
                pass
    
        unfiltered_detections = dset.get_ndetection().sum()
        total_summary = {"z": redshift, "n_det": total_results, "n_det_unfiltered": unfiltered_detections}
    
        for col in ["x1", "c", "t0", "magabs", "ra", "dec"]:
            total_summary[col] = snia_models.data[col].iloc[0]
    
        all_summaries.append(total_summary)
    
    total_summary = pd.DataFrame(all_summaries)
    
    return total_summary
