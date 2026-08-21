import pandas as pd
from lightcurve import lsst_bands

def make_visibility_row(lc_by_band, limit_mag_dict = limit_mag_dict, in_mag = True):
    #initialise every band as -1 meaning not available.
    row = {band: 0 for band in limit_mag_dict}
    
    for band, y_vals in lc_by_band.items():
        if in_mag:
            row[band] = int(
                np.any(np.asarray(y_vals) < limit_mag_dict[band]["mag"])
            ) #returns 0 (falls within filter wavelength range but too faint), 1 if visible
        else:
            row[band] = int(
                np.any(np.asarray(y_vals) < limit_mag_dict[band]["flux"])
            )

    return row

def five_sigma_detection_by_band(dataset, targets, index=0):
    z = targets.data.loc[index, "z"]

    valid_bands = get_valid_bands(model=targets.template.sncosmo_model,redshift=z,bands=lsst_bands,)
    # row = {band: -1 if band not in valid_bands else 0 for band in lsst_bands} #initialise row dict as 0

    row = {band: 0 for band in lsst_bands}
    detection_rows = dataset.get_data(index=index,detection=True)

    for band in valid_bands:
        band_data = detection_rows[detection_rows["band"] == band]
        if len(band_data) > 0:
            row[band] = 1

    return row

def set_selection_criteria(total_points, n_filters, min_points_per_filter):
    """
    Create a dict of selection criteria of a given source which is flexible
    
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

def five_sigma_detection_by_band_multiband(dataset, targets, index=0, selection_criteria = None):

    # very first filter will be

    z = targets.data.loc[index, "z"]

    valid_bands = get_valid_bands(model=targets.template.sncosmo_model,redshift=z,bands=lsst_bands,)
    # row = {band: -1 if band not in valid_bands else 0 for band in lsst_bands} #initialise row dict as 0

    row = {band: 0 for band in lsst_bands}
    detection_rows = dataset.get_data(index=index,detection=True)

    if len(detection_rows) == 0:
        return row

    # Count 5-sigma detections per band
    indiv_data = detection_rows["band"].value_counts()

    if not exist_in_multiband_check(indiv_data, selection_criteria = selection_criteria):
        return row #all 0s
    
    for band in valid_bands:
        band_data = detection_rows[detection_rows["band"] == band]
        if len(band_data) > 0:
            row[band] = 1

    return row

def flux_limit_detection_by_band(dataset, targets, index=0):
    """
    Third condition: skysurvey five-sigma AND above our flux limit.
    How do we confirm the source though?
    """
    z = targets.data.loc[index, "z"]
    valid_bands = get_valid_bands(model=targets.template.sncosmo_model,redshift=z,bands=lsst_bands)
    # row = {band: 0 if band not in valid_bands else 0 for band in lsst_bands}
    row = {band: 0 for band in lsst_bands}
    detection_rows = dataset.get_data(index=index,detection=True).copy()

    if len(detection_rows) == 0:
        return row

    flux_lookup = {band: limit_mag_dict[band]["flux"] for band in limit_mag_dict}

    detection_rows["limit_flux"] = detection_rows["band"].map(flux_lookup)

    for band in valid_bands:
        band_data = detection_rows[detection_rows["band"] == band]
        if np.any(band_data["flux"] >= band_data["limit_flux"]):
            row[band] = 1
        
    return row

def flux_limit_detection_by_band_multiband(dataset, targets, index=0):
    """
    Third condition: skysurvey five-sigma AND above our flux limit.
    How do we confirm the source though?
    """
    z = targets.data.loc[index, "z"]
    valid_bands = get_valid_bands(model=targets.template.sncosmo_model,redshift=z,bands=lsst_bands)
    # row = {band: 0 if band not in valid_bands else 0 for band in lsst_bands}
    row = {band: 0 for band in lsst_bands}
    detection_rows = dataset.get_data(index=index,detection=True).copy() #you have index as input so this 1 row


    if len(detection_rows) == 0:
        return row

    # Count detections in each band
    indiv_data = detection_rows["band"].value_counts()

    if not exist_in_multiband_check(indiv_data):
        return row

    flux_lookup = {band: limit_mag_dict[band]["flux"] for band in limit_mag_dict}

    detection_rows["limit_flux"] = detection_rows["band"].map(flux_lookup)

    for band in valid_bands:
        band_data = detection_rows[detection_rows["band"] == band]
        if np.any(band_data["flux"] >= band_data["limit_flux"]):
            row[band] = 1
        
    return row

def compare_detection_conditions(dataset,targets,data_models,index=0):

    theoretical = determine_visibility(data_models[index],return_dict=True,in_mag=False,)
    five_sigma = five_sigma_detection_by_band(dataset,targets,index=index)
    flux_limit = flux_limit_detection_by_band(dataset,targets,index=index)
    five_sigma_multi = five_sigma_detection_by_band_multiband(dataset,targets,index=index)
    flux_limit_multi = flux_limit_detection_by_band_multiband(dataset,targets,index=index)

    result = pd.DataFrame(
        [theoretical, five_sigma, flux_limit, five_sigma_multi, flux_limit_multi],
        index =["theoretical","5_sigma","5_sigma_flux_limit", "five_sigma_multi", "flux_limit_multi"],)

    result.index.name = "condition"
    result.insert(0, "target", index)

    return result.reset_index()


# n_detections code

from lightcurve import lsst_bands

def get_number_detections_band_five_sigma(dataset, targets, index=0):
    z = targets.data.loc[index, "z"]

    # valid_bands = get_valid_bands(model=targets.template.sncosmo_model,redshift=z,bands=lsst_bands,)
    # row = {band: -1 if band not in valid_bands else 0 for band in lsst_bands} #initialise row dict as 0

    row = {band: 0 for band in lsst_bands} #initiate
    detection_rows = dataset.get_data(index=index,detection=True)

    for band in lsst_bands:
        band_data = detection_rows[detection_rows["band"] == band]
        row[band] = len(band_data)

    return row

def get_number_detections_band_five_sigma_multiband(dataset, targets, index=0):
    z = targets.data.loc[index, "z"]

    # valid_bands = get_valid_bands(model=targets.template.sncosmo_model,redshift=z,bands=lsst_bands,)
    # row = {band: -1 if band not in valid_bands else 0 for band in lsst_bands} #initialise row dict as 0

    row = {band: 0 for band in lsst_bands} #initiate
    detection_rows = dataset.get_data(index=index,detection=True)

    indiv_data = detection_rows["band"].value_counts()

    if not exist_in_multiband_check(indiv_data): #we should expect >=5 for all plots
        return row

    for band in lsst_bands:
        band_data = detection_rows[detection_rows["band"] == band]
        row[band] = len(band_data)

    return row

def get_number_detections_band_flux_limit(dataset, targets, index=0):
    """
    Third condition: skysurvey five-sigma AND above our flux limit.
    """
    # z = targets.data.loc[index, "z"]
    # valid_bands = get_valid_bands(model=targets.template.sncosmo_model,redshift=z,bands=lsst_bands)
    # row = {band: 0 if band not in valid_bands else 0 for band in lsst_bands}
    row = {band: 0 for band in lsst_bands}
    detection_rows = dataset.get_data(index=index,detection=True).copy()

    if len(detection_rows) == 0:
        return row

    flux_lookup = {band: limit_mag_dict[band]["flux"] for band in limit_mag_dict}
    detection_rows["limit_flux"] = detection_rows["band"].map(flux_lookup)

    for band in lsst_bands:
        band_data = detection_rows[detection_rows["band"] == band]
        filtered_data = band_data[band_data["flux"] >= band_data["limit_flux"]]
        row[band] = len(filtered_data)
    return row

def get_number_detections_band_flux_limit_multiband(dataset, targets, index=0):
    """
    Third condition: skysurvey five-sigma AND above our flux limit.
    """
    # z = targets.data.loc[index, "z"]
    # valid_bands = get_valid_bands(model=targets.template.sncosmo_model,redshift=z,bands=lsst_bands)
    # row = {band: 0 if band not in valid_bands else 0 for band in lsst_bands}
    row = {band: 0 for band in lsst_bands}
    detection_rows = dataset.get_data(index=index,detection=True).copy()

    if len(detection_rows) == 0:
        return row

    indiv_data = detection_rows["band"].value_counts()

    if not exist_in_multiband_check(indiv_data): #we should expect >=5 for all plots
        return row

    flux_lookup = {band: limit_mag_dict[band]["flux"] for band in limit_mag_dict}
    detection_rows["limit_flux"] = detection_rows["band"].map(flux_lookup)

    for band in lsst_bands:
        band_data = detection_rows[detection_rows["band"] == band]
        filtered_data = band_data[band_data["flux"] >= band_data["limit_flux"]]
        row[band] = len(filtered_data)
    return row


def compare_ndetections_by_condition(dataset,targets, index=0):
    """
    Just two conditions since theoretical lightcurve does not give discrete datapoints (there is actl 500 but says nothing about obs style)
    
    data_model: snia_model.data
    """
    if targets is None:
        raise ValueError('Input model required.')
    
    five_sigma = get_number_detections_band_five_sigma(dataset,targets,index=index)

    flux_limit = get_number_detections_band_flux_limit(dataset,targets,index=index)

    five_sigma_multi = get_number_detections_band_five_sigma_multiband(dataset,targets,index=index)

    flux_limit_multi = get_number_detections_band_flux_limit_multiband(dataset,targets,index=index)
    
    result = pd.DataFrame(
        [five_sigma, flux_limit, five_sigma_multi, flux_limit_multi],
        index=["5_sigma", "5_sigma_flux_limit", "5_sigma_multi","5_sigma_flux_limit_multi"]
    )

    result.index.name = "condition"

    metadata = {}
    metadata["target"] = index
    metadata["z"] = targets.data["z"].loc[index]  # Get z from data, not hardcoded
   

    for param in ['x1', 'c', 't0', 'magabs', 'ra', 'dec']:
        metadata[param] = targets.data[param].loc[index]
    
    # Insert all metadata columns at the beginning
    for col_name, col_value in metadata.items():
        result.insert(0, col_name, col_value)

    return result.reset_index()