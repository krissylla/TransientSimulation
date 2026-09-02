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
    dset_target_indices = set(dset.data.index.get_level_values(0).unique())


    target_data = snia_models.data[["z", "x1", "c", "t0", "magabs", "magobs", "ra", "dec"]]

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