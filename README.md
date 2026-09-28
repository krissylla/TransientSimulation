# TransientSimulation
Simulation of detectability of a transient class by an optical survey (LSST)


We need to setop a virtual enviorment with micromamba:
```
micromamba create -n TransientLightcurves numpy
micromamba activate TransientLightcurves
micromamba install pip
pip install astropy matplotlib ligo.skymap
```

Some useful links:
* [skysurvey](https://skysurvey.readthedocs.io/en/latest/) package for simulating lightcurves
* [Paper](https://www.aanda.org/articles/aa/full_html/2024/06/aa49012-23/aa49012-23.html) Paper by Petrecca et. al 2024: "Recovered supernova Ia rate from simulated LSST images" which we compare our recovery efficiency $\\epsilon$ with. Note: This paper used very different selection methods.

### General File Structure:
* `params.py`: Includes all input parameters of the simulation.
    * `p_cosmology`: dictionary of cosomlogial parameters. Here the local rate of transients $R_{0,k} is defined. 
    * `p`: dictionary or general saving parameters. Used mainly for icecube alerts. `save_dir` defines where the plot from `plotting_functions` is saved to.
* `transient_rates.py`: Definition for the volumetric rate $R_k(z)$ of a transient of type $k$ assuming a local rate $R_{0,k}$ defined in `params.py`.
* `random_population.py`: Includes the main simulation method to generate transients
    *  `get_N_k`: Computes the total number of transients up to a redshift of $z_{max}$ by integrating the volumetric rate (see `Tutorials/volumetric_rate.ipynb` for more information).
    * `generate_random_transients`: Generates a transient simulation of size $N_{tot}$, where the source's [ra, dec] i sunifeormly sampled, and the redshift is sampled from a distirbution that follows the volumetric rate evolution function $R_k(z)$. 
* `plotting_functions.py`: Includes functions that plot a given population in the Mollweide projection
    * `plot_population': Plots only the transient population
    * `plot_popoulation_with_alerts`: plots the transient population overlapped with $N$ healpix maps. Useful for overplotting the transients with neutrino alerts
* `lightcurve.py`: Main module for lightcurve simulation. This is where we want to add all the methods that use `skysurvey` with the population paramters given by `generate_random_transients`. In a jupyter notebook, these three functions are run in order: `generate_snia_dict()`, `generate_ordered_parameter_list()`, and `generate_snia_lightcurve()` which generates from particular distributions.

# added by Denzel

* `lsst_functions.py`: Includes all the key functions that analyse the LSST observation plans ("opsim"s)

* `skysurvey_prelim.ipynb`: Notebook to generate a test population and conduct preliminary studies on detection limits (redshift limit where n_sources detected or n_alerts drop to zero.)

* `script_instructions.txt`: Written instructions on how to simulate SNIa populations on the cluster, combine the parquets from each batch (since it saves computational time to run multiple small batches in parallel) and how to extract .npz file if the population is too large (>few hundred MB) to analyse in a jupyter notebook.

* `skysurvey_detections.ipynb`: A notebook for generating a custom SNIa population, combining it with what LSST will observe, and then creating a parquet which contains the parameter values and some other statistics of each SNIa.

* `population_analysis.ipynb`: This notebook is used for a large enough parquet (but filesize not too big i.e. < few hundred MB>). It includes a default parquet file which includes 11299884 SNIa which is sent separately.

* `skysurvey_fullpop_analysis.ipynb`: After an .npz file is extracted from all the 456 parquets x 200000 batch_size, this notebook visualises the distribution of parameter values, alerts per band per night, average n_alerts in each band per SNIa. 

* `job.py`: Code which takes input arguments to set the selection criteria and generate the SNIa population, before finding n_alerts per band for each SNIa source. This file produces a parquet of all the generated SNIa, with information about the param value of each source, their detectability, and detections in each band.

* `job.sub`: Submission script (which runs `job.py`) for HTCondor which runs the functions inside `lightcurve.py`and `lsst_functions.py` to generate a large population. If the population of SNIa is less than 10 million, the filesize should be < 300 MB, which might not need a cluster. However, it is encouraged to use this to save computational time + save a fixed population.

* `combine_parquets.py`: This script is run using `combine.sub` which is submitted to the cluster. Before using, ensure that you change the input_dir and output_dir accordingly. 

* `combine.sub`: Script (which runs `combine_parquets.py`) to be sent to the HTCondor cluster which combines parquets together. To be used when a large population was split into smaller `batch_size`s and then exported to the folder as separate small parquets (e.g. 213 x 50000 `batch_size`)

* `aggregate.py`: Extract an .npz file with data for each redshift bin of size z = 0.01. Includes n_detections per band. This can be run in the terminal using >>> python aggregate.py. See also the function `aggregate_by_param.py` if you want to bin data using other params.

* `aggregate_by_param.py`: Similar to .npz, but also bins each of the 8 parameters (c, x1, z, t0, ra, dec, magabs, magobs) into 100 bins. Even though this script is an imporvement to `aggregate.py`, it is kept separate as you mostly only need data for each redshift bin.
 
* `tripp1998_nicolas2021.py`: Reference script which includes the default sampling functions for magabs (Tripp1998) and the SNIa stretch modelling (NIcolas2021). Copied directly from skysurvey source code.

* `test_parquet.parquet`: pre-computed dataset of 1000 sources observed by LSST in ONE year. Used in `skysurvey_detections.ipynb`.

* `SNParquets/my_test_parquet.parquet`: 10000 sources generated across 1 year + 100*u.d. Used in `skysurvey_prelim.ipynb`.


# notebooks that you might have, but not used

* `skysurvey.ipynb`: One of the first scripts used to learn an experiment different skysurvey functions. The key functions and results were transferred into `lightcurve.py`, `lsst_functions.py` and the other jupyter notebooks.

* `skysurvey_make_10yrparquet.ipynb`: You can combine your smaller parquets into a large one here and return it to the folder. I used this to combine my 10 year population of 91 million SNIa together (~3.9 GB). Another way to analyse this large dataset will be to loop through each small parquet (e.g. 200000 SNIa), append data into specific dataframes, close it, and repeat forthe next one, to prevent crashing VSCode.

* `smallparquet.parquet`: Could be an older version of test_parquet.parquet