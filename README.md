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
* `lightcurve.py`: Main module for lightcurve simulation. This is where we want to add all the methods that use `skysurvey` with the population paramters given by `generate_random_transients`
 



