import sys
import os


import numpy as np
import matplotlib.pyplot as plt
import scipy
import astropy.units as u
from astropy.time import Time
import astropy.cosmology as acosmo
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import pandas as pd
from scipy.interpolate import interp1d
import ligo.skymap.plot #used implicitely
import mhealpy as mhp
import healpy as hp
import copy


sys.path.insert(1, '../.')
from transient_rates import R_SFR, R_TDE, R_sGRB_gaus, plot_rates
from params import p_cosmology
from ic_visualization import read_map, plot_map, evaluate_map, get_map_area, \
    get_moc_map_metadata, get_max_ra_dec_from_map, is_source_in_contour_area



def plot_population_with_alerts(population, alerts, r=1, p=0.9, plot_specific_alert=None):

    '''
    population: dict. Transient poopulation
    alerts: list of list. Randmom;y generated alerts. Each element is [QTable, HealpixMap]
    '''
    ra = population['ra']
    dec= population['dec']

    

    
    #get total map for plotting
    if isinstance(plot_specific_alert, int):
         original_alert = alerts.pop(plot_specific_alert)
    else:
        original_alert = alerts.pop() 
    sel_alert = copy.deepcopy(original_alert)
    total_map = copy.deepcopy(sel_alert)[1]
    ra_zoom, dec_zoom = get_max_ra_dec_from_map(total_map) #we zoom in on the first generated alert

    
    
    
    
    #plotting
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.axes(
        [0.05, 0.05, 0.8, 0.9],
        projection='astro degrees mollweide',
        center=SkyCoord(0 * u.rad, 0 * u.rad, frame='icrs'))
    
    ax_inset = plt.axes(
            [1.03, 0.3, 0.42, 0.42],
            projection='astro degrees zoom',
            center=SkyCoord(ra_zoom * u.deg, dec_zoom * u.deg, frame='icrs'),
            radius=r * u.deg)
    for key in ['ra', 'dec']:
        ax_inset.coords[key].set_ticks_visible(False)


    
    

    
    # ax.set_title(r'$N_{total} =$ ' + np.format_float_scientific(len(ra), 1) + f" time = {population['time_window'].to(u.d)}  z_max={population['z_max']}")

    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")
    ax.grid()
    ax.locator_params(nbins=10) #test!
    ax.mark_inset_axes(ax_inset)
    ax.connect_inset_axes(ax_inset, 'upper left')
    ax.connect_inset_axes(ax_inset, 'lower left')
    ax_inset.scalebar((0.1, 0.1), 1 * u.deg).label()
    ax_inset.compass(0.9, 0.1, 0.2)
    ax_inset.grid()
    
    
    ax_inset.set_title(fr'RA: {ra_zoom:.2f}$\degree$  DEC: {dec_zoom:.2f}$\degree$')


    
    #plot alerts
    img_inset = sel_alert[1].get_wcs_img(ax_inset)
    im = ax_inset.imshow(img_inset, cmap='Greens')
    c = plt.colorbar(im, label=r'Probability')
    
    for _, m in alerts:
        total_map += m
    img_inset = total_map.get_wcs_img(ax)
    im = ax.imshow(img_inset, cmap='Greens')

    

    #plot SN
    ax.plot(ra, dec, '.', c='orange', ms=0.01, transform=ax.get_transform('world'))

    
    #50 & 90% ontour
    skymap = copy.deepcopy(sel_alert)[0]
    skymap.sort("PROBDENSITY", reverse=True)
    sorted_metadata = get_moc_map_metadata(skymap)
    # prob = sorted_metadata['PIXELAREA'] * skymap["PROBDENSITY"]
    cumprob = np.cumsum(skymap['PROBDENSITY'])
    # searchsorted gives the FIRST index where cumprob >= target.
    i_50 = cumprob.searchsorted(0.5)
    i_90 = cumprob.searchsorted(0.9)
    
    m_prob = mhp.HealpixMap(data=cumprob, uniq=skymap['UNIQ'], density=True)
    img = m_prob.get_wcs_img(ax_inset)#, coord = coord, rasterize = rasterize)
    im = ax_inset.contour(img, levels=[0.5, 0.9], colors=['black', 'black'], linestyles=['solid', 'dashed'])

    area_50 = sorted_metadata['PIXELAREA'][:i_50].sum().to(u.deg**2)
    area_90 = sorted_metadata['PIXELAREA'][:i_90].sum().to(u.deg**2)

    #plot inset sn

    catalog = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    center_point = SkyCoord(ra=ra_zoom * u.deg, dec=dec_zoom * u.deg)
    radius = 5 * np.sqrt(area_90) 

    
    separation = center_point.separation(catalog)
    selection_zoom = separation < radius

    ax_inset.plot(
        ra[selection_zoom],
        dec[selection_zoom],
        'o',
        c='orange', ms=1, transform=ax_inset.get_transform('world'))
    
    #we also colour the accepted pixels within a p_value
    ra_accepted, dec_accepted = is_source_in_contour_area(original_alert[1], ra[selection_zoom], dec[selection_zoom],p=p)
    ax_inset.plot(
        ra_accepted,
        dec_accepted,
        'o',
        c='cyan', ms=1, transform=ax_inset.get_transform('world'),
        label=f'{100 * p:.1f}%')
    

    

    print("Area 50%: ", area_50)
    print("Area 90%: ", area_90)

    #plot
    ax_inset.legend(framealpha=1)
    plt.show()

    return original_alert

def plot_population(population, ra_dec_zoom=None, r=1):
    '''
    Plots a simulated population in the Mollweide projection:
    
    Params:
    -------
    population: dict of arrays. Population parameter values
    ra_dec_zoom: list: [ra, dec] in degrees
    '''

    ra = population['ra']
    dec= population['dec']

    if ra_dec_zoom:
        ra_zoom, dec_zoom = ra_dec_zoom
    else:
        ra_zoom, dec_zoom = ra[0], dec[0] #we just zoom into the first alert

    
    
    #plotting
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.axes(
        [0.05, 0.05, 0.8, 0.9],
        projection='astro degrees mollweide',
        center=SkyCoord(0 * u.rad, 0 * u.rad, frame='icrs'))
    
    ax_inset = plt.axes(
            [1.03, 0.3, 0.42, 0.42],
            projection='astro degrees zoom',
            center=SkyCoord(ra_zoom * u.deg, dec_zoom * u.deg, frame='icrs'),
            radius=r * u.deg)
    for key in ['ra', 'dec']:
        ax_inset.coords[key].set_ticks_visible(False)


    
    


    ax.set_title(r'$N_{total} =$ ' + np.format_float_scientific(len(ra), 1) + f" time = {population['time_window'].to(u.d)}  z_max={population['z_max']}")

    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")
    ax.grid()
    ax.locator_params(nbins=10) #test!
    ax.mark_inset_axes(ax_inset)
    ax.connect_inset_axes(ax_inset, 'upper left')
    ax.connect_inset_axes(ax_inset, 'lower left')
    ax_inset.scalebar((0.1, 0.1), 1 * u.deg).label()
    ax_inset.compass(0.9, 0.1, 0.2)
    ax_inset.grid()
    
    
    ax_inset.set_title(fr'RA: {ra_zoom:.2f}$\degree$  DEC: {dec_zoom:.2f}$\degree$')


    


    

    #plot popualation
    ax.plot(ra, dec, '.', c='orange', ms=0.01, transform=ax.get_transform('world'))

    
    

    #plot inset (zoomed in) popoulation

    catalog = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    center_point = SkyCoord(ra=ra_zoom * u.deg, dec=dec_zoom * u.deg)
    radius = 5 * r * u.deg

    
    separation = center_point.separation(catalog)
    selection_zoom = separation < radius

    ax_inset.plot(
        ra[selection_zoom],
        dec[selection_zoom],
        'o',
        c='orange', ms=1, transform=ax_inset.get_transform('world'))
    
    
    

    # print("Area 50%: ", area_50)
    # print("Area 90%: ", area_90)

    #plot
    # ax_inset.legend(framealpha=1)
    plt.show()

    return None