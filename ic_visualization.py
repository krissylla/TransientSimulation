import os
import json
import yaml
import requests #for url downloading
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import mhealpy as mhp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable
import astropy_healpix as ah
from ligo.skymap.io.fits import read_sky_map
import ligo.skymap.plot #used impliciteely

from params import p

import warnings #for import lal inside ligo.skymap
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

def read_map(map_name, map_dir=p['map_dir']):
    '''
    Opens a map at map_path. If map_path is a url, it downloads the map in the save_dir folder
    '''

    #check if map_name is an url
    if 'https' in map_name:
        file_name = map_name.split('/')[-1]
        alert_name = file_name.split('_')[0]

        #first we download the map
        response = requests.get(map_name)
        map_path = os.path.join(map_dir, file_name)
        with open(map_path, 'wb') as file: #also deletes previously existing file and redownloads alert. 
            file.write(response.content)
    
    #if not url, then it could be a full path:
    elif os.path.exists(map_name):
        map_path = map_name
        alert_name = os.path.basename(map_path).split('_')[0]
    
    #Check if its just the filename:
    elif os.path.exists(os.path.join(map_dir, map_name)):
        map_path = os.path.join(map_dir, map_name)
        alert_name = os.path.basename(map_path).split('_')[0]
    
    #In case just the alert name like IceCube-260610A. We can check if this alert was already downloaded.
    #this will be usefull in case I want to pull up a specific alert on comand
    else:
        downloaded_maps = {alert.split('_')[0]: alert for alert in os.listdir(map_dir)}
        if map_name in downloaded_maps.keys():
            print('Opening downloaded alert: ', downloaded_maps[map_name])
            map_path = os.path.join(map_dir, downloaded_maps[map_name])
            alert_name = os.path.basename(map_path).split('_')[0]
        else:
            raise AttributeError(f'Incorrect Map name inpu: {map_name} \n. Please provide a url, map_path, or alert name in the format IceCube-YYMMDDA')


    #we open the map
    skymap = QTable.read(map_path)
    _, header = fits.getdata(map_path, header=True)

    #Check if array has masked values. We fill them with 0.0
    try:
        if skymap['PROBDENSITY'].mask.any():
            print("Map has masked values. reeplacing them with 0")
            skymap['PROBDENSITY'] = skymap['PROBDENSITY'].filled(0)
    except AttributeError:
        pass #array has no masked values

    map_data = {
        'skymap': skymap,
        'header': header,
        'map_path': map_path,
        'alert_name': alert_name
    }

    return map_data

def get_moc_map_metadata(skymap):
    '''
    Not sure if useful, but returns a dictionary with the metavalues for the multi-ordered map
    '''
    level, ipix = ah.uniq_to_level_ipix(skymap['UNIQ'])
    nside = ah.level_to_nside(level)
    pixel_area = ah.nside_to_pixel_area(nside)
    
    metadata = {
        'PROBDENSITY': skymap['PROBDENSITY'],
        'UNIQ': skymap['UNIQ'],
        'LEVEL': level,
        'IPIX': ipix,
        'NSIDE': nside,
        'PIXELAREA': pixel_area
    }

    return metadata

def plot_map(map_data, r=3, plot_prob_density=False, save_dir=p['save_dir'],
             coord='icrs', nest=True, show_fig=True, plot_ztf_lsst=True, plot_source=False,
             plot_only_zoomed_frame=False):
    
    #plotting info for ligo.skymap.plot
    ligo_coord = {
        'E': 'geo',
        'icrs': 'astro',
        'G': 'galactic'
    }
    astropy_frame = {
        'icrs': 'icrs',
        'G': 'galactic'
    }

    #get data and metadata
    skymap = map_data['skymap']
    header = map_data['header']
    alert_name = map_data['alert_name']
    metadata = get_moc_map_metadata(skymap)

    if not header:
        #we get ra and dec from maximum value of map
        arg_max = np.max(skymap['PROBDENSITY'])
        max_i = list(skymap['PROBDENSITY']).index(arg_max)
        max_ipix = metadata['IPIX'][max_i]
        ra, dec = hp.pix2ang(metadata['NSIDE'][max_i], max_ipix, lonlat=True, nest=True)
        ra = ra * u.deg
        dec = dec * u.deg

    else:
        header_dict = dict(header)
        ra = header_dict['RA'] * u.deg
        dec = header_dict['DEC'] * u.deg
    
    #zoom box center
    center = SkyCoord(ra.to('rad'), dec.to('rad'), frame=astropy_frame[coord])
    
    fig = plt.figure(figsize=(6, 6), dpi=100)

    ax = plt.axes(
        [0.05, 0.05, 0.8, 0.9],
        projection=f'{ligo_coord[coord]} degrees mollweide',
        center=SkyCoord(0 * u.rad, 0 * u.rad, frame=astropy_frame[coord]))
    ax_inset = plt.axes(
        [1.03, 0.3, 0.42, 0.42],
        projection=f'{ligo_coord[coord]} degrees zoom',
        center=center,
        radius=r * u.deg)
    
    if coord == 'icrs':
        for key in ['ra', 'dec']:
            ax_inset.coords[key].set_ticks_visible(False)
    
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
    ax.set_title(alert_name)
    ax_inset.set_title(fr'RA: {ra.value:.2f}$\degree$  DEC: {dec.value:.2f}$\degree$')


    #map processing
    if plot_prob_density:
        prob_density = skymap['PROBDENSITY'].to_value(u.deg**-2)

        m = mhp.HealpixMap(data=prob_density, uniq=skymap['UNIQ'], density=True)
        img = m.get_wcs_img(ax)#, coord = coord, rasterize = rasterize)
        ax.imshow(img, cmap='OrRd')
        img_inset = m.get_wcs_img(ax_inset)
        im = ax_inset.imshow(img_inset, cmap='OrRd')
        c = plt.colorbar(im, label=r'Prob density [$deg^{-2}$]')
    else:
        #we show absolute probability = prob_density * pixel_area
        prob = metadata['PIXELAREA'] * skymap['PROBDENSITY']

        m = mhp.HealpixMap(data=prob, uniq=skymap['UNIQ'], density=False)
        img = m.get_wcs_img(ax)#, coord = coord, rasterize = rasterize)
        ax.imshow(img, cmap='OrRd')
        img_inset = m.get_wcs_img(ax_inset)
        im = ax_inset.imshow(img_inset, cmap='OrRd')
        c = plt.colorbar(im, label=r'Probability')
    
    #get 90% and  50% confidence regions
    skymap.sort("PROBDENSITY", reverse=True)
    sorted_metadata = get_moc_map_metadata(skymap)
    prob = sorted_metadata['PIXELAREA'] * skymap["PROBDENSITY"]
    cumprob = np.cumsum(prob)
    # searchsorted gives the FIRST index where cumprob >= target.
    i_50 = cumprob.searchsorted(0.5)
    i_90 = cumprob.searchsorted(0.9)
    
    m_prob = mhp.HealpixMap(data=cumprob, uniq=skymap['UNIQ'], density=True)
    img = m_prob.get_wcs_img(ax_inset)#, coord = coord, rasterize = rasterize)
    im = ax_inset.contour(img, levels=[0.5, 0.9], colors=['black', 'black'], linestyles=['solid', 'dashed']) #nested=True,

    area_50 = sorted_metadata['PIXELAREA'][:i_50].sum().to(u.deg**2)
    area_90 = sorted_metadata['PIXELAREA'][:i_90].sum().to(u.deg**2)

    #area text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text_str = f'Area 50%:\n {area_50:.3f} \n\nArea 90%:\n {area_90:.3f}'
    ax.text(1.9, 0.35, text_str, fontsize=14, bbox=props, transform=ax.transAxes)#, horizontalalignment='right', va='top')

    #extra plotting features
    if plot_ztf_lsst:
        lsst_limit =  [np.linspace(0,360, 50), np.ones(50) * 30]
        ztf_limit = [np.linspace(0,360, 50), np.ones(50) * -30]
        ax.plot(lsst_limit[0], lsst_limit[1], c='purple', linestyle='dotted', transform=ax.get_transform('world'))
        ax.plot(ztf_limit[0], ztf_limit[1], c='green', linestyle='dotted', transform=ax.get_transform('world'))

        ax_inset.plot(lsst_limit[0], lsst_limit[1], c='purple', linestyle='dotted', transform=ax_inset.get_transform('world'))
        ax_inset.plot(ztf_limit[0], ztf_limit[1], c='green', linestyle='dotted', transform=ax_inset.get_transform('world'))
        
        ax.text(175, 20, r'$\downarrow$LSST', c='purple', transform=ax.get_transform('world'))
        ax.text(175, -28, r'$\uparrow$ZTF', c='green', transform=ax.get_transform('world'))

    if isinstance(plot_source, dict):
        ax_inset.plot(plot_source['ra'],
                    plot_source['dec'], 'o',
                    label=plot_source['name'],
                    c='cyan',  
                    transform=ax_inset.get_transform('world'))    
        ax_inset.legend(framealpha=1)

    elif isinstance(plot_source, list):
        for source in plot_source:
            ax_inset.plot(source['ra'],
                        source['dec'], 'o',
                        label=source['name'],
                        transform=ax_inset.get_transform('world'))    
            ax_inset.legend(framealpha=1)

    if isinstance(save_dir, str):
        fig_path = os.path.join(save_dir, f'{alert_name}.png')
        plt.savefig(fig_path, bbox_inches='tight')
    if show_fig:
        plt.show()
        
    return fig_path

def evaluate_map(skymap, ra, dec, plot_source=None, **kwargs):
    '''
    Reads the absolute spatial probability from the map at a sky location. 
    
    Params:
    -------
    skymap: QTable object, original healpix map where skymap[PROBDENSITY] is in 1/sr
    ra: float, right assention in degrees. It could also have astropy.unit of degree
    dec: float, declination in degrees. It could also have astropy.unit of degree
    plot_source: list or dict or None. If given a source (dict) or a list of sources, then it 
        plots the source on the map with its probabilty value. 
    **kwargs: extra plotting arguments for plot_map

    Returns:
    -------
    spatial_prob: float, spatial probaility at ra, dec. 
    '''
    metadata = get_moc_map_metadata(skymap)
    abs_prob = metadata['PROBDENSITY'] * metadata['PIXELAREA']

    mhpx_map = mhp.HealpixMap(data=abs_prob, uniq=metadata['UNIQ'], density=False)

    spatial_prob = mhpx_map.get_interp_val(ra, dec, lonlat=True)

    if isinstance(plot_source, dict) or isinstance(plot_source, list):
        plot_map(skymap, plot_source=plot_source, **kwargs)
    
    return spatial_prob

def get_map_area(map_data, p=0.9):
    '''
    Computes the containment area up to a value of p. p=0.9 corresponds to the 90% probability area
    '''

    skymap = map_data['skymap']
    skymap.sort("PROBDENSITY", reverse=True)
    sorted_metadata = get_moc_map_metadata(skymap)
    prob = sorted_metadata['PIXELAREA'] * skymap["PROBDENSITY"]
    cumprob = np.cumsum(prob)
    # searchsorted gives the FIRST index where cumprob >= target.
    p_i = cumprob.searchsorted(p)

    p_area = sorted_metadata['PIXELAREA'][:p_i].sum().to(u.deg**2)

    return p_area

def get_max_ra_dec_from_map(m):
    ''''
    Computes the ra and dec coordinate for the maximum value in the healpy map. 
    
    
    m: HealpixMap object from mhealpy. 
    '''
    arg_max = np.argmax(m.data)
    ra, dec = hp.pix2ang(m.nside, arg_max, lonlat=True, nest=m.is_nested)
    return ra, dec

def is_source_in_contour_area(skymap, ra, dec, p=0.9, return_bool=False):
    '''
    Checks if a source's position is inside the top p contoour area. 

    m: mhealpy HealpixMap object 
    ra: float or 1darray
    dec: float or 1darray
    p: contour area to check. p=0.9 corresponds to the 90% percent contour area

    Returns:
    accepted_ra, accepted_dec: np.array. Returns the ra & dec posiitons of the objects that 
        where within the p contour region. Returns empty array if no sources accepted the criteria
    '''

    if isinstance(ra, float) or isinstance(ra, int):
        ra = np.array(ra)
    if isinstance(dec, float) or isinstance(dec, int):
        dec = np.array(dec)

    # skymap = QTable({
    #     'PROBDENSITY': m.data,
    #     'UNIQ': m.uniq
    # })

    #get uniq at ra, dec value
    metadata = get_moc_map_metadata(skymap)

    if 'ABSPROB' in skymap.keys():
        m = mhp.HealpixMap(data=skymap['ABSPROB'], nside=metadata['NSIDE'])
    else:
        
        abs_prob = skymap['PROBDENSITY'] * metadata['PIXELAREA']
        m = mhp.HealpixMap(data=abs_prob, nside=metadata['NSIDE'])

    pix_at_radec = m.ang2pix(ra, dec, lonlat=True)
    uniq_at_radec = m.pix2uniq(pix_at_radec)


    #get area contour
    
    skymap.sort("PROBDENSITY", reverse=True)
    cumprob = np.cumsum(skymap["PROBDENSITY"] * metadata['PIXELAREA'])
    # searchsorted gives the FIRST index where cumprob >= target.
    i_p = cumprob.searchsorted(p)

    accepted_uniq = skymap['UNIQ'][:i_p]

    
    bool_array = np.isin(uniq_at_radec, accepted_uniq)
    if return_bool:
        return bool_array

    accepted_ra, accepted_dec = ra[bool_array], dec[bool_array]
    return accepted_ra, accepted_dec

    
    

    #evaluate if uniq is in area



def get_neutrino_properties(map_data, alert):
    '''
    Returns a dict with the relevant neutrino properties for the Rubicube framework from the json alert'''

    neutrino_properties = dict()

    neutrino_properties['E'] = alert['nu_energy']
    neutrino_properties['ra'] = alert['RA']
    neutrino_properties['dec'] = alert['DEC']
    neutrino_properties['t_nu'] = alert['DATE-OBS']
    neutrino_properties['A_90'] = get_map_area(map_data, p=0.9)

    return neutrino_properties
    

   



if __name__ == '__main__':
    map_data =read_map('IceCube-260610A')
    plot_map(map_data, show_fig=False, plot_prob_density=True)
    print(evaluate_map(map_data['skymap'], 309, 37))
