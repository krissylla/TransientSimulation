import astropy.units as u
#parameter dictionary for the visualization codes
p = {
    'map_dir': '/Users/cristozilleruelo/Rubicube-Ampel-HU-astro/RubiCube/ic_alerts',
    'save_dir': '/Users/cristozilleruelo/Rubicube-Ampel-HU-astro/RubiCube/plots',
    'file_dir': '/Users/cristozilleruelo/Rubicube-Ampel-HU-astro/RubiCube',
    'json_alerts': 'filtered_alerts.json',
    'selected_alerts': 'selected_alerts.json',
    'yaml_test': '/Users/cristozilleruelo/Rubicube-Ampel-HU-astro/RubiCube/test.yaml',
    'Eff_area': {
        'dir': '/Users/cristozilleruelo/Desktop/DESY/Thesis/IceCat-1/eff_area/',
        'gfu-bronze': 'Effa_gfubronze.txt',
        'gfu-gold': 'Effa_gfugold.txt',
        'hese-bronze': 'Effa_hesebronze.txt',
        'hese-gold': 'Effa_hesegold.txt',
        'ehe-gold': 'Effa_ehegold.txt',
        'all-gold': 'Effa_all_streams_gold_only.txt',
        'all-gold-bronze': 'Effa_all_streams_gold_bronze.txt'}
}
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
        'TDE': 8e-7 * u.Mpc **  -3  * u.yr ** -1, #?  sun et al maybe
        'KN': 59.3 * u.Gpc ** -3 * u.yr ** -1, #BTS ZTF, upper limit
    }
}