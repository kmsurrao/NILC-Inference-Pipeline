import numpy as np
import os
import subprocess
import healpy as hp
import itertools

def setup_output_dir(inp, env, scaling=False):
    '''
    Sets up directory for output files

    ARGUMENTS
    ---------
    inp: Info() object containing input specifications
    env: environment object
    scaling: Bool, whether multiplicative scale factors are used to fit parameter dependence

    RETURNS
    -------
    None
    '''
    if not os.path.isdir(inp.output_dir):
        subprocess.call(f'mkdir {inp.output_dir}', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/maps'):
        subprocess.call(f'mkdir {inp.output_dir}/maps', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/pyilc_yaml_files'):
        subprocess.call(f'mkdir {inp.output_dir}/pyilc_yaml_files', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/pyilc_outputs'):
        subprocess.call(f'mkdir {inp.output_dir}/pyilc_outputs', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/n_point_funcs'):
        subprocess.call(f'mkdir {inp.output_dir}/n_point_funcs', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/data_vecs'):
        subprocess.call(f'mkdir {inp.output_dir}/data_vecs', shell=True, env=env)
    if scaling:
        scalings = [list(i) for i in itertools.product([0, 1], repeat=5)]
        for s in scalings:
            scaling_str = ''.join(str(e) for e in s) 
            new_dir_pyilc_inputs = f'{inp.output_dir}/pyilc_yaml_files/{scaling_str}'
            if not os.path.isdir(new_dir_pyilc_inputs):
                subprocess.call(f'mkdir {new_dir_pyilc_inputs}', shell=True, env=env)
            new_dir_pyilc_outputs = f'{inp.output_dir}/pyilc_outputs/{scaling_str}'
            if not os.path.isdir(new_dir_pyilc_outputs):
                subprocess.call(f'mkdir {new_dir_pyilc_outputs}', shell=True, env=env)
            new_dir_maps = f'{inp.output_dir}/maps/{scaling_str}'
            if not os.path.isdir(new_dir_maps):
                subprocess.call(f'mkdir {new_dir_maps}', shell=True, env=env)

    return 

def tsz_spectral_response(freqs): #input frequency in GHz
    '''
    ARGUMENTS
    ---------
    freqs: 1D numpy array, contains frequencies (GHz) for which to calculate tSZ spectral response

    RETURNS
    ---------
    1D array containing tSZ spectral response to each frequency
    '''
    T_cmb = 2.726
    h = 6.62607004*10**(-34)
    kb = 1.38064852*10**(-23)
    response = []
    for freq in freqs:
        x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
        response.append(T_cmb*(x*1/np.tanh(x/2)-4)) #was factor of tcmb microkelvin before
    return np.array(response)

def GaussianNeedlets(inp, taper_width=0):
    '''
    Function from pyilc (https://github.com/jcolinhill/pyilc, https://arxiv.org/pdf/2307.01043.pdf)

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    taper_width: int, ell-space width of high ell taper. If 0, no taper is applied

    RETURNS
    --------
    ell: array of ell values
    filters: (N_scales, ellmax+1) numpy array containing filters at each scale

    '''
    FWHM_arcmin = np.array(inp.GN_FWHM_arcmin)
    if ( any( i <= j for i, j in zip(FWHM_arcmin, FWHM_arcmin[1:]))):
        raise AssertionError
    ell = np.arange(inp.ell_sum_max+1)
    N_scales = len(FWHM_arcmin) + 1
    filters = np.zeros((N_scales, inp.ell_sum_max+1))
    FWHM = FWHM_arcmin * np.pi/(180.*60.)
    # define gaussians
    Gaussians = np.zeros((N_scales-1, inp.ell_sum_max+1))
    for i in range(N_scales-1):
        Gaussians[i] = hp.gauss_beam(FWHM[i], lmax=inp.ell_sum_max)
    # define needlet filters in harmonic space
    filters[0] = Gaussians[0]
    for i in range(1,N_scales-1):
        filters[i] = np.sqrt(Gaussians[i]**2. - Gaussians[i-1]**2.)
    filters[N_scales-1] = np.sqrt(1. - Gaussians[N_scales-2]**2.)
    # simple check to ensure that sum of squared transmission is unity as needed for NILC algorithm
    assert (np.absolute( np.sum( filters**2., axis=0 ) - np.ones(inp.ell_sum_max+1,dtype=float)) < 1.e-3).all(), "wavelet filter transmission check failed"
    if taper_width:
        taper_func = (1.0 - 0.5*(np.tanh(0.025*(ell - (inp.ell_sum_max - taper_width))) + 1.0)) #smooth taper to zero from ELLMAX-taper_width to ELLMAX
        # taper_func *= 0.5*(np.tanh(0.5*(ell-10)))+0.5 #smooth taper to zero for low ell
    else:
        taper_func = np.ones_like(ell)
    for i, filt in enumerate(filters):
        filters[i] = filters[i]*taper_func
    return ell, filters

def build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=None):
    '''
    Note that pyilc checks which frequencies to use for every filter scale
    We include all freqs for all filter scales here

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    h: (N_scales, ellmax+1) ndarray containing needlet filters at each scale
    CMB_wt_maps: (Nscales, Nfreqs=2, npix (variable for each scale and freq)) nested list,
                contains NILC weight maps for preserved CMB
    tSZ_wt_maps: (Nscales, Nfreqs=2, npix (variable for each scale and freq)) nested list,
                contains NILC weight maps for preserved tSZ
    freq_maps: (Nfreqs=2, 12*nside**2) ndarray containing simulated map at 
                each frequency to use in NILC map construction

    RETURNS
    -------
    NILC_maps: (2 for CMB or tSZ preserved NILC map, 12*nside**2) ndarray
    '''

    if freq_maps is None:
        freq_map1 = hp.read_map(f'{inp.output_dir}/maps/sim{sim}_freq1.fits')
        freq_map2 = hp.read_map(f'{inp.output_dir}/maps/sim{sim}_freq2.fits')
        freq_maps = [freq_map1, freq_map2]
    
    NILC_maps = []
    for p in range(2):
        if p==0:
            wt_maps = CMB_wt_maps
        else:
            wt_maps = tSZ_wt_maps
        all_maps = np.zeros((inp.Nscales, 12*inp.nside**2)) #index as all_maps[scale][pixel]
        for i in range(len(inp.freqs)):
            map_ = freq_maps[i]
            map_ = hp.ud_grade(map_, inp.nside) #changed
            alm_orig = hp.map2alm(map_) #changed
            for n in range(inp.Nscales):
                alm = hp.almxfl(alm_orig, h[n]) #initial needlet filtering
                map_ = hp.alm2map(alm, inp.nside)
                NILC_weights = hp.ud_grade(wt_maps[n][i],inp.nside)
                map_ = map_*NILC_weights #application of weight map
                all_maps[n] = np.add(all_maps[n], map_) #add maps at all frequencies for each scale
        T_ILC_n = None
        for n in range(inp.Nscales):
            T_ILC_alm = hp.map2alm(all_maps[n]) #changed
            tmp = hp.almxfl(T_ILC_alm, h[n]) #final needlet filtering
            if T_ILC_n is None:
                T_ILC_n = np.zeros((inp.Nscales,len(tmp)),dtype=np.complex128)
            T_ILC_n[n]=tmp
        T_ILC = np.sum(np.array([hp.alm2map(T_ILC_n[n], inp.nside) for n in range(len(T_ILC_n))]), axis=0) #adding maps from all scales
        NILC_maps.append(T_ILC)
    return NILC_maps


