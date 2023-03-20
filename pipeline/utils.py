import numpy as np
import os
import subprocess
import healpy as hp

def setup_output_dir(inp, env):
    '''
    Sets up directory for output files

    ARGUMENTS
    ---------
    inp: Info() object containing input specifications
    env: environment object

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

def GaussianNeedlets(ELLMAX, FWHM_arcmin=np.array([600., 60., 30., 15.])):
    '''
    Function from pyilc (https://github.com/jcolinhill/pyilc)

    ARGUMENTS
    ---------
    ELLMAX: int, maximum ell for needlet filters
    FWHM_arcmin: array of FWHM used for constrution of Gaussians (needlet filters
        are differences of two Gaussians). FWHM need to be in strictly decreasing 
        order, otherwise you'll get nonsense


    RETURNS
    --------
    ell: array of ell values
    filters: (N_scales, ellmax+1) numpy array containing filters at each scale

    '''
    if ( any( i <= j for i, j in zip(FWHM_arcmin, FWHM_arcmin[1:]))):
        raise AssertionError
    ell = np.arange(ELLMAX+1)
    N_scales = len(FWHM_arcmin) + 1
    filters = np.zeros((N_scales, ELLMAX+1))
    FWHM = FWHM_arcmin * np.pi/(180.*60.)
    # define gaussians
    Gaussians = np.zeros((N_scales-1, ELLMAX+1))
    for i in range(N_scales-1):
        Gaussians[i] = hp.gauss_beam(FWHM[i], lmax=ELLMAX)
    # define needlet filters in harmonic space
    filters[0] = Gaussians[0]
    for i in range(1,N_scales-1):
        filters[i] = np.sqrt(Gaussians[i]**2. - Gaussians[i-1]**2.)
    filters[N_scales-1] = np.sqrt(1. - Gaussians[N_scales-2]**2.)
    # simple check to ensure that sum of squared transmission is unity as needed for NILC algorithm
    assert (np.absolute( np.sum( filters**2., axis=0 ) - np.ones(ELLMAX+1,dtype=float)) < 1.e-3).all(), "wavelet filter transmission check failed"
    # taper_width = 20.
    # taper_func = (1.0 - 0.5*(np.tanh(0.025*(ell - (ELLMAX - taper_width))) + 1.0)) #smooth taper to zero from ELLMAX-taper_width to ELLMAX
    # taper_func *= 0.5*(np.tanh(0.5*(ell-10)))+0.5 #smooth taper to zero for low ell
    # for i, filt in enumerate(filters):
    #     filters[i] = filters[i]*taper_func
    return ell, filters