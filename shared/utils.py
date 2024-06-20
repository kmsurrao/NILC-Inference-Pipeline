import numpy as np
import os
import subprocess
import healpy as hp
import itertools


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
    if not os.path.isdir(f'{inp.output_dir}/data_vecs'):
        subprocess.call(f'mkdir {inp.output_dir}/data_vecs', shell=True, env=env)
    if not os.path.isdir(f'{inp.output_dir}/posteriors'):
        subprocess.call(f'mkdir {inp.output_dir}/posteriors', shell=True, env=env)
    return 



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


def build_NILC_maps(inp, sim, h, all_wt_maps, freq_maps=None, split=None):
    '''
    Note that pyilc checks which frequencies to use for every filter scale
    We include all freqs for all filter scales here

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    h: (N_scales, ellmax+1) ndarray containing needlet filters at each scale
    all_wt_maps: (Ncomps, Nscales, Nfreqs, Npix) ndarray containing NILC weight maps for each component
    freq_maps: (Nfreqs=2, 12*nside**2) ndarray containing simulated map at 
                each frequency to use in NILC map construction
    split: None or int, either 1 or 2 representing which split of data is used

    RETURNS
    -------
    NILC_maps: (Ncomps, 12*nside**2) ndarray
    '''
    Ncomps = len(inp.comps)
    if freq_maps is None:
        split_str = f'_split{split}' if split is not None else ''
        freq_maps = [hp.read_map(f'{inp.output_dir}/maps/sim{sim}_freq{i+1}{split_str}.fits') for i in range(len(inp.freqs))]
    
    NILC_maps = []
    for p in range(Ncomps):
        wt_maps = all_wt_maps[p]
        all_maps = np.zeros((inp.Nscales, 12*inp.nside**2)) #index as all_maps[scale][pixel]
        for i in range(len(inp.freqs)):
            map_ = freq_maps[i]
            map_ = hp.ud_grade(map_, inp.nside)
            alm_orig = hp.map2alm(map_)
            for n in range(inp.Nscales):
                alm = hp.almxfl(alm_orig, h[n]) #initial needlet filtering
                map_ = hp.alm2map(alm, inp.nside)
                NILC_weights = hp.ud_grade(wt_maps[n][i],inp.nside)
                map_ = map_*NILC_weights #application of weight map
                all_maps[n] = np.add(all_maps[n], map_) #add maps at all frequencies for each scale
        T_ILC_n = None
        for n in range(inp.Nscales):
            T_ILC_alm = hp.map2alm(all_maps[n])
            tmp = hp.almxfl(T_ILC_alm, h[n]) #final needlet filtering
            if T_ILC_n is None:
                T_ILC_n = np.zeros((inp.Nscales,len(tmp)),dtype=np.complex128)
            T_ILC_n[n]=tmp
        T_ILC = np.sum(np.array([hp.alm2map(T_ILC_n[n], inp.nside) for n in range(len(T_ILC_n))]), axis=0) #adding maps from all scales
        NILC_maps.append(T_ILC)
    return NILC_maps


def get_scalings(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    scalings: list of lists, each of length 5
            idx0: takes on values from 0 to len(inp.scaling_factors)-1,
                  indicating by which scaling factor the input maps are scaled
            idx i: 0 for unscaled component i-1, 1 for scaled component i-1
    '''
    scalings_init = [list(i) for i in itertools.product([0, 1], repeat=len(inp.comps))]
    scalings = []
    for i in range(len(inp.scaling_factors)):
        for s in scalings_init:
            scalings.append(np.array([i]+s))
    return scalings


def get_naming_str(inp, pipeline):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    pipeline: str, pipeline being run ('multifrequency', 'HILC', or 'NILC')

    RETURNS
    -------
    name: str, string to add on at the end of file names, providing information about the run
    '''
    assert pipeline in {'multifrequency', 'HILC', 'NILC'}, "pipeline must be 'multifrequency', 'HILC', or 'NILC'"
    name = f'{pipeline}_'
    if 'tsz' in inp.comps:
        tsz_idx = inp.comps.index('tsz')
        gaussian_str = 'gaussiantsz_' if inp.use_Gaussian[tsz_idx] else 'nongaussiantsz_'
        name += gaussian_str
    if pipeline == 'HILC':
        wts_str = 'weightsonce_' if inp.compute_weights_once else 'weightsvary_'
        name += wts_str
    if inp.Nsims % 1000 == 0:
        sims_str = f'{inp.Nsims//1000}ksims'
    else:
        sims_str = f'{int(inp.Nsims)}sims'
    name += sims_str
    if 'tsz' in inp.comps:
        name += f'_tszamp{int(inp.amp_factors[tsz_idx])}'
    if inp.use_lfi:
        name += '_lfi'
    else:
        name += '_gaussianlkl'
        if pipeline == 'HILC':
            if inp.compute_weights_once and not inp.use_symbolic_regression:
                name += '_analytic'
            else:
                name += '_sr'
    if pipeline == 'NILC':
        name += f'_{inp.Nscales}scales'
    return name



def spectral_response(freqs, comp):
    '''
    ARGUMENTS
    ---------
    freqs: 1D numpy array, contains frequencies (GHz) for which to calculate the spectral response
    comp: str, name of component. Currently only has 'cmb', 'tsz', and 'cib' implemented, but
            this function can be modified to add others.

    RETURNS
    ---------
    1D array containing spectral response of comp to each frequency
    '''
    if comp == 'cmb':
        return np.ones(len(freqs), dtype=np.float32)


    if comp == 'tsz':
        T_cmb = 2.726
        h = 6.62607004*10**(-34)
        kb = 1.38064852*10**(-23)
        response = []
        for freq in freqs:
            x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
            response.append(T_cmb*(x*1/np.tanh(x/2)-4))
        return np.array(response)
    

    elif comp == 'cib':
        # CIB = modified blackbody here
        # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units

        TCMB = 2.726 #Kelvin
        TCMB_uK = 2.726e6 #micro-Kelvin
        hplanck=6.626068e-34 #MKS
        kboltz=1.3806503e-23 #MKS
        clight=299792458.0 #MKS

        # function needed for Planck bandpass integration/conversion following approach in Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf
        # blackbody derivative
        # units are 1e-26 Jy/sr/uK_CMB
        def dBnudT(nu_ghz):
            nu = 1.e9*np.asarray(nu_ghz)
            X = hplanck*nu/(kboltz*TCMB)
            return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK

        # conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
        #   i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
        #   i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
        def ItoDeltaT(nu_ghz):
            return 1./dBnudT(nu_ghz)

        Tdust_CIB = 20.0       #CIB effective dust temperature [K] (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf)
        beta_CIB = 1.45         #CIB modified blackbody spectral index (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf ; Table 10 of that paper contains CIB monopoles)
        nu0_CIB_ghz = 353.0    #CIB pivot frequency [GHz]

        nu_ghz = freqs
        nu = 1.e9*np.asarray(nu_ghz).astype(float)
        X_CIB = hplanck*nu/(kboltz*Tdust_CIB)
        nu0_CIB = nu0_CIB_ghz*1.e9
        X0_CIB = hplanck*nu0_CIB/(kboltz*Tdust_CIB)
        resp = (nu/nu0_CIB)**(3.0+(beta_CIB)) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0)) * (ItoDeltaT(np.asarray(nu_ghz).astype(float))/ItoDeltaT(nu0_CIB_ghz))
        resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
        return resp



def sublist_idx(outer_list, sublist):
    '''
    ARGUMENTS
    ---------
    outer_list: 

    RETURNS
    -------
    '''
    for i, elt in enumerate(outer_list):
        if np.array_equal(elt, sublist):
            return i
