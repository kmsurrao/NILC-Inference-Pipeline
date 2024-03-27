import healpy as hp
import numpy as np
from utils import tsz_spectral_response, cib_spectral_response


def generate_freq_maps(inp, sim=None, save=True, scaling=None, pars=None, include_noise=True, map_tmpdir=None):

    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number (if None, will generate a random simulation number)
    save: Bool, whether to save frequency map files
    scaling: None or list of length Ncomps+1
            idx0: takes on values from 0 to len(inp.scaling_factors)-1,
                indicating by which scaling factor the input maps are scaled
            idx i+1: 0 for unscaled ith component, 1 for scaled ith component
    pars: array of floats [Acomp1, Acomp2, etc.] (if not provided, all assumed to be 1)
    include_noise: Bool, whether to include noise in the simulations (if False, same_noise is ignored)
    map_tmpdir: str, temporary directory in which to save maps. Must be defined if save is True.

    RETURNS
    -------
    comp_spectra: ndarray of shape (Ncomps, ellmax+1) containing power spectrum of each component
    comp_maps: ndarray of shape (Ncomps, Npix) healpix galactic coordinate maps of each component
        (maps are amplified depending on scaling and pars)
    noise_maps: (Nfreqs, Nsplits, Npix) ndarray of noise maps

    '''
    if save:
        assert map_tmpdir is not None, "map_tmpdir must be defined if save=True in generate_freq_maps"

    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)
    np.random.seed(sim)
    Nfreqs = len(inp.freqs)
    Ncomps = len(inp.comps)
    Npix = 12*inp.nside**2
    Nsplits = 2

    #Determine which components to scale
    extra_amps = np.ones(Ncomps)
    if scaling is not None:
        scale_factor = inp.scaling_factors[scaling[0]]
        multiplier = scale_factor*scaling[1:]
        multiplier[multiplier==0] = 1.
        extra_amps *= multiplier 
    if pars is not None:
        old_pars = pars
        pars = np.sqrt(np.array(pars))
        extra_amps *= pars

    # Get maps and power spectra of all components 
    comp_maps = np.zeros((Ncomps, Npix), dtype=np.float32)
    comp_spectra = np.zeros((Ncomps, inp.ellmax+1), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        comp_path = inp.paths_to_comps[c]
        isGaussian = inp.use_Gaussian[c]
        if isGaussian:
            map_ = hp.ud_grade(hp.read_map(comp_path), inp.nside)
            cl = hp.anafast(map_, lmax=3*inp.nside-1)
            map_ = hp.synfast(cl, nside=inp.nside)
        else:
            map_ = hp.ud_grade(hp.read_map(f'{comp_path}/{comp}_{sim:05d}.fits'), inp.nside)
        comp_maps[c] = inp.amp_factors[c]*extra_amps[c]*map_
        comp_spectra[c] = hp.anafast(comp_maps[c], lmax=inp.ellmax)
    
    # noise map realizations
    if not include_noise:
        noise_maps = np.zeros((Nfreqs, Nsplits, Npix), dtype=np.float32)
    else:
        theta_fwhm = (1.4/60.)*(np.pi/180.)
        sigma = theta_fwhm/np.sqrt(8.*np.log(2.))
        W_arr = (np.array(inp.noise)/60.)*(np.pi/180.)
        ells = np.arange(3*inp.nside)
        noise_cl = np.zeros((Nfreqs, Nsplits, len(ells)), dtype=np.float32)
        noise_maps = np.zeros((Nfreqs, Nsplits, Npix), dtype=np.float32)
        for i in range(Nfreqs):
            for s in range(Nsplits):
                noise_cl[i,s] = W_arr[i]**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
                noise_maps[i,s] = hp.synfast(noise_cl[i,s], inp.nside)

    # spectral response vectors
    sed_arr = np.ones((Ncomps, Nfreqs), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        if comp == 'tsz':
            sed_arr[c] = tsz_spectral_response(inp.freqs)
        elif comp == 'cib':
            sed_arr[c] = cib_spectral_response(inp.freqs)

    #create maps at different freqs (in GHz) and splits 
    sim_maps = np.zeros((Nfreqs, Nsplits, Npix), dtype=np.float32)
    for i in range(Nfreqs):
        for s in range(Nsplits):
            sim_maps[i,s] = np.sum(np.array([sed_arr[c,i]*comp_maps[c] for c in range(Ncomps)]), axis=0) + noise_maps[i,s]
            if save:
                if pars is not None:
                    pars_str = f'_pars{old_pars[0]:.3f}_{old_pars[1]:.3f}_'
                else:
                    pars_str = ''
                map_fname = f'{map_tmpdir}/sim{sim}_freq{i+1}_split{s+1}{pars_str}.fits'
                hp.write_map(map_fname, sim_maps[i,s], overwrite=True, dtype=np.float32)
    if inp.verbose and save and pars is None:
        print(f'created {map_fname} and similarly for other freqs and splits', flush=True)

    return comp_spectra, comp_maps, noise_maps



def save_scaled_freq_maps(inp, sim, scaling, map_tmpdir, comp_maps_unscaled, noise_maps_unscaled):

    '''
    Given unscaled maps, save frequency maps containing scaled versions of those same realizations
   
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number 
    scaling: None or list of length 3
            idx0: takes on values from 0 to len(inp.scaling_factors)-1,
                indicating by which scaling factor the input maps are scaled
            idx i+1: 0 for unscaled component i, 1 for scaled component i
    map_tmpdir: str, temporary directory in which to save maps. Must be defined if save is True.
    comp_maps_unscaled: ndarray of shape (Ncomps, Npix) containing unscaled component maps
            (but amplified according to input yaml)
    noise_maps_unscaled: (Nfreqs, Nsplits, Npix) ndarray of unscaled maps of noise

    RETURNS
    -------
    comp_maps: (Ncomps, Npix) healpix galactic coordinate maps of component maps
        (maps are amplified depending on scaling)
    noise_maps: (Nfreqs, Nsplits, Npix) ndarray of noise maps

    '''
    Nfreqs = len(inp.freqs)
    Ncomps = len(inp.comps)
    Nsplits = 2
    Npix = 12*inp.nside**2
    
    #Determine which components to scale
    extra_amps = np.ones(Ncomps)
    if scaling is not None:
        scale_factor = inp.scaling_factors[scaling[0]]
        multiplier = scale_factor*scaling[1:]
        multiplier[multiplier==0] = 1.
        extra_amps *= multiplier 

    # Scale the components
    comp_maps = np.zeros_like(comp_maps_unscaled)
    for c in range(Ncomps):
        comp_maps[c] = extra_amps[c]*comp_maps_unscaled[c]
    noise_maps = noise_maps_unscaled

    # spectral response vectors
    sed_arr = np.ones((Ncomps, Nfreqs), dtype=np.float32)
    for c, comp in enumerate(inp.comps):
        if comp == 'tsz':
            sed_arr[c] = tsz_spectral_response(inp.freqs)
        elif comp == 'cib':
            sed_arr[c] = cib_spectral_response(inp.freqs)
    
    #create maps at different freqs (in GHz) and splits 
    sim_maps = np.zeros((Nfreqs, Nsplits, Npix), dtype=np.float32)
    for i in range(Nfreqs):
        for s in range(Nsplits):
            sim_maps[i,s] = np.sum(np.array([sed_arr[c,i]*comp_maps[c] for c in range(Ncomps)]), axis=0) + noise_maps[i,s]
            map_fname = f'{map_tmpdir}/sim{sim}_freq{i+1}_split{s+1}.fits'
            hp.write_map(map_fname, sim_maps[i,s], overwrite=True, dtype=np.float32)
    if inp.verbose:
        print(f'created {map_fname} and similarly for other freqs and splits', flush=True)
    
    return comp_maps, noise_maps