import healpy as hp
import numpy as np
from utils import tsz_spectral_response, cib_spectral_response


def generate_freq_maps(inp, sim=None, save=True, scaling=None, same_noise=True, pars=None, include_noise=True, map_tmpdir=None, cib_path=None):

    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number (if None, will generate a random simulation number)
    save: Bool, whether to save frequency map files
    scaling: None or list of length 3
            idx0: takes on values from 0 to len(inp.scaling_factors)-1,
                indicating by which scaling factor the input maps are scaled
            idx1: 0 for unscaled CMB, 1 for scaled CMB
            idx2: 0 for unscaled ftSZ, 1 for scaled ftSZ
    same_noise: Bool, whether to use the same or different noise in the two frequently channels
            (currently, if False, sets the noise in the higher frequency channel to be slightly higher)
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)
    include_noise: Bool, whether to include noise in the simulations (if False, same_noise is ignored)
    map_tmpdir: str, temporary directory in which to save maps. Must be defined if save is True.
    cib_path: str, path to CIB map file (assumed in uK) at 150 GHz. If this is provided, it assumed that 
            a Gaussian CIB realization will be added to the maps. If not provided, the map will consist 
            only of CMB and tSZ.

    RETURNS
    -------
    cmb_cl, tsz_cl: power spectra of CMB and tSZ (CC, T)
    cmb_map, tsz_map: healpix galactic coordinate maps of CMB and tSZ
        (maps are amplified depending on scaling and pars)
    noise_maps: (Nfreqs, Nsplits, Npix) ndarray of noise maps

    '''
    if save:
        assert map_tmpdir is not None, "map_tmpdir must be defined if save=True in generate_freq_maps"

    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)
    np.random.seed(sim)
    Nfreqs = len(inp.freqs)

    #Determine which components to scale
    CMB_amp, tSZ_amp_extra= 1, 1
    if scaling is not None:
        scale_factor = inp.scaling_factors[scaling[0]]
        if scaling[1]: CMB_amp = scale_factor
        if scaling[2]: tSZ_amp_extra = scale_factor
    if pars is not None:
        old_pars = pars
        pars = np.sqrt(np.array(pars))
        CMB_amp *= pars[0]
        tSZ_amp_extra *= pars[1]

    #Read tSZ halosky map
    if tSZ_amp_extra == 0:
        tsz_map = np.zeros(12*inp.nside**2)
        tsz_cl = np.zeros(inp.ell_sum_max+1)
    else:
        if not inp.use_Gaussian_tSZ:
            tsz_map = hp.read_map(f'{inp.halosky_maps_path}/tsz_{sim:05d}.fits')
        else:
            tsz_map = hp.read_map(f'{inp.halosky_maps_path}/tsz_00000.fits')
            tsz_cl = hp.anafast(tsz_map, lmax=3*inp.nside-1)
            tsz_map = hp.synfast(tsz_cl, nside=inp.nside)
        tsz_map = inp.tsz_amp*tSZ_amp_extra*hp.ud_grade(tsz_map, inp.nside)
        tsz_cl = hp.anafast(tsz_map, lmax=inp.ell_sum_max)

    #realization of CMB from lensed alm
    if CMB_amp == 0:
        cmb_map = np.zeros(12*inp.nside**2)
        cmb_cl = np.zeros(inp.ell_sum_max+1)
    else:
        cmb_map = hp.read_map(inp.cmb_map_file)
        cmb_map = CMB_amp*hp.ud_grade(cmb_map, inp.nside)
        cmb_cl = hp.anafast(cmb_map, lmax=inp.ell_sum_max)
        cmb_map = hp.synfast(cmb_cl, inp.nside)
        cmb_cl = hp.anafast(cmb_map, lmax=inp.ell_sum_max)

    #noise map realizations
    if not include_noise or inp.noise==0:
        noise_maps = np.zeros((Nfreqs, 2, 12*inp.nside**2), dtype=np.float32) #shape Nfreqs, Nsplits, Npix
    else:
        theta_fwhm = (1.4/60.)*(np.pi/180.)
        sigma = theta_fwhm/np.sqrt(8.*np.log(2.))
        W1 = (inp.noise/60.)*(np.pi/180.)
        if same_noise:
            W2 = W1
        else:
            W2 = (inp.noise*np.sqrt(1.5)/60.)*(np.pi/180.)
        ells = np.arange(3*inp.nside)
        noise_cl = np.zeros((Nfreqs, 2, len(ells)), dtype=np.float32) #shape Nfreqs, Nsplits, len(ells)
        noise_maps = np.zeros((Nfreqs, 2, 12*inp.nside**2), dtype=np.float32) #shape Nfreqs, Nsplits, Npix
        W_arr = [W1, W2]
        for i in range(2): #iterate over frequencies
            for s in range(2): #iterate over splits
                noise_cl[i,s] = W_arr[i]**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
                noise_maps[i,s] = hp.synfast(noise_cl[i,s], inp.nside)
    
    #Gaussian CIB map realization (if including CIB)
    if cib_path is not None:
        cib_map = 10**(-6)*hp.ud_grade(hp.read_map(cib_path), inp.nside) #convert uK to K
        cib_map /= cib_spectral_response([150])

    #tSZ spectral response (and CIB if included)
    g_vec = tsz_spectral_response(inp.freqs)
    h_vec = cib_spectral_response(inp.freqs)

    #create maps at freqs (in GHz) and 2 splits 
    sim_maps = np.zeros((Nfreqs, Nfreqs, 12*inp.nside**2), dtype=np.float32)
    for i in range(Nfreqs):
        for s in range(2):
            sim_maps[i,s] = cmb_map + g_vec[i]*tsz_map + noise_maps[i,s]
            if cib_path is not None:
                sim_maps[i,s] += h_vec[i]*cib_map
            if save:
                if pars is not None:
                    pars_str = f'_pars{old_pars[0]:.3f}_{old_pars[1]:.3f}_'
                else:
                    pars_str = ''
                map_fname = f'{map_tmpdir}/sim{sim}_freq{i+1}_split{s+1}{pars_str}.fits'
                hp.write_map(map_fname, sim_maps[i,s], overwrite=True, dtype=np.float32)
    if inp.verbose and save and pars is None:
        print(f'created {map_fname} and similarly for other freqs and splits', flush=True)

    if cib_path is not None:
        cib_cl = hp.anafast(cib_map, lmax=inp.ellmax)
        return cmb_cl, tsz_cl, cib_cl, cmb_map, tsz_map, cib_map, noise_maps
    
    return cmb_cl, tsz_cl, cmb_map, tsz_map, noise_maps



def save_scaled_freq_maps(inp, sim, scaling, map_tmpdir, CMB_map_unscaled, tSZ_map_unscaled, noise_maps_unscaled):

    '''
    Given unscaled maps, save frequency maps containing scaled versions of those same realizations
   
     ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number 
    scaling: None or list of length 3
            idx0: takes on values from 0 to len(inp.scaling_factors)-1,
                indicating by which scaling factor the input maps are scaled
            idx1: 0 for unscaled CMB, 1 for scaled CMB
            idx2: 0 for unscaled ftSZ, 1 for scaled ftSZ
    map_tmpdir: str, temporary directory in which to save maps. Must be defined if save is True.
    CMB_map_unscaled: unscaled map of CMB 
    tSZ_map_unscaled: unscaled map of tSZ (but amplified according to input yaml) 
    noise_maps_unscaled: unscaled maps of noise

    RETURNS
    -------
    cmb_map, tsz_map: healpix galactic coordinate maps of CMB and tSZ
        (maps are amplified depending on scaling)
    noise_maps: (Nfreqs, Nsplits, Npix) ndarray of noise maps

    '''
    #Determine which components to scale
    CMB_amp, tSZ_amp_extra= 1, 1
    if scaling is not None:
        scale_factor = inp.scaling_factors[scaling[0]]
        if scaling[1]: CMB_amp = scale_factor
        if scaling[2]: tSZ_amp_extra = scale_factor
    
    cmb_map = CMB_amp*CMB_map_unscaled
    tsz_map = tSZ_amp_extra*tSZ_map_unscaled
    noise_maps = noise_maps_unscaled

    #tSZ spectral response
    g_vec = tsz_spectral_response(inp.freqs)
    Nfreqs = len(inp.freqs)

    #create maps at each freq (in GHz) and 2 splits 
    sim_maps = np.zeros((Nfreqs, Nfreqs, 12*inp.nside**2), dtype=np.float32)
    for i in range(Nfreqs):
        for s in range(2):
            sim_maps[i,s] = cmb_map + g_vec[i]*tsz_map + noise_maps[i,s]
            map_fname = f'{map_tmpdir}/sim{sim}_freq{i+1}_split{s+1}.fits'
            hp.write_map(map_fname, sim_maps[i,s], overwrite=True, dtype=np.float32)
    if inp.verbose:
        print(f'created {map_fname} and similarly for other freqs and splits', flush=True)

    return cmb_map, tsz_map, noise_maps