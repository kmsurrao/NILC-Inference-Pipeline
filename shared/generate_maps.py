import healpy as hp
import numpy as np
from utils import tsz_spectral_response


def generate_freq_maps(inp, sim=None, save=True, band_limit=False, scaling=None, same_noise=True, pars=None, include_noise=True):

    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number (if None, will generate a random simulation number)
    save: Bool, whether to save frequency map files
    band_limit: Bool, whether or not to remove all power in weight maps above ellmax
    scaling: None or list of length 5
            idx0: takes on values from 0 to len(inp.scaling_factors)-1,
                indicating by which scaling factor the input maps are scaled
            idx1: 0 for unscaled CMB, 1 for scaled CMB
            idx2: 0 for unscaled ftSZ, 1 for scaled ftSZ
    same_noise: Bool, whether to use the same or different noise in the two frequently channels
            (currently, if False, sets the noise in the higher frequency channel to be slightly higher)
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)
    include_noise: Bool, whether to include noise in the simulations (if False, same_noise is ignored)

    RETURNS
    -------
    cmb_cl, tsz_cl: power spectra of CMB and tSZ (CC, T)
    cmb_map, tsz_map: healpix galactic coordinate maps of CMB and tSZ
        (maps are amplified depending on scaling and pars)
    noise_maps: (Nfreqs, Nsplits, Npix) ndarray of noise maps

    '''
    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)

    np.random.seed(sim)
    l_arr, m_arr = hp.Alm.getlm(3*inp.nside-1)

    #Determine which components to scale
    CMB_amp, tSZ_amp_extra= 1, 1
    if scaling is not None:
        scale_factor = inp.scaling_factors[scaling[0]]
        if scaling[1]: CMB_amp = scale_factor
        if scaling[2]: tSZ_amp_extra = scale_factor
    if pars is not None:
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
        if band_limit:
            tsz_alm = hp.map2alm(tsz_map)
            tsz_alm = tsz_alm*(l_arr<=inp.ellmax)
            tsz_map = hp.alm2map(tsz_alm, nside=inp.nside)
        tsz_cl = hp.anafast(tsz_map, lmax=inp.ell_sum_max)

    #realization of CMB from lensed alm
    if CMB_amp == 0:
        cmb_map = np.zeros(12*inp.nside**2)
        cmb_cl = np.zeros(inp.ell_sum_max+1)
    else:
        cmb_map = hp.read_map(inp.cmb_map_file)
        cmb_map = CMB_amp*hp.ud_grade(cmb_map, inp.nside)
        if band_limit:
            cmb_alm = hp.map2alm(cmb_map)
            cmb_alm = cmb_alm*(l_arr<=inp.ellmax)
            cmb_map = hp.alm2map(cmb_alm, nside=inp.nside)
        cmb_cl = hp.anafast(cmb_map, lmax=inp.ell_sum_max)
        cmb_map = hp.synfast(cmb_cl, inp.nside)
        cmb_cl = hp.anafast(cmb_map, lmax=inp.ell_sum_max)

    #noise map realizations
    if not include_noise:
        noise_maps = np.zeros((2,2,12*inp.nside**2), dtype=np.float32) #shape Nfreqs, Nsplits, Npix
    else:
        theta_fwhm = (1.4/60.)*(np.pi/180.)
        sigma = theta_fwhm/np.sqrt(8.*np.log(2.))
        W1 = (inp.noise/60.)*(np.pi/180.)
        if same_noise:
            W2 = W1
        else:
            W2 = (inp.noise*np.sqrt(1.5)/60.)*(np.pi/180.)
        ells = np.arange(3*inp.nside)
        noise_cl = np.zeros((2,2,len(ells)), dtype=np.float32) #shape Nfreqs, Nsplits, len(ells)
        noise_maps = np.zeros((2,2,12*inp.nside**2), dtype=np.float32) #shape Nfreqs, Nsplits, Npix
        W_arr = [W1, W2]
        for i in range(2): #iterate over frequencies
            for s in range(2): #iterate over splits
                noise_cl[i,s] = W_arr[i]**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
                noise_maps[i,s] = hp.synfast(noise_cl[i,s], inp.nside)
                if band_limit:
                    alm = hp.map2alm(noise_maps[i,s])
                    alm = alm*(l_arr<=inp.ellmax)
                    noise_maps[i,s] = hp.map2alm(alm, nside=inp.nside)

    #tSZ spectral response
    g1, g2 = tsz_spectral_response(inp.freqs)
    g_vec = [g1, g2]

    #create maps at freq1 and freq2 (in GHz) and 2 splits 
    sim_maps = np.zeros((2,2,12*inp.nside**2), dtype=np.float32)
    for i in range(2):
        for s in range(2):
            sim_maps[i,s] = cmb_map + g_vec[i] + noise_maps[i,s]
            if save:
                if not scaling:
                    map_fname = f'{inp.output_dir}/maps/sim{sim}_freq{i+1}_split{s+1}.fits'
                else:
                    scaling_str = ''.join(str(e) for e in scaling) 
                    map_fname = f'{inp.output_dir}/maps/{scaling_str}/sim{sim}_freq{i+1}_split{s+1}.fits'
                hp.write_map(map_fname, sim_maps[i,s], overwrite=True, dtype=np.float32)
    if inp.verbose and save:
        print(f'created {map_fname} and similarly for other freqs and splits', flush=True)

    return cmb_cl, tsz_cl, cmb_map, tsz_map, noise_maps
