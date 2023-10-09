import healpy as hp
import numpy as np
from utils import tsz_spectral_response


def generate_freq_maps(sim, inp, save=True, band_limit=False, scaling=None, same_noise=True, pars=None):

    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    save: Bool, whether to save frequency map files
    band_limit: Bool, whether or not to remove all power in weight maps above ellmax
    scaling: None or list of length 5
            idx0: takes on values from 0 to len(inp.scaling_factors)-1,
                indicating by which scaling factor the input maps are scaled
            idx1: 0 for unscaled CMB, 1 for scaled CMB
            idx2: 0 for unscaled ftSZ, 1 for scaled ftSZ
            idx3: 0 for unscaled noise90, 1 for scaled noise90
            idx4: 0 for unscaled noise150, 1 for scaled noise150
    same_noise: Bool, whether to use the same or different noise in the two frequently channels
            (currently, if False, sets the noise in the higher frequency channel to be slightly higher)
    pars: array of floats [Acmb, Atsz, Anoise1, Anoise2] (if not provided, all assumed to be 1)

    RETURNS
    -------
    power spectra of CMB, tSZ, and noise (CC, T, N)
    cmb_map, tsz_map, noise1_map, noise2_map (amplified depending on scaling)

    '''

    np.random.seed(sim)
    l_arr, m_arr = hp.Alm.getlm(3*inp.nside-1)

    #Determine which components to scale
    CMB_amp, tSZ_amp_extra, noise1_amp, noise2_amp = 1, 1, 1, 1
    if scaling is not None:
        scale_factor = inp.scaling_factors[scaling[0]]
        if scaling[1]: CMB_amp = scale_factor
        if scaling[2]: tSZ_amp_extra = scale_factor
        if scaling[3]: noise1_amp = scale_factor
        if scaling[4]: noise2_amp = scale_factor
    if pars is not None:
        CMB_amp *= pars[0]
        tSZ_amp_extra *= pars[1]
        noise1_amp *= pars[2]
        noise2_amp *= pars[3]

    #Read tSZ halosky map
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
    theta_fwhm = (1.4/60.)*(np.pi/180.)
    sigma = theta_fwhm/np.sqrt(8.*np.log(2.))
    W1 = (inp.noise/60.)*(np.pi/180.)
    if same_noise:
        W2 = W1
    else:
        W2 = (inp.noise*np.sqrt(1.5)/60.)*(np.pi/180.)
    ells = np.arange(3*inp.nside)
    noise1_cl = W1**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
    noise2_cl = W2**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
    noise1_map = noise1_amp*hp.synfast(noise1_cl, inp.nside)
    if sim==0:
        np.random.seed(1)
    else:
        np.random.seed(inp.Nsims+sim)
    noise2_map = noise2_amp*hp.synfast(noise2_cl, inp.nside)
    np.random.seed(sim)
    if band_limit:
        noise1_alm = hp.map2alm(noise1_map)
        noise1_alm = noise1_alm*(l_arr<=inp.ellmax)
        noise1_map = hp.alm2map(noise1_alm, nside=inp.nside)
        noise2_alm = hp.map2alm(noise2_map)
        noise2_alm = noise2_alm*(l_arr<=inp.ellmax)
        noise2_map = hp.alm2map(noise2_alm, nside=inp.nside)
    noise1_cl = hp.anafast(noise1_map, lmax=inp.ell_sum_max)
    noise2_cl = hp.anafast(noise2_map, lmax=inp.ell_sum_max)

    #tSZ spectral response
    g1, g2 = tsz_spectral_response(inp.freqs)

    #create maps at freq1 and freq2 (in GHz)
    sim_map_1 = cmb_map + g1*tsz_map + noise1_map
    sim_map_2 = cmb_map + g2*tsz_map + noise2_map #make noise different in both maps
    if save:
        if not scaling:
            map1_fname = f'{inp.output_dir}/maps/sim{sim}_freq1.fits'
            map2_fname = f'{inp.output_dir}/maps/sim{sim}_freq2.fits'
        else:
            scaling_str = ''.join(str(e) for e in scaling) 
            map1_fname = f'{inp.output_dir}/maps/{scaling_str}/sim{sim}_freq1.fits'
            map2_fname = f'{inp.output_dir}/maps/{scaling_str}/sim{sim}_freq2.fits'
        hp.write_map(map1_fname, sim_map_1, overwrite=True, dtype=np.float32)
        hp.write_map(map2_fname, sim_map_2, overwrite=True, dtype=np.float32)
        if inp.verbose:
            print(f'created {map1_fname} and {map2_fname}', flush=True)

    return cmb_cl, tsz_cl, noise1_cl, noise2_cl, cmb_map, tsz_map, noise1_map, noise2_map
