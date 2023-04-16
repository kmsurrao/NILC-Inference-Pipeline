import healpy as hp
import numpy as np
from utils import tsz_spectral_response


def generate_freq_maps(sim, inp, save=True, band_limit=False):

    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    save: Bool, whether to save frequency map files
    band_limit: Bool, whether or not to remove all power in weight maps above ellmax

    RETURNS
    -------
    power spectra of CMB, tSZ, and noise (CC, T, N)
    '''

    np.random.seed(sim)
    l_arr, m_arr = hp.Alm.getlm(3*inp.nside-1)

    #Read tSZ halosky map
    tsz_map = hp.read_map(f'{inp.halosky_maps_path}/tsz_{sim:05d}.fits')
    tsz_map = inp.tsz_amp*hp.ud_grade(tsz_map, inp.nside)
    if band_limit:
        tsz_alm = hp.map2alm(tsz_map)
        tsz_alm = tsz_alm*(l_arr<=inp.ellmax)
        tsz_map = hp.alm2map(tsz_alm, nside=inp.nside)
    tsz_cl = hp.anafast(tsz_map, lmax=inp.ell_sum_max)

    #realization of CMB from lensed alm
    cmb_map = hp.read_map(inp.cmb_map_file)
    cmb_map = hp.ud_grade(cmb_map, inp.nside)
    if band_limit:
        cmb_alm = hp.map2alm(cmb_map)
        cmb_alm = cmb_alm*(l_arr<=inp.ellmax)
        cmb_map = hp.alm2map(cmb_alm, nside=inp.nside)
    cmb_cl = hp.anafast(cmb_map, lmax=inp.ell_sum_max)
    cmb_map = hp.synfast(cmb_cl, inp.nside)

    #noise map realization
    theta_fwhm = (1.4/60.)*(np.pi/180.)
    sigma = theta_fwhm/np.sqrt(8.*np.log(2.))
    W = (inp.noise/60.)*(np.pi/180.)
    ells = np.arange(3*inp.nside)
    noise_cl = W**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
    noise_map = hp.synfast(noise_cl, inp.nside)
    if band_limit:
        noise_alm = hp.map2alm(noise_map)
        noise_alm = noise_alm*(l_arr<=inp.ellmax)
        noise_map = hp.alm2map(noise_alm, nside=inp.nside)
    noise_cl = hp.anafast(noise_map, lmax=inp.ell_sum_max)

    #tSZ spectral response
    g1, g2 = tsz_spectral_response(inp.freqs)

    #create maps at freq1 and freq2 (in GHz)
    sim_map_1 = cmb_map + g1*tsz_map + noise_map
    sim_map_2 = cmb_map + g2*tsz_map + 1.5*noise_map #make noise different in both maps
    if save:
        hp.write_map(f'{inp.output_dir}/maps/sim{sim}_freq1.fits', sim_map_1, overwrite=True)
        hp.write_map(f'{inp.output_dir}/maps/sim{sim}_freq2.fits', sim_map_2, overwrite=True)
        if inp.verbose:
            print(f'created {inp.output_dir}/maps/sim{sim}_freq1.fits and {inp.output_dir}/maps/sim{sim}_freq2.fits', flush=True)

    return cmb_cl, tsz_cl, noise_cl, cmb_map, tsz_map, noise_map
