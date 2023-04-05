import healpy as hp
import numpy as np
from utils import tsz_spectral_response


def generate_freq_maps(sim, inp):

    '''
    saves freq map files 
    returns power spectra of CMB, tSZ, and noise (CC, T, N)
    '''

    np.random.seed(sim)

    #Read tSZ halosky map
    tsz_map = hp.read_map(f'{inp.halosky_maps_path}/tsz_{sim:05d}.fits')
    tsz_map = inp.tsz_amp*hp.ud_grade(tsz_map, inp.nside)
    tsz_cl = hp.anafast(tsz_map, lmax=inp.ell_sum_max)

    #realization of CMB from lensed alm
    cmb_map = hp.read_map(inp.cmb_map_file)
    cmb_map = hp.ud_grade(cmb_map, inp.nside)
    cmb_cl = hp.anafast(cmb_map, lmax=inp.ell_sum_max)
    cmb_map = hp.synfast(cmb_cl, inp.nside)

    #noise map realization
    theta_fwhm = (1.4/60.)*(np.pi/180.)
    sigma = theta_fwhm/np.sqrt(8.*np.log(2.))
    W = (inp.noise/60.)*(np.pi/180.)
    ells = np.arange(3*inp.nside)
    noise_cl = W**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
    noise_map = hp.synfast(noise_cl, inp.nside)
    noise_cl = hp.anafast(noise_map, lmax=inp.ell_sum_max)

    #tSZ spectral response
    g1, g2 = tsz_spectral_response(inp.freqs)

    #create maps at freq1 and freq2 (in GHz)
    sim_map_1 = cmb_map + g1*tsz_map + noise_map
    sim_map_2 = cmb_map + g2*tsz_map + 1.5*noise_map #make noise different in both maps
    hp.write_map(f'{inp.output_dir}/maps/sim{sim}_freq1.fits', sim_map_1, overwrite=True)
    hp.write_map(f'{inp.output_dir}/maps/sim{sim}_freq2.fits', sim_map_2, overwrite=True)
    if inp.verbose:
        print(f'created {inp.output_dir}/maps/sim{sim}_freq1.fits and {inp.output_dir}/maps/sim{sim}_freq2.fits', flush=True)

    return cmb_cl, tsz_cl, noise_cl, cmb_map, tsz_map, noise_map
