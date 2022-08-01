import os
import healpy as hp
import numpy as np
import subprocess
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)


def generate_freq_maps(sim, freqs, tsz_amp, nside, ellmax, cmb_alm_file, halosky_scripts_path, verbose, include_noise=True):

    '''
    saves freq map files 
    returns power spectra of CMB, tSZ, and noise (CC, T, N)
    '''

    my_env = os.environ.copy()

    #create tSZ map from halosky
    subprocess.run([f"python {halosky_scripts_path}/example.py {sim}"], shell=True, env=my_env)
    if verbose:
        print(f'finished creating tSZ map sim {sim} from halosky')
    tsz_map = hp.read_map(f'maps/{sim}_tsz_00000.fits')
    tsz_map = tsz_amp*hp.ud_grade(tsz_map, nside) 
    tsz_cl = hp.anafast(tsz_map, lmax=ellmax)

    #realization of CMB from lensed alm
    cmb_alm = hp.read_alm(cmb_alm_file)
    cmb_cl = hp.alm2cl(cmb_alm)*10**(-12)
    cmb_map = hp.synfast(cmb_cl, nside)
    cmb_cl = hp.anafast(cmb_map, lmax=ellmax)
    hp.write_map(f'maps/{sim}_cmb_map.fits', cmb_map, overwrite=True)

    #noise map realization
    if include_noise:
        theta_fwhm = (1.4/60.)*(np.pi/180.)
        sigma = theta_fwhm/np.sqrt(8.*np.log(2.))
        W = (1/60.)*(np.pi/180.)
        ells = np.arange(3*nside)
        noise_cl = W**2*np.exp(ells*(ells+1)*sigma**2)*10**(-12)
        noise_map = hp.synfast(noise_cl, nside)
        noise_cl = hp.anafast(noise_map, lmax=ellmax)

    #tSZ spectral response
    T_cmb = 2.726
    h = 6.62607004*10**(-34)
    kb = 1.38064852*10**(-23)
    f = 1. #fsky
    def tsz_spectral_response(freq): #input frequency in GHz
        x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
        return T_cmb*(x*1/np.tanh(x/2)-4) #was factor of tcmb microkelvin before
    g1, g2 = tsz_spectral_response(freqs[0]), tsz_spectral_response(freqs[1])

    #create maps at freq1 and freq2 (in GHz)
    if include_noise:
        sim_map_1 = cmb_map + g1*tsz_map + noise_map
        sim_map_2 = cmb_map + g2*tsz_map + noise_map
    else:
        sim_map_1 = cmb_map + g1*tsz_map
        sim_map_2 = cmb_map + g2*tsz_map
    hp.write_map(f'maps/sim{sim}_freq1.fits', sim_map_1, overwrite=True)
    hp.write_map(f'maps/sim{sim}_freq2.fits', sim_map_2, overwrite=True)
    if verbose:
        print(f'created maps/sim{sim}_freq1.fits and maps/sim{sim}_freq2.fits', flush=True)

    if include_noise:
        return cmb_cl[:ellmax+1], tsz_cl[:ellmax+1], noise_cl[:ellmax+1]
    return cmb_cl[:ellmax+1], tsz_cl[:ellmax+1]