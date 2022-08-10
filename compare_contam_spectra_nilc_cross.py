import sys
import os
import subprocess
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
from input import Info
from nilc_power_spectrum_calc import calculate_all_cl
from generate_maps import *
from wt_map_spectra import *
from data_spectra import *
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
hp.disable_warnings()


def sim_propagation(wt_maps, sim_map, spectral_response, inp):
    ell, filters = GaussianNeedlets(inp.ellmax, FWHM_arcmin=inp.GN_FWHM_arcmin)
    npix = 12*inp.nside**2
    nfreqs = len(inp.freqs)
    all_maps = np.zeros((inp.Nscales,npix)) #index as all_maps[n][pixel]
    for i in range(nfreqs):
        map_ = spectral_response[i]*sim_map
        alm_orig = hp.map2alm(map_)
        for n in range(inp.Nscales):
            alm = hp.almxfl(alm_orig,filters[n]) #initial needlet filtering
            map_ = hp.alm2map(alm, inp.nside)
            NILC_weights = hp.ud_grade(wt_maps[n][i],inp.nside)
            map_ = map_*NILC_weights #application of weight map
            all_maps[n] = np.add(all_maps[n],map_) #add maps at all frequencies for each scale
    T_ILC_n = None
    for n in range(inp.Nscales):
        T_ILC_alm = hp.map2alm(all_maps[n])
        tmp = hp.almxfl(T_ILC_alm,filters[n]) #final needlet filtering
        if T_ILC_n is None:
            T_ILC_n = np.zeros((inp.Nscales,len(tmp)),dtype=np.complex128)
        T_ILC_n[n]=tmp
    T_ILC = np.sum(np.array([hp.alm2map(T_ILC_n[n],inp.nside) for n in range(len(T_ILC_n))]), axis=0) #adding maps from all scales
    return T_ILC


if __name__=='__main__':
    print('starting script compare_contam_spectra_nilc_cross.py', flush=True)

    # main input file containing most specifications
    try:
        input_file = (sys.argv)[1]
    except IndexError:
        input_file = 'example.yaml'

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # current environment, also environment in which to run subprocesses
    my_env = os.environ.copy()

    #set sim number to 101 (to not conflict with runs from main.py)
    sim = 101

    # Generate frequency maps with include_noise=False and get CC, T
    include_noise = True
    CC, T, N = generate_freq_maps(sim, inp.freqs, inp.tsz_amp, inp.nside, inp.ellmax, inp.cmb_alm_file, inp.halosky_maps_path, inp.scratch_path, inp.verbose, include_noise=include_noise)

    #get NILC weight maps for preserved component CMB and preserved component tSZ
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/CMB_preserved.yml {sim}"], shell=True, env=my_env)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component CMB, sim {sim}', flush=True)
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/tSZ_preserved.yml {sim}"], shell=True, env=my_env)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component tSZ, sim {sim}', flush=True)

    # Get weight map power spectra
    wt_map_power_spectrum = get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.nside, inp.verbose, inp.scratch_path, comps=['CMB', 'tSZ'])
    if inp.verbose:
        print(f'calculated weight map spectra for sim {sim}', flush=True)

    # Calculate propagation of CC and T to tSZ NILC CMB NILC cross spectrum
    M = wt_map_power_spectrum[1]
    del wt_map_power_spectrum #free up memory
    wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
    nfreqs = len(inp.freqs)
    h = GaussianNeedlets(inp.ellmax, inp.GN_FWHM_arcmin)[1]
    a = np.array([1., 1.])
    g = tsz_spectral_response(inp.freqs)
    CC_nilc = calculate_all_cl(nfreqs, inp.ellmax, h, a, CC, M, wigner) #CMB propagation from our equation
    T_nilc = calculate_all_cl(nfreqs, inp.ellmax, h, g, T, M, wigner) #tSZ propagation from our equation, should be unbiased
    if inp.verbose:
        print('calculated CC_nilc and T_nilc', flush=True)
    del wigner #free up memory




    #find CC from simulation directly
    CMB_wt_maps = load_wt_maps(sim, inp.Nscales, inp.nside, inp.scratch_path, comps=['CMB'])[0]
    tSZ_wt_maps = load_wt_maps(sim, inp.Nscales, inp.nside, inp.scratch_path, comps=['tSZ'])[1]
    cmb_map = hp.read_map(f'maps/{sim}_cmb_map.fits')
    CMB_in_CMB_NILC = sim_propagation(CMB_wt_maps, cmb_map, a, inp)
    CMB_in_tSZ_NILC = sim_propagation(tSZ_wt_maps, cmb_map, a, inp)
    CC_sim = hp.anafast(CMB_in_CMB_NILC, CMB_in_tSZ_NILC, lmax=inp.ellmax)

    #plot comparison of our approach and simulation for CMB
    ells = np.arange(inp.ellmax+1)
    plt.plot(ells[2:], (ells*(ells+1)*CC_sim/(2*np.pi))[2:],label='CMB directly calculated from simulation')
    plt.plot(ells[2:], (ells*(ells+1)*CC_nilc/(2*np.pi))[2:],label='CMB from analytic model')
    plt.legend()
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
    # plt.yscale('log')
    plt.savefig(f'contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_cross_compCMB_includenoise{include_noise}.png')
    if inp.verbose:
        print(f'saved contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_cross_compCMB_includenoise{include_noise}.png', flush=True)

    #find T from simulation directly
    tsz_map = inp.tsz_amp*hp.read_map(f'{inp.halosky_maps_path}/tsz_{sim:05d}.fits')
    tSZ_in_CMB_NILC = sim_propagation(CMB_wt_maps, tsz_map, g, inp)
    tSZ_in_tSZ_NILC = sim_propagation(tSZ_wt_maps, tsz_map, g, inp)
    T_sim = hp.anafast(tSZ_in_CMB_NILC, tSZ_in_tSZ_NILC, lmax=inp.ellmax)


    #plot comparison of our approach and simulation for tSZ
    ells = np.arange(inp.ellmax+1)
    plt.clf()
    plt.plot(ells[2:], (ells*(ells+1)*T_sim/(2*np.pi))[2:],label='tSZ directly calculated from simulation')
    plt.plot(ells[2:], (ells*(ells+1)*T_nilc/(2*np.pi))[2:],label='tSZ from analytic model')
    plt.legend()
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
    # plt.yscale('log')
    plt.savefig(f'contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_cross_comptSZ_includenoise{include_noise}.png')
    if inp.verbose:
        print(f'saved contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_cross_comptSZ_includenoise{include_noise}.png', flush=True)


    #delete files
    if inp.remove_files:
        subprocess.call(f'rm {inp.scratch_path}/wt_maps/CMB/{sim}_*', shell=True, env=my_env)
        subprocess.call(f'rm {inp.scratch_path}/wt_maps/tSZ/{sim}_*', shell=True, env=my_env)
        subprocess.call(f'rm {inp.scratch_path}/maps/sim{sim}_freq1.fits {inp.scratch_path}/maps/sim{sim}_freq2.fits', shell=True, env=my_env)
        subprocess.call(f'rm {inp.scratch_path}/maps/{sim}_cmb_map.fits', shell=True, env=my_env)