print('starting script compare_contam_spectra_preserved_cmb.py', flush=True)

import sys
sys.path.insert(0, "./../" )
sys.path.insert(0, "./../pipeline" )
import os
import subprocess
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
from input import Info
from nilc_power_spectrum_calc import calculate_all_cl_corrected
from generate_maps import *
from wt_map_spectra import *
from data_spectra import *
from wigner3j import *
from compare_contam_spectra_nilc_cross import sim_propagation
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
hp.disable_warnings()


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

# Generate frequency maps and get CC, T
CC, T, N = generate_freq_maps(sim, inp.freqs, inp.tsz_amp, inp.noise, inp.nside, inp.ellmax, inp.cmb_alm_file, inp.halosky_maps_path, inp.scratch_path, inp.verbose)

# Get NILC weight maps just for preserved CMB
# subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/CMB_preserved.yml {sim}"], shell=True, env=my_env)
if inp.verbose:
    print(f'generated NILC weight maps for preserved component CMB, sim {sim}', flush=True)

# Get weight map power spectra
wt_map_power_spectrum = get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.verbose, inp.scratch_path, comps=['CMB'])
if inp.verbose:
    print(f'calculated weight map spectra for sim {sim}', flush=True)
M = wt_map_power_spectrum[0]
del wt_map_power_spectrum #free up memory

#load component maps
tsz_map = inp.tsz_amp*hp.read_map(f'{inp.halosky_maps_path}/tsz_{sim:05d}.fits')
# tsz_map = hp.read_map(f'{inp.scratch_path}/test_maps/{sim}_tsz_map.fits') #remove this later
cmb_map = hp.read_map(f'{inp.scratch_path}/maps/{sim}_cmb_map.fits')

# Get component map and weight map cross spectra
W_tsz = get_comp_and_wt_map_cross_spectra(sim, inp.ellmax, inp.Nscales, tsz_map, inp.verbose, inp.scratch_path, ['CMB'])[0]
W_cmb = get_comp_and_wt_map_cross_spectra(sim, inp.ellmax, inp.Nscales, cmb_map, inp.verbose, inp.scratch_path, ['CMB'])[0]
if inp.verbose:
    print(f'calculated component map and weight map cross spectra for sim {sim}', flush=True)

# Calculate propagation of T and CC to NILC preserved CMB weight map
wigner_zero_m = get_wigner3j_zero_m(inp, save=False)
wigner_nonzero_m = get_wigner3j_nonzero_m(inp, save=False)
nfreqs = len(inp.freqs)
h = GaussianNeedlets(inp.ellmax, inp.GN_FWHM_arcmin)[1]
a = np.array([1., 1.])
g = tsz_spectral_response(inp.freqs)
T_nilc = calculate_all_cl_corrected(nfreqs, inp.ellmax, h, g, T, M, W_tsz, W_tsz, wigner_zero_m, wigner_nonzero_m) #tSZ propagation from our equation
CC_nilc = calculate_all_cl_corrected(nfreqs, inp.ellmax, h, a, CC, M, W_cmb, W_cmb, wigner_zero_m, wigner_nonzero_m) #CMB propagation from our equation, should be unbiased
if inp.verbose:
    print('calculated T_nilc and CC_nilc', flush=True)
del wigner #free up memory


#find T from simulation directly
wt_maps = load_wt_maps(sim, inp.Nscales, inp.nside, inp.scratch_path, comps=['CMB'])[0]
tSZ_in_CMB_NILC = sim_propagation(wt_maps, tsz_map, g, inp)
T_sim = hp.anafast(tSZ_in_CMB_NILC, lmax=inp.ellmax)


#plot comparison of our approach and simulation for tSZ
ells = np.arange(inp.ellmax+1)
plt.plot(ells[2:], (ells*(ells+1)*T/(2*np.pi))[2:], label='tSZ input')
plt.plot(ells[2:], (ells*(ells+1)*T_sim/(2*np.pi))[2:], label='tSZ directly calculated from simulation')
plt.plot(ells[2:], (ells*(ells+1)*T_nilc/(2*np.pi))[2:], label='tSZ from analytic model')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{yy}}{2\pi}$ [$\mathrm{K}^2$]')
plt.yscale('log')
plt.savefig(f'contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png')
plt.close('all')
if inp.verbose:
    print(f'saved contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png', flush=True)


#find CC from simulation directly
CMB_in_CMB_NILC = sim_propagation(wt_maps, cmb_map, a, inp)
CC_sim = hp.anafast(CMB_in_CMB_NILC, lmax=inp.ellmax)


#plot comparison of our approach and simulation for CMB
ells = np.arange(inp.ellmax+1)
plt.clf()
plt.plot(ells[2:], (ells*(ells+1)*CC/(2*np.pi))[2:], label='CMB input')
plt.plot(ells[2:], (ells*(ells+1)*CC_sim/(2*np.pi))[2:], label='CMB directly calculated from simulation')
plt.plot(ells[2:], (ells*(ells+1)*CC_nilc/(2*np.pi))[2:], label='CMB from analytic model')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.savefig(f'contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
plt.close('all')
if inp.verbose:
    print(f'saved contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)


#plot ratio T_nilc/T_sim
plt.clf()
plt.plot(ells[2:], (T_nilc/T_sim)[2:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{T_{\mathrm{nilc}}}{T_{\mathrm{sim}}}$')
plt.savefig(f'ratio_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png')
plt.close('all')
if inp.verbose:
    print(f'saved ratio_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png', flush=True)

#plot ratio CC_nilc/CC_sim
plt.clf()
plt.plot(ells[2:], (CC_nilc/CC_sim)[2:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{CC_{\mathrm{nilc}}}{CC_{\mathrm{sim}}}$')
plt.savefig(f'ratio_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
plt.close('all')
if inp.verbose:
    print(f'saved ratio_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)

#plot ratio CC_nilc/CC
plt.clf()
plt.plot(ells[2:], (CC_nilc/CC)[2:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{CC_{\mathrm{nilc}}}{CC}$')
plt.savefig(f'ratioCCnilcCC_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
plt.close('all')
if inp.verbose:
    print(f'saved ratioCCnilcCC_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)

#plot ratio CC_sim/CC
plt.clf()
plt.plot(ells[2:], (CC_sim/CC)[2:])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{CC_{\mathrm{sim}}}{CC}$')
plt.savefig(f'ratioCCsimCC_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
plt.close('all')
if inp.verbose:
    print(f'saved ratioCCsimCC_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)

#delete files
if inp.remove_files:
    subprocess.call(f'rm {inp.scratch_path}/wt_maps/CMB/{sim}_*', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/sim{sim}_freq1.fits {inp.scratch_path}/maps/sim{sim}_freq2.fits', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/{sim}_cmb_map.fits', shell=True, env=my_env)