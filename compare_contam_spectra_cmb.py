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
CC, T = generate_freq_maps(sim, inp.freqs, inp.tsz_amp, inp.nside, inp.ellmax, inp.cmb_alm_file, inp.halosky_scripts_path, inp.verbose, include_noise=True)

# # Get NILC weight maps just for preserved tSZ
# subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/tSZ_preserved.yml {sim}"], shell=True, env=my_env)
# if inp.verbose:
#     print(f'generated NILC weight maps for preserved component tSZ, sim {sim}', flush=True)

# Get weight map power spectra
wt_map_power_spectrum = get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.nside, inp.verbose, comps=['tSZ'])
#get final NILC map and then don't need pyilc outputs anymore
NILC_map = hp.read_map(f'wt_maps/tSZ/{sim}_needletILCmap_component_tSZ.fits')
if inp.verbose:
    print(f'calculated weight map spectra for sim {sim}', flush=True)

# Calculate propagation of CC to NILC preserved tSZ weight map
M = wt_map_power_spectrum[2]
del wt_map_power_spectrum #free up memory
wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
nfreqs = len(inp.freqs)
h = GaussianNeedlets(inp.ellmax, inp.GN_FWHM_arcmin)[1]
a = np.array([1., 1.])
g = tsz_spectral_response(inp.freqs)
CC_nilc = calculate_all_cl(nfreqs, inp.ellmax, h, a, CC, M, wigner) #CMB propagation from our equation
if inp.verbose:
    print('calculated CC_nilc', flush=True)
del wigner #free up memory

# Compute power spectrum of NILC map and subtract T. This is CC from simulation
NILC_map_spectrum = hp.anafast(NILC_map, lmax=inp.ellmax)
CC_sim = np.array(NILC_map_spectrum)-np.array(T) #tSZ calculated directly from simulation
if inp.verbose:
    print('calculated CC_sim', flush=True)
    print(f'NILC_map_spectrum: {NILC_map_spectrum[500:510]}', flush=True)
    print(f'T: {T[500:510]}', flush=True)
    print(f'CC_sim: {CC_sim[500:510]}', flush=True)
    print(f'CC_nilc: {CC_nilc[500:510]}', flush=True)

#plot comparison of our approach and simulation
ells = np.arange(inp.ellmax+1)
plt.plot(ells[2:750], (ells*(ells+1)*CC_nilc/(2*np.pi))[2:750],label='CMB from our approach', 'o')
plt.plot(ells[2:750], (ells*(ells+1)*CC_sim/(2*np.pi))[2:750],label='CMB directly calculated from simulation', 'o')
# plt.plot(ells[2:750], (ells*(ells+1)*NILC_map_spectrum/(2*np.pi))[2:750],label='tSZ NILC map spectrum')
# plt.plot(ells[2:750], (ells*(ells+1)*T/(2*np.pi))[2:750],label='tSZ input map spectrum')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.savefig(f'contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_preservedtSZ.png')
if inp.verbose:
    print(f'saved contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_preservedtSZ.png', flush=True)

#delete files
if inp.remove_files:
    # subprocess.call(f'rm wt_maps/tSZ/{sim}_*', shell=True, env=my_env)
    subprocess.call(f'rm maps/sim{sim}_freq1.fits maps/sim{sim}_freq2.fits', shell=True, env=my_env)
    subprocess.call('rm maps/tsz_00000*', shell=True, env=my_env)