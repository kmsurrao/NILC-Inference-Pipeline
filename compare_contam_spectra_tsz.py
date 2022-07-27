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
CC, T, N = generate_freq_maps(sim, inp.freqs, inp.tsz_amp, inp.nside, inp.ellmax, inp.cmb_alm_file, inp.halosky_scripts_path, inp.verbose, include_noise=True)

# # Get NILC weight maps just for preserved CMB
# subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/CMB_preserved.yml {sim}"], shell=True, env=my_env)
# if inp.verbose:
#     print(f'generated NILC weight maps for preserved component CMB, sim {sim}', flush=True)

# Get weight map power spectra
wt_map_power_spectrum = get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.nside, inp.verbose, comps=['CMB'])
#get final NILC map and then don't need pyilc outputs anymore
NILC_map = hp.read_map(f'wt_maps/CMB/{sim}_needletILCmap_component_CMB.fits')*10**(-6)
if inp.verbose:
    print(f'calculated weight map spectra for sim {sim}', flush=True)

# Calculate propagation of T to NILC preserved CMB weight map
M = wt_map_power_spectrum[0]
del wt_map_power_spectrum #free up memory
wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
nfreqs = len(inp.freqs)
h = GaussianNeedlets(inp.ellmax, inp.GN_FWHM_arcmin)[1]
a = np.array([1., 1.])
g = tsz_spectral_response(inp.freqs)
T_nilc = calculate_all_cl(nfreqs, inp.ellmax, h, g, T, M, wigner) #tSZ propagation from our equation
if inp.verbose:
    print('calculated T_nilc', flush=True)
del wigner #free up memory


#find T from simulation directly
wt_maps = load_wt_maps(sim, inp.Nscales, inp.nside, comps=['CMB'])[0]
ell, filters = GaussianNeedlets(inp.ellmax, FWHM_arcmin=inp.GN_FWHM_arcmin)
npix = 12*inp.nside**2
nfreqs = len(inp.freqs)
all_maps = np.zeros((inp.Nscales,npix)) #index as all_maps[n][pixel]
for i in range(nfreqs):
    map_ = g[i]*hp.read_map('maps/tsz_00000.fits')
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
T_sim = hp.anafast(T_ILC, lmax=inp.ellmax)


#plot comparison of our approach and simulation
ells = np.arange(inp.ellmax+1)
plt.plot(ells[2:], (ells*(ells+1)*T_nilc/(2*np.pi))[2:], label='tSZ from our approach')
plt.plot(ells[2:], (ells*(ells+1)*T_sim/(2*np.pi))[2:], label='tSZ directly calculated from simulation')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{yy}}{2\pi}$ [$\mathrm{K}^2$]')
plt.yscale('log')
plt.savefig(f'contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_preservedCMB.png')
plt.close('all')
if inp.verbose:
    print(f'saved contam_spectra_comparison_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_preservedCMB.png', flush=True)

#delete files
if inp.remove_files:
    # subprocess.call(f'rm wt_maps/CMB/{sim}_*', shell=True, env=my_env)
    subprocess.call(f'rm maps/sim{sim}_freq1.fits maps/sim{sim}_freq2.fits', shell=True, env=my_env)
    subprocess.call('rm maps/tsz_00000*', shell=True, env=my_env)