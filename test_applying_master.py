import sys
import os
import subprocess
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
from input import Info
from generate_maps import *
from wt_map_spectra import *
from data_spectra import *
from py3nj import *
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
hp.disable_warnings()


#define scale and frequency of interest
n = 4
m = 4
i = 1
j = 1

def calculate_all_cl(scale_n, scale_m, freq_i, freq_j, nfreqs, ellmax, h, a, cl, M, wigner, delta_ij=False):
    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    M = M.astype(np.float32)[:,:,:,:,:ellmax+1]
    Cl = float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,i,j,nmijq->nmijl',2*l2+1,2*l3+1,wigner,wigner,cl,a,a,M,optimize=True) #one scale, one frequency with no needlet filters
    return Cl[scale_n][scale_m][freq_i][freq_j]

def sim_propagation(n, m, i, j, wt_maps, sim_map, spectral_response, inp):
    ell, filters = GaussianNeedlets(inp.ellmax, FWHM_arcmin=inp.GN_FWHM_arcmin)
    npix = 12*inp.nside**2
    map1 = spectral_response[i]*sim_map
    map1 = hp.ud_grade(map1, inp.nside)
    NILC_weights1 = hp.ud_grade(wt_maps[n][i],inp.nside)
    map1 = map1*NILC_weights1 #application of weight map
    map2 = spectral_response[j]*sim_map
    map2 = hp.ud_grade(map2, inp.nside)
    NILC_weights2 = hp.ud_grade(wt_maps[m][j],inp.nside)
    map2 = map2*NILC_weights2 #application of weight map
    return hp.anafast(map1, map2, lmax=inp.ellmax)

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
sim = 103

# Generate frequency maps and get CC, T
CC, T, N = generate_freq_maps(sim, inp.freqs, inp.tsz_amp, inp.noise, inp.nside, inp.ellmax, inp.cmb_alm_file, inp.halosky_maps_path, inp.scratch_path, inp.verbose)

# Get NILC weight maps just for preserved CMB
# subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/CMB_preserved.yml {sim}"], shell=True, env=my_env)
if inp.verbose:
    print(f'generated NILC weight maps for preserved component CMB, sim {sim}', flush=True)

# Get weight map power spectra
wt_map_power_spectrum = get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.nside, inp.verbose, inp.scratch_path, comps=['CMB'])
if inp.verbose:
    print(f'calculated weight map spectra for sim {sim}', flush=True)

# Calculate propagation of T and CC to NILC preserved CMB map
M = wt_map_power_spectrum[0]
del wt_map_power_spectrum #free up memory
wigner_zero_m = get_wigner3j_zero_m(inp, save=False)
wigner_nonzero_m = get_wigner3j_nonzero_m(inp, save=False)
nfreqs = len(inp.freqs)
h = GaussianNeedlets(inp.ellmax, inp.GN_FWHM_arcmin)[1]
a = np.array([1., 1.])
g = tsz_spectral_response(inp.freqs)
T_nilc = calculate_all_cl(n, m, i, j, nfreqs, inp.ellmax, h, g, T, M, wigner) #tSZ propagation from our equation
CC_nilc = calculate_all_cl(n, m, i, j, nfreqs, inp.ellmax, h, a, CC, M, wigner) #CMB propagation from our equation, should be unbiased
if inp.verbose:
    print('calculated T_nilc and CC_nilc', flush=True)
del wigner #free up memory


#find T from simulation directly
wt_maps = load_wt_maps(sim, inp.Nscales, inp.nside, inp.scratch_path, comps=['CMB'])[0]
tsz_map = inp.tsz_amp*hp.read_map(f'{inp.halosky_maps_path}/tsz_{sim:05d}.fits')
# tsz_map = hp.read_map(f'{inp.scratch_path}/test_maps/{sim}_tsz_map.fits') #remove this later
T_sim = sim_propagation(n, m, i, j, wt_maps, tsz_map, g, inp)


#plot comparison of our approach and simulation for tSZ
ells = np.arange(inp.ellmax+1)
plt.plot(ells[2:], (ells*(ells+1)*T_sim/(2*np.pi))[2:], label='tSZ directly calculated from simulation')
plt.plot(ells[2:], (ells*(ells+1)*T_nilc/(2*np.pi))[2:], label='tSZ from analytic model')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{yy}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.savefig(f'test_applying_master_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png')
plt.close('all')
if inp.verbose:
    print(f'saved test_applying_master_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png', flush=True)


#find CC from simulation directly
cmb_map = hp.read_map(f'{inp.scratch_path}/maps/{sim}_cmb_map.fits')
CC_sim = sim_propagation(n, m, i, j, wt_maps, cmb_map, a, inp)


#plot comparison of our approach and simulation for CMB
ells = np.arange(inp.ellmax+1)
plt.clf()
plt.plot(ells[2:], (ells*(ells+1)*CC_sim/(2*np.pi))[2:], label='CMB directly calculated from simulation')
plt.plot(ells[2:], (ells*(ells+1)*CC_nilc/(2*np.pi))[2:], label='CMB from analytic model')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.savefig(f'test_applying_master_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
plt.close('all')
if inp.verbose:
    print(f'saved test_applying_master_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)


#delete files
if inp.remove_files:
    # subprocess.call(f'rm {inp.scratch_path}/wt_maps/CMB/{sim}_*', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/sim{sim}_freq1.fits {inp.scratch_path}/maps/sim{sim}_freq2.fits', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/{sim}_cmb_map.fits', shell=True, env=my_env)