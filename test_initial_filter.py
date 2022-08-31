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
from wigner3j import *
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
hp.disable_warnings()


#define scales of interest
n = 4
m = 5


def calculate_all_cl(scale_n, scale_m, nfreqs, ellmax, h, a, cl, M, wigner, delta_ij=False):
    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    M = M.astype(np.float32)[:,:,:,:,:ellmax+1]
    Cl = float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,np,mp,i,j,nmijq->nml',2*l2+1,2*l3+1,wigner,wigner,cl,h,h,a,a,M,optimize=True) #one scale, one frequency with no needlet filters
    return Cl[scale_n][scale_m]

def sim_propagation(scale_n, scale_m, wt_maps, sim_map, spectral_response, inp):
    ell, filters = GaussianNeedlets(inp.ellmax, FWHM_arcmin=inp.GN_FWHM_arcmin)
    # filters = np.ones(filters.shape) #remove
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
    return hp.anafast(all_maps[scale_n], all_maps[scale_m], lmax=inp.ellmax)

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
subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/CMB_preserved.yml {sim}"], shell=True, env=my_env)
if inp.verbose:
    print(f'generated NILC weight maps for preserved component CMB, sim {sim}', flush=True)

# Get weight map power spectra
wt_map_power_spectrum = get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.verbose, inp.scratch_path, comps=['CMB'])
if inp.verbose:
    print(f'calculated weight map spectra for sim {sim}', flush=True)

# Calculate propagation of T and CC to NILC preserved CMB map
M = wt_map_power_spectrum[0]
del wt_map_power_spectrum #free up memory
wigner_zero_m = get_wigner3j_zero_m(inp, save=False)
wigner_nonzero_m = get_wigner3j_nonzero_m(inp, save=False)
nfreqs = len(inp.freqs)
h = GaussianNeedlets(inp.ellmax, inp.GN_FWHM_arcmin)[1]
# h = np.ones(h.shape) #remove
a = np.array([1., 1.])
g = tsz_spectral_response(inp.freqs)
T_nilc = calculate_all_cl(n, m, nfreqs, inp.ellmax, h, g, T, M, wigner) #tSZ propagation from our equation
CC_nilc = calculate_all_cl(n, m, nfreqs, inp.ellmax, h, a, CC, M, wigner) #CMB propagation from our equation, should be unbiased
if inp.verbose:
    print('calculated T_nilc and CC_nilc', flush=True)
del wigner #free up memory


#find T from simulation directly
wt_maps = load_wt_maps(sim, inp.Nscales, inp.nside, inp.scratch_path, comps=['CMB'])[0]
tsz_map = inp.tsz_amp*hp.read_map(f'{inp.halosky_maps_path}/tsz_{sim:05d}.fits')
# tsz_map = hp.read_map(f'{inp.scratch_path}/test_maps/{sim}_tsz_map.fits') #remove this later
T_sim = sim_propagation(n, m, wt_maps, tsz_map, g, inp)


#plot comparison of our approach and simulation for tSZ
ells = np.arange(inp.ellmax+1)
plt.plot(ells[2:], (ells*(ells+1)*T_sim/(2*np.pi))[2:], label='tSZ directly calculated from simulation')
plt.plot(ells[2:], (ells*(ells+1)*T_nilc/(2*np.pi))[2:], label='tSZ from analytic model')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{yy}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.savefig(f'test_initial_filter_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png')
plt.close('all')
if inp.verbose:
    print(f'saved test_initial_filter_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png', flush=True)


#find CC from simulation directly
cmb_map = hp.read_map(f'{inp.scratch_path}/maps/{sim}_cmb_map.fits')
CC_sim = sim_propagation(n, m, wt_maps, cmb_map, a, inp)


#plot comparison of our approach and simulation for CMB
ells = np.arange(inp.ellmax+1)
plt.clf()
plt.plot(ells[2:], (ells*(ells+1)*CC_sim/(2*np.pi))[2:], label='CMB directly calculated from simulation')
plt.plot(ells[2:], (ells*(ells+1)*CC_nilc/(2*np.pi))[2:], label='CMB from analytic model')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.savefig(f'test_initial_filter_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
plt.close('all')
if inp.verbose:
    print(f'saved test_initial_filter_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)


#delete files
if inp.remove_files:
    # subprocess.call(f'rm {inp.scratch_path}/wt_maps/CMB/{sim}_*', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/sim{sim}_freq1.fits {inp.scratch_path}/maps/sim{sim}_freq2.fits', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/{sim}_cmb_map.fits', shell=True, env=my_env)