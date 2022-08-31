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



def calculate_all_cl(nfreqs, ellmax, h, a, cl, M, wigner, delta_ij=False):
    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    M = M.astype(np.float32)[:,:,:,:,:ellmax+1]
    Cl = float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,np,mp,i,j,nmijq->l',2*l2+1,2*l3+1,wigner,wigner,cl,h,h,a,a,M,optimize=True) #without final needlet filter
    return Cl

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
        tmp = T_ILC_alm #remove and uncomment line below for original
        # tmp = hp.almxfl(T_ILC_alm,filters[n]) #final needlet filtering
        if T_ILC_n is None:
            # T_ILC_n = np.zeros((inp.Nscales,len(tmp)),dtype=np.complex128)
            T_ILC_n = np.zeros((inp.Nscales,len(tmp)),dtype=np.complex128)
        T_ILC_n[n]=tmp
    T_ILC = np.sum(np.array([hp.alm2map(T_ILC_n[n],inp.nside) for n in range(len(T_ILC_n))]), axis=0) #adding maps from all scales
    return hp.anafast(T_ILC, lmax=inp.ellmax)

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
T_nilc = calculate_all_cl(nfreqs, inp.ellmax, h, g, T, M, wigner) #tSZ propagation from our equation
CC_nilc = calculate_all_cl(nfreqs, inp.ellmax, h, a, CC, M, wigner) #CMB propagation from our equation, should be unbiased
if inp.verbose:
    print('calculated T_nilc and CC_nilc', flush=True)
del wigner #free up memory


#find T from simulation directly
wt_maps = load_wt_maps(sim, inp.Nscales, inp.nside, inp.scratch_path, comps=['CMB'])[0]
tsz_map = inp.tsz_amp*hp.read_map(f'{inp.halosky_maps_path}/tsz_{sim:05d}.fits')
# tsz_map = hp.read_map(f'{inp.scratch_path}/test_maps/{sim}_tsz_map.fits') #remove this later
T_sim = sim_propagation(wt_maps, tsz_map, g, inp)
# T_sim = hp.anafast(tSZ_in_CMB_NILC, lmax=inp.ellmax)


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
CC_sim = sim_propagation(wt_maps, cmb_map, a, inp)
# CC_sim = hp.anafast(CMB_in_CMB_NILC, lmax=inp.ellmax)


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


# #plot ratio T_nilc/T_sim
# plt.clf()
# plt.plot(ells[2:], (T_nilc/T_sim)[2:])
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$\frac{T_{\mathrm{nilc}}}{T_{\mathrm{sim}}}$')
# plt.savefig(f'ratio_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png')
# plt.close('all')
# if inp.verbose:
#     print(f'saved ratio_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png', flush=True)

# #plot ratio CC_nilc/CC_sim
# plt.clf()
# plt.plot(ells[2:], (CC_nilc/CC_sim)[2:])
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$\frac{CC_{\mathrm{nilc}}}{CC_{\mathrm{sim}}}$')
# plt.savefig(f'ratio_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
# plt.close('all')
# if inp.verbose:
#     print(f'saved ratio_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)

# #plot ratio CC_nilc/CC
# plt.clf()
# plt.plot(ells[2:], (CC_nilc/CC)[2:])
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$\frac{CC_{\mathrm{nilc}}}{CC}$')
# plt.savefig(f'ratioCCnilcCC_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
# plt.close('all')
# if inp.verbose:
#     print(f'saved ratioCCnilcCC_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)

# #plot ratio CC_sim/CC
# plt.clf()
# plt.plot(ells[2:], (CC_sim/CC)[2:])
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$\frac{CC_{\mathrm{sim}}}{CC}$')
# plt.savefig(f'ratioCCsimCC_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
# plt.close('all')
# if inp.verbose:
#     print(f'saved ratioCCsimCC_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)

#delete files
if inp.remove_files:
    # subprocess.call(f'rm {inp.scratch_path}/wt_maps/CMB/{sim}_*', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/sim{sim}_freq1.fits {inp.scratch_path}/maps/sim{sim}_freq2.fits', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/{sim}_cmb_map.fits', shell=True, env=my_env)