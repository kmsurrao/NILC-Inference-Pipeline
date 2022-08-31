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


#define scale and frequency of interest
n = 4
m = 4

def calculate_all_cl(scale_n, scale_m, nfreqs, ellmax, h, a, cl, M, wigner, constituent_plot=False, delta_ij=False):
    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    M = M.astype(np.float32)[:,:,:,:,:ellmax+1]
    # Cl = float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,i,j,nmijq->nml',2*l2+1,2*l3+1,wigner,wigner,cl,a,a,M,optimize=True) #one scale, one frequency with no needlet filters
    nscales = len(h)
    Cl = np.zeros((nscales, nscales, ellmax+1))
    # for (i,j) in [(0,0), (0,1), (1,0), (1,1)]:
    for (i,j) in [(0,0), (0,1), (1,0), (1,1)]:
        M_tmp = M[:,:,i,j,:]
        Cl_at_ij = a[i]*a[j]*float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,nmq->nml',2*l2+1,2*l3+1,wigner,wigner,cl,M_tmp,optimize=True)
        # Cl_at_ij = a[i]*a[j]*float(1/(4*np.pi))*np.einsum('p,lpq,lpq,p,nmq->nml',2*l2+1,wigner,wigner,cl,M_tmp,optimize=True) #remove, test to see if had extra factor 2ell3+1
        Cl += Cl_at_ij
        if constituent_plot:
            plt.plot(np.arange(2, ellmax+1), Cl_at_ij[scale_n][scale_m][2:], label=f'i={i}, j={j}')
    if constituent_plot:
        plt.plot(np.arange(2, ellmax+1), Cl[scale_n][scale_m][2:], label=f'total')
        plt.legend()
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{i(n),j(m)}}{2\pi}$')
        # plt.yscale('log')
        plt.savefig('test_sums/test_sum_freqs_constituents.png')
        print('saved test_sums/test_sum_freqs_constituents.png')
    return Cl[scale_n][scale_m]

def sim_propagation(n, m, wt_maps, sim_map, spectral_response, inp):
    ell, filters = GaussianNeedlets(inp.ellmax, FWHM_arcmin=inp.GN_FWHM_arcmin)
    npix = 12*inp.nside**2
    map1_tot, map2_tot = np.zeros(npix), np.zeros(npix)
    for i in range(2):
        map1 = spectral_response[i]*sim_map
        map1 = hp.ud_grade(map1, inp.nside)
        NILC_weights1 = hp.ud_grade(wt_maps[n][i],inp.nside)
        map1 = map1*NILC_weights1 #application of weight map
        map1_tot += map1
    for i in range(2):
        map2 = spectral_response[i]*sim_map
        map2 = hp.ud_grade(map2, inp.nside)
        NILC_weights2 = hp.ud_grade(wt_maps[m][i],inp.nside)
        map2 = map2*NILC_weights2 #application of weight map
        map2_tot += map2
    return hp.anafast(map1_tot, map2_tot, lmax=inp.ellmax)


def compare(n, m, wt_maps, sim_map, spectral_response, inp, nfreqs, ellmax, h, a, cl, M, wigner):
    ell, filters = GaussianNeedlets(inp.ellmax, FWHM_arcmin=inp.GN_FWHM_arcmin)
    npix = 12*inp.nside**2
    sim_tot = np.zeros(inp.ellmax+1)

    l2 = np.arange(ellmax+1)
    l3 = np.arange(ellmax+1)
    M = M.astype(np.float32)[:,:,:,:,:ellmax+1]
    nscales = len(h)
    analytic_tot = np.zeros((nscales, nscales, ellmax+1))

    for (i,j) in [(0,0), (0,1), (1,0), (1,1)]:

        #direct calculation
        map1 = spectral_response[i]*sim_map
        map1 = hp.ud_grade(map1, inp.nside)
        NILC_weights1 = hp.ud_grade(wt_maps[n][i],inp.nside)
        map1 = map1*NILC_weights1 #application of weight map
        map2 = spectral_response[j]*sim_map
        map2 = hp.ud_grade(map2, inp.nside)
        NILC_weights2 = hp.ud_grade(wt_maps[m][j],inp.nside)
        map2 = map2*NILC_weights2 #application of weight map
        sim_ij = hp.anafast(map1, map2, lmax=inp.ellmax)
        sim_tot += sim_ij

        #analytic equation
        M_tmp = M[:,:,i,j,:]
        analytic_ij = a[i]*a[j]*float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,nmq->nml',2*l2+1,2*l3+1,wigner,wigner,cl,M_tmp,optimize=True)
        # analytic_ij = a[i]*a[j]*float(1/(4*np.pi))*np.einsum('p,lpq,lpq,p,nmq->nml',2*l2+1,wigner,wigner,cl,M_tmp,optimize=True) #remove, test to see if had extra factor 2ell3+1
        analytic_tot += analytic_ij

        ells = np.arange(2, ellmax+1)
        plt.clf()
        plt.plot(ells, ells*(ells+1)*sim_ij[2:]/(2*np.pi), label=f'sim')
        plt.plot(ells, ells*(ells+1)*analytic_ij[n][m][2:]/(2*np.pi), label=f'analytic')
        plt.legend()
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{nm}}{2\pi}$')
        plt.title(f'i={i}, j={j}, n={n}, m={m}')
        plt.savefig(f'test_sums/test_constituents_ij{i}{j}_scales{n}{m}.png')
        print(f'saved test_sums/fig test_constituents_ij{i}{j}_scales{n}{m}.png')

        plt.clf()
        plt.plot(ells, analytic_ij[n][m][2:]/sim_ij[2:])
        plt.xlabel(r'$\ell$')
        plt.ylabel('analytic/simulation')
        plt.title(f'Ratio plot i={i}, j={j}, n={n}, m={m}')
        plt.savefig(f'test_sums/test_constituents_ratio_ij{i}{j}_scales{n}{m}.png')
        print(f'saved fig test_sums/test_constituents_ratio_ij{i}{j}_scales{n}{m}.png')

    plt.clf()
    plt.plot(ells, ells*(ells+1)*sim_tot[2:]/(2*np.pi), label=f'sim')
    plt.plot(ells, ells*(ells+1)*analytic_tot[n][m][2:]/(2*np.pi), label=f'analytic')
    plt.legend()
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{nm}}{2\pi}$')
    plt.title(f'summed over frequencies n={n}, m={m}')
    plt.savefig(f'test_sums/test_constituents_tot_scales{n}{m}.png')
    print(f'saved fig test_sums/test_constituents_tot_scales{n}{m}.png')

    plt.clf()
    plt.plot(ells, analytic_tot[n][m][2:]/sim_tot[2:])
    plt.xlabel(r'$\ell$')
    plt.ylabel('analytic/simulation')
    plt.title(f'Ratio plot for summed over frequencies n={n}, m={m}')
    plt.savefig(f'test_sums/test_constituents_ratio_tot_scales{n}{m}.png')
    print(f'saved fig test_sums/test_constituents_ratio_tot_scales{n}{m}.png')

    return sim_tot, analytic_tot[n][m]


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
#use sim 103 to use random weight maps or 104 for wt maps perfectly correlated to tSZ. comment out pyilc line
sim = 101

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
T_nilc = calculate_all_cl(n, m, nfreqs, inp.ellmax, h, g, T, M, wigner, constituent_plot=True) #tSZ propagation from our equation
CC_nilc = calculate_all_cl(n, m, nfreqs, inp.ellmax, h, a, CC, M, wigner) #CMB propagation from our equation, should be unbiased
if inp.verbose:
    print('calculated T_nilc and CC_nilc', flush=True)


#find T from simulation directly
wt_maps = load_wt_maps(sim, inp.Nscales, inp.nside, inp.scratch_path, comps=['CMB'])[0]
tsz_map = inp.tsz_amp*hp.read_map(f'{inp.halosky_maps_path}/tsz_{sim:05d}.fits')
# tsz_map = hp.read_map(f'{inp.scratch_path}/test_maps/{sim}_tsz_map.fits') #remove this later, keep for gaussian tSZ
T_sim = sim_propagation(n, m, wt_maps, tsz_map, g, inp)


#plot comparison of our approach and simulation for tSZ
ells = np.arange(inp.ellmax+1)
plt.clf()
plt.plot(ells[2:], (ells*(ells+1)*T_sim/(2*np.pi))[2:], label='tSZ directly calculated from simulation')
plt.plot(ells[2:], (ells*(ells+1)*T_nilc/(2*np.pi))[2:], label='tSZ from analytic model')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{yy}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.savefig(f'test_sums/test_sum_freqs_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png')
plt.close('all')
if inp.verbose:
    print(f'saved test_sums/test_sum_freqs_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_comptSZ.png', flush=True)

#make plots of tSZ propagation from direct and analytic calculation, total and at each frequency pair
tsz_compare_sim_tot, _ = compare(n, m, wt_maps, tsz_map, g, inp, nfreqs, inp.ellmax, h, g, T, M, wigner)
del wigner #free up memory
plt.clf()
plt.plot(ells[2:], (ells*(ells+1)*T_sim/(2*np.pi))[2:], label='directly calculated from simulation')
plt.plot(ells[2:], (ells*(ells+1)*tsz_compare_sim_tot/(2*np.pi))[2:], label='direct calculation components summed over freqs')
plt.plot(ells[2:], (ells*(ells+1)*T_nilc/(2*np.pi))[2:], label='tSZ from analytic model')
plt.title(f'tSZ to CMB preserved NILC without filtering, scale {n}, {m}')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{yy}}{2\pi}$ [$\mathrm{K}^2$]')
plt.savefig(f'test_sums/test_compare_direct_calc.png')
if inp.verbose:
    print(f'saved test_sums/test_compare_direct_calc.png', flush=True)
plt.clf()
plt.plot(ells[2:], (T_sim/tsz_compare_sim_tot)[2:])
plt.title(f'tSZ to CMB preserved NILC without filtering, scale {n}, {m}')
plt.xlabel(r'$\ell$')
plt.ylabel('direct calculation/ sum direct calculation over freqs')
plt.savefig(f'test_sums/test_compare_direct_calc_ratio.png')
if inp.verbose:
    print(f'saved test_sums/test_compare_direct_calc_ratio.png', flush=True)




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
plt.savefig(f'test_sum_freqs_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png')
plt.close('all')
if inp.verbose:
    print(f'saved test_sum_freqs_nside{inp.nside}_ellmax{inp.ellmax}_tSZamp{int(inp.tsz_amp)}_noise{int(inp.noise)}_preservedCMB_compCMB.png', flush=True)


#delete files
if inp.remove_files:
    # subprocess.call(f'rm {inp.scratch_path}/wt_maps/CMB/{sim}_*', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/sim{sim}_freq1.fits {inp.scratch_path}/maps/sim{sim}_freq2.fits', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/{sim}_cmb_map.fits', shell=True, env=my_env)