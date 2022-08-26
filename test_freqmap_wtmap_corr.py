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
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

n = 0
i = 0

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
wt_maps = load_wt_maps(sim, inp.Nscales, inp.nside, inp.scratch_path, comps=['CMB'])[0]

#find correlation of ith frequency map with ith weight map at scale n
freq_map = hp.read_map(f'{inp.scratch_path}/maps/sim{sim}_freq{i+1}.fits')
freq_map = hp.ud_grade(freq_map, inp.nside)
wt_map = hp.ud_grade(wt_maps[n][i], inp.nside)
corr = hp.anafast(freq_map, wt_map, lmax=inp.ellmax)

#plot cross power spectrum
plt.clf()
ells = np.arange(2, inp.ellmax+1)
plt.plot(ells, ells*(ells+1)*corr[2:]/(2*np.pi))
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TM}}{2\pi}$')
plt.title(f'Cross power spectrum of freq {i} map and weight map at freq {i} scale {n}')
plt.savefig(f'freqmap_wtmap_corr_i{i}_n{n}.png')
print(f'saved fig freqmap_wtmap_corr_i{i}_n{n}.png')


#delete files
if inp.remove_files:
    # subprocess.call(f'rm {inp.scratch_path}/wt_maps/CMB/{sim}_*', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/sim{sim}_freq1.fits {inp.scratch_path}/maps/sim{sim}_freq2.fits', shell=True, env=my_env)
    subprocess.call(f'rm {inp.scratch_path}/maps/{sim}_cmb_map.fits', shell=True, env=my_env)