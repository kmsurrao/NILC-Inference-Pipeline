import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import subprocess
from input import Info
from generate_maps import *
from wt_map_spectra import *
from data_spectra import *

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
sim = 102

# Generate frequency maps with include_noise=False and get CC, T
CC, T, N = generate_freq_maps(sim, inp.freqs, inp.tsz_amp, inp.nside, inp.ellmax, inp.cmb_alm_file, inp.halosky_scripts_path, inp.verbose, include_noise=True)

# Get NILC weight maps just for preserved tSZ
subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/tSZ_preserved.yml {sim}"], shell=True, env=my_env)
if inp.verbose:
    print(f'generated NILC weight maps for preserved component tSZ, sim {sim}', flush=True)

# Get weight map power spectra
wt_map_power_spectrum = get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.nside, inp.verbose, comps=['tSZ'])
#get final NILC map and then don't need pyilc outputs anymore
NILC_map = hp.read_map(f'wt_maps/tSZ/{sim}_needletILCmap_component_tSZ.fits')
if inp.verbose:
    print(f'calculated weight map spectra for sim {sim}', flush=True)

# Calculate propagation of CC to NILC preserved tSZ weight map
M = wt_map_power_spectrum[2]
del wt_map_power_spectrum #free up memory
wigner_zero_m = get_wigner3j_zero_m(inp, save=False)
wigner_nonzero_m = get_wigner3j_nonzero_m(inp, save=False)
nfreqs = len(inp.freqs)
h = GaussianNeedlets(inp.ellmax, inp.GN_FWHM_arcmin)[1]
a = np.array([1., 1.])
g = tsz_spectral_response(inp.freqs)
l2 = np.arange(inp.ellmax+1)
l3 = np.arange(inp.ellmax+1)
M = M.astype(np.float32)[:,:,:,:,:inp.ellmax+1]
Cl = float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,nl,ml,np,mp,i,j,nmijq->lpq',2*l2+1,2*l3+1,wigner,wigner,CC,h[:,:inp.ellmax+1],h[:,:inp.ellmax+1],h,h,a,a,M,optimize=True) #l1 x l2 x l3
Cl = Cl[501][400:600,0:200]

#create heat map
fig, ax = plt.subplots()
c = ax.imshow(Cl, extent=[0, 200, 400, 600])
plt.colorbar(c)
plt.xlabel(r'$\ell_3$')
plt.ylabel(r'$\ell_2$')
plt.savefig('heatmap_ell500.png')
if inp.verbose:
    print('saved fig heatmap_ell500.png')

