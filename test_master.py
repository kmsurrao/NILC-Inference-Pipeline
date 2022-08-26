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

#load wigner3j symbols
wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]

#get halosky map and weight map and calculate their power spectra
tsz_map = 1000*hp.read_map(f'{inp.halosky_maps_path}/tsz_00000.fits')
tsz_map = hp.ud_grade(tsz_map, inp.nside)
# wt_map = hp.read_map(f'{inp.scratch_path}/wt_maps/tSZ/102_weightmap_freq1_scale4_component_tSZ.fits')
wt_map = hp.read_map(f'{inp.scratch_path}/wt_maps/CMB/4_weightmap_freq0_scale0_component_CMB.fits')*10**(-6)
wt_map = hp.ud_grade(wt_map, inp.nside)
tsz_cl = hp.anafast(tsz_map, lmax=inp.ellmax)
wt_map_cl = hp.anafast(wt_map, lmax=inp.ellmax)

#get masked map and calculate spectra
masked_map = tsz_map*wt_map
masked_map_cl = hp.anafast(masked_map, lmax=inp.ellmax)

#calculate spectra of masked map from MASTER approach
l2 = np.arange(inp.ellmax+1)
l3 = np.arange(inp.ellmax+1)
master_cl = float(1/(4*np.pi))*np.einsum('p,q,lpq,lpq,p,q->l',2*l2+1,2*l3+1,wigner,wigner,tsz_cl,wt_map_cl,optimize=True)

#make comparison plot of masked_map_cl and master_cl
ells = np.arange(inp.ellmax+1)
plt.clf()
plt.plot(ells[2:], (ells*(ells+1)*masked_map_cl/(2*np.pi))[2:], label='LHS of MASTER equation')
plt.plot(ells[2:], (ells*(ells+1)*master_cl/(2*np.pi))[2:], label='RHS of MASTER equation')
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\frac{\ell(\ell+1)C_{\ell}^{TT}}{2\pi}$ [$\mathrm{K}^2$]')
# plt.yscale('log')
plt.savefig(f'master_test_nongaussian.png')
print(f'saved fig master_test_nongaussian.png')
plt.close('all')
print(masked_map_cl/master_cl)
print('masked_map_cl: ', masked_map_cl)
print('master_cl: ', master_cl)
print('tsz_cl: ', tsz_cl)
