import sys
import os
import subprocess
import multiprocessing as mp
from input import Info
from generate_maps import *
from wt_map_spectra import *
from data_spectra import *
from acmb_atsz_nilc import *

# main input file containing most specifications 
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = 'example.yaml'

# read in the input file and set up relevant info object
inp = Info(input_file)

# current environment, also environment in which to run subprocesses
my_env = os.environ.copy()

def one_sim(sim, inp=inp, env=my_env):
    #create frequency maps (GHz) consisting of CMB, tSZ, and noise
    #get power spectra of component maps (CC, T, and N)
    CC, T, N = generate_freq_maps(sim, inp.freqs, inp.tsz_amp, inp.nside, inp.ellmax, inp.cmb_alm_file, inp.halosky_scripts_path, inp.verbose)
    
    #get NILC weight maps for preserved component CMB and preserved component tSZ
    #note: need to remove after each sim run
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/CMB_preserved.yml {sim}"], shell=True, env=env)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component CMB, sim {sim}')
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/tSZ_preserved.yml {sim}"], shell=True, env=env)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component tSZ, sim {sim}')
    if inp.remove_files: #don't need frequency maps anymore
        subprocess.call(f'rm maps/sim{sim}_freq1.fits maps/sim{sim}_freq2.fits', shell=True, env=env)

    #get power spectra of weight maps--dimensions (3,Nscales,Nscales,Nfreqs,Nfreqs,ellmax)
    get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.nside, inp.verbose)
    #don't need pyilc outputs anymore
    subprocess.call(f'rm wt_maps/CMB/{sim}*', shell=True, env=env)
    subprocess.call(f'rm wt_maps/tSZ/{sim}*', shell=True, env=env)
    

    #get contributions to ClTT, ClTy, and Clyy from Acmb, Atsz, and noise components
    #0th index is sim number; 1st index is 0 for Acmb, 1 for Atsz, 2 for noise; 2nd index is ell
    get_data_spectra(sim, inp.freqs, inp.Nscales, inp.tsz_amp, inp.ellmax, inp.wigner_file, CC, T, N, inp.GN_FWHM_arcmin, inp.verbose)
    if inp.remove_files: #don't need weight map spectra anymore
        subprocess.call(f'rm wt_maps/sim{sim}_wt_map_spectra.p', shell=True, env=env)


for i in range(inp.Nsims):
    one_sim(i, inp, my_env)


lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = get_parameter_cov_matrix(inp.Nsims, inp.ellmax, inp.verbose)
print(f'Acmb={mean_acmb}+{upper_acmb-mean_acmb}-{mean_acmb-lower_acmb}')
print(f'Atsz={mean_atsz}+{upper_atsz-mean_atsz}-{mean_atsz-lower_atsz}')
