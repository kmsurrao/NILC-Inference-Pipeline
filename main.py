import sys
import os
import subprocess
import multiprocessing as mp
from input import Info
import pickle
from __future__ import print_function
from scoop import futures
from generate_maps import *
from wt_map_spectra import *
from data_spectra import *
from acmb_atsz_nilc import *
from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
hp.disable_warnings()


def one_sim(sim, inp=inp, env=my_env):
    #create frequency maps (GHz) consisting of CMB, tSZ, and noise
    #get power spectra of component maps (CC, T, and N)
    CC, T, N = generate_freq_maps(sim, inp.freqs, inp.tsz_amp, inp.nside, inp.ellmax, inp.cmb_alm_file, inp.halosky_scripts_path, inp.verbose)
    
    #get NILC weight maps for preserved component CMB and preserved component tSZ
    #note: need to remove after each sim run
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/CMB_preserved.yml {sim}"], shell=True, env=env)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component CMB, sim {sim}', flush=True)
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/tSZ_preserved.yml {sim}"], shell=True, env=env)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component tSZ, sim {sim}', flush=True)
    if inp.remove_files: #don't need frequency maps anymore
        subprocess.call(f'rm maps/sim{sim}_freq1.fits maps/sim{sim}_freq2.fits', shell=True, env=env)
        subprocess.call('rm maps/{sim}_tsz_00000.fits', shell=True, env=env)
        subprocess.call('rm maps/tsz_00000*', shell=True, env=env)
        subprocess.call('rm maps/{sim}_cmb_map.fits', shell=True, env=env)

    #get power spectra of weight maps--dimensions (3,Nscales,Nscales,Nfreqs,Nfreqs,ellmax)
    wt_map_power_spectrum = get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.nside, inp.verbose)
    #don't need pyilc outputs anymore
    subprocess.call(f'rm wt_maps/CMB/{sim}_*', shell=True, env=env)
    subprocess.call(f'rm wt_maps/tSZ/{sim}_*', shell=True, env=env)
    if inp.verbose:
        print(f'calculated weight map spectra for sim {sim}', flush=True)

    #get contributions to ClTT, ClTy, and Clyy from Acmb, Atsz, and noise components
    #has dim (3 for ClTT ClTy Clyy, 3 for CMB tSZ noise components, ellmax+1)
    data_spectra = get_data_spectra(sim, inp.freqs, inp.Nscales, inp.tsz_amp, inp.ellmax, inp.wigner_file, CC, T, N, wt_map_power_spectrum, inp.GN_FWHM_arcmin, inp.verbose)
    if inp.verbose:
        print(f'calculated data spectra for sim {sim}', flush=True)
    #don't need weight map spectra anymore
    del wt_map_power_spectrum #free up memory

    return data_spectra

# pool = mp.Pool(inp.num_parallel)
# results = pool.starmap(one_sim, [(sim, inp, my_env) for sim in range(inp.Nsims)])
# pool.close()
# results = np.asarray(results, dtype=np.float32) #dim (Nsims, 3 for ClTT ClTy Clyy, 3 for CMB tSZ noise components, ellmax+1)

if __name__ == "__main__":

    # main input file containing most specifications 
    try:
        input_file = (sys.argv)[1]
    except IndexError:
        input_file = 'example.yaml'

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # current environment, also environment in which to run subprocesses
    my_env = os.environ.copy()


    results = list(futures.map(one_sim, [(sim, inp, my_env) for sim in range(inp.Nsims)]))
    results = np.asarray(results, dtype=np.float32) #dim (Nsims, 3 for ClTT ClTy Clyy, 3 for CMB tSZ noise components, ellmax+1)

    try:
        pickle.dump(results, open('data_spectra.p', 'wb'), protocol=4)
        if inp.verbose:
            print('created file data_spectra.p', flush=True)
    except Exception:
        print('could not create file data_spectra.p', flush=True)

    lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = get_parameter_cov_matrix(inp.Nsims, inp.ellmax, inp.verbose)
    print(f'Acmb={mean_acmb}+{upper_acmb-mean_acmb}-{mean_acmb-lower_acmb}', flush=True)
    print(f'Atsz={mean_atsz}+{upper_atsz-mean_atsz}-{mean_atsz-lower_atsz}', flush=True)
