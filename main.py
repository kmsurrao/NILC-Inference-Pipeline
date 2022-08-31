# from mpi4py import MPI
# from mpi4py.futures import MPICommExecutor
import sys
import os
import subprocess
import multiprocessing as mp
from input import Info
import pickle
import py3nj
from generate_maps import *
from wt_map_spectra import *
from data_spectra import *
from acmb_atsz_nilc import *
from wigner3j import *
from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
hp.disable_warnings()


def one_sim(sim, inp, env, wigner_zero_m, wigner_nonzero_m):

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N = generate_freq_maps(sim, inp.freqs, inp.tsz_amp, inp.noise, inp.nside, inp.ellmax, inp.cmb_alm_file, inp.halosky_maps_path, inp.scratch_path, inp.verbose)
    
    #get NILC weight maps for preserved component CMB and preserved component tSZ
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/CMB_preserved.yml {sim}"], shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component CMB, sim {sim}', flush=True)
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.pyilc_path}/input/tSZ_preserved.yml {sim}"], shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component tSZ, sim {sim}', flush=True)
    if inp.remove_files: #don't need frequency maps anymore
        subprocess.call(f'rm {inp.scratch_path}/maps/sim{sim}_freq1.fits {inp.scratch_path}/maps/sim{sim}_freq2.fits', shell=True, env=env)
        subprocess.call(f'rm {inp.scratch_path}/maps/{sim}_cmb_map.fits', shell=True, env=env)

    #get power spectra of weight maps--dimensions (3,Nscales,Nscales,Nfreqs,Nfreqs,ellmax)
    wt_map_power_spectrum = get_wt_map_spectra(sim, inp.ellmax, inp.Nscales, inp.verbose, inp.scratch_path)
    if inp.verbose:
        print(f'calculated weight map spectra for sim {sim}', flush=True)
    if inp.remove_files: #don't need pyilc outputs anymore
        subprocess.call(f'rm {inp.scratch_path}/wt_maps/CMB/{sim}_*', shell=True, env=env)
        subprocess.call(f'rm {inp.scratch_path}/wt_maps/tSZ/{sim}_*', shell=True, env=env)

    #get contributions to ClTT, ClTy, and Clyy from Acmb, Atsz, and noise components
    #has dim (3 for ClTT ClTy Clyy, 3 for CMB tSZ noise components, ellmax+1)
    data_spectra = get_data_spectra(sim, inp.freqs, inp.Nscales, inp.tsz_amp, inp.ellmax, wigner_zero_m, wigner_nonzero_m, CC, T, N, wt_map_power_spectrum, inp.GN_FWHM_arcmin, inp.scratch_path, inp.verbose)
    if inp.verbose:
        print(f'calculated data spectra for sim {sim}', flush=True)
    #don't need weight map spectra anymore
    del wt_map_power_spectrum #free up memory

    return 0



if __name__ == '__main__':
    # main input file containing most specifications 
    try:
        input_file = (sys.argv)[1]
    except IndexError:
        input_file = 'example.yaml'

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # current environment, also environment in which to run subprocesses
    my_env = os.environ.copy()

    #load wigner3j symbols
    # wigner_zero_m = get_wigner3j_zero_m(inp, save=False)
    wigner_nonzero_m = get_wigner3j_nonzero_m(inp, save=True)


    # mpi_rank = MPI.COMM_WORLD.Get_rank()
    # mpi_size = MPI.COMM_WORLD.Get_size()
    # print(mpi_rank, mpi_size, flush=True)
    
    # with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
    #     if executor is not None:
    #         print('executor is not None', flush=True)
    #         iterable = [(sim, inp, my_env, wigner_zero_m, wigner_nonzero_m) for sim in range(inp.Nsims)]
    #         for result in executor.starmap(one_sim, iterable): #dim (Nsims, 3 for ClTT ClTy Clyy, 3 for CMB tSZ noise components, ellmax+1))
    #             print('got result', flush=True)
    #         lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = get_parameter_cov_matrix(inp.Nsims, inp.ellmax, inp.scratch_path, inp.verbose)
    #         print(f'Acmb={mean_acmb}+{upper_acmb-mean_acmb}-{mean_acmb-lower_acmb}', flush=True)
    #         print(f'Atsz={mean_atsz}+{upper_atsz-mean_atsz}-{mean_atsz-lower_atsz}', flush=True)



    # pool = mp.Pool(inp.num_parallel)
    # results = pool.starmap(one_sim, [(sim, inp, my_env) for sim in range(inp.Nsims)])
    # pool.close()
    # results = np.asarray(results, dtype=np.float32) #dim (Nsims, 3 for ClTT ClTy Clyy, 3 for CMB tSZ noise components, ellmax+1)

