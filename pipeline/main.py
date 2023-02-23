# from mpi4py import MPI
# from mpi4py.futures import MPICommExecutor
import sys
import os
import subprocess
import multiprocessing as mp
from input import Info
import pickle
from utils import setup_output_dir
from generate_maps import generate_freq_maps
from pyilc_interface import setup_pyilc
from load_weight_maps import load_wt_maps
from calculate_n_point_functions import *
from data_spectra import *
from acmb_atsz_nilc import *
from wigner3j import *
from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
hp.disable_warnings()


def one_sim(sim, inp, env):

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N, CMB_map, tSZ_map, noise_map = generate_freq_maps(sim, inp)
    
    #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
    setup_pyilc(sim, inp, env)

    #load weight maps and then remove pyilc weight map files
    CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim)
    if inp.remove_files: #don't need pyilc output files anymore
        subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/sim{sim}*', shell=True, env=env)
        subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/sim{sim}*', shell=True, env=env)

    #get power spectra of components
    Clzz = get_Clzz(CC, T, N)
    if inp.verbose:
        print(f'calculated component map spectra for sim {sim}', flush=True)

    #get power spectra of weight maps
    Clw1w2 = get_Clw1w2(inp, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated weight map spectra for sim {sim}', flush=True)

    #get cross-spectra of component maps and weight maps
    Clzw = get_Clzw(inp, CMB_wt_maps, tSZ_wt_maps, CMB_map, tSZ_map, noise_map)
    if inp.verbose:
        print(f'calculated component and weight map cross-spectra for sim {sim}', flush=True)
    
    #get means of weight maps
    w = get_w(inp, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated weight map means for sim {sim}', flush=True)

    #get means of component maps
    a = get_a(CMB_map, tSZ_map, noise_map)
    if inp.verbose:
        print(f'calculated component map means for sim {sim}', flush=True)
    
    #get bispectrum for two factors of map and one factor of weight map
    bispectrum_zzw = get_bispectrum_zzw(inp, CMB_map, tSZ_map, noise_map, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated bispectra for map, map, weight map for sim {sim}', flush=True)
    
    #get bispectrum for two weight maps and one factor of map
    bispectrum_wzw = get_bispectrum_wzw(inp, CMB_map, tSZ_map, noise_map, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated bispectra for weight map, map, weight map for sim {sim}', flush=True)
    
    #get unnormalized trispectrum estimator rho
    Rho = get_rho(inp, CMB_map, tSZ_map, noise_map, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated unnormalized trispectrum estimator rho for sim {sim}', flush=True)


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

    #set up output directory
    setup_output_dir(inp, my_env)

    #get wigner 3j symbols
    if inp.wigner_file != '':
        inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ell_sum_max+1, :inp.ell_sum_max+1, :inp.ell_sum_max+1]
    else:
        inp.wigner3j = compute_3j(inp.ell_sum_max)


    # mpi_rank = MPI.COMM_WORLD.Get_rank()
    # mpi_size = MPI.COMM_WORLD.Get_size()
    # print(mpi_rank, mpi_size, flush=True)
    
    # with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
    #     if executor is not None:
    #         print('executor is not None', flush=True)
    #         iterable = [(sim, inp, my_env, wigner3j, wigner_nonzero_m) for sim in range(inp.Nsims)]
    #         for result in executor.starmap(one_sim, iterable): #dim (Nsims, 3 for ClTT ClTy Clyy, 3 for CMB tSZ noise components, ellmax+1))
    #             print('got result', flush=True)
    #         lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = get_parameter_cov_matrix(inp.Nsims, inp.ellmax, inp.scratch_path, inp.verbose)
    #         print(f'Acmb={mean_acmb}+{upper_acmb-mean_acmb}-{mean_acmb-lower_acmb}', flush=True)
    #         print(f'Atsz={mean_atsz}+{upper_atsz-mean_atsz}-{mean_atsz-lower_atsz}', flush=True)



    # pool = mp.Pool(inp.num_parallel)
    # results = pool.starmap(one_sim, [(sim, inp, my_env) for sim in range(inp.Nsims)])
    # pool.close()
    # results = np.asarray(results, dtype=np.float32) #dim (Nsims, 3 for ClTT ClTy Clyy, 3 for CMB tSZ noise components, ellmax+1)

