import sys
sys.path.append('../shared')
import numpy as np
import os
import multiprocessing as mp
from input import Info
import pickle
import subprocess
import time
import healpy as hp
from generate_maps import generate_freq_maps
from pyilc_interface import setup_pyilc
from load_weight_maps import load_wt_maps
from utils import setup_output_dir, tsz_spectral_response, GaussianNeedlets, build_NILC_maps
from acmb_atsz_nilc import *
from wigner3j import *


def get_data_vectors(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    Clpq: (N_preserved_comps=2, N_preserved_comps=2, N_comps=3, N_comps=3, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra. 
        preserved_comps = CMB, tSZ
        comps = CMB, tSZ, noise
        For example, Clpq[0,1,1,2] is cross-spectrum of tSZ propagation to 
        CMB-preserved NILC map and noise propagation to tSZ-preserved NILC map.
    '''
    N_preserved_comps = 2 #components to create NILC maps for: CMB, tSZ
    N_comps = 3 #CMB, tSZ, noise

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N, CMB_map, tSZ_map, noise_map = generate_freq_maps(sim, inp)
    
    #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
    setup_pyilc(sim, inp, env, suppress_printing=True)

    #load weight maps
    CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim)

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = tsz_spectral_response(inp.freqs)
    g_cmb = np.ones(len(inp.freqs))
    g_noise = [1.,1.5] #based on how we defined noise spectra

    #define and fill in array of data vectors
    Clpq = np.zeros((N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1))

    for y in range(N_comps):

        if y==0: #CMB
            compy = CMB_map
            g_vec = g_cmb
        elif y==1: #tSZ
            compy = tSZ_map
            g_vec = g_tsz
        else: #noise
            compy = noise_map
            g_vec = g_noise
        compy_freq1, compy_freq2 = g_vec[0]*compy, g_vec[1]*compy
        
        for z in range(N_comps):

            if z==0: #CMB
                compz = CMB_map
                g_vec = g_cmb
            elif z==1: #tSZ
                compz = tSZ_map
                g_vec = g_tsz
            else: #noise
                compz = noise_map
                g_vec = g_noise
            compz_freq1, compz_freq2 = g_vec[0]*compz, g_vec[1]*compz


            y_to_CMB_preserved, y_to_tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[compy_freq1, compy_freq2])
            z_to_CMB_preserved, z_to_tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[compz_freq1, compz_freq2])
            Clpq[0,0,y,z] = hp.anafast(y_to_CMB_preserved, z_to_CMB_preserved, lmax=inp.ellmax)
            Clpq[1,1,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
            Clpq[0,1,y,z] = hp.anafast(y_to_CMB_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
            Clpq[1,0,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_CMB_preserved, lmax=inp.ellmax)
    
    if inp.remove_files:

        #remove pyilc outputs
        subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/sim{sim}*', shell=True, env=env)
        subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/sim{sim}*', shell=True, env=env)

        #remove frequency map files
        subprocess.call(f'rm {inp.output_dir}/maps/sim{sim}_freq1.fits and {inp.output_dir}/maps/sim{sim}_freq2.fits', shell=True, env=env)
    
    return Clpq


def main(inp, env):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    lower_acmb: float, 1sigma below mean for acmb
    upper_acmb: float, 1sigma above mean for acmb
    mean_acmb: float, mean value of acmb
    lower_atsz: float, 1sigma below mean for atsz
    upper_atsz: float, 1sigma above mean for atsz
    mean_atsz: float, mean value of atsz
    '''

    pool = mp.Pool(inp.num_parallel)
    Clpq = pool.starmap(get_data_vectors, [(sim, inp, env) for sim in range(inp.Nsims)])
    pool.close()
    Clpq = np.asarray(Clpq, dtype=np.float32) #shape (Nsims, N_preserved_comps=2, N_preserved_comps=2, N_comps=3, N_comps=3, ellmax+1)
    if inp.save_files:
        pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/Clpq.p')
    
    acmb_array, atsz_array = get_all_acmb_atsz(inp, Clpq)
    lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = get_parameter_cov_matrix(acmb_array, atsz_array, nbins=100, smoothing_factor=0.065) 

    return lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz



if __name__ == '__main__':

    start_time = time.time()

    # main input file containing most specifications 
    try:
        input_file = (sys.argv)[1]
    except IndexError:
        input_file = 'laptop.yaml'

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    inp.ell_sum_max = inp.ellmax

    # current environment, also environment in which to run subprocesses
    my_env = os.environ.copy()

    #set up output directory
    setup_output_dir(inp, my_env)

    #get wigner 3j symbols
    if inp.wigner_file != '':
        inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
    else:
        inp.wigner3j = compute_3j(inp.ellmax)
    
    #set up output directory
    setup_output_dir(inp, my_env)

    lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = main(inp, my_env)
    print(f'Acmb = {mean_acmb} + {upper_acmb-mean_acmb} - {mean_acmb-lower_acmb}', flush=True)
    print(f'Atsz = {mean_atsz} + {upper_atsz-mean_atsz} - {mean_atsz-lower_atsz}', flush=True)
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)

