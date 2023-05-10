import sys
import numpy as np
sys.path.append('../shared')
import os
import multiprocessing as mp
from input import Info
import pickle
import subprocess
import time
import healpy as hp
from generate_maps import generate_freq_maps
from utils import setup_output_dir, tsz_spectral_response
from wigner3j import *
from acmb_atsz import get_all_acmb_atsz, get_parameter_cov_matrix

def get_data_vectors(sim, inp):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    Clij: (Nfreqs=2, Nfreqs=2, Ncomps=4, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    '''
    Ncomps = 4 #CMB, tSZ, noise 90 nGHz, noise 150 GHz
    Nfreqs = len(inp.freqs)

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(sim, inp, save=False)
    all_spectra = [CC, T, N1, N2]

    #get spectral responses
    g_cmb = np.ones(len(inp.freqs))
    g_tsz = tsz_spectral_response(inp.freqs)
    g_noise1 = np.array([1.,0.])
    g_noise2 = np.array([0.,1.])
    all_g_vecs = np.array([g_cmb, g_tsz, g_noise1, g_noise2])

    #define and fill in array of data vectors
    Clij = np.zeros((Nfreqs, Nfreqs, Ncomps, inp.ellmax+1))
    for i in range(Nfreqs):
      for j in range(Nfreqs):
         for y in range(Ncomps):
            Clij[i,j,y] = all_g_vecs[y,i]*all_g_vecs[y,j]*all_spectra[y]
    
    return Clij



def main(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

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
    Clij = pool.starmap(get_data_vectors, [(sim, inp) for sim in range(inp.Nsims)])
    pool.close()
    Clij = np.asarray(Clij, dtype=np.float32) #shape (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=3, ellmax+1)
    if inp.save_files:
        pickle.dump(Clij, open(f'{inp.output_dir}/data_vecs/Clij.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/Clij.p')
    
    acmb_array, atsz_array, anoise1_array, anoise2_array = get_all_acmb_atsz(inp, Clij)
    lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = get_parameter_cov_matrix(acmb_array, atsz_array, anoise1_array, anoise2_array, nbins=100, smoothing_factor=0.065) 

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
    
    #set up output directory
    setup_output_dir(inp, my_env)

    lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = main(inp)
    print(f'Acmb = {mean_acmb} + {upper_acmb-mean_acmb} - {mean_acmb-lower_acmb}', flush=True)
    print(f'Atsz = {mean_atsz} + {upper_atsz-mean_atsz} - {mean_atsz-lower_atsz}', flush=True)
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)

