import sys
import numpy as np
sys.path.append('../shared')
import os
import multiprocessing as mp
from input import Info
import pickle
import subprocess
import time
import argparse
from scipy import stats
import healpy as hp
from generate_maps import generate_freq_maps
from utils import setup_output_dir, tsz_spectral_response
from wigner3j import *
from acmb_atsz import get_all_acmb_atsz

def get_data_vectors(sim, inp):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    Clij: (Nfreqs=2, Nfreqs=2, Ncomps=4, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    '''
    Ncomps = 4 #CMB, tSZ, noise 90 nGHz, noise 150 GHz
    Nfreqs = len(inp.freqs)

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(sim, inp, save=False)
    all_spectra_orig = [CC, T, N1, N2]
    all_spectra = []
    ells = np.arange(inp.ellmax+1)
    for Cl in all_spectra_orig:
        Dl = ells*(ells+1)/2/np.pi*Cl
        res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
        mean_ells = (res[1][:-1]+res[1][1:])/2
        all_spectra.append(res[0]/(mean_ells*(mean_ells+1)/2/np.pi))



    #get spectral responses
    g_cmb = np.ones(len(inp.freqs))
    g_tsz = tsz_spectral_response(inp.freqs)
    g_noise1 = np.array([1.,0.])
    g_noise2 = np.array([0.,1.])
    all_g_vecs = np.array([g_cmb, g_tsz, g_noise1, g_noise2])

    #define and fill in array of data vectors
    Clij = np.zeros((Nfreqs, Nfreqs, Ncomps, inp.Nbins))
    for i in range(Nfreqs):
      for j in range(Nfreqs):
         for y in range(Ncomps):
            Clij[i,j,y] = all_g_vecs[y,i]*all_g_vecs[y,j]*all_spectra[y]
    
    return Clij



def main():
    '''
    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation
    '''

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Covariance from template-fitting approach.")
    parser.add_argument("--config", default="stampede.yaml")
    args = parser.parse_args()
    input_file = args.config

    start_time = time.time()

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    inp.ell_sum_max = inp.ellmax

    # current environment, also environment in which to run subprocesses
    my_env = os.environ.copy()

    #set up output directory
    setup_output_dir(inp, my_env)

    pool = mp.Pool(inp.num_parallel)
    Clij = pool.starmap(get_data_vectors, [(sim, inp) for sim in range(inp.Nsims)])
    pool.close()
    Clij = np.asarray(Clij, dtype=np.float32) #shape (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=3, Nbins)
    if inp.save_files:
        pickle.dump(Clij, open(f'{inp.output_dir}/data_vecs/Clij.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/Clij.p')

    # Clij = pickle.load(open(f'{inp.output_dir}/data_vecs/Clij.p', 'rb')) #remove this line and uncomment above
    
    acmb_array, atsz_array, anoise1_array, anoise2_array = get_all_acmb_atsz(inp, Clij)
    
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return acmb_array, atsz_array, anoise1_array, anoise2_array


if __name__ == '__main__':
    main()



