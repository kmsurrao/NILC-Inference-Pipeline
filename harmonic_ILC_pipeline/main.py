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
import healpy as hp
from harmonic_ILC import get_data_vecs
from generate_maps import generate_freq_maps
from utils import setup_output_dir, tsz_spectral_response
from acmb_atsz_hilc import get_all_acmb_atsz

def get_freq_power_spec(sim, inp):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    Clij: (Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim2: index0 is total power in Clij, other indices are power from each component
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
    Clij = np.zeros((Nfreqs, Nfreqs, 1+Ncomps, inp.ellmax+1))
    for i in range(Nfreqs):
      for j in range(Nfreqs):
        map_i = CMB_map + g_tsz[i]*tSZ_map + g_noise1[i]*noise1_map + g_noise2[i]*noise2_map
        map_j = CMB_map + g_tsz[j]*tSZ_map + g_noise1[j]*noise1_map + g_noise2[j]*noise2_map
        spectrum = hp.anafast(map_i, map_j, lmax=inp.ellmax)
        Clij[i,j,0] = spectrum
        for y in range(Ncomps):
            Clij[i,j,1+y] = all_g_vecs[y,i]*all_g_vecs[y,j]*all_spectra[y]
    
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
    parser = argparse.ArgumentParser(description="Covariance from harmonic ILC power spectrum template-fitting approach.")
    parser.add_argument("--config", default="example.yaml")
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
    Clij = pool.starmap(get_freq_power_spec, [(sim, inp) for sim in range(inp.Nsims)])
    pool.close()
    Clij = np.asarray(Clij, dtype=np.float32) #shape (Nsims, Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1)
    if inp.save_files:
        pickle.dump(Clij, open(f'{inp.output_dir}/data_vecs/Clij_HILC.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/Clij_HILC.p')
    
    inp.Clij_theory = np.mean(Clij, axis=0)
    pool = mp.Pool(inp.num_parallel)
    Clpq = pool.starmap(get_data_vecs, [(inp, Clij[sim]) for sim in range(inp.Nsims)])
    pool.close()
    Clpq = np.asarray(Clpq, dtype=np.float32) #shape (N_preserved_comps=2, N_preserved_comps=2, 1+Ncomps, Nbins)
    if inp.save_files:
        pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq_HILC.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/Clpq_HILC.p')
    
    if inp.compute_weights_once:
        acmb_array, atsz_array, anoise1_array, anoise2_array = get_all_acmb_atsz(inp, Clpq)
    else:
        acmb_array, atsz_array, anoise1_array, anoise2_array = get_all_acmb_atsz(inp, Clpq, Clij=Clij)
    
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return acmb_array, atsz_array, anoise1_array, anoise2_array


if __name__ == '__main__':
    main()



