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
    Clij: (Nfreqs=2, Nfreqs=2, Ncomps=3, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    '''
    Ncomps = 3 #CMB, tSZ, noise
    Nfreqs = len(inp.freqs)

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N, CMB_map, tSZ_map, noise_map = generate_freq_maps(sim, inp, save=False)
    all_spectra = [CC, T, N]

    #get spectral responses
    g_cmb = np.ones(len(inp.freqs))
    g_tsz = tsz_spectral_response(inp.freqs)
    g_noise = np.array([1.,1.5]) #based on how we defined noise spectra
    all_g_vecs = np.array([g_cmb, g_tsz, g_noise])

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
    
    acmb_array, atsz_array = get_all_acmb_atsz(inp, Clij)
    lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = get_parameter_cov_matrix(acmb_array, atsz_array, nbins=100, smoothing_factor=0.065) 

    return lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz

