import sys
import numpy as np
sys.path.append('../../shared')
sys.path.append('..')
import os
import multiprocessing as mp
from input import Info
import pickle
import time
import argparse
import tqdm
import healpy as hp
from scipy import stats
import scipy
from utils import setup_output_dir, get_naming_str, tsz_spectral_response, cib_spectral_response
from generate_maps import generate_freq_maps

def get_freq_power_spec(inp, sim=None, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number (if sim is None, a random simulation number will be used)
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)

    RETURNS
    -------
    Clij: (Nsplits=2, Nsplits=2, Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim4: index0 is total power in Clij, other indices are power from each component
    '''

    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)

    Ncomps = 3 #CMB, tSZ
    Nfreqs = len(inp.freqs)
    Nsplits = 2

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    cib_path = '/scratch/09334/ksurrao/ACT_sims/agora/agora_act_150ghz_lcibNG_uk_nside128.fits'
    CC, T, CIB, CMB_map, tSZ_map, CIB_map, noise_maps = generate_freq_maps(inp, sim, save=False, pars=pars, cib_path=cib_path)
    all_spectra = [CC, T, CIB]

    #get spectral responses
    g_cmb = np.ones(len(inp.freqs))
    g_tsz = tsz_spectral_response(inp.freqs)
    g_cib = cib_spectral_response(inp.freqs)
    all_g_vecs = np.array([g_cmb, g_tsz, g_cib])

    #define and fill in array of data vectors
    Clij = np.zeros((Nsplits, Nsplits, Nfreqs, Nfreqs, 1+Ncomps, inp.ellmax+1))
    for i in range(Nfreqs):
      for j in range(Nfreqs):
        for s1 in range(Nsplits):
            for s2 in range(Nsplits):
                map_i = CMB_map + g_tsz[i]*tSZ_map + g_cib[i]*CIB_map + noise_maps[i,s1]
                map_j = CMB_map + g_tsz[j]*tSZ_map + g_cib[j]*CIB_map + noise_maps[j,s2]
                spectrum = hp.anafast(map_i, map_j, lmax=inp.ellmax)
                Clij[s1,s2,i,j,0] = spectrum
                for y in range(Ncomps):
                    Clij[s1,s2,i,j,1+y] = all_g_vecs[y,i]*all_g_vecs[y,j]*all_spectra[y]
    
    return Clij


def get_freq_power_spec_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function get_freq_power_spec

    RETURNS
    -------
    function of *args, get_freq_power_spec(inp, sim=None, pars=None)
    '''
    return get_freq_power_spec(*args)


def get_Rlij_inv(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nsplits=2, Nsplits=2, Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim4: index0 is total power in Clij, other indices are power from each component
    
    RETURNS
    -------
    Rlij_inv: (Nsplits=2, Nsplits=2, ellmax+1, Nfreqs=2, Nfreqs=2) 
        ndarray containing inverse Rij matrix at each ell
    '''
    ells = np.arange(inp.ellmax+1)
    prefactor = (2*ells+1)/(4*np.pi)
    Nsplits = 2
    Nfreqs = len(inp.freqs)
    Rlij_inv = np.zeros((Nsplits, Nsplits, inp.ellmax+1, Nfreqs, Nfreqs), dtype=np.float32)
    for s0 in range(Nsplits):
        for s1 in range(Nsplits):
            Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, Clij[s0,s1,:,:,0,:])
            if not inp.delta_l:
                Rlij = Rlij_no_binning
            else:
                Rlij = np.zeros((len(inp.freqs), len(inp.freqs), inp.ellmax+1)) 
                for i in range(len(inp.freqs)):
                    for j in range(len(inp.freqs)):
                        Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], np.ones(2*inp.delta_l+1)))[inp.delta_l:inp.ellmax+1+inp.delta_l]
            Rlij_inv[s0,s1] = np.array([np.linalg.inv(Rlij[:,:,l]) for l in range(inp.ellmax+1)]) 
    return Rlij_inv #index as Rlij_inv[s0,s1,l,i,j]
    

def weights(Rlij_inv, spectral_response, spectral_response2=None):
    '''
    ARGUMENTS
    ---------
    Rlij_inv: (Nsplits=2, Nsplits=2, ellmax+1, Nfreqs=2, Nfreqs=2) 
        ndarray containing inverse Rij matrix at each ell
    spectral_response: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    spectral_response2: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency for second component if producing
        ILC cross-spectrum of two different components
    
    RETURNS
    -------
    w1: (Nfreqs, ellmax+1) ndarray of harmonic ILC weights for split 1 for component with spectral_response SED
    w2: (Nfreqs, ellmax+1) ndarray of harmonic ILC weights for split 2 for component with spectral_response2 SED
        if provided, otherwise for component with spectral_response SED
    '''
    numerator1 = np.einsum('lij,j->il', Rlij_inv[0,0], spectral_response)
    denominator1 = np.einsum('lkm,k,m->l', Rlij_inv[0,0], spectral_response, spectral_response)
    w1 = numerator1/denominator1 #index as w1[i][l]
    if spectral_response2 is None:
        spectral_response2 = spectral_response
    numerator2 = np.einsum('lij,j->il', Rlij_inv[1,1], spectral_response2)
    denominator2 = np.einsum('lkm,k,m->l', Rlij_inv[1,1], spectral_response2, spectral_response2)
    w2 = numerator2/denominator2 #index as w2[i][l]
    return w1, w2


def HILC_spectrum(inp, Clij, spectral_response, spectral_response2=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nsplits=2, Nsplits=2, Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim4: index0 is total power in Clij, other indices are power from each component
    spectral_response: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency
    spectral_response2: array-like of length Nfreqs containing spectral response
        of component of interest at each frequency for second component if producing
        ILC cross-spectrum of two different components

    RETURNS
    -------
    Clpq: (1+Ncomps, ellmax+1) ndarray containing contributions of each component
        to the power spectrum of harmonic ILC map p and harmonic ILC map q
        dim0: index0 is total power spectrum of HILC map p and HILC map q

    '''
    if inp.compute_weights_once:
        Rlij_inv = get_Rlij_inv(inp, inp.Clij_theory)
    else:
        Rlij_inv = get_Rlij_inv(inp, Clij)
    w1, w2 = weights(Rlij_inv, spectral_response, spectral_response2=spectral_response2)
    Clpq = np.einsum('il,jl,ijal->al', w1, w2, Clij[0,1])   
    return Clpq

    

def get_data_vecs(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij: (Nsplits=2, Nsplits=2, Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        dim4: index0 is total power in Clij, other indices are power from each component

    RETURNS
    -------
    Clpq: (N_preserved_comps=3, N_preserved_comps=3, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
    '''

    N_preserved_comps = 3
    Ncomps = 3
    
    #get spectral responses
    g_cmb = np.ones(len(inp.freqs))
    g_tsz = tsz_spectral_response(inp.freqs)
    g_cib = cib_spectral_response(inp.freqs)
    all_g_vecs = np.array([g_cmb, g_tsz, g_cib])

    #HILC auto- and cross-spectra
    Clpq_orig = np.zeros((N_preserved_comps, N_preserved_comps, 1+Ncomps, inp.ellmax+1))
    for p in range(N_preserved_comps):
        for q in range(N_preserved_comps):
            Clpq_orig[p,q] = HILC_spectrum(inp, Clij, all_g_vecs[p], spectral_response2=all_g_vecs[q])
    
    #binning
    Clpq = np.zeros((N_preserved_comps, N_preserved_comps, 1+Ncomps, inp.Nbins))
    ells = np.arange(inp.ellmax+1)
    for p in range(N_preserved_comps):
        for q in range(N_preserved_comps):
            for y in range(1+Ncomps):
                Dl = ells*(ells+1)/2/np.pi*Clpq_orig[p,q,y]
                res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
                mean_ells = (res[1][:-1]+res[1][1:])/2
                Clpq[p,q,y] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
    
    return Clpq[:,:,0]


def get_data_vecs_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function get_data_vecs

    RETURNS
    -------
    function of *args, get_data_vecs(inp, Clij)
    '''
    return get_data_vecs(*args)


def get_PScov_sim(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, N_preserved_comps=3, N_preserved_comps=3, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
    
    RETURNS
    -------
    cov: (9*Nbins, 9*Nbins) ndarray containing covariance matrix Cov_{pq,rs}
        index as cov[(0-8 for component spectra)*Nbins + bin1, (0-8 for component spectra)*Nbins + bin2]
    '''
    Clpq_tmp = np.array([Clpq[:,0,0], Clpq[:,1,1], Clpq[:,2,2], \
                         Clpq[:,0,1], Clpq[:,1,0], Clpq[:,0,2], \
                         Clpq[:,2,0], Clpq[:,1,2], Clpq[:,2,1]])
    Clpq_tmp = np.transpose(Clpq_tmp, axes=(0,2,1)) #shape (9 for different component spectra, Nbins, Nsims)
    Clpq_tmp = np.reshape(Clpq_tmp, (inp.Nbins*9, -1))
    cov = np.cov(Clpq_tmp)
    return cov


def main():

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Covariance from harmonic ILC power spectrum template-fitting approach.")
    parser.add_argument("--config", default="../example_yaml_files/weights_vary_SR.yaml")
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
    print(f'Running {inp.Nsims} simulations for frequency-frequency power spectra...', flush=True)
    inputs = [(inp, sim) for sim in range(inp.Nsims)]
    Clij = list(tqdm.tqdm(pool.imap(get_freq_power_spec_star, inputs), total=inp.Nsims))
    pool.close()
    Clij = np.asarray(Clij, dtype=np.float32)
    if inp.save_files:
        naming_str = get_naming_str(inp, 'HILC')
        pickle.dump(Clij, open(f'{inp.output_dir}/data_vecs/Clij_{naming_str}.p', 'wb'), protocol=4)
        print(f'\nsaved {inp.output_dir}/data_vecs/Clij_{naming_str}.p', flush=True)
    
    pool = mp.Pool(inp.num_parallel)
    print(f'\nRunning {inp.Nsims} simulations for HILC spectra...', flush=True)
    inp.Clij_theory = np.mean(Clij, axis=0)
    inputs = [(inp, Clij[sim]) for sim in range(inp.Nsims)]
    Clpq = list(tqdm.tqdm(pool.imap(get_data_vecs_star, inputs), total=inp.Nsims))
    pool.close()
    Clpq = np.asarray(Clpq, dtype=np.float32)
    if inp.save_files:
        pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq_{naming_str}.p', 'wb'), protocol=4)
        print(f'\nsaved {inp.output_dir}/data_vecs/Clpq_{naming_str}.p', flush=True)
    

    determinants = []
    eigenvals = []
    for Nsims in range(100, inp.Nsims, 50):
        cov = get_PScov_sim(inp, Clpq[:Nsims])
        determinants.append(np.linalg.det(cov))
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvals.append(eigenvalues)
    pickle.dump(determinants, open(f'{inp.output_dir}/determinants_{naming_str}.p', 'wb'))
    pickle.dump(eigenvals, open(f'{inp.output_dir}/eigenvals_{naming_str}.p', 'wb'))
    print(f'saved {inp.output_dir}/determinants_{naming_str}.p', flush=True)
    print(f'saved {inp.output_dir}/eigenvals_{naming_str}.p', flush=True)
    print('determinants: ', determinants, flush=True)

    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return 


if __name__ == '__main__':
    main()



