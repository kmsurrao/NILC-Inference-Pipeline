import sys
sys.path.append('../../shared')
sys.path.append('..')
import numpy as np
import os
import multiprocessing as mp
from input import Info
import pickle
import time
import argparse
import tqdm
import shutil
import tempfile
import healpy as hp
from scipy import stats
import scipy
from generate_maps import generate_freq_maps
from pyilc_interface import setup_pyilc, load_wt_maps
from utils import setup_output_dir, get_naming_str, spectral_response, GaussianNeedlets, build_NILC_maps


def get_maps_and_wts(sim, inp, env, pars=None):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)

    RETURNS
    -------
    CMB_map, tSZ_map, CIB_map, noise_maps: maps of all the components
    all_wt_maps: (Nsplits, N_preserved_comps, Nscales, Nfreqs, Npix) ndarray containing all weight maps

    '''
    #create temporary directory to place maps
    map_tmpdir = tempfile.mkdtemp(dir=inp.output_dir)

    #array for all weight maps
    Nsplits = 2
    N_preserved_comps = 3
    all_wt_maps = np.zeros((Nsplits, N_preserved_comps, inp.Nscales, len(inp.freqs), 12*inp.nside**2))

    #create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T)
    cib_path = '/scratch/09334/ksurrao/ACT_sims/agora/agora_act_150ghz_lcibNG_uk_nside128.fits'
    CC, T, CIB, CMB_map, tSZ_map, CIB_map, noise_maps = generate_freq_maps(inp, sim, pars=pars, map_tmpdir=map_tmpdir, cib_path=cib_path)
       
    #generate and save files containing frequency maps and then run pyilc
    for split in [1,2]:
        pyilc_tmpdir = setup_pyilc(sim, split, inp, env, map_tmpdir, suppress_printing=True, pars=pars, cib=True) #set suppress_printing=False to debug pyilc runs
        CMB_wt_maps, tSZ_wt_maps, CIB_wt_maps = load_wt_maps(inp, sim, split, pyilc_tmpdir, pars=pars, cib=True) #load weight maps
        all_wt_maps[split-1] = np.array([CMB_wt_maps, tSZ_wt_maps, CIB_wt_maps])
        shutil.rmtree(pyilc_tmpdir)
    
    shutil.rmtree(map_tmpdir)

    return CMB_map, tSZ_map, CIB_map, noise_maps, all_wt_maps



def get_data_vectors(inp, env, sim=None, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object
    sim: int, simulation number (if sim is None, a random simulation number will be used)
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)

    RETURNS
    -------
    Clpq: (N_preserved_comps=3, N_preserved_comps=3, Nbins) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        preserved_comps = CMB, ftSZ, CIB
    '''

    if sim is None:
        sim = np.random.randint(0, high=inp.Nsims, size=None, dtype=int)
    
    N_preserved_comps = 3 #components to create NILC maps for: CMB, ftSZ, CIB

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = spectral_response(inp.freqs, 'tsz')
    g_cib = spectral_response(inp.freqs, 'cib')

    #get maps and weight maps
    CMB_map, tSZ_map, CIB_map, noise_maps, all_wt_maps = get_maps_and_wts(sim, inp, env, pars=pars)

    #get map level propagation of components
    Npix = 12*inp.nside**2
    Nsplits = 2
    all_map_level_prop = np.zeros((Nsplits, N_preserved_comps, Npix)) 

    for split in [1,2]:
        map_0 = CMB_map + g_tsz[0]*tSZ_map + g_cib[0]*CIB_map + noise_maps[0,split-1]
        map_1 = CMB_map + g_tsz[1]*tSZ_map + g_cib[1]*CIB_map + noise_maps[1,split-1]
        CMB_wt_maps, tSZ_wt_maps, CIB_wt_maps = all_wt_maps[split-1]
        all_map_level_prop[split-1] = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, CIB_wt_maps=CIB_wt_maps, freq_maps=[map_0, map_1])

    #define and fill in array of data vectors
    Clpq_tmp = np.zeros((N_preserved_comps, N_preserved_comps, inp.ellmax+1)) #unbinned
    Clpq = np.zeros((N_preserved_comps, N_preserved_comps, inp.Nbins)) #binned

    CMB_preserved_s1,  CMB_preserved_s2 = all_map_level_prop[:,0]
    tSZ_preserved_s1,  tSZ_preserved_s2 = all_map_level_prop[:,1]
    CIB_preserved_s1,  CIB_preserved_s2 = all_map_level_prop[:,2]

    Clpq_tmp[0,0] = hp.anafast(CMB_preserved_s1, CMB_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[1,1] = hp.anafast(tSZ_preserved_s1, tSZ_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[2,2] = hp.anafast(CIB_preserved_s1, CIB_preserved_s2, lmax=inp.ellmax)

    Clpq_tmp[0,1] = hp.anafast(CMB_preserved_s1, tSZ_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[1,0] = hp.anafast(tSZ_preserved_s1, CMB_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[0,2] = hp.anafast(CMB_preserved_s1, CIB_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[2,0] = hp.anafast(CIB_preserved_s1, CMB_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[1,2] = hp.anafast(tSZ_preserved_s1, CIB_preserved_s2, lmax=inp.ellmax)
    Clpq_tmp[2,1] = hp.anafast(CIB_preserved_s1, tSZ_preserved_s2, lmax=inp.ellmax)

    ells = np.arange(inp.ellmax+1)
    all_spectra = [Clpq_tmp[0,0], Clpq_tmp[1,1], Clpq_tmp[2,2], \
                   Clpq_tmp[0,1], Clpq_tmp[1,0], Clpq_tmp[0,2], \
                   Clpq_tmp[2,0], Clpq_tmp[1,2], Clpq_tmp[2,1]]
    
    index_mapping = {0:(0,0), 1:(1,1), 2:(2,2), 3:(0,1), 4:(1,0), \
                     5:(0,2), 6:(2,0), 7:(1,2), 8:(2,1)} #maps idx to p,q for all_spectra
    for idx, Cl in enumerate(all_spectra):
        Dl = ells*(ells+1)/2/np.pi*Cl
        res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
        mean_ells = (res[1][:-1]+res[1][1:])/2
        p,q = index_mapping[idx]
        Clpq[p,q] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)

    return Clpq


def get_data_vectors_star(args):
    '''
    Useful for using multiprocessing imap
    (imap supports tqdm but starmap does not)

    ARGUMENTS
    ---------
    args: arguments to function get_data_vectors

    RETURNS
    -------
    function of *args, get_data_vectors(inp, env, sim=None, pars=None)
    '''
    return get_data_vectors(*args)


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
    parser = argparse.ArgumentParser(description="Covariance from NILC approach.")
    parser.add_argument("--config", default="../example_yaml_files/gaussian_likelihood.yaml")
    args = parser.parse_args()
    input_file = args.config

    start_time = time.time()

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    inp.ell_sum_max = inp.ellmax

    # current environment, also environment in which to run subprocesses
    env = os.environ.copy()

    #set up output directory
    setup_output_dir(inp, env)

    pool = mp.Pool(inp.num_parallel)
    inputs = [(inp, env, sim) for sim in range(inp.Nsims)]
    print(f'Running {inp.Nsims} simulations...', flush=True)
    Clpq = list(tqdm.tqdm(pool.imap(get_data_vectors_star, inputs), total=inp.Nsims))
    pool.close()
    Clpq = np.asarray(Clpq, dtype=np.float32)
    if inp.save_files:
        naming_str = get_naming_str(inp, 'NILC')
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


