import sys
sys.path.append('../shared')
import numpy as np
import os
import multiprocessing as mp
from input import Info
import pickle
import subprocess
import time
import argparse
import healpy as hp
from generate_maps import generate_freq_maps
from pyilc_interface import setup_pyilc, weight_maps_exist
from load_weight_maps import load_wt_maps
from utils import setup_output_dir, tsz_spectral_response, GaussianNeedlets, build_NILC_maps
from acmb_atsz_nilc import *

def get_scaled_maps_and_wts(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    CMB_map, tSZ_map, noise1_map, noise2_map: unscaled maps of all the components
    all_wt_maps: 2*N_comps+1 for each scaled low component, followed by each scaled high component, then all unscaled;
                 N_preserved_comps, Nscales, Nfreqs, Npix) ndarray containing all weight maps
    '''

    N_comps = 4 #CMB, tSZ, noise1, noise2
    N_preserved_comps = 2
    comps = ['CMB', 'tSZ', 'noise1', 'noise2']

    #array for all weight maps, shape (2*N_comps+1, N_preserved_comps, Nscales, Nfreqs, Npix)
    all_wt_maps = np.zeros((2*N_comps+1, N_preserved_comps, inp.Nscales, len(inp.freqs), 12*inp.nside**2))

    for y in range(2*N_comps+1):

        if y==2*N_comps: 
            scaling=None
        else: 
            if y < N_comps: scaling = [inp.scaling_factors[0], comps[y%N_comps]]
            else: scaling = [inp.scaling_factors[1], comps[y%N_comps]]

        #create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N1, N2)
        CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(sim, inp, scaling=scaling)
        
        #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
        if not weight_maps_exist(sim, inp, scaling=scaling): #check if not all the weight maps already exist
            #remove any existing weight maps for this sim and scaling to prevent pyilc errors
            if scaling:
                scaling_type = 'low' if scaling[0] < 1.0 else 'high'                                                     
                subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/scaled_{scaling_type}_{scaling[1]}/sim{sim}*', shell=True, env=env)
            else:
                subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/unscaled/sim{sim}*', shell=True, env=env)
            setup_pyilc(sim, inp, env, suppress_printing=True, scaling=scaling) #set suppress_printing=False to debug pyilc runs

        #load weight maps
        CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim, scaling=scaling)
        all_wt_maps[y] = np.array([CMB_wt_maps, tSZ_wt_maps])
    
    return CMB_map, tSZ_map, noise1_map, noise2_map, all_wt_maps



def get_data_vectors(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    Clpq: (2*N_comps+1, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        Size of dimension 0 is N_comps+1 for each scaled component, then all unscaled.
        preserved_comps = CMB, ftSZ
        comps = CMB, ftSZ, noise 90 GHz, noise 150 GHz
        For example, Clpq[3,0,1,1,2] is cross-spectrum of ftSZ propagation to 
        CMB-preserved NILC map and 90 GHz noise propagation to ftSZ-preserved NILC map 
        when 150 GHz noise is scaled low
    '''
    
    N_preserved_comps = 2 #components to create NILC maps for: CMB, ftSZ
    N_comps = 4 #CMB, ftSZ, noise1, noise2

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = tsz_spectral_response(inp.freqs)
    g_cmb = np.ones(len(inp.freqs))
    g_noise1 = [1.,0.]
    g_noise2 = [0.,1.]

    #get maps and weight maps, all_wt_maps is (2*N_comps+1, N_preserved_comps, Nscales, Nfreqs, Npix) ndarray
    CMB_map, tSZ_map, noise1_map, noise2_map, all_wt_maps = get_scaled_maps_and_wts(sim, inp, env)


    #get map level propagation of components
    Npix = 12*inp.nside**2
    all_map_level_prop = np.zeros((N_preserved_comps, N_comps, 2*N_comps+1, Npix)) 
    
    for y in range(N_comps):
        for s in range(2*N_comps+1): #2*N_comps+1 for each scaled low comp, each scaled high comp, then all unscaled

            if y==0: compy, g_vecy = np.copy(CMB_map), g_cmb #CMB
            elif y==1: compy, g_vecy = np.copy(tSZ_map), g_tsz #ftSZ
            elif y==2: compy, g_vecy = np.copy(noise1_map), g_noise1 #noise 90 GHz
            elif y==3: compy, g_vecy = np.copy(noise2_map), g_noise2 #noise 150 GHz
            
            if s==y: #if component y is the one that's scaled and is scaled down
                compy *= inp.scaling_factors[0]
            elif s==y+N_comps: #if component y is the one that's scaled and is scaled up
                compy *= inp.scaling_factors[1]
            CMB_wt_maps, tSZ_wt_maps = all_wt_maps[s]
            compy_freq1, compy_freq2 = g_vecy[0]*compy, g_vecy[1]*compy

            y_to_CMB_preserved, y_to_tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[compy_freq1, compy_freq2])
            all_map_level_prop[0,y,s] = y_to_CMB_preserved
            all_map_level_prop[1,y,s] = y_to_tSZ_preserved

    #define and fill in array of data vectors (dim 0 has size 2*N_comps+1 for each scaled low component, each scaled high comp, and then all unscaled)
    Clpq = np.zeros((2*N_comps+1, N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1))

    for y in range(N_comps):
        for z in range(N_comps):
            for s in range(2*N_comps+1): #each scaled low component, each scaled high comp, and then all unscaled
                
                y_to_CMB_preserved = all_map_level_prop[0,y,s]
                y_to_tSZ_preserved = all_map_level_prop[1,y,s]
                z_to_CMB_preserved = all_map_level_prop[0,z,s]
                z_to_tSZ_preserved = all_map_level_prop[1,z,s]
            
                Clpq[s,0,0,y,z] = hp.anafast(y_to_CMB_preserved, z_to_CMB_preserved, lmax=inp.ellmax)
                Clpq[s,1,1,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
                Clpq[s,0,1,y,z] = hp.anafast(y_to_CMB_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
                Clpq[s,1,0,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_CMB_preserved, lmax=inp.ellmax)


    if inp.remove_files:
        #remove pyilc outputs
        subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/*/sim{sim}*', shell=True, env=env)
        #remove frequency map files
        subprocess.call(f'rm {inp.output_dir}/maps/*/sim{sim}_freq*.fits', shell=True, env=env)

    return Clpq



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
    parser = argparse.ArgumentParser(description="Covariance from NILC approach.")
    parser.add_argument("--config", default="stampede.yaml")
    args = parser.parse_args()
    input_file = args.config

    start_time = time.time()

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    inp.ell_sum_max = inp.ellmax

    # current environment, also environment in which to run subprocesses
    env = os.environ.copy()

    #set up output directory
    setup_output_dir(inp, env, scaling=True)

    pool = mp.Pool(inp.num_parallel)
    Clpq = pool.starmap(get_data_vectors, [(sim, inp, env) for sim in range(inp.Nsims)])
    pool.close()
    Clpq = np.asarray(Clpq, dtype=np.float32) #shape (Nsims, 2*N_comps+1 for scalings, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1)
    if inp.save_files:
        pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/Clpq.p')

    # Clpq = pickle.load(open(f'{inp.output_dir}/data_vecs/Clpq.p', 'rb'))
    
    acmb_array, atsz_array, anoise1_array, anoise2_array = get_all_acmb_atsz(inp, Clpq)
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return acmb_array, atsz_array, anoise1_array, anoise2_array


if __name__ == '__main__':
    main()


