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
from pyilc_interface import setup_pyilc, weight_maps_exist
from load_weight_maps import load_wt_maps
from utils import setup_output_dir, tsz_spectral_response, GaussianNeedlets, build_NILC_maps
from acmb_atsz_nilc import *

def get_scaled_maps_and_wts(sim, inp, env, scale_factor):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object
    scale_factor: float, multiplicative factor by which to test scaling components

    RETURNS
    -------
    CMB_map, tSZ_map, noise1_map, noise2_map: unscaled maps of all the components
    all_wt_maps: (N_comps+1 for each scaled component followed by all unscaled, N_preserved_comps, 
                Nscales, Nfreqs, Npix) ndarray containing all weight maps
    '''

    N_comps = 4 #CMB, tSZ, noise1, noise2
    N_preserved_comps = 2
    comps = ['CMB', 'tSZ', 'noise1', 'noise2']

    #array for all weight maps, shape (N_comps+1, N_preserved_comps, Nscales, Nfreqs, Npix)
    all_wt_maps = np.zeros((N_comps+1, N_preserved_comps, inp.Nscales, len(inp.freqs), 12*inp.nside**2))

    for y in range(N_comps+1):

        if y==N_comps: scaling=None
        else: scaling = [scale_factor, comps[y]]

        #create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N1, N2)
        CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(sim, inp, scaling=scaling)
        
        #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
        if not weight_maps_exist(sim, inp, scaling=scaling): #check if not all the weight maps already exist
            #remove any existing weight maps for this sim and scaling to prevent pyilc errors
            if scaling:                                                     
                subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/scaled_{scaling[1]}/sim{sim}*', shell=True, env=env)
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
    Clpq: (N_comps+1, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        Size of dimension 0 is N_comps+1 for each scaled component, then all unscaled.
        preserved_comps = CMB, ftSZ
        comps = CMB, ftSZ, noise 90 GHz, noise 150 GHz
        For example, Clpq[3,0,1,1,2] is cross-spectrum of ftSZ propagation to 
        CMB-preserved NILC map and 90 GHz noise propagation to ftSZ-preserved NILC map 
        when 150 GHz noise is scaled 
    '''
    
    N_preserved_comps = 2 #components to create NILC maps for: CMB, tSZ
    N_comps = 4 #CMB, tSZ, noise1, noise2
    comps = ['CMB', 'tSZ', 'noise1', 'noise2']
    scale_factor = 1.1

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = tsz_spectral_response(inp.freqs)
    g_cmb = np.ones(len(inp.freqs))
    g_noise1 = [1.,0.]
    g_noise2 = [0.,1.]

    #get maps and weight maps, all_wt_maps is (N_comps+1, N_preserved_comps, Nscales, Nfreqs, Npix) ndarray
    CMB_map, tSZ_map, noise1_map, noise2_map, all_wt_maps = get_scaled_maps_and_wts(sim, inp, env, scale_factor)


    #get map level propagation of components
    Npix = 12*inp.nside**2
    all_map_level_prop = np.zeros((N_preserved_comps, N_comps, N_comps+1, Npix)) 
    
    for y in range(N_comps):
        for s in range(N_comps+1): #N_comps+1 for each scaled comp, then all unscaled

            if y==0: compy, g_vecy = CMB_map, g_cmb #CMB
            elif y==1: compy, g_vecy = tSZ_map, g_tsz #ftSZ
            elif y==2: compy, g_vecy = noise1_map, g_noise1 #noise 90 GHz
            elif y==3: compy, g_vecy = noise2_map, g_noise2 #noise 150 GHz
            
            if s==y: compy *= scale_factor #if component y is the one that's scaled
            CMB_wt_maps, tSZ_wt_maps = all_wt_maps[s]
            compy_freq1, compy_freq2 = g_vecy[0]*compy, g_vecy[1]*compy

            y_to_CMB_preserved, y_to_tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[compy_freq1, compy_freq2])
            all_map_level_prop[0,y,s] = y_to_CMB_preserved
            all_map_level_prop[1,y,s] = y_to_tSZ_preserved

    
    #define and fill in array of data vectors (dim 0 has size N_comps+1 for each scaled component and then all unscaled)
    Clpq = np.zeros((N_comps+1, N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1))

    for y in range(N_comps):
        for z in range(N_comps):
            for s in range(N_comps+1): #each scaled component and then all unscaled
                
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



def main(inp, env):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    acmb_vals: [lower_acmb, upper_acmb, mean_acmb]
    atsz_vals: [lower_atsz, upper_atsz, mean_atsz]
    anoise1_vals: [lower_anoise1, upper_anoise1, mean_anoise1]
    anoise2_vals: [lower_anoise2, upper_anoise2, mean_anoise2]
    
    where
    lower_a: float, lower bound of parameter A (68% confidence)
    upper_a: float, upper bound of parameter A (68% confidence)
    mean_a: float, mean value of parameter A
    '''

    pool = mp.Pool(inp.num_parallel)
    Clpq = pool.starmap(get_data_vectors, [(sim, inp, env) for sim in range(inp.Nsims)])
    pool.close()
    Clpq = np.asarray(Clpq, dtype=np.float32) #shape (Nsims, 2 for unscaled/scaled, 2 for unscaled/scaled, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1)
    if inp.save_files:
        pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/Clpq.p')

    # Clpq = pickle.load(open('/scratch/09334/ksurrao/NILC/outputs_weight_dep/data_vecs/Clpq.p', 'rb'))
    
    acmb_array, atsz_array, anoise1_array, anoise2_array = get_all_acmb_atsz(inp, Clpq)
    acmb_vals, atsz_vals, anoise1_vals, anoise2_vals = get_parameter_cov_matrix(acmb_array, atsz_array, anoise1_array, anoise2_array, nbins=100, smoothing_factor=0.065) 

    return acmb_vals, atsz_vals, anoise1_vals, anoise2_vals




# def get_data_vectors(sim, inp, env):
#     '''
#     ARGUMENTS
#     ---------
#     sim: int, simulation number
#     inp: Info object containing input parameter specifications
#     env: environment object

#     RETURNS
#     -------
#     Clpq: (2, 2, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
#         containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
#         Size of dimensions 0 and 1 is 2 for unscaled or scaled.
#         preserved_comps = CMB, ftSZ
#         comps = CMB, ftSZ, noise 90 GHz, noise 150 GHz
#         For example, Clpq[0,1,0,1,1,2] is cross-spectrum of unscaled ftSZ propagation to 
#         CMB-preserved NILC map and scaled 90 GHz noise propagation to ftSZ-preserved NILC map
#     '''
    
#     N_preserved_comps = 2 #components to create NILC maps for: CMB, tSZ
#     N_comps = 4 #CMB, tSZ, noise1, noise2
#     comps = ['CMB', 'tSZ', 'noise1', 'noise2']
#     scale_factor = 1.1

#     #get needlet filters and spectral responses
#     h = GaussianNeedlets(inp)[1]
#     g_tsz = tsz_spectral_response(inp.freqs)
#     g_cmb = np.ones(len(inp.freqs))
#     g_noise1 = [1.,0.]
#     g_noise2 = [0.,1.]

#     #get maps and weight maps, all_wt_maps is (N_comps+1, 2, Nscales, Nfreqs, Npix) array
#     CMB_map, tSZ_map, noise1_map, noise2_map, all_wt_maps = get_scaled_maps_and_wts(sim, inp, env, scale_factor)


#     #get map level propagation of components
#     Npix = 12*inp.nside**2
#     all_map_level_prop = np.zeros((N_preserved_comps, N_comps, 2, Npix)) #2 for unscaled and scaled
    
#     for y in range(N_comps):
#         for s in range(2): #0 for no scaling and 1 for scaling

#             if y==0: compy, g_vecy = CMB_map, g_cmb #CMB
#             elif y==1: compy, g_vecy = tSZ_map, g_tsz #ftSZ
#             elif y==2: compy, g_vecy = noise1_map, g_noise1 #noise 90 GHz
#             elif y==3: compy, g_vecy = noise2_map, g_noise2 #noise 150 GHz

#             if s==1: compy *= scale_factor
#             compy_freq1, compy_freq2 = g_vecy[0]*compy, g_vecy[1]*compy

#             if s==0:
#                 CMB_wt_maps, tSZ_wt_maps = all_wt_maps[-1]
#             else:
#                 CMB_wt_maps, tSZ_wt_maps = all_wt_maps[y]
            
#             y_to_CMB_preserved, y_to_tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[compy_freq1, compy_freq2])
#             all_map_level_prop[0,y,s] = y_to_CMB_preserved
#             all_map_level_prop[1,y,s] = y_to_tSZ_preserved

    
#     #define and fill in array of data vectors (dims 0 and 1 have size 2 for unscaled or scaled)
#     Clpq = np.zeros((2, 2, N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1))

#     for y in range(N_comps):
#         for sy in range(2): #unscaled and scaled component y
#             for z in range(N_comps):
#                 for sz in range(2): #unscaled and scaled component z

#                     y_to_CMB_preserved = all_map_level_prop[0,y,sy]
#                     y_to_tSZ_preserved = all_map_level_prop[1,y,sy]
#                     z_to_CMB_preserved = all_map_level_prop[0,z,sz]
#                     z_to_tSZ_preserved = all_map_level_prop[1,z,sz]
                
#                     Clpq[sy,sz,0,0,y,z] = hp.anafast(y_to_CMB_preserved, z_to_CMB_preserved, lmax=inp.ellmax)
#                     Clpq[sy,sz,1,1,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
#                     Clpq[sy,sz,0,1,y,z] = hp.anafast(y_to_CMB_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
#                     Clpq[sy,sz,1,0,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_CMB_preserved, lmax=inp.ellmax)


#     if inp.remove_files:
#         #remove pyilc outputs
#         subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/*/sim{sim}*', shell=True, env=env)
#         #remove frequency map files
#         subprocess.call(f'rm {inp.output_dir}/maps/*/sim{sim}_freq*.fits', shell=True, env=env)

#     return Clpq



# def main(inp, env):
#     '''
#     ARGUMENTS
#     ---------
#     inp: Info object containing input parameter specifications
#     env: environment object

#     RETURNS
#     -------
#     acmb_vals: [lower_acmb, upper_acmb, mean_acmb]
#     atsz_vals: [lower_atsz, upper_atsz, mean_atsz]
#     anoise1_vals: [lower_anoise1, upper_anoise1, mean_anoise1]
#     anoise2_vals: [lower_anoise2, upper_anoise2, mean_anoise2]
    
#     where
#     lower_a: float, lower bound of parameter A (68% confidence)
#     upper_a: float, upper bound of parameter A (68% confidence)
#     mean_a: float, mean value of parameter A
#     '''

#     # pool = mp.Pool(inp.num_parallel)
#     # Clpq = pool.starmap(get_data_vectors, [(sim, inp, env) for sim in range(inp.Nsims)])
#     # pool.close()
#     # Clpq = np.asarray(Clpq, dtype=np.float32) #shape (Nsims, 2 for unscaled/scaled, 2 for unscaled/scaled, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1)
#     # if inp.save_files:
#     #     pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq.p', 'wb'), protocol=4)
#     #     if inp.verbose:
#     #         print(f'saved {inp.output_dir}/data_vecs/Clpq.p')

#     Clpq = pickle.load(open('/scratch/09334/ksurrao/NILC/outputs_weight_dep/data_vecs/Clpq.p', 'rb'))
    
#     acmb_array, atsz_array, anoise1_array, anoise2_array = get_all_acmb_atsz(inp, Clpq)
#     acmb_vals, atsz_vals, anoise1_vals, anoise2_vals = get_parameter_cov_matrix(acmb_array, atsz_array, anoise1_array, anoise2_array, nbins=100, smoothing_factor=0.065) 

#     return acmb_vals, atsz_vals, anoise1_vals, anoise2_vals



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
    setup_output_dir(inp, my_env, scaling=True)


    acmb_vals, atsz_vals, anoise1_vals, anoise2_vals = main(inp, my_env)
    print_result('Acmb', acmb_vals)
    print_result('Atsz', atsz_vals)
    print_result('Anoise1', anoise1_vals)
    print_result('Anoise2', anoise2_vals)
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)

