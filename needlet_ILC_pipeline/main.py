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
from scipy import stats
from generate_maps import generate_freq_maps
from pyilc_interface import setup_pyilc, weight_maps_exist
from load_weight_maps import load_wt_maps
from utils import setup_output_dir, tsz_spectral_response, GaussianNeedlets, build_NILC_maps, get_scalings
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
    CMB_map_unscaled, tSZ_map_unscaled, noise1_map_unscaled, noise2_map_unscaled: unscaled maps of all the components
    all_wt_maps: (Nscalings, 2, 2, 2, 2, N_preserved_comps, Nscales, Nfreqs, Npix) ndarray containing all weight maps
                dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
                      idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
                dim1: idx0 for unscaled CMB, idx1 for scaled CMB
                dim2: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
                dim3: idx0 for unscaled noise90, idx1 for scaled noise90
                dim4: idx0 for unscaled noise150, idx1 for scaled noise150
                Note: for sim >= Nsims_for_fits, results are meaningless except for scaling 00000 (all unscaled)

    '''

    N_preserved_comps = 2

    #array for all weight maps
    Nscalings = len(inp.scaling_factors)
    all_wt_maps = np.zeros((Nscalings, 2, 2, 2, 2, N_preserved_comps, inp.Nscales, len(inp.freqs), 12*inp.nside**2))
    scalings = get_scalings(inp)

    for s, scaling in enumerate(scalings):

        #create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N1, N2)
        CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(sim, inp, scaling=scaling)
        if s==0:
            CMB_map_unscaled, tSZ_map_unscaled, noise1_map_unscaled, noise2_map_unscaled = CMB_map, tSZ_map, noise1_map, noise2_map
        
        #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
        if not weight_maps_exist(sim, inp, scaling=scaling): #check if not all the weight maps already exist
            #remove any existing weight maps for this sim and scaling to prevent pyilc errors
            if scaling is not None:  
                scaling_str = ''.join(str(e) for e in scaling)                                                  
                subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/{scaling_str}/sim{sim}*', shell=True, env=env)
            else:
                subprocess.call(f'rm -f {inp.output_dir}/pyilc_outputs/sim{sim}*', shell=True, env=env)
            setup_pyilc(sim, inp, env, suppress_printing=True, scaling=scaling) #set suppress_printing=False to debug pyilc runs

        #load weight maps
        CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim, scaling=scaling)
        all_wt_maps[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4]] = np.array([CMB_wt_maps, tSZ_wt_maps])

        if sim >= inp.Nsims_for_fits:
            break #only need unscaled version after getting Nsims_for_fits scaled maps and weights
    
    return CMB_map_unscaled, tSZ_map_unscaled, noise1_map_unscaled, noise2_map_unscaled, all_wt_maps



def get_data_vectors(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    Clpq: (Nscalings, 2, 2, 2, 2, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, Nbins) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        dim0: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
            idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim1: idx0 for unscaled CMB, idx1 for scaled CMB
        dim2: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
        dim3: idx0 for unscaled noise90, idx1 for scaled noise90
        dim4: idx0 for unscaled noise150, idx1 for scaled noise150
        preserved_comps = CMB, ftSZ
        comps = CMB, ftSZ, noise 90 GHz, noise 150 GHz
        For example, Clpq[1,0,1,0,1,0,1,1,2] is cross-spectrum of ftSZ propagation to 
        CMB-preserved NILC map and 90 GHz noise propagation to ftSZ-preserved NILC map 
        when ftSZ and 150 GHz noise are scaled according to inp.scaling_factors[1]
        Note: for sim >= Nsims_for_fits, results are meaningless except for scaling 00000 (all unscaled)
    '''
    
    N_preserved_comps = 2 #components to create NILC maps for: CMB, ftSZ
    N_comps = 4 #CMB, ftSZ, noise1, noise2

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = tsz_spectral_response(inp.freqs)
    g_cmb = np.ones(len(inp.freqs))
    g_noise1 = [1.,0.]
    g_noise2 = [0.,1.]

    #get maps and weight maps
    CMB_map, tSZ_map, noise1_map, noise2_map, all_wt_maps = get_scaled_maps_and_wts(sim, inp, env)

    #get map level propagation of components
    Npix = 12*inp.nside**2
    Nscalings = len(inp.scaling_factors)
    all_map_level_prop = np.zeros((Nscalings,2,2,2,2, N_preserved_comps, N_comps, Npix)) 
    scalings = get_scalings(inp)
    
    for y in range(N_comps):
        for scaling in scalings:

            if y==0: compy, g_vecy = np.copy(CMB_map), g_cmb #CMB
            elif y==1: compy, g_vecy = np.copy(tSZ_map), g_tsz #ftSZ
            elif y==2: compy, g_vecy = np.copy(noise1_map), g_noise1 #noise 90 GHz
            elif y==3: compy, g_vecy = np.copy(noise2_map), g_noise2 #noise 150 GHz

            if scaling[y+1]==1: #component y scaled
                compy *= inp.scaling_factors[scaling[0]]
            
            CMB_wt_maps, tSZ_wt_maps = all_wt_maps[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4]]
            compy_freq1, compy_freq2 = g_vecy[0]*compy, g_vecy[1]*compy

            y_to_CMB_preserved, y_to_tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[compy_freq1, compy_freq2])
            all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0, y] = y_to_CMB_preserved
            all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1, y] = y_to_tSZ_preserved

            if sim >= inp.Nsims_for_fits:
                break #only need unscaled version after getting Nsims_for_fits scaled maps and weights

    #define and fill in array of data vectors (dim 0 has size Nscalings for which scaling in inp.scaling_factors is used)
    Clpq_tmp = np.zeros((Nscalings,2,2,2,2, N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1)) #unbinned
    Clpq = np.zeros((Nscalings,2,2,2,2, N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.Nbins)) #binned

    for y in range(N_comps):
        for z in range(N_comps):
            for scaling in scalings:

                y_to_CMB_preserved = all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0, y]
                y_to_tSZ_preserved = all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1, y]
                z_to_CMB_preserved = all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0, z]
                z_to_tSZ_preserved = all_map_level_prop[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1, z]
            
                Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0,0,y,z] = hp.anafast(y_to_CMB_preserved, z_to_CMB_preserved, lmax=inp.ellmax)
                Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1,1,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
                Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0,1,y,z] = hp.anafast(y_to_CMB_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
                Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1,0,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_CMB_preserved, lmax=inp.ellmax)

                ells = np.arange(inp.ellmax+1)
                all_spectra = [Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4],0,0,y,z], 
                               Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4],1,1,y,z], 
                               Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4],0,1,y,z], 
                               Clpq_tmp[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4],1,0,y,z]]
                for idx, Cl in enumerate(all_spectra):
                    Dl = ells*(ells+1)/2/np.pi*Cl
                    res = stats.binned_statistic(ells[2:], Dl[2:], statistic='mean', bins=inp.Nbins)
                    mean_ells = (res[1][:-1]+res[1][1:])/2
                    if idx==0: 
                        Clpq[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0,0,y,z] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
                    elif idx==1: 
                        Clpq[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1,1,y,z] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
                    elif idx==2: 
                        Clpq[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 0,1,y,z] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
                    elif idx==3: 
                        Clpq[scaling[0], scaling[1], scaling[2], scaling[3], scaling[4], 1,0,y,z] = res[0]/(mean_ells*(mean_ells+1)/2/np.pi)
                
                if sim >= inp.Nsims_for_fits:
                    break #only need unscaled version after getting Nsims_for_fits scaled maps and weights


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
    parser.add_argument("--config", default="example.yaml")
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
    Clpq = np.asarray(Clpq, dtype=np.float32)
    if inp.save_files:
        pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq.p', 'wb'), protocol=4)
        print(f'saved {inp.output_dir}/data_vecs/Clpq.p', flush=True)
    
    acmb_array, atsz_array, anoise1_array, anoise2_array = get_all_acmb_atsz(inp, Clpq, env)
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return acmb_array, atsz_array, anoise1_array, anoise2_array


if __name__ == '__main__':
    main()


