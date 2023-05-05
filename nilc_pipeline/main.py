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


def get_data_vectors(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object

    RETURNS
    -------
    Clpq: (N_amps, N_amps, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra. 
        preserved_comps = CMB, ftSZ
        comps = CMB, ftSZ, noise 90 GHz, noise 150 GHz
        For example, Clpq[0,0,0,1,1,2] is cross-spectrum of ftSZ propagation to 
        CMB-preserved NILC map and 90 GHz noise propagation to ftSZ-preserved NILC map with
        a scaling of amp=1 for the ftSZ amplitude and amp=1 for the noise amplitude.
    '''
    
    N_preserved_comps = 2 #components to create NILC maps for: CMB, tSZ
    N_comps = 4 #CMB, tSZ, noise1, noise2
    comps = ['CMB', 'tSZ', 'noise1', 'noise2']
    scalings = [1, 10, 50, 100]

    #define array of data vectors
    Clpq = np.zeros((len(scalings), len(scalings), N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1))

    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_tsz = tsz_spectral_response(inp.freqs)
    g_cmb = np.ones(len(inp.freqs))
    g_noise1 = [1.,0.]
    g_noise2 = [0.,1.]

    for y in range(N_comps):
        for z in range(N_comps):
            for s1 in range(len(scalings)):
                for s2 in range(len(scalings)):
                
                    #scaling parameters to feed into functions
                    scaling = [[scalings[s1], comps[y]], [scalings[s2], comps[z]]]

                    #create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N1, N2)
                    CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(sim, inp, scaling=scaling)
                    
                    #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
                    setup_pyilc(sim, inp, env, suppress_printing=True, scaling=scaling)

                    #load weight maps
                    CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim, scaling=scaling)


                    #fill in array of data vectors

                    if y==0: compy, g_vecy = CMB_map, g_cmb #CMB
                    elif y==1: compy, g_vecy = tSZ_map, g_tsz #ftSZ
                    elif y==2: compy, g_vecy = noise1_map, g_noise1 #noise 90 GHz
                    elif y==3: compy, g_vecy = noise2_map, g_noise2 #noise 150 GHz
                    compy_freq1, compy_freq2 = g_vecy[0]*compy, g_vecy[1]*compy

                    if z==0: compz, g_vecz = CMB_map, g_cmb #CMB
                    elif z==1: compz, g_vecz = tSZ_map, g_tsz #ftSZ
                    elif z==2: compz, g_vecz = noise1_map, g_noise1 #noise 90 GHz
                    elif z==3: compz, g_vecz = noise2_map, g_noise2 #noise 150 GHz
                    compz_freq1, compz_freq2 = g_vecz[0]*compz, g_vecz[1]*compz

                    y_to_CMB_preserved, y_to_tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[compy_freq1, compy_freq2])
                    z_to_CMB_preserved, z_to_tSZ_preserved = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, freq_maps=[compz_freq1, compz_freq2])
                    Clpq[s1,s2,0,0,y,z] = hp.anafast(y_to_CMB_preserved, z_to_CMB_preserved, lmax=inp.ellmax)
                    Clpq[s1,s2,1,1,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
                    Clpq[s1,s2,0,1,y,z] = hp.anafast(y_to_CMB_preserved, z_to_tSZ_preserved, lmax=inp.ellmax)
                    Clpq[s1,s2,1,0,y,z] = hp.anafast(y_to_tSZ_preserved, z_to_CMB_preserved, lmax=inp.ellmax)
            
                    if inp.remove_files:
                        #remove pyilc outputs
                        subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/scaling{s1}{comps[y]}_scaling{s2}{comps[z]}/sim{sim}*', shell=True, env=env)
                        #remove frequency map files
                        subprocess.call(f'rm {inp.output_dir}/maps/scaling{s1}{comps[y]}_scaling{s2}{comps[z]}/sim{sim}_freq*.fits', shell=True, env=env)
                
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
    Clpq = np.asarray(Clpq, dtype=np.float32) #shape (Nsims, N_amps, N_amps, N_preserved_comps=2, N_preserved_comps=2, N_comps=3, N_comps=3, ellmax+1)
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
    setup_output_dir(inp, my_env, scalings=[1, 10, 50, 100])


    lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = main(inp, my_env)
    print(f'Acmb = {mean_acmb} + {upper_acmb-mean_acmb} - {mean_acmb-lower_acmb}', flush=True)
    print(f'Atsz = {mean_atsz} + {upper_atsz-mean_atsz} - {mean_atsz-lower_atsz}', flush=True)
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)

