import sys
sys.path.append('../shared')
import os
import subprocess
import pickle
import time
import argparse
from input import Info
from generate_maps import generate_freq_maps
from pyilc_interface import setup_pyilc
from load_weight_maps import load_wt_maps
from utils import setup_output_dir, tsz_spectral_response, GaussianNeedlets, build_NILC_maps
from nilc_power_spectrum_calc import calculate_all_cl
from calculate_n_point_functions import *
from wigner3j import *


def get_data_vectors(sim, inp, env):
    '''
    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object, contains input parameter specifications
    env: environment object, current environment in which to run subprocesses

    RETURNS
    -------
    Clpq: (N_preserved_comps, N_preserved_comps, N_comps, 4, ellmax+1) numpy array,
            contains contributions from each component to the power spectrum of NILC maps
            with preserved components p and q,
            index as Clpq[p,q,z,reMASTERed term,l]
    Clpq_direct: (N_preserved_comps, N_preserved_comps, ellmax+1) numpy array,
            directly computed auto- and cross- spectra of NILC maps from pyilc,
            index as Clpq_direct[p,q,l]
    '''

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N1, N2, CMB_map, tSZ_map, noise1_map, noise2_map = generate_freq_maps(sim, inp, band_limit=True)
    
    #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
    setup_pyilc(sim, inp, env)

    #load weight maps
    CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim, band_limit=True)

    #get power spectra of components, index as Clzz[z,l]
    Clzz = get_Clzz(CC, T)
    if inp.verbose:
        print(f'calculated component map spectra for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(Clzz, open(f'{inp.output_dir}/n_point_funcs/sim{sim}_Clzz.p', 'wb'))

    #get power spectra of weight maps, index as Clw1w2[p,q,n,m,i,j,l]
    Clw1w2 = get_Clw1w2(inp, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated weight map spectra for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(Clw1w2, open(f'{inp.output_dir}/n_point_funcs/sim{sim}_Clw1w2.p', 'wb'))

    #get cross-spectra of component maps and weight maps, index as Clzw[z,p,n,i,l]
    Clzw = get_Clzw(inp, CMB_wt_maps, tSZ_wt_maps, CMB_map, tSZ_map)
    if inp.verbose:
        print(f'calculated component and weight map cross-spectra for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(Clzw, open(f'{inp.output_dir}/n_point_funcs/sim{sim}_Clzw.p', 'wb'))
        
    #get means of weight maps, index as w[p,n,i]
    w = get_w(inp, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated weight map means for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(w, open(f'{inp.output_dir}/n_point_funcs/sim{sim}_w.p', 'wb'))

    #get means of component maps, index as a[z]
    a = get_a(CMB_map, tSZ_map)
    if inp.verbose:
        print(f'calculated component map means for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(a, open(f'{inp.output_dir}/n_point_funcs/sim{sim}_a.p', 'wb'))
        
    #get bispectrum for two factors of map and one factor of weight map, index as bispectrum_zzw[z,q,m,j,l1,l2,l3]
    bispectrum_zzw = get_bispectrum_zzw(inp, CMB_map, tSZ_map, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated bispectra for map, map, weight map for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(bispectrum_zzw, open(f'{inp.output_dir}/n_point_funcs/sim{sim}_bispectrum_zzw.p', 'wb'))
    
    # get bispectrum for two weight maps and one factor of map, index as bispectrum_wzw[p,n,i,z,q,m,j,l1,l2,l3]
    bispectrum_wzw = get_bispectrum_wzw(inp, CMB_map, tSZ_map, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated bispectra for weight map, map, weight map for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(bispectrum_wzw, open(f'{inp.output_dir}/n_point_funcs/sim{sim}_bispectrum_wzw.p', 'wb'))
    
    # get unnormalized trispectrum estimator rho, index as rho[z,p,n,i,q,m,j,l2,l4,l3,l5,l1]
    Rho = get_rho(inp, CMB_map, tSZ_map, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated unnormalized trispectrum estimator rho for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(Rho, open(f'{inp.output_dir}/n_point_funcs/sim{sim}_Rho.p', 'wb'))
        
    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp)[1]
    g_cmb = np.ones(len(inp.freqs))
    g_tsz = tsz_spectral_response(inp.freqs)

    #get contributions to Clpq from each comp, index as Cl_{comp}[p,q,l] and contrib_{comp}[reMASTERed term, p,q,l]
    Cl_CMB, contrib_CMB = calculate_all_cl(inp, h, g_cmb, Clzz[0], Clw1w2, Clzw[0], w, a[0],
            bispectrum_zzw[0], bispectrum_wzw[:,:,:,0,:,:,:,:,:,:], Rho[0], delta_ij=False)
    Cl_tSZ, contrib_tSZ = calculate_all_cl(inp, h, g_tsz, Clzz[1], Clw1w2, Clzw[1], w, a[1],
            bispectrum_zzw[1], bispectrum_wzw[:,:,:,1,:,:,:,:,:,:], Rho[1], delta_ij=False)
    Clpq = np.array([contrib_CMB, contrib_tSZ], dtype=np.float32) #indices (z, reMASTERed term, p,q,l)
    Clpq = np.transpose(Clpq, axes=(2,3,0,1,4)) #indices (p,q,z, reMASTERed term,l)

    # # code below is to get directly computed Clpq of NILC maps from pyilc
    # # note subtlety that pyilc only includes certain frequencies at each filter scale
    # # thus, we compute it manually instead for comparison to the analytic result
    # CMB_NILC_map = 10**(-6)*hp.read_map(f'{inp.output_dir}/pyilc_outputs/sim{sim}needletILCmap_component_CMB.fits')
    # tSZ_NILC_map = hp.read_map(f'{inp.output_dir}/pyilc_outputs/sim{sim}needletILCmap_component_tSZ.fits')

    #get directly computed Clpq of NILC maps
    CMB_NILC_map, tSZ_NILC_map = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps)
    Clpq_direct = np.zeros((2, 2, inp.ellmax+1))
    Clpq_direct[0][0] = hp.anafast(CMB_NILC_map, lmax=inp.ellmax)
    Clpq_direct[1][1] = hp.anafast(tSZ_NILC_map, lmax=inp.ellmax)
    Clpq_direct[0][1] = hp.anafast(CMB_NILC_map, tSZ_NILC_map, lmax=inp.ellmax)
    Clpq_direct[1][0] = Clpq_direct[0][1]

    #use code below to get directly computed propagation of CMB and tSZ individually to CMB and tSZ NILC maps
    CMB_prop_to_CMB_NILC_map, CMB_prop_to_tSZ_NILC_map = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, [CMB_map, CMB_map])
    tSZ_prop_to_CMB_NILC_map, tSZ_prop_to_tSZ_NILC_map = build_NILC_maps(inp, sim, h, CMB_wt_maps, tSZ_wt_maps, [g_tsz[0]*tSZ_map, g_tsz[1]*tSZ_map])
    directly_computed_prop_to_NILC_PS = np.zeros((2, 2, inp.ellmax+1)) #index as [0 or 1 for CMB or tSZ propagation, 0 of 1 for CMB or tSZ preserved NILC map, ell]
    directly_computed_prop_to_NILC_PS[0,0] = hp.anafast(CMB_prop_to_CMB_NILC_map, lmax=inp.ellmax)
    directly_computed_prop_to_NILC_PS[0,1] = hp.anafast(CMB_prop_to_tSZ_NILC_map, lmax=inp.ellmax)
    directly_computed_prop_to_NILC_PS[1,0] = hp.anafast(tSZ_prop_to_CMB_NILC_map, lmax=inp.ellmax)
    directly_computed_prop_to_NILC_PS[1,1] = hp.anafast(tSZ_prop_to_tSZ_NILC_map, lmax=inp.ellmax)

    if inp.save_files:
        pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/sim{sim}_Clpq.p', 'wb'), protocol=4)
        pickle.dump(Clpq_direct, open(f'{inp.output_dir}/data_vecs/sim{sim}_Clpq_direct.p', 'wb'), protocol=4)
        pickle.dump(directly_computed_prop_to_NILC_PS, open(f'{inp.output_dir}/data_vecs/sim{sim}_directly_computed_prop_to_NILC_PS.p', 'wb'), protocol=4)

    if inp.remove_files: #don't need pyilc output files anymore
        subprocess.call(f'rm {inp.output_dir}/pyilc_outputs/sim{sim}*', shell=True, env=env)
    
    return Clpq, Clpq_direct




def main():
    '''
    RETURNS
    -------
    Clpq: (N_preserved_comps, N_preserved_comps, N_comps, 4, ellmax+1) numpy array,
            contains contributions from each component to the power spectrum of NILC maps
            with preserved components p and q,
            index as Clpq[p,q,z,reMASTERed term,l]
    Clpq_direct: (N_preserved_comps, N_preserved_comps, ellmax+1) numpy array,
            directly computed auto- and cross- spectra of NILC maps from pyilc,
            index as Clpq_direct[p,q,l]
    '''
    

    start_time = time.time()

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Analytic NILC power spectrum.")
    parser.add_argument("--config", default="example.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # current environment, also environment in which to run subprocesses
    my_env = os.environ.copy()

    #get wigner 3j symbols
    if inp.wigner_file != '':
        inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ell_sum_max+1, :inp.ell_sum_max+1, :inp.ell_sum_max+1]
    else:
        inp.wigner3j = compute_3j(inp.ell_sum_max)
    
    #set up output directory
    setup_output_dir(inp, my_env)
    
    Clpq, Clpq_direct = get_data_vectors(0, inp, my_env)

    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)

    return Clpq, Clpq_direct




if __name__ == '__main__':
    main()

    

