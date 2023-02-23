import sys
import os
import subprocess
import pickle
from input import Info
from generate_maps import generate_freq_maps
from pyilc_interface import setup_pyilc
from load_weight_maps import load_wt_maps
from utils import setup_output_dir, tsz_spectral_response, GaussianNeedlets
from nilc_power_spectrum_calc import calculate_all_cl
from calculate_n_point_functions import *
from wigner3j import *
from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
hp.disable_warnings()


def one_sim(sim, inp, env):

    #Create frequency maps (GHz) consisting of CMB, tSZ, and noise. Get power spectra of component maps (CC, T, and N)
    CC, T, N, CMB_map, tSZ_map, noise_map = generate_freq_maps(sim, inp)
    
    #get NILC weight maps for preserved component CMB and preserved component tSZ using pyilc
    setup_pyilc(sim, inp, env)

    #load weight maps and then remove pyilc weight map files
    CMB_wt_maps, tSZ_wt_maps = load_wt_maps(inp, sim)
    if inp.remove_files: #don't need pyilc output files anymore
        subprocess.call(f'rm {inp.scratch_path}/pyilc_outputs/sim{sim}*', shell=True, env=env)
        subprocess.call(f'rm {inp.scratch_path}/pyilc_outputs/sim{sim}*', shell=True, env=env)

    #get power spectra of components, index as Clzz[z,l]
    Clzz = get_Clzz(CC, T, N)
    if inp.verbose:
        print(f'calculated component map spectra for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(Clzz, open(f'{inp.output_dir}/n_point_funcs/Clzz.p', 'wb'))

    #get power spectra of weight maps, index as Clw1w2[p,q,n,m,i,j,l]
    Clw1w2 = get_Clw1w2(inp, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated weight map spectra for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(Clw1w2, open(f'{inp.output_dir}/n_point_funcs/Clw1w2.p', 'wb'))

    #get cross-spectra of component maps and weight maps, index as Clzw[z,p,n,i,l]
    Clzw = get_Clzw(inp, CMB_wt_maps, tSZ_wt_maps, CMB_map, tSZ_map, noise_map)
    if inp.verbose:
        print(f'calculated component and weight map cross-spectra for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(Clzw, open(f'{inp.output_dir}/n_point_funcs/Clzw.p', 'wb'))
    
    #get means of weight maps, index as w[p,n,i]
    w = get_w(inp, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated weight map means for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(w, open(f'{inp.output_dir}/n_point_funcs/w.p', 'wb'))

    #get means of component maps, index as a[z]
    a = get_a(CMB_map, tSZ_map, noise_map)
    if inp.verbose:
        print(f'calculated component map means for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(a, open(f'{inp.output_dir}/n_point_funcs/a.p', 'wb'))
    
    #get bispectrum for two factors of map and one factor of weight map, index as bispectrum_zzw[z,q,m,j,b1,b2,b3]
    bispectrum_zzw = get_bispectrum_zzw(inp, CMB_map, tSZ_map, noise_map, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated bispectra for map, map, weight map for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(bispectrum_zzw, open(f'{inp.output_dir}/n_point_funcs/bispectrum_zzw.p', 'wb'))
    
    #get bispectrum for two weight maps and one factor of map, index as bispectrum_wzw[p,n,i,z,q,m,j,b1,b2,b3]
    bispectrum_wzw = get_bispectrum_wzw(inp, CMB_map, tSZ_map, noise_map, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated bispectra for weight map, map, weight map for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(bispectrum_wzw, open(f'{inp.output_dir}/n_point_funcs/bispectrum_wzw.p', 'wb'))
    
    #get unnormalized trispectrum estimator rho, index as rho[z,p,n,i,q,m,j,b2,b4,b3,b5,b1]
    Rho = get_rho(inp, CMB_map, tSZ_map, noise_map, CMB_wt_maps, tSZ_wt_maps)
    if inp.verbose:
        print(f'calculated unnormalized trispectrum estimator rho for sim {sim}', flush=True)
    if inp.save_files:
        pickle.dump(Rho, open(f'{inp.output_dir}/n_point_funcs/Rho.p', 'wb'))
    
    #get needlet filters and spectral responses
    h = GaussianNeedlets(inp.ellmax, inp.FWHM_arcmin)[1]
    g_cmb = np.array([1., 1.])
    g_tsz = tsz_spectral_response(inp.freqs)

    #get contributions to Clpq from each comp, index as Cl_{comp}[p,q,l] and contrib_{comp}[reMASTERed term, p,q,l]
    Cl_CMB, contrib_CMB = calculate_all_cl(inp, h, g_cmb, Clzz[0], Clw1w2, Clzw[0], w, a[0],
            bispectrum_zzw[0], bispectrum_wzw[:,:,:,0,:,:,:,:,:,:], Rho[0], delta_ij=False)
    Cl_tSZ, contrib_tSZ = calculate_all_cl(inp, h, g_cmb, Clzz[1], Clw1w2, Clzw[1], w, a[1],
            bispectrum_zzw[1], bispectrum_wzw[:,:,:,1,:,:,:,:,:,:], Rho[1], delta_ij=False)
    Cl_noise, contrib_noise = calculate_all_cl(inp, h, g_cmb, Clzz[2], Clw1w2, Clzw[2], w, a[2],
            bispectrum_zzw[2], bispectrum_wzw[:,:,:,2,:,:,:,:,:,:], Rho[2], delta_ij=True)
    Clpq = np.array([contrib_CMB, contrib_tSZ, contrib_noise], dtype=np.float32) #indices (z, reMASTERed term, p,q,l)
    Clpq = np.transpose(Clpq, axes=(2,3,0,1,4)) #indices (p,q,z, reMASTERed term,l)

    if inp.save_files:
        pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq.p', 'wb'), protocol=4)

    return Clpq



if __name__ == '__main__':
    # main input file containing most specifications 
    try:
        input_file = (sys.argv)[1]
    except IndexError:
        input_file = 'example.yaml'

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
    
    Clpq = one_sim(0, inp, my_env)
    

