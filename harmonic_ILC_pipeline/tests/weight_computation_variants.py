import numpy as np
import argparse
import time
import os
import multiprocessing as mp
import pickle
import subprocess 
import sys
sys.path.append('../../shared')
from utils import setup_output_dir
sys.path.append('..')
from input import Info
import hilc_analytic



def main():
    '''
    Computes variants of weight computation in harmonic ILC
    - determining weights once, getting power spectrum as w^i_\ell w^j_\ell C_\ell^{ij}
    - determining weights for each sim, getting power spectrum as w^i_\ell w^j_\ell C_\ell^{ij}
    '''

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Covariance from harmonic ILC power spectrum template-fitting approach.")
    parser.add_argument("--config", default="../example.yaml")
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
    if not os.path.isdir(f'{inp.output_dir}/HILC_tests'):
        subprocess.call(f'mkdir {inp.output_dir}/HILC_tests', shell=True, env=my_env)

    inp.compute_weights_once = True
    pool = mp.Pool(inp.num_parallel)
    results = pool.starmap(hilc_analytic.get_freq_power_spec, [(sim, inp, True) for sim in range(inp.Nsims)])
    pool.close()
    Clij = np.asarray(results, dtype=np.float32) #shape (Nsims, Nfreqs=2, Nfreqs=2, 1+Ncomps, ellmax+1)

    # Determine weights once, compute HILC power spectrum analytically
    inp.compute_weights_once = True
    inp.Clij_theory = np.mean(Clij, axis=0)
    pool = mp.Pool(inp.num_parallel)
    Clpq1 = pool.starmap(hilc_analytic.get_data_vecs, [(inp, Clij[sim]) for sim in range(inp.Nsims)])
    pool.close()
    Clpq1 = np.asarray(Clpq1, dtype=np.float32)[:,:,:,0,:] #shape (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins)
    if inp.save_files:
        pickle.dump(Clpq1, open(f'{inp.output_dir}/HILC_tests/Clpq_weights_once.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/HILC_tests/Clpq_weights_once.p')
    
    
    # Determine weights for each realization, compute HILC power spectrum analytically
    inp.compute_weights_once = False
    pool = mp.Pool(inp.num_parallel)
    Clpq2 = pool.starmap(hilc_analytic.get_data_vecs, [(inp, Clij[sim]) for sim in range(inp.Nsims)])
    pool.close()
    Clpq2 = np.asarray(Clpq2, dtype=np.float32)[:,:,:,0,:] #shape (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins)
    if inp.save_files:
        pickle.dump(Clpq2, open(f'{inp.output_dir}/HILC_tests/Clpq_weights_vary.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/HILC_tests/Clpq_weights_vary.p')
    

    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return

if __name__ == '__main__':
    main()