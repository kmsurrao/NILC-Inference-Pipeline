import sys
import numpy as np
sys.path.append('../shared')
import os
import multiprocessing as mp
from input import Info
import pickle
import time
import argparse
import tqdm
from utils import setup_output_dir, get_naming_str
import hilc_SR
import hilc_analytic
import param_cov_SR
import param_cov_analytic
from likelihood_free_inference import get_posterior


def main():
    '''
    RETURNS
    -------
    a_array: (Ncomps, Nsims) ndarray containing best fit parameters for each simulation
    '''

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Covariance from harmonic ILC power spectrum template-fitting approach.")
    parser.add_argument("--config", default="example_yaml_files/weights_vary_LFI.yaml")
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

    if not inp.use_lfi:
        
        pool = mp.Pool(inp.num_parallel)
        print(f'Running {inp.Nsims} simulations for frequency-frequency power spectra...', flush=True)
        if inp.use_symbolic_regression:
            inputs = [(sim, inp) for sim in range(inp.Nsims)]
            Clij = list(tqdm.tqdm(pool.imap(hilc_SR.get_freq_power_spec_star, inputs), total=inp.Nsims))
        else:
            inputs = [(inp, sim) for sim in range(inp.Nsims)]
            Clij = list(tqdm.tqdm(pool.imap(hilc_analytic.get_freq_power_spec_star, inputs), total=inp.Nsims))
        pool.close()
        Clij = np.asarray(Clij, dtype=np.float32)
        if inp.save_files:
            naming_str = get_naming_str(inp, 'HILC')
            pickle.dump(Clij, open(f'{inp.output_dir}/data_vecs/Clij_{naming_str}.p', 'wb'), protocol=4)
            print(f'\nsaved {inp.output_dir}/data_vecs/Clij_{naming_str}.p', flush=True)
        
        pool = mp.Pool(inp.num_parallel)
        print(f'\nRunning {inp.Nsims} simulations for HILC spectra...', flush=True)
        if inp.use_symbolic_regression:
            inp.Clij_theory = np.mean(Clij[:,0,0,0], axis=0)
            inputs = [(inp, Clij[sim], sim) for sim in range(inp.Nsims)]
            Clpq = list(tqdm.tqdm(pool.imap(hilc_SR.get_data_vecs_star, inputs), total=inp.Nsims))
        else:
            inp.Clij_theory = np.mean(Clij, axis=0)
            inputs = [(inp, Clij[sim]) for sim in range(inp.Nsims)]
            Clpq = list(tqdm.tqdm(pool.imap(hilc_analytic.get_data_vecs_star, inputs), total=inp.Nsims))
        pool.close()
        Clpq = np.asarray(Clpq, dtype=np.float32)
        if inp.save_files:
            pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq_{naming_str}.p', 'wb'), protocol=4)
            print(f'\nsaved {inp.output_dir}/data_vecs/Clpq_{naming_str}.p', flush=True)
        
        if inp.use_symbolic_regression:
            a_array = param_cov_SR.get_all_a_vec(inp, Clpq, HILC=True)
        else:
            a_array = param_cov_analytic.get_all_a_vec(inp, Clpq)
    
    else:
        samples = get_posterior(inp, 'HILC', my_env)
        a_array = np.array(samples, dtype=np.float32).T
        
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return a_array


if __name__ == '__main__':
    main()



