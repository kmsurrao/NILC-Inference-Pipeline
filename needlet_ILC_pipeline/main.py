import sys
sys.path.append('../shared')
import numpy as np
import os
import multiprocessing as mp
from input import Info
import pickle
import time
import argparse
from utils import setup_output_dir
import param_cov_SR
from nilc_data_vecs import get_scaled_data_vectors
from likelihood_free_inference import get_posterior

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

    if not inp.use_lfi:
        pool = mp.Pool(inp.num_parallel)
        Clpq = pool.starmap(get_scaled_data_vectors, [(sim, inp, env) for sim in range(inp.Nsims)])
        pool.close()
        Clpq = np.asarray(Clpq, dtype=np.float32)
        if inp.save_files:
            pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq.p', 'wb'), protocol=4)
            print(f'saved {inp.output_dir}/data_vecs/Clpq.p', flush=True)
        acmb_array, atsz_array, anoise1_array, anoise2_array = param_cov_SR.get_all_acmb_atsz(inp, Clpq, env, HILC=False)
    
    else:
        samples = get_posterior(inp, 'NILC', env)
        acmb_array, atsz_array, anoise1_array, anoise2_array = samples
        print('Results from Likelihood-Free Inference', flush=True)
        print('----------------------------------------------', flush=True)
        print(f'Acmb = {np.mean(acmb_array)} +/- {np.std(acmb_array)}', flush=True)
        print(f'Atsz = {np.mean(atsz_array)} +/- {np.std(atsz_array)}', flush=True)
        print(f'Anoise1 = {np.mean(anoise1_array)} +/- {np.std(anoise1_array)}', flush=True)
        print(f'Anoise2 = {np.mean(anoise2_array)} +/- {np.std(anoise2_array)}', flush=True)

    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return acmb_array, atsz_array, anoise1_array, anoise2_array


if __name__ == '__main__':
    main()


