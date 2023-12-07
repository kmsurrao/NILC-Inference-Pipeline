import sys
sys.path.append('../shared')
import numpy as np
import os
import multiprocessing as mp
from input import Info
import pickle
import time
import argparse
import tqdm
from utils import setup_output_dir, get_naming_str
import param_cov_SR
from nilc_data_vecs import get_scaled_data_vectors_star
from likelihood_free_inference import get_posterior

def main():
    '''
    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    '''

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Covariance from NILC approach.")
    parser.add_argument("--config", default="example_yaml_files/lfi.yaml")
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

    if not inp.use_lfi:
        pool = mp.Pool(inp.num_parallel)
        inputs = [(sim, inp, env) for sim in range(inp.Nsims)]
        print(f'Running {inp.Nsims} simulations...', flush=True)
        Clpq = list(tqdm.tqdm(pool.imap(get_scaled_data_vectors_star, inputs), total=inp.Nsims))
        pool.close()
        Clpq = np.asarray(Clpq, dtype=np.float32)
        if inp.save_files:
            naming_str = get_naming_str(inp, 'NILC')
            pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq_{naming_str}.p', 'wb'), protocol=4)
            print(f'\nsaved {inp.output_dir}/data_vecs/Clpq_{naming_str}.p', flush=True)
        acmb_array, atsz_array = param_cov_SR.get_all_acmb_atsz(inp, Clpq, HILC=False)
    
    else:
        samples = get_posterior(inp, 'NILC', env)
        acmb_array, atsz_array = np.array(samples, dtype=np.float32).T

    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return acmb_array, atsz_array


if __name__ == '__main__':
    main()


