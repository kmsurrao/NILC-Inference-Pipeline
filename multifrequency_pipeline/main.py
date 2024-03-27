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
from param_cov import get_all_a_vec
from multifrequency_data_vecs import get_data_vectors_star
from likelihood_free_inference import get_posterior


def main():
    '''
    RETURNS
    -------
    a_array: (Ncomps, Nsims) ndarray containing best fit parameters for each simulation
    '''

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Covariance from multifrequency template-fitting approach.")
    parser.add_argument("--config", default="example_yaml_files/lfi.yaml")
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
        inputs = [(inp, sim) for sim in range(inp.Nsims)]
        print(f'Running {inp.Nsims} simulations...', flush=True)
        Clij = list(tqdm.tqdm(pool.imap(get_data_vectors_star, inputs), total=inp.Nsims))
        pool.close()
        Clij = np.asarray(Clij, dtype=np.float32) #shape (Nsims, Nfreqs, Nfreqs, 1+Ncomps, Nbins)
        if inp.save_files:
            naming_str = get_naming_str(inp, 'multifrequency')
            pickle.dump(Clij, open(f'{inp.output_dir}/data_vecs/Clij_{naming_str}.p', 'wb'), protocol=4)
            print(f'\nsaved {inp.output_dir}/data_vecs/Clij_{naming_str}.p')
        # naming_str = get_naming_str(inp, 'multifrequency') #remove and uncomment above
        # Clij = pickle.load(open(f'{inp.output_dir}/data_vecs/Clij_{naming_str}.p', 'rb')) #remove and uncomment above
        a_array = get_all_a_vec(inp, Clij)
    
    else:
        samples = get_posterior(inp, 'multifrequency', my_env)
        a_array = np.array(samples, dtype=np.float32).T
    
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return a_array


if __name__ == '__main__':
    main()



