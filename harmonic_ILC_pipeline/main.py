import sys
import numpy as np
sys.path.append('../shared')
import os
import multiprocessing as mp
from input import Info
import pickle
import time
import argparse
from utils import setup_output_dir
import hilc_SR
import hilc_analytic
import param_cov_SR
import param_cov_analytic


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
    parser = argparse.ArgumentParser(description="Covariance from harmonic ILC power spectrum template-fitting approach.")
    parser.add_argument("--config", default="example.yaml")
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

    pool = mp.Pool(inp.num_parallel)
    if inp.use_symbolic_regression:
        Clij = pool.starmap(hilc_SR.get_freq_power_spec, [(sim, inp) for sim in range(inp.Nsims)])
    else:
        Clij = pool.starmap(hilc_analytic.get_freq_power_spec, [(sim, inp) for sim in range(inp.Nsims)])
    pool.close()
    Clij = np.asarray(Clij, dtype=np.float32)
    if inp.save_files:
        pickle.dump(Clij, open(f'{inp.output_dir}/data_vecs/Clij_HILC.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/Clij_HILC.p')
       
    pool = mp.Pool(inp.num_parallel)
    if inp.use_symbolic_regression:
        inp.Clij_theory = np.mean(Clij[:,0,0,0,0,0], axis=0)
        Clpq = pool.starmap(hilc_SR.get_data_vecs, [(inp, Clij[sim]) for sim in range(inp.Nsims)])
    else:
        inp.Clij_theory = np.mean(Clij, axis=0)
        Clpq = pool.starmap(hilc_analytic.get_data_vecs, [(inp, Clij[sim]) for sim in range(inp.Nsims)])
    pool.close()
    Clpq = np.asarray(Clpq, dtype=np.float32)
    if inp.save_files:
        pickle.dump(Clpq, open(f'{inp.output_dir}/data_vecs/Clpq_HILC.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/Clpq_HILC.p')
    
    if inp.use_symbolic_regression:
        acmb_array, atsz_array, anoise1_array, anoise2_array = param_cov_SR.get_all_acmb_atsz(inp, Clpq, my_env, HILC=True)
    else:
        acmb_array, atsz_array, anoise1_array, anoise2_array = param_cov_analytic.get_all_acmb_atsz(inp, Clpq)
    
    print('PROGRAM FINISHED RUNNING')
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
    return acmb_array, atsz_array, anoise1_array, anoise2_array


if __name__ == '__main__':
    main()



