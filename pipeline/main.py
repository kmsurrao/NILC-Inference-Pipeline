import sys
import os
import multiprocessing as mp
from input import Info
import pickle
from utils import setup_output_dir
from check_analytic_model import get_data_vectors
from acmb_atsz_nilc import *
from wigner3j import *
from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)


def main(inp, env):

    Clpq = np.zeros((inp.Nsims, 2, 2, 3, 4, inp.ellmax+1))

    pool = mp.Pool(inp.num_parallel)
    results = pool.starmap(get_data_vectors, [(sim, inp, env) for sim in range(inp.Nsims)])
    pool.close()
    Clpq = np.asarray(results[:,0], dtype=np.float32)
    
    acmb_array, atsz_array = get_all_acmb_atsz(inp, Clpq)
    lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = get_parameter_cov_matrix(acmb_array, atsz_array, nbins=100, smoothing_factor=0.065) 

    return lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz



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

    #set up output directory
    setup_output_dir(inp, my_env)

    #get wigner 3j symbols
    if inp.wigner_file != '':
        inp.wigner3j = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ell_sum_max+1, :inp.ell_sum_max+1, :inp.ell_sum_max+1]
    else:
        inp.wigner3j = compute_3j(inp.ell_sum_max)
    
    #set up output directory
    setup_output_dir(inp, my_env)

    lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz = main(inp, my_env)
    print(f'Acmb = {mean_acmb} + {upper_acmb-mean_acmb} - {mean_acmb-lower_acmb}', flush=True)
    print(f'Atsz = {mean_atsz} + {upper_atsz-mean_atsz} - {mean_atsz-lower_atsz}', flush=True)

