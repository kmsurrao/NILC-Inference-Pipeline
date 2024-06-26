##########################################################################
## This script computes posteriors for a sky model consisting of only   ##
## tSZ at a single frequency. The only parameter is Atsz.               ##
## It compares the results using likelihood-free inference and an       ##
## explicit Gaussian likelihood (which is only accurate when using      ##
##                   Gaussian tSZ realizations).                        ##
##########################################################################

import sys
sys.path.append('../..')
sys.path.append('../../../shared')
import numpy as np
import multiprocessing as mp
import argparse
from lfi_posteriors import get_posterior
from likelihood_posteriors import get_all_acmb_atsz
from multifrequency_data_vecs import get_data_vectors
from input import Info

def main():

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Comparing posteriors for tSZ.")
    parser.add_argument("--config", default="../../example_yaml_files/lfi.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    inp.ell_sum_max = inp.ellmax
    inp.use_Gaussian_tSZ = True
    inp.noise = 0

    pool = mp.Pool(inp.num_parallel)
    pars = [0., 1.] #tSZ only
    Clij = pool.starmap(get_data_vectors, [(inp, sim, pars) for sim in range(inp.Nsims)])
    pool.close()
    Clij = np.asarray(Clij, dtype=np.float32)[:,0,0,0,:] #shape (Nsims, Nbins)

    print(flush=True)
    print('Getting results using an explicit Gaussian likelihood...', flush=True)
    a1_array = get_all_acmb_atsz(inp, Clij)
    
    print(flush=True)
    print('Getting results using neural posterior estimation...', flush=True)
    prior_half_widths = [4*np.std(a1_array)]
    get_posterior(inp, prior_half_widths, Clij, 'tSZ')

if __name__ == '__main__':
    main()
