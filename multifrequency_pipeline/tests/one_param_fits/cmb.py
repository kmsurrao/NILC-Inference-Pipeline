##########################################################################
## This script computes posteriors for a sky model consisting of only   ##
## CMB at a single frequency. The only parameter is Acmb. It            ##
## compares the results using likelihood-free inference and an          ##
##                    explicit Gaussian likelihood                      ##
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
    parser = argparse.ArgumentParser(description="Comparing posteriors for CMB.")
    parser.add_argument("--config", default="../../example.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    inp.ell_sum_max = inp.ellmax

    pool = mp.Pool(inp.num_parallel)
    pars = [1., 0., 0., 0.] #CMB only
    Clij = pool.starmap(get_data_vectors, [(inp, sim, pars) for sim in range(inp.Nsims)])
    pool.close()
    Clij = np.asarray(Clij, dtype=np.float32)[:,0,0,0,:] #shape (Nsims, Nbins)

    min_bin = 0
    Clij = Clij[:,min_bin:] #cut off bins with high variance                                                          
    inp.Nbins -= min_bin

    print(flush=True)
    print('Getting results using an explicit Gaussian likelihood...', flush=True)
    a1_array = get_all_acmb_atsz(inp, Clij)
    
    print(flush=True)
    print('Getting results using neural posterior estimation...', flush=True)
    prior_half_widths = [4*np.std(a1_array)]
    get_posterior(inp, prior_half_widths, Clij, 'CMB')

if __name__ == '__main__':
    main()
