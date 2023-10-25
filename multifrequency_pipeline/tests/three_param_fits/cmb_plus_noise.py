##########################################################################
## This script computes posteriors for a sky model consisting of only   ##
## CMB and noise at two frequencies. The parameters are Acmb, Anoise1,  ##
## and Anoise2. It compares the results using likelihood-free inference ##
##             and an explicit Gaussian likelihood.                     ##
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
    parser = argparse.ArgumentParser(description="Comparing posteriors for CMB+noise.")
    parser.add_argument("--config", default="../../example.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    inp.ell_sum_max = inp.ellmax

    pool = mp.Pool(inp.num_parallel)
    pars = [1., 0., 1., 1.] #CMB and noise only
    Clij = pool.starmap(get_data_vectors, [(inp, sim, pars) for sim in range(inp.Nsims)])
    pool.close()
    Clij = np.asarray(Clij, dtype=np.float32) #shape (Nsims, Nfreqs=2, Nfreqs=2, 1+4, Nbins)
    Clij = np.concatenate((Clij[:,:,:,:2,:], Clij[:,:,:,3:,:]), axis=3) #get rid of tSZ component

    print(flush=True)
    print('Getting results using an explicit Gaussian likelihood...', flush=True)
    a1_array, a2_array, a3_array = get_all_acmb_atsz(inp, Clij)
    
    print(flush=True)
    print('Getting results using neural posterior estimation...', flush=True)
    prior_half_widths = []
    for arr in [a1_array, a2_array, a3_array]:
        prior_half_widths.append(4*np.std(arr))
    get_posterior(inp, prior_half_widths, Clij, omit_tsz=True)

if __name__ == '__main__':
    main()
