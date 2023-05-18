## must be run after template_fitting_pipeline/main.py with save_files=True

import numpy as np
import pickle
import argparse
import sys
sys.path.append('../shared')
from input import Info
from utils import tsz_spectral_response

def main():

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Analytic covariance from template-fitting approach.")
    parser.add_argument("--config", default="stampede.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)
    ells = np.arange(inp.ellmax+1)

    Clij = pickle.load(open(f'{inp.output_dir}/data_vecs/Clij.p', 'rb')) #dim (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=4, ellmax+1)
    Clij = np.mean(Clij, axis=0) #dim (Nfreqs=2, Nfreqs=2, Ncomps=4, ellmax+1)

    g1, g2 = tsz_spectral_response(inp.freqs) #tSZ spectral response at 90 and 150 GHz

    CC = Clij[0,0,0] #CMB
    T = Clij[0,0,1]/g1**2 #tSZ (in Compton-y)
    N1 = Clij[0,0,2] #noise 90 GHz
    N2 = Clij[1,1,3] #noise 150 GHz

    Nmodes = inp.ellmax+1

    #use analytic expressions derived from Mathematica notebook
    CMB_var = (2* (CC* (g1 - g2)**2 + g2**2*N1 + g1**2*N2)**2 *(g1**2*N2*T +
       CC* (N1 + N2 + (g1 - g2)**2*T) + N1* (N2 + g2**2*T)))/(CC**2 *(g1 -
       g2)**2 *Nmodes *(2*g2**2*N1**2 + g1**2*N1*N2 + 2*g1*g2*N1*N2 +
       g2**2*N1*N2 + 2*g1**2*N2**2 + (g1 - g2)**2 *(g2**2*N1 + g1**2*N2)*T +
       CC*(g1 - g2)**2 *(N1 + N2 + (g1 - g2)**2 *T)))
    CMB_tSZ_covar = -((2 *(g2*N1 + g1*N2)**2 *(g1**2*N2*T + CC
       *(N1 + N2 + (g1 - g2)**2*T) + N1 *(N2 + g2**2*T)))/
       (CC *(g1 - g2)**2*Nmodes*T *(2*g2**2*N1**2 + g1**2*N1*N2 + 2*g1*g2*N1*N2 +
       g2**2*N1*N2 + 2*g1**2*N2**2 + (g1 - g2)**2 *(g2**2*N1 + g1**2*N2)*T +
       CC*(g1 - g2)**2 *(N1 + N2 + (g1 - g2)**2*T))))
    tSZ_var = (2 *(N1 + N2 + (g1 - g2)**2*T)**2 *(g1**2*N2*T +
      CC *(N1 + N2 + (g1 - g2)**2*T) + N1 *(N2 + g2**2*T)))/((g1 -
      g2)**2 *Nmodes *T**2 *(2*g2**2*N1**2 + g1**2*N1*N2 + 2*g1*g2*N1*N2 +
      g2**2*N1*N2 + 2*g1**2*N2**2 + (g1 - g2)**2 *(g2**2*N1 + g1**2*N2)*T +
      CC*(g1 - g2)**2 *(N1 + N2 + (g1 - g2)**2 *T)))

    print('variance on Acmb: ', CMB_var, flush=True)
    print('variance on Atsz: ', tSZ_var, flush=True)
    print('covariance of Acmb and Atsz: ', CMB_tSZ_covar, flush=True)

    full_covar = np.array([CMB_var, CMB_tSZ_covar, tSZ_var])
    if inp.save_files:
        pickle.dump(full_covar, open(f'{inp.output_dir}/template_fiting_analytic_covar.p'))


if __name__=='__main__':
    main()