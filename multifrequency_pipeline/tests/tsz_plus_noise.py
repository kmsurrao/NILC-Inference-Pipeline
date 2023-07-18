## must be run after template_fitting_pipeline/main.py with save_files=True

import numpy as np
import pickle
import argparse
import scipy
from scipy.optimize import minimize
import sys
sys.path.append('..')
sys.path.append('../../shared')
from input import Info

def get_PScov_sim(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=3, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
        Note that CMB should be excluded from Clij
    
    RETURNS
    -------
    cov: (ellmax+1,3,3) ndarray containing covariance matrix Cov_{ij,kl}
        index as cov[l, freq1, freq2]
    '''
    Clij_tmp = np.sum(Clij, axis=3)
    Clij_tmp = np.array([Clij_tmp[:,0,0], Clij_tmp[:,0,1], Clij_tmp[:,1,1]])
    Clij_tmp = np.transpose(Clij_tmp, axes=(2,0,1)) #shape (ellmax+1, 3 for Cl00 Cl01 and Cl11, Nsims)
    cov = np.array([np.cov(Clij_tmp[l]) for l in range(inp.ellmax+1)]) #shape (ellmax+1,3,3)
    assert cov.shape == (inp.ellmax+1, 3, 3), f"covariance shape is {cov.shape} but should be ({inp.ellmax+1},3,3)"
    return cov


def get_all_acmb_atsz(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=4, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j

    RETURNS
    -------
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation

    '''

    def ClijA(Atsz, Anoise1, Anoise2):
        '''
        Model for theoretical spectra Clpq including Acmb and Atsz parameters

        ARGUMENTS
        ---------
        Atsz: float, scaling parameter for ftSZ power spectrum
        Anoise1: float, scaling parameter for 90 GHz noise power spectrum
        Anoise2: float, scaling parameter for 150 GHz noise power spectrum

        RETURNS
        -------
        (ellmax+1, 2, 2) ndarray, 
        index as array[l;  0-2 or ij=00, 01, 11]

        '''

        Clij_with_A_00 = Atsz*Clij00[0] + Anoise1*Clij00[1] + Anoise2*Clij00[2]
        Clij_with_A_01 = Atsz*Clij01[0] + Anoise1*Clij01[1] + Anoise2*Clij01[2]
        Clij_with_A_10 = Atsz*Clij10[0] + Anoise1*Clij10[1] + Anoise2*Clij10[2]
        Clij_with_A_11 = Atsz*Clij11[0] + Anoise1*Clij11[1] + Anoise2*Clij11[2]
        return np.array([[[Clij_with_A_00[l], Clij_with_A_01[l]],[Clij_with_A_10[l], Clij_with_A_11[l]]] for l in range(inp.ellmax+1)])


    def lnL(pars, f, inp): 
        '''
        Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)
        Let Clpqd be the data spectra obtained by averaging over all the theory spectra from each sim

        ARGUMENTS
        ---------
        pars: parameters to function f (not manually inputted but used by minimizer)
        f: function that returns theory model in terms of Acmb and Atsz
        sim: int, simulation number
        inp: Info object containing input parameter specifications

        RETURNS
        -------
        negative log likelihood for one simulation, combined over multipoles 
        '''
        model = f(*pars)
        Clij00d = np.mean(np.sum(Clij00_all_sims, axis=1), axis=0)
        Clij01d = np.mean(np.sum(Clij01_all_sims, axis=1), axis=0)
        Clij11d = np.mean(np.sum(Clij11_all_sims, axis=1), axis=0)
        assert Clij00d.shape == (inp.ellmax+1,), f"Clij00d.shape is {Clij00d.shape}, should be ({inp.ellmax+1},)"
        return np.sum([1/2* \
         ((model[l][0,0]-Clij00d[l])*PScov_sim_Inv[l][0,0]*(model[l][0,0]-Clij00d[l]) + (model[l][0,0]-Clij00d[l])*PScov_sim_Inv[l][0,1]*(model[l][0,1]-Clij01d[l]) + (model[l][0,0]-Clij00d[l])*PScov_sim_Inv[l][0,2]*(model[l][1,1]-Clij11d[l]) \
        + (model[l][0,1]-Clij01d[l])*PScov_sim_Inv[l][1,0]*(model[l][0,0]-Clij00d[l]) + (model[l][0,1]-Clij01d[l])*PScov_sim_Inv[l][1,1]*(model[l][0,1]-Clij01d[l]) + (model[l][0,1]-Clij01d[l])*PScov_sim_Inv[l][1,2]*(model[l][1,1]-Clij11d[l]) \
        + (model[l][1,1]-Clij11d[l])*PScov_sim_Inv[l][2,0]*(model[l][0,0]-Clij00d[l]) + (model[l][1,1]-Clij11d[l])*PScov_sim_Inv[l][2,1]*(model[l][0,1]-Clij01d[l]) + (model[l][1,1]-Clij11d[l])*PScov_sim_Inv[l][2,2]*(model[l][1,1]-Clij11d[l])) \
        for l in range(2, inp.ellmax+1)]) 

    def acmb_atsz():
        '''
        Maximize likelihood with respect to Acmb and Atsz for one sim

        ARGUMENTS
        ---------
        sim: int, simulation number

        RETURNS
        -------
        best fit Acmb, Atsz, Anoise1, Anoise2 (floats)
        '''
        atsz_start = 1.0
        anoise1_start = 1.0
        anoise2_start = 1.0
        res = minimize(lnL, x0 = [atsz_start, anoise1_start, anoise2_start], args = (ClijA, inp), method='Nelder-Mead') #default method is BFGS
        return res.x #atsz, anoise1, anoise2
    
    PScov_sim = get_PScov_sim(inp, Clij)
    PScov_sim_Inv = np.array([scipy.linalg.inv(PScov_sim[l]) for l in range(inp.ellmax+1)])

    Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims = Clij[:,0,0], Clij[:,0,1], Clij[:,1,0], Clij[:,1,1]

    atsz_array = np.ones(inp.Nsims, dtype=np.float32)
    anoise1_array = np.ones(inp.Nsims, dtype=np.float32)
    anoise2_array = np.ones(inp.Nsims, dtype=np.float32)
    for sim in range(inp.Nsims):
        Clij00, Clij01, Clij10, Clij11 = Clij00_all_sims[sim], Clij01_all_sims[sim], Clij10_all_sims[sim], Clij11_all_sims[sim]
        atsz, anoise1, anoise2 = acmb_atsz()
        atsz_array[sim] = atsz
        anoise1_array[sim] = anoise1
        anoise2_array[sim] = anoise2
    
    if inp.verbose:
        print('atsz_array: ', atsz_array, flush=True)
        print('noise1_array: ', anoise1_array, flush=True)
        print('noise2_array: ', anoise2_array, flush=True)
    
    print(f'Atsz = {np.mean(atsz_array)} +/- {np.std(atsz_array)}', flush=True)
    print(f'Anoise1 = {np.mean(anoise1_array)} +/- {np.std(anoise1_array)}', flush=True)
    print(f'Anoise2 = {np.mean(anoise2_array)} +/- {np.std(anoise2_array)}', flush=True)
   
    return atsz_array, anoise1_array, anoise2_array



def main():
    '''
    RETURNS
    -------
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation
    '''

    # main input file containing most specifications 
    parser = argparse.ArgumentParser(description="Analytic covariance from template-fitting approach.")
    parser.add_argument("--config", default="../stampede.yaml")
    args = parser.parse_args()
    input_file = args.config

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    Clij = pickle.load(open(f'{inp.output_dir}/data_vecs/Clij.p', 'rb')) #dim (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=4, ellmax+1)
    Clij = np.delete(Clij, 0, axis=3) #remove CMB 

    atsz_array, anoise1_array, anoise2_array = get_all_acmb_atsz(inp, Clij)
    return atsz_array, anoise1_array, anoise2_array
    

    

 

if __name__=='__main__':
    main()
