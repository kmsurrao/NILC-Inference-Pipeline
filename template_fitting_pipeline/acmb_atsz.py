import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
import multiprocessing as mp


def get_PScov_sim(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=4, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    
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



def ClijA(Acmb, Atsz, Anoise1, Anoise2, inp, Clij00, Clij01, Clij10, Clij11):
    '''
    Model for theoretical spectra Clpq including Acmb and Atsz parameters

    ARGUMENTS
    ---------
    USED BY MINIMIZER
    Acmb: float, scaling parameter for CMB power spectrum
    Atsz: float, scaling parameter for tSZ power spectrum
    Anoise1: float, scaling parameter for 90 GHz noise power spectrum
    Anoise2: float, scaling parameter for 150 GHz noise power spectrum

    CONSTANT ARGS
    inp: Info object containing input parameter specifications
    Clij{i}{j}: (N_comps=4, ellmax+1) ndarray containing contribution of components to Clij

    RETURNS
    -------
    (ellmax+1, 2, 2) ndarray, 
    index as array[l;  0-2 or ij=00, 01, 11]

    '''

    Clij_with_A_00 = Acmb*Clij00[0] + Atsz*Clij00[1] + Anoise1*Clij00[2] + Anoise2*Clij00[3]
    Clij_with_A_01 = Acmb*Clij01[0] + Atsz*Clij01[1] + Anoise1*Clij01[2] + Anoise2*Clij01[3]
    Clij_with_A_10 = Acmb*Clij10[0] + Atsz*Clij10[1] + Anoise1*Clij10[2] + Anoise2*Clij10[3]
    Clij_with_A_11 = Acmb*Clij11[0] + Atsz*Clij11[1] + Anoise1*Clij11[2] + Anoise2*Clij11[3]
    return np.array([[[Clij_with_A_00[l], Clij_with_A_01[l]],[Clij_with_A_10[l], Clij_with_A_11[l]]] for l in range(inp.ellmax+1)])


def lnL(pars, f, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)
    Let Clpqd be the data spectra obtained by averaging over all the theory spectra from each sim

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb and Atsz
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, N_comps=4, ellmax+1) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (ellmax+1, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    negative log likelihood for one simulation, combined over multipoles 
    '''
    Clij00 = Clij00_all_sims[sim]
    Clij01 = Clij01_all_sims[sim]
    Clij10 = Clij10_all_sims[sim]
    Clij11 = Clij11_all_sims[sim]
    model = f(*pars, inp, Clij00, Clij01, Clij10, Clij11)
    Clij00d = np.mean(np.sum(Clij00_all_sims, axis=1), axis=0)
    Clij01d = np.mean(np.sum(Clij01_all_sims, axis=1), axis=0)
    Clij11d = np.mean(np.sum(Clij11_all_sims, axis=1), axis=0)
    assert Clij00d.shape == (inp.ellmax+1,), f"Clij00d.shape is {Clij00d.shape}, should be ({inp.ellmax+1},)"
    return np.sum([1/2* \
        ((model[l][0,0]-Clij00d[l])*PScov_sim_Inv[l][0,0]*(model[l][0,0]-Clij00d[l]) + (model[l][0,0]-Clij00d[l])*PScov_sim_Inv[l][0,1]*(model[l][0,1]-Clij01d[l]) + (model[l][0,0]-Clij00d[l])*PScov_sim_Inv[l][0,2]*(model[l][1,1]-Clij11d[l]) \
    + (model[l][0,1]-Clij01d[l])*PScov_sim_Inv[l][1,0]*(model[l][0,0]-Clij00d[l]) + (model[l][0,1]-Clij01d[l])*PScov_sim_Inv[l][1,1]*(model[l][0,1]-Clij01d[l]) + (model[l][0,1]-Clij01d[l])*PScov_sim_Inv[l][1,2]*(model[l][1,1]-Clij11d[l]) \
    + (model[l][1,1]-Clij11d[l])*PScov_sim_Inv[l][2,0]*(model[l][0,0]-Clij00d[l]) + (model[l][1,1]-Clij11d[l])*PScov_sim_Inv[l][2,1]*(model[l][0,1]-Clij01d[l]) + (model[l][1,1]-Clij11d[l])*PScov_sim_Inv[l][2,2]*(model[l][1,1]-Clij11d[l])) \
    for l in range(2, inp.ellmax+1)]) 

def acmb_atsz(inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb and Atsz for one sim

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, N_comps=4, ellmax+1) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (ellmax+1, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit Acmb, Atsz, Anoise1, Anoise2 (floats)
    '''
    acmb_start = 1.0
    atsz_start = 1.0
    anoise1_start = 1.0
    anoise2_start = 1.0
    res = minimize(lnL, x0 = [acmb_start, atsz_start, anoise1_start, anoise2_start], args = (ClijA, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv), method='Nelder-Mead') #default method is BFGS
    return res.x #acmb, atsz, anoise


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
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation

    '''

    PScov_sim = get_PScov_sim(inp, Clij)
    PScov_sim_Inv = np.array([scipy.linalg.inv(PScov_sim[l]) for l in range(inp.ellmax+1)])

    Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims = Clij[:,0,0], Clij[:,0,1], Clij[:,1,0], Clij[:,1,1]

    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(acmb_atsz, [(inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv) for sim in range(inp.Nsims)])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, 4 for Acmb Atsz Anoise1 Anoise2)
    acmb_array = param_array[:,0]
    atsz_array = param_array[:,1]
    anoise1_array = param_array[:,2]
    anoise2_array = param_array[:,3]
    
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_array_template_fitting.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_array_template_fitting.p', 'wb'))
    pickle.dump(anoise1_array, open(f'{inp.output_dir}/anoise1_array_template_fitting.p', 'wb'))
    pickle.dump(anoise2_array, open(f'{inp.output_dir}/anoise2_array_template_fitting.p', 'wb'))
    if inp.verbose:
        print(f'created {inp.output_dir}/acmb_array_template_fitting.p and atsz and anoise1 and anoise2', flush=True)
    
    print(f'Acmb = {np.mean(acmb_array)} +/- {np.std(acmb_array)}', flush=True)
    print(f'Atsz = {np.mean(atsz_array)} +/- {np.std(atsz_array)}', flush=True)
    print(f'Anoise1 = {np.mean(anoise1_array)} +/- {np.std(anoise1_array)}', flush=True)
    print(f'Anoise2 = {np.mean(anoise2_array)} +/- {np.std(anoise2_array)}', flush=True)
   
    return acmb_array, atsz_array, anoise1_array, anoise2_array

