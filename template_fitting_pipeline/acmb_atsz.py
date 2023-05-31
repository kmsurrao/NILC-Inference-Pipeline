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
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=4, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    
    RETURNS
    -------
    cov: (Nbins, Nbins, 3, 3) ndarray containing covariance matrix Cov_{ij b1, kl b2}
        index as cov[b1, b2, 0-2 for Cl00 Cl01 Cl11, 0-2 for Cl00 Cl01 Cl11]
    '''
    cov = np.zeros((inp.Nbins, inp.Nbins, 3, 3))
    Clij_tmp = np.sum(Clij, axis=3) #shape (Nsims, Nfreqs=2, Nfreqs=2, Nbins)
    Clij_tmp = np.array([Clij_tmp[:,0,0], Clij_tmp[:,0,1], Clij_tmp[:,1,1]]) #shape (3, Nsims, Nbins)
    Clij_tmp = np.transpose(Clij_tmp, axes=(2,0,1)) #shape (Nbins, 3 for Cl00 Cl01 and Cl11, Nsims)
    Clij_tmp_means = np.mean(Clij_tmp, axis=2)
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for i in range(3):
                for j in range(3):
                    for sim in range(inp.Nsims):
                        cov[b1,b2,i,j] += (Clij_tmp[b1,i,sim]-Clij_tmp_means[b1,i])*(Clij_tmp[b2,j,sim]-Clij_tmp_means[b2,j])
    cov /= (inp.Nsims-1)
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
    Clij{i}{j}: (N_comps=4, Nbins) ndarray containing contribution of components to Clij

    RETURNS
    -------
    (Nbins, 2, 2) ndarray, 
    index as array[bin;  0-2 or ij=00, 01, 11]

    '''

    Clij_with_A_00 = Acmb*Clij00[0] + Atsz*Clij00[1] + Anoise1*Clij00[2] + Anoise2*Clij00[3]
    Clij_with_A_01 = Acmb*Clij01[0] + Atsz*Clij01[1] + Anoise1*Clij01[2] + Anoise2*Clij01[3]
    Clij_with_A_10 = Acmb*Clij10[0] + Atsz*Clij10[1] + Anoise1*Clij10[2] + Anoise2*Clij10[3]
    Clij_with_A_11 = Acmb*Clij11[0] + Atsz*Clij11[1] + Anoise1*Clij11[2] + Anoise2*Clij11[3]
    return np.array([[[Clij_with_A_00[b], Clij_with_A_01[b]],[Clij_with_A_10[b], Clij_with_A_11[b]]] for b in range(inp.Nbins)])


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
    Clij{i}{j}_all_sims: (Nsims, N_comps=4, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

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
    return np.sum([[1/2* \
     ((model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l2,0,0]*(model[l2][0,0]-Clij00d[l2]) + (model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l2,0,1]*(model[l2][0,1]-Clij01d[l2]) + (model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l2,0,2]*(model[l2][1,1]-Clij11d[l2]) \
    + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l2,1,0]*(model[l2][0,0]-Clij00d[l2]) + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l2,1,1]*(model[l2][0,1]-Clij01d[l2]) + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l2,1,2]*(model[l2][1,1]-Clij11d[l2]) \
    + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l2,2,0]*(model[l2][0,0]-Clij00d[l2]) + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l2,2,1]*(model[l2][0,1]-Clij01d[l2]) + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l2,2,2]*(model[l2][1,1]-Clij11d[l2])) \
    for l1 in range(inp.Nbins)] for l2 in range(inp.Nbins)]) 

def acmb_atsz(inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb and Atsz for one sim

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, N_comps=4, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

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
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=4, Nbins) ndarray 
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
    PScov_sim_alt = np.zeros((3*inp.Nbins, 3*inp.Nbins))
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for i in range(3):
                for j in range(3):
                    PScov_sim_alt[i*inp.Nbins+b1, j*inp.Nbins+b2] = PScov_sim[b1,b2,i,j]
    PScov_sim_alt_Inv = scipy.linalg.inv(PScov_sim_alt)
    PScov_sim_Inv = np.zeros((inp.Nbins, inp.Nbins, 3, 3))
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for i in range(3):
                for j in range(3):
                    PScov_sim_Inv[b1, b2, i, j] = PScov_sim_alt_Inv[i*inp.Nbins+b1, j*inp.Nbins+b2]

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

