import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
import multiprocessing as mp
from fits import fit_func, call_fit, get_parameter_dependence



def get_PScov_sim(inp, Clpq_unscaled):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq_unscaled: (Nsims, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra
    
    RETURNS
    -------
    cov: (ellmax+1,3,3) ndarray containing covariance matrix Cov_{pq,rs}
        index as cov[l, 0-2 for ClTT ClTy Clyy, 0-2 for ClTT ClTy Clyy]
    '''
    Clpq_tmp = np.sum(Clpq_unscaled, axis=(3,4))
    Clpq_tmp = np.array([Clpq_tmp[:,0,0], Clpq_tmp[:,0,1], Clpq_tmp[:,1,1]])
    Clpq_tmp = np.transpose(Clpq_tmp, axes=(2,0,1)) #shape (ellmax+1, 3 for ClTT, ClTy, Clyy, Nsims)
    Clpq_tmp = np.reshape(Clpq_tmp, (3*(inp.ellmax+1),inp.Nsims))
    cov = np.cov(Clpq_tmp) #shape (3*(ellmax+1), 3*(ellmax+1))
    return cov


def ClpqA(Acmb, Atsz, Anoise1, Anoise2, inp, ClTT, ClTy, ClyT, Clyy, best_fits):
    '''
    Model for theoretical spectra Clpq including Acmb, Atsz, and Anoise parameters

    ARGUMENTS
    ---------
    USED BY MINIMIZER
    Acmb: float, scaling parameter for CMB power spectrum
    Atsz: float, scaling parameter for tSZ power spectrum
    Anoise1: float, scaling parameter for 90 GHz noise power spectrum
    Anoise2: float, scaling parameter for 150 GHz noise power spectrum
    
    CONSTANT ARGS
    inp: Info object containing input parameter specifications
    Cl{p}{q}: (N_comps=4, N_comps=4, ellmax+1) ndarray containing contribution of components to Clpq
    best_fits: (N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, N_comps) ndarray
        containing best fits to Acmb, Atsz, Anoise1, Anoise2; N_comps is for exponent params

    RETURNS
    -------
    theory_model: (ellmax+1, 2, 2) ndarray for ClTT, ClTy, ClyT, and Clyy in terms of A_y and A_z parameters

    '''
    theory_model = np.zeros((inp.ellmax+1, 2, 2))

    for l in range(inp.ellmax+1):
        for p,q in [(0,0), (0,1), (1,0), (1,1)]:

            if p==0 and q==0: 
                best_fits_here, Clpq_here = best_fits[0,0], ClTT
            elif p==0 and q==1:
                best_fits_here, Clpq_here = best_fits[0,1], ClTy
            elif p==1 and q==0:
                best_fits_here, Clpq_here = best_fits[1,0], ClyT
            elif p==1 and q==1:
                best_fits_here, Clpq_here = best_fits[1,1], Clyy
            A_vec = [Acmb, Atsz, Anoise1, Anoise2]
            theory_model[l,p,q] = \
              call_fit(A_vec, best_fits_here[0,0,l])*Clpq_here[0,0,l]  + call_fit(A_vec, best_fits_here[0,1,l])*Clpq_here[0,1,l]  + call_fit(A_vec, best_fits_here[0,2,l])*Clpq_here[0,2,l]  + call_fit(A_vec, best_fits_here[0,3,l])*Clpq_here[0,3,l]\
            + call_fit(A_vec, best_fits_here[1,0,l])*Clpq_here[1,0,l]  + call_fit(A_vec, best_fits_here[1,1,l])*Clpq_here[1,1,l]  + call_fit(A_vec, best_fits_here[1,2,l])*Clpq_here[1,2,l]  + call_fit(A_vec, best_fits_here[1,3,l])*Clpq_here[1,3,l] \
            + call_fit(A_vec, best_fits_here[2,0,l])*Clpq_here[2,0,l]  + call_fit(A_vec, best_fits_here[2,1,l])*Clpq_here[2,1,l]  + call_fit(A_vec, best_fits_here[2,2,l])*Clpq_here[2,2,l]  + call_fit(A_vec, best_fits_here[2,3,l])*Clpq_here[2,3,l] \
            + call_fit(A_vec, best_fits_here[3,0,l])*Clpq_here[3,0,l]  + call_fit(A_vec, best_fits_here[3,1,l])*Clpq_here[3,1,l]  + call_fit(A_vec, best_fits_here[3,2,l])*Clpq_here[3,2,l]  + call_fit(A_vec, best_fits_here[3,3,l])*Clpq_here[3,3,l]

    return theory_model



def lnL(pars, f, inp, sim, ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims, PScov_sim_Inv, best_fits): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)
    Let Clpqd be the data spectra obtained by averaging over all the theory spectra from each sim

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb, Atsz, Anoise1, and Anoise2
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Cl{p}{q}_all_sims: (Nsims, N_comps=4, N_comps=4, ellmax+1) ndarray containing contribution of components to Clpq
    PScov_sim_Inv: (ellmax+1, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing inverse of power spectrum covariance matrix
    best_fits: (N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, N_comps) ndarray
        containing best fits to Acmb, Atsz, Anoise1, Anoise2; N_comps is for exponent params


    RETURNS
    -------
    negative log likelihood for one simulation, combined over multipoles 
    '''
    ClTT = ClTT_all_sims[sim]
    ClTy = ClTy_all_sims[sim]
    ClyT = ClyT_all_sims[sim]
    Clyy = Clyy_all_sims[sim]
    model = f(*pars, inp, ClTT, ClTy, ClyT, Clyy, best_fits)
    ClTTd = np.mean(np.sum(ClTT_all_sims, axis=(1,2)), axis=0)
    ClTyd = np.mean(np.sum(ClTy_all_sims, axis=(1,2)), axis=0)
    Clyyd = np.mean(np.sum(Clyy_all_sims, axis=(1,2)), axis=0)
    return np.sum([[1/2* \
         ((model[l1][0,0]-ClTTd[l1])*PScov_sim_Inv[l1,0,l2,0]*(model[l2][0,0]-ClTTd[l2]) + (model[l1][0,0]-ClTTd[l1])*PScov_sim_Inv[l1,0,l2,1]*(model[l2][0,1]-ClTyd[l2]) + (model[l1][0,0]-ClTTd[l1])*PScov_sim_Inv[l1,0,l2,2]*(model[l2][1,1]-Clyyd[l2]) \
        + (model[l1][0,1]-ClTyd[l1])*PScov_sim_Inv[l1,1,l2,0]*(model[l2][0,0]-ClTTd[l2]) + (model[l1][0,1]-ClTyd[l1])*PScov_sim_Inv[l1,1,l2,1]*(model[l2][0,1]-ClTyd[l2]) + (model[l1][0,1]-ClTyd[l1])*PScov_sim_Inv[l1,1,l2,2]*(model[l2][1,1]-Clyyd[l2]) \
        + (model[l1][1,1]-Clyyd[l1])*PScov_sim_Inv[l1,2,l2,0]*(model[l2][0,0]-ClTTd[l2]) + (model[l1][1,1]-Clyyd[l1])*PScov_sim_Inv[l1,2,l2,1]*(model[l2][0,1]-ClTyd[l2]) + (model[l1][1,1]-Clyyd[l1])*PScov_sim_Inv[l1,2,l2,2]*(model[l2][1,1]-Clyyd[l2])) \
    for l1 in range(2, inp.ellmax+1)] for l2 in range(2, inp.ellmax+1)]) 

def acmb_atsz(inp, sim, ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims, PScov_sim_Inv, best_fits):
    '''
    Maximize likelihood with respect to Acmb and Atsz for one sim

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Cl{p}{q}_all_sims: (Nsims, N_comps=4, N_comps=4, ellmax+1) ndarray containing contribution of components to Clpq
    PScov_sim_Inv: (ellmax+1, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing inverse of power spectrum covariance matrix
    best_fits: (N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, N_comps) ndarray
        containing best fits to Acmb, Atsz, Anoise1, Anoise2; N_comps is for exponent params

    RETURNS
    -------
    best fit Acmb, Atsz, Anoise1, Anoise2 (floats)
    '''
    acmb_start = 1.0
    atsz_start = 1.0
    anoise1_start = 1.0
    anoise2_start = 1.0
    bounds = ((0.0, None), (0.0, None), (0.0, None), (0.0, None))
    res = minimize(lnL, x0 = [acmb_start, atsz_start, anoise1_start, anoise2_start], args = (ClpqA, inp, sim, ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims, PScov_sim_Inv, best_fits), method='Nelder-Mead', bounds=bounds) #default method is BFGS
    return res.x #acmb, atsz, anoise1, anoise2


def get_all_acmb_atsz(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, 2*N_comps+1, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra

    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation

    '''
    
    N_comps = 4
    best_fits = get_parameter_dependence(inp, Clpq) #(N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 2*N_comps)
    Clpq_unscaled = Clpq[:,2*N_comps]

    PScov_sim = get_PScov_sim(inp, Clpq_unscaled)
    PScov_sim_Inv = scipy.linalg.inv(PScov_sim)
    PScov_sim_Inv = np.reshape(PScov_sim_Inv, (inp.ellmax+1, 3, inp.ellmax+1, 3))

    ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims = Clpq_unscaled[:,0,0], Clpq_unscaled[:,0,1], Clpq_unscaled[:,1,0], Clpq_unscaled[:,1,1]

    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(acmb_atsz, [(inp, sim, ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims, PScov_sim_Inv, best_fits) for sim in range(inp.Nsims)])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, 4 for Acmb Atsz Anoise1 Anoise2)
    acmb_array = param_array[:,0]
    atsz_array = param_array[:,1]
    anoise1_array = param_array[:,2]
    anoise2_array = param_array[:,3]
    
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_array_nilc.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_array_nilc.p', 'wb'))
    pickle.dump(anoise1_array, open(f'{inp.output_dir}/anoise1_array_nilc.p', 'wb'))
    pickle.dump(anoise2_array, open(f'{inp.output_dir}/anoise2_array_nilc.p', 'wb'))
    if inp.verbose:
        print(f'created {inp.output_dir}/acmb_array_nilc.p, atsz_array_nilc.p, anoise1_array_nilc.p, anoise2_array_nilc.p', flush=True)
        print('acmb_array: ', acmb_array)
        print('atsz_array: ', atsz_array)
        print('anoise1_array: ', anoise1_array)
        print('anoise2_array: ', anoise2_array)
   
    # #remove section below and uncomment above
    # acmb_array = pickle.load(open(f'{inp.output_dir}/acmb_array_nilc.p', 'rb'))
    # atsz_array = pickle.load(open(f'{inp.output_dir}/atsz_array_nilc.p', 'rb'))
    # anoise1_array = pickle.load(open(f'{inp.output_dir}/anoise1_array_nilc.p', 'rb'))
    # anoise2_array = pickle.load(open(f'{inp.output_dir}/anoise2_array_nilc.p', 'rb'))

    print(f'Acmb = {np.mean(acmb_array)} +/- {np.std(acmb_array)}', flush=True)
    print(f'Atsz = {np.mean(atsz_array)} +/- {np.std(atsz_array)}', flush=True)
    print(f'Anoise1 = {np.mean(anoise1_array)} +/- {np.std(anoise1_array)}', flush=True)
    print(f'Anoise2 = {np.mean(anoise2_array)} +/- {np.std(anoise2_array)}', flush=True)

    return acmb_array, atsz_array, anoise1_array, anoise2_array
