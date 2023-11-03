############################################################################################
# This script contains functions for computing the parameter covariance matrix 
# when using analytic parameter dependence.
############################################################################################

import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
import multiprocessing as mp
import emcee

##############################################
#####  POWER SPECTRUM COVARIANCE MATRIX  #####
##############################################

def get_PScov_sim(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
    
    RETURNS
    -------
    cov: (4*Nbins, 4*Nbins) ndarray containing covariance matrix Cov_{pq,rs}
        index as cov[(0-4 for ClTT ClTy ClyT Clyy)*Nbins + bin1, (0-3 for ClTT ClTy ClyT Clyy)*Nbins + bin2]
    '''
    Clpq_tmp = np.array([Clpq[:,0,0], Clpq[:,0,1], Clpq[:,1,0], Clpq[:,1,1]])
    Clpq_tmp = np.transpose(Clpq_tmp, axes=(0,2,1)) #shape (4 for ClTT ClTy ClyT Clyy, Nbins Nsims)
    Clpq_tmp = np.reshape(Clpq_tmp, (inp.Nbins*4, -1))
    cov = np.cov(Clpq_tmp)
    return cov


##############################################
#########      NUMERICAL MLE      ############
##############################################


def ClpqA(Acmb, Atsz, inp, Clpq):
    '''
    Model for theoretical spectra Clpq including Acmb and Atsz parameters

    ARGUMENTS
    ---------
    USED BY MINIMIZER
    Acmb: float, scaling parameter for CMB power spectrum
    Atsz: float, scaling parameter for tSZ power spectrum
    
    CONSTANT ARGS
    inp: Info object containing input parameter specifications
    Clpq: (N_preserved_comps, N_preserved_comps, 1+Ncomps, Nbins) ndarray
        containing contribution of components to Clpq


    RETURNS
    -------
    theory_model: (2, 2, Nbins) ndarray for ClTT, ClTy, ClyT, and Clyy in terms of A_y and A_z parameters

    '''
    theory_model = np.zeros((2, 2, inp.Nbins))
    A_vec = [Acmb, Atsz]
    for p,q in [(0,0), (0,1), (1,0), (1,1)]:
        theory_model[p,q] = np.einsum('a,ab->b', A_vec, Clpq[p,q,1:])
    return theory_model


def neg_lnL(pars, f, inp, sim, Clpq, PScov_sim_Inv): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb, Atsz, Anoise1, and Anoise2
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clpq: (Nsims, N_preserved_comps, N_preserved_comps, 1+Ncomps, Nbins) ndarray 
        containing contribution of components to Clpq
    PScov_sim_Inv: (Nbins, Nbins, 2,2,2,2) ndarray 
        containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    neg_log_lkl: float, negative log likelihood for one simulation, combined over multipole bins
    '''    
    model = f(*pars, inp, np.mean(Clpq, axis=0))
    Clpqd = Clpq[sim, :, :, 0]
    neg_log_lkl = 1/2*np.einsum('ijb,bcijkl,klc->', model-Clpqd, PScov_sim_Inv, model-Clpqd)
    return neg_log_lkl


def acmb_atsz(inp, sim, Clpq, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb and Atsz for one sim

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clpq: (Nsims, N_preserved_comps, N_preserved_comps, 1+Ncomps, Nbins) ndarray 
        containing contribution of components to Clpq
    PScov_sim_Inv: (Nbins, Nbins, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray 
        containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit Acmb, Atsz, Anoise1, Anoise2 (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start, start] #acmb_start, atsz_start
        res = minimize(neg_lnL, x0 = start_array, args = (ClpqA, inp, sim, Clpq, PScov_sim_Inv), method='Nelder-Mead') #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x



def get_MLE_arrays(inp, Clpq, PScov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, 1+Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
    PScov_sim_Inv: (Nbins, Nbins, 2,2,2,2) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation

    '''
    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(acmb_atsz, [(inp, sim, Clpq, PScov_sim_Inv) for sim in range(inp.Nsims)])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, 2 for Acmb Atsz)
    acmb_array = param_array[:,0]
    atsz_array = param_array[:,1]
    
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_array_HILC.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_array_HILC.p', 'wb'))
    if inp.verbose:
        print(f'created {inp.output_dir}/acmb_array_HILC.p, atsz_array_HILC.p, anoise1_array_HILC.p, anoise2_array_HILC.p', flush=True)
    print('Results from maximum likelihood estimation', flush=True)
    print('----------------------------------------------', flush=True)
    print(f'Acmb = {np.mean(acmb_array)} +/- {np.std(acmb_array)}', flush=True)
    print(f'Atsz = {np.mean(atsz_array)} +/- {np.std(atsz_array)}', flush=True)

    return acmb_array, atsz_array


###############################
### FISHER MATRIX FORECAST  ###
###############################

def Fisher_inversion(inp, Clpq, PScov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, 1+Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim2: index0 is total power in Clpq, other indices are power from each component
    PScov_sim_Inv: (Nbins, Nbins, 2,2,2,2) ndarray containing 
        inverse of power spectrum covariance matrix

    RETURNS
    -------
    acmb_std, atsz_std, anoise1_std, anoise2_std: predicted standard deviations of Acmb, etc.
        found by computing the Fisher matrix and inverting
    '''

    Ncomps = 2
    Clpq_mean = np.mean(Clpq, axis=0)
    deriv_vec = np.zeros((Ncomps, 2, 2, inp.Nbins))
    
    for A in range(Ncomps):
        for p in range(2):
            for q in range(2):
                deriv_vec[A,p,q] = Clpq_mean[p,q,1+A]

    Fisher = np.einsum('Aijb,bcijkl,Bklc->AB', deriv_vec, PScov_sim_Inv, deriv_vec)
    final_cov = np.linalg.inv(Fisher)
    acmb_std = np.sqrt(final_cov[0,0])
    atsz_std = np.sqrt(final_cov[1,1])

    print('Results from inverting Fisher matrix', flush=True)
    print('----------------------------------------', flush=True)
    print('Acmb std dev: ', acmb_std, flush=True)
    print('Atsz std dev: ', atsz_std, flush=True)
    return acmb_std, atsz_std


##############################################
########   MCMC WITH ONE SIMULATION  #########
##############################################

def pos_lnL(pars, f, inp, sim, Clpq, PScov_sim_Inv): 
    '''
    Expression for positive log likelihood for one sim

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb, Atsz, Anoise1, and Anoise2
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clpq: (Nsims, N_preserved_comps, N_preserved_comps, 1+Ncomps, Nbins) ndarray 
        containing contribution of components to Clpq
    PScov_sim_Inv: (Nbins, Nbins, 2,2,2,2) ndarray 
        containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    log likelihood for one simulation, combined over multipoles 
    '''
    return -neg_lnL(pars, f, inp, sim, Clpq, PScov_sim_Inv)


def MCMC(inp, Clpq, PScov_sim_Inv, sim=0):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clpq: (Nsims, N_preserved_comps, N_preserved_comps, 1+Ncomps, Nbins) ndarray 
        containing contribution of components to Clpq
    PScov_sim_Inv: (Nbins, Nbins, 2,2,2,2) ndarray containing 
        inverse of power spectrum covariance matrix
    sim: int, simulation number

    RETURNS
    -------
    acmb_std, atsz_std: predicted standard deviations of Acmb, Atsz found from MCMC
    '''

    np.random.seed(0)
    ndim = 2
    nwalkers = 10
    p0 = np.random.random((nwalkers, ndim))*(1.2-0.8)+0.8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[ClpqA, inp, sim, Clpq, PScov_sim_Inv])
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(state, 1000)
    samples = sampler.get_chain() #dimensions (1000, nwalkers, Ncomps=2)
    
    acmb_std = np.mean(np.array([np.std(samples[:,walker,0]) for walker in range(nwalkers)]))
    atsz_std = np.mean(np.array([np.std(samples[:,walker,1]) for walker in range(nwalkers)]))

    print('Results from MCMC', flush=True)
    print('------------------------------------', flush=True)
    print('Acmb std dev: ', acmb_std, flush=True)
    print('Atsz std dev: ', atsz_std, flush=True)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), flush=True)
    return acmb_std, atsz_std


##############################################
## COMPARE RESULTS FROM DIFFERENT METHODS  ###
##############################################


def get_all_acmb_atsz(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, 1+Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim2: index0 is total power in Clpq, other indices are power from each component

    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    '''
    Clpq_unscaled = Clpq[:,:,:,0]

    PScov_sim = get_PScov_sim(inp, Clpq_unscaled)
    PScov_sim_alt_Inv = scipy.linalg.inv(PScov_sim)
    PScov_sim_Inv = np.zeros((inp.Nbins, inp.Nbins, 2,2,2,2), dtype=np.float32)
    idx_mapping = {(0,0):0, (0,1):1, (1,0):2, (1,1):3}
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for p,q in [(0,0), (0,1), (1,0), (1,1)]:
                for r,s in [(0,0), (0,1), (1,0), (1,1)]:
                    pq = idx_mapping[(p,q)]
                    rs = idx_mapping[(r,s)]
                    PScov_sim_Inv[b1,b2,p,q,r,s] = PScov_sim_alt_Inv[pq*inp.Nbins+b1, rs*inp.Nbins+b2]
    PScov_sim_Inv *= (inp.Nsims-(inp.Nbins*3)-2)/(inp.Nsims-1) #correction factor from https://arxiv.org/pdf/astro-ph/0608064.pdf

    acmb_array, atsz_array = get_MLE_arrays(inp, Clpq, PScov_sim_Inv)
    
    print(flush=True)
    Fisher_inversion(inp, Clpq, PScov_sim_Inv)

    print(flush=True)
    MCMC(inp, Clpq, PScov_sim_Inv, sim=0)
   
    return acmb_array, atsz_array