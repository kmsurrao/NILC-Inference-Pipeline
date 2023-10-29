############################################################################################
# This script contains functions for computing the parameter covariance matrix 
# when fitting parameter dependence with symbolic regression.
# Can be used for harmonic ILC or needlet ILC.
############################################################################################

import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
import multiprocessing as mp
import emcee
from fits import call_fit, get_parameter_dependence

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
        dim3: index0 is total power in Clpq, other indices are power from each component
    
    RETURNS
    -------
    cov: (3*Nbins, 3*Nbins) ndarray containing covariance matrix Cov_{pq,rs}
        index as cov[(0-2 for ClTT ClTy Clyy)*Nbins + bin1, (0-2 for ClTT ClTy Clyy)*Nbins + bin2]
    '''
    Clpq_tmp = np.array([Clpq[:,0,0], Clpq[:,0,1], Clpq[:,1,1]])
    Clpq_tmp = np.transpose(Clpq_tmp, axes=(0,2,1)) #shape (3 for ClTT, ClTy, Clyy, Nbins, Nsims)
    Clpq_tmp = np.reshape(Clpq_tmp, (inp.Nbins*3, -1))
    cov = np.cov(Clpq_tmp)
    return cov


##############################################
#########      NUMERICAL MLE      ############
##############################################


def ClpqA(Acmb, Atsz, Anoise1, Anoise2, inp, Clpq, best_fits):
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
    Clpq: (N_preserved_comps, N_preserved_comps, Nbins) ndarray containing power spectrum of HILC maps p and q
        (mean over all realizations)
    best_fits: (N_preserved_comps, N_preserved_comps, Nbins) ndarray
        containing best fit sympy expressions to Acmb, Atsz, Anoise1, Anoise2

    RETURNS
    -------
    theory_model: (Nbins, 2, 2) ndarray for ClTT, ClTy, ClyT, and Clyy in terms of parameters

    '''
    theory_model = np.zeros((inp.Nbins, 2, 2))
    A_vec = [Acmb, Atsz, Anoise1, Anoise2]
    for p,q in [(0,0), (0,1), (1,0), (1,1)]:
        for b in range(inp.Nbins):
            theory_model[b,p,q] = call_fit(A_vec, best_fits[p][q][b])*Clpq[p,q,b]
    return theory_model


def neg_lnL(pars, f, inp, sim, Clpq, PScov_sim_Inv, best_fits): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb, Atsz, Anoise1, and Anoise2
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clpq: (Nsims, N_preserved_comps, N_preserved_comps, Nbins) ndarray containing power spectra of maps p and q
    PScov_sim_Inv: (Nbins, Nbins, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing inverse of power spectrum covariance matrix
    best_fits: (N_preserved_comps, N_preserved_comps, Nbins) ndarray
        containing best fit sympy expressions to Acmb, Atsz, Anoise1, Anoise2

    RETURNS
    -------
    neg_log_likelihood: float, negative log likelihood for one simulation, combined over multipole bins
    '''
    index_mapping = {0: (0,0), 1: (0,1), 2: (1,1)} #map pq to p,q or rs to r,s
    Clpq_mean = np.mean(Clpq, axis=0)
    model = f(*pars, inp, Clpq_mean, best_fits)
    data = Clpq[sim]
    neg_log_likelihood = 0.
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for pq in range(3):
                for rs in range(3):
                    p,q = index_mapping[pq]
                    r,s = index_mapping[rs]
                    neg_log_likelihood += 1/2*(model[b1,p,q]-data[p,q,b1])*PScov_sim_Inv[b1,b2,pq,rs]*(model[b2,r,s]-data[r,s,b2])
    return neg_log_likelihood


def acmb_atsz(inp, sim, Clpq, PScov_sim_Inv, best_fits):
    '''
    Maximize likelihood with respect to Acmb and Atsz for one sim

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
    PScov_sim_Inv: (Nbins, Nbins, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing inverse of power spectrum covariance matrix
    best_fits: (N_preserved_comps, N_preserved_comps, Nbins) ndarray
        containing best fit sympy expressions to Acmb, Atsz, Anoise1, Anoise2

    RETURNS
    -------
    best fit Acmb, Atsz, Anoise1, Anoise2 (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start, start, start, start] #acmb_start, atsz_start, anoise1_start, anoise2_start
        res = minimize(neg_lnL, x0 = start_array, args = (ClpqA, inp, sim, Clpq, PScov_sim_Inv, best_fits), method='Nelder-Mead') #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x



def get_MLE_arrays(inp, Clpq, PScov_sim_Inv, best_fits, HILC=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
    PScov_sim_Inv: (Nbins, Nbins, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing inverse of power spectrum covariance matrix
    best_fits: (N_preserved_comps, N_preserved_comps, Nbins) ndarray
        containing best fit sympy expressions to Acmb, Atsz, Anoise1, Anoise2
    HILC: Bool, True is using harmonic ILC pipeline, False if using needlet ILC pipeline

    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation

    '''
    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(acmb_atsz, [(inp, sim, Clpq, PScov_sim_Inv, best_fits) for sim in range(inp.Nsims)])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, 4 for Acmb Atsz Anoise1 Anoise2)
    acmb_array = param_array[:,0]
    atsz_array = param_array[:,1]
    anoise1_array = param_array[:,2]
    anoise2_array = param_array[:,3]
    
    string = 'HILC' if HILC else 'NILC'
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_array_{string}.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_array_{string}.p', 'wb'))
    pickle.dump(anoise1_array, open(f'{inp.output_dir}/anoise1_array_{string}.p', 'wb'))
    pickle.dump(anoise2_array, open(f'{inp.output_dir}/anoise2_array_{string}.p', 'wb'))
    if inp.verbose:
        print(f'created {inp.output_dir}/acmb_array_{string}.p, atsz_array_{string}.p, anoise1_array_{string}.p, anoise2_array_{string}.p', flush=True)
    print('Results from maximum likelihood estimation', flush=True)
    print('----------------------------------------------', flush=True)
    print(f'Acmb = {np.mean(acmb_array)} +/- {np.std(acmb_array)}', flush=True)
    print(f'Atsz = {np.mean(atsz_array)} +/- {np.std(atsz_array)}', flush=True)
    print(f'Anoise1 = {np.mean(anoise1_array)} +/- {np.std(anoise1_array)}', flush=True)
    print(f'Anoise2 = {np.mean(anoise2_array)} +/- {np.std(anoise2_array)}', flush=True)

    return acmb_array, atsz_array, anoise1_array, anoise2_array


###############################
### FISHER MATRIX FORECAST  ###
###############################

def Fisher_inversion(inp, Clpq, PScov_sim_Inv, best_fits):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim2: index0 is total power in Clpq, other indices are power from each component
    PScov_sim_Inv: (Nbins, Nbins, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing 
        inverse of power spectrum covariance matrix
    best_fits: (N_preserved_comps, N_preserved_comps, Nbins) ndarray
        containing best fit sympy expressions to Acmb, Atsz, Anoise1, Anoise2

    RETURNS
    -------
    acmb_std, atsz_std, anoise1_std, anoise2_std: predicted standard deviations of Acmb, etc.
        found by computing the Fisher matrix and inverting
    '''

    Ncomps = 4
    index_mapping = {0: (0,0), 1: (0,1), 2: (1,1)} #map pq to p,q or rs to r,s
    Clpq_mean = np.mean(Clpq, axis=0)
    deriv_vec = np.zeros((Ncomps, 3, inp.Nbins))
    
    for A in range(Ncomps):
        h = 0.0001
        pars_high, pars_low = np.ones(Ncomps), np.ones(Ncomps)
        pars_high[A] += h
        pars_low[A] -= h
        for pq in range(3):
            p,q = index_mapping[pq]
            Clpq_high, Clpq_low = np.zeros(inp.Nbins), np.zeros(inp.Nbins)
            for b in range(inp.Nbins):
                Clpq_high[b] = call_fit(pars_high, best_fits[p,q,b])*Clpq_mean[p,q,b]
                Clpq_low[b] = call_fit(pars_low, best_fits[p,q,b])*Clpq_mean[p,q,b]
            deriv_vec[A,pq] = (Clpq_high-Clpq_low)/(2*h)

    Fisher = np.einsum('Aib,bcij,Bjc->AB', deriv_vec, PScov_sim_Inv, deriv_vec)
    final_cov = np.linalg.inv(Fisher)
    acmb_std = np.sqrt(final_cov[0,0])
    atsz_std = np.sqrt(final_cov[1,1])
    anoise1_std = np.sqrt(final_cov[2,2])
    anoise2_std = np.sqrt(final_cov[3,3])

    print('Results from inverting Fisher matrix', flush=True)
    print('----------------------------------------', flush=True)
    print('Acmb std dev: ', acmb_std, flush=True)
    print('Atsz std dev: ', atsz_std, flush=True)
    print('Anoise1 std dev: ', anoise1_std, flush=True)
    print('Anoise2 std dev: ', anoise2_std, flush=True)
    return acmb_std, atsz_std, anoise1_std, anoise2_std


##############################################
########   MCMC WITH ONE SIMULATION  #########
##############################################

def pos_lnL(pars, f, inp, sim, Clpq, PScov_sim_Inv, best_fits): 
    '''
    Expression for positive log likelihood for one sim

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb, Atsz, Anoise1, and Anoise2
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clpq: (Nsims, N_preserved_comps, N_preserved_comps, Nbins) ndarray containing power spectra of maps p and q
    PScov_sim_Inv: (Nbins, Nbins, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing inverse of power spectrum covariance matrix
    best_fits: (N_preserved_comps, N_preserved_comps, Nbins) ndarray
        containing best fit sympy expressions to Acmb, Atsz, Anoise1, Anoise2

    RETURNS
    -------
    log likelihood for one simulation, combined over multipoles 
    '''
    return -neg_lnL(pars, f, inp, sim, Clpq, PScov_sim_Inv, best_fits)


def MCMC(inp, Clpq, PScov_sim_Inv, best_fits, sim=0):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clpq: (Nsims, N_preserved_comps, N_preserved_comps, Nbins) ndarray containing power spectra of maps p and q
    PScov_sim_Inv: (Nbins, Nbins, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing 
        inverse of power spectrum covariance matrix
    best_fits: (N_preserved_comps, N_preserved_comps, Nbins) ndarray
        containing best fit sympy expressions to Acmb, Atsz, Anoise1, Anoise2
    sim: int, simulation number

    RETURNS
    -------
    acmb_std, atsz_std, anoise1_std, anoise2_std: predicted standard deviations of Acmb, etc.
        found from MCMC
    '''

    np.random.seed(0)
    ndim = 4
    nwalkers = 10
    p0 = np.random.random((nwalkers, ndim))*(1.2-0.8)+0.8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[ClpqA, inp, sim, Clpq, PScov_sim_Inv, best_fits])
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(state, 1000)
    samples = sampler.get_chain() #dimensions (1000, nwalkers, Ncomps=4)
    
    acmb_std = np.mean(np.array([np.std(samples[:,walker,0]) for walker in range(nwalkers)]))
    atsz_std = np.mean(np.array([np.std(samples[:,walker,1]) for walker in range(nwalkers)]))
    anoise1_std = np.mean(np.array([np.std(samples[:,walker,2]) for walker in range(nwalkers)]))
    anoise2_std = np.mean(np.array([np.std(samples[:,walker,3]) for walker in range(nwalkers)]))

    print('Results from MCMC', flush=True)
    print('------------------------------------', flush=True)
    print('Acmb std dev: ', acmb_std, flush=True)
    print('Atsz std dev: ', atsz_std, flush=True)
    print('Anoise1 std dev: ', anoise1_std, flush=True)
    print('Anoise2 std dev: ', anoise2_std, flush=True)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), flush=True)
    return acmb_std, atsz_std, anoise1_std, anoise2_std


##############################################
## COMPARE RESULTS FROM DIFFERENT METHODS  ###
##############################################


def get_all_acmb_atsz(inp, Clpq, env, HILC=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, Nscalings, 2,2,2,2, N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim1: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
              idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim2: idx0 for unscaled CMB, idx1 for scaled CMB
        dim3: idx0 for unscaled ftSZ, idx1 for scaled ftSZ
        dim4: idx0 for unscaled noise90, idx1 for scaled noise90
        dim5: idx0 for unscaled noise150, idx1 for scaled noise150
    env: environment object
    HILC: Bool, True is using harmonic ILC pipeline, False if using needlet ILC pipeline

    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation
    '''
    best_fits = get_parameter_dependence(inp, Clpq[:inp.Nsims_for_fits], env, HILC=HILC)
    Clpq_unscaled = Clpq[:,0,0,0,0,0]

    PScov_sim = get_PScov_sim(inp, Clpq_unscaled)
    PScov_sim_alt_Inv = scipy.linalg.inv(PScov_sim)
    PScov_sim_Inv = np.zeros((inp.Nbins, inp.Nbins, 3, 3))
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for i in range(3):
                for j in range(3):
                    PScov_sim_Inv[b1, b2, i, j] = PScov_sim_alt_Inv[i*inp.Nbins+b1, j*inp.Nbins+b2]
    PScov_sim_Inv *= (inp.Nsims-(inp.Nbins*3)-2)/(inp.Nsims-1) #correction factor from https://arxiv.org/pdf/astro-ph/0608064.pdf

    acmb_array, atsz_array, anoise1_array, anoise2_array = get_MLE_arrays(inp, Clpq_unscaled, PScov_sim_Inv, best_fits, HILC=HILC)
    
    print(flush=True)
    Fisher_inversion(inp, Clpq_unscaled, PScov_sim_Inv, best_fits)

    print(flush=True)
    MCMC(inp, Clpq_unscaled, PScov_sim_Inv, best_fits, sim=0)
   
    return acmb_array, atsz_array, anoise1_array, anoise2_array