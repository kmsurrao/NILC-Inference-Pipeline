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
from getdist import MCSamples
import itertools
from sym_reg import call_fit, get_parameter_dependence
from utils import get_naming_str

##############################################
#####  POWER SPECTRUM COVARIANCE MATRIX  #####
##############################################

def get_PScov_sim(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, Ncomps, Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
    
    RETURNS
    -------
    cov: (Ncomps**2*Nbins, Ncomps**2*Nbins) ndarray containing covariance matrix Cov_{pq,rs}
        index as cov[(0 to Ncomps**2)*Nbins + bin1, (0 to Ncomps**2)*Nbins + bin2]
    '''
    Ncomps = len(inp.comps)
    Clpq_tmp = np.zeros((Ncomps**2, Clpq.shape[0], Clpq.shape[-1]), dtype=np.float32) #shape (Ncomps**2, Nsims, Nbins)
    idx = 0
    for p in range(Ncomps):
        for q in range(Ncomps):
            Clpq_tmp[idx] = Clpq[:,p,q]
            idx += 1
    Clpq_tmp = np.transpose(Clpq_tmp, axes=(0,2,1)) #shape (Ncomps**2, Nbins, Nsims)
    Clpq_tmp = np.reshape(Clpq_tmp, (inp.Nbins*Ncomps**2, -1))
    cov = np.cov(Clpq_tmp)
    return cov


##############################################
#########      NUMERICAL MLE      ############
##############################################


def ClpqA(inp, Clpq, best_fits, *pars):
    '''
    Model for theoretical spectra Clpq including amplitude parameters

    ARGUMENTS
    ---------
    CONSTANT ARGS
    inp: Info object containing input parameter specifications
    Clpq: (Ncomps, Ncomps, Nbins) ndarray containing power spectrum of HILC maps p and q
        (mean over all realizations)
    best_fits: (Ncomps, Ncomps, Nbins) ndarray
        containing best fit sympy expressions in terms of parameters
    
    USED BY MINIMIZER
    pars: floats, scaling parameters for component power spectra

    RETURNS
    -------
    theory_model: (Ncomps, Ncomps, Nbins) ndarray for HILC spectra in terms of A_y and A_z parameters

    '''
    Ncomps = len(inp.comps)
    theory_model = np.zeros((Ncomps, Ncomps, inp.Nbins))
    for p in range(Ncomps):
        for q in range(Ncomps):
            for b in range(inp.Nbins):
                theory_model[p,q,b] = call_fit(pars, best_fits[p][q][b])*Clpq[p,q,b]
    return theory_model


def neg_lnL(pars, f, inp, sim, Clpq, PScov_sim_Inv, best_fits): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of parameters
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clpq: (Nsims, Ncomps, Ncomps, Nbins) ndarray containing power spectra of maps p and q
    PScov_sim_Inv: (Nbins, Nbins, Ncomps, Ncomps, Ncomps, Ncomps) ndarray
        containing inverse of power spectrum covariance matrix
    best_fits: (Ncomps, Ncomps, Nbins) ndarray
        containing best fit sympy expressions in terms of parameters

    RETURNS
    -------
    neg_log_lkl: float, negative log likelihood for one simulation, combined over multipole bins
    '''
    model = f(inp, np.mean(Clpq, axis=0), best_fits, *pars)
    Clpqd = Clpq[sim]
    neg_log_lkl = 1/2*np.einsum('ijb,bcijkl,klc->', model-Clpqd, PScov_sim_Inv, model-Clpqd)
    return neg_log_lkl


def a_vec(inp, sim, Clpq, PScov_sim_Inv, best_fits):
    '''
    Maximize likelihood with respect to parameters for one sim

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clpq: (Nsims, Ncomps, Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
    PScov_sim_Inv: (Nbins, Nbins, Ncomps, Ncomps, Ncomps, Ncomps) ndarray
        containing inverse of power spectrum covariance matrix
    best_fits: (Ncomps, Ncomps, Nbins) ndarray
        containing best fit sympy expressions in terms of parameters

    RETURNS
    -------
    best fit parameters (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start]*len(inp.comps)
        res = minimize(neg_lnL, x0 = start_array, args = (ClpqA, inp, sim, Clpq, PScov_sim_Inv, best_fits), method='Nelder-Mead') #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x



def get_MLE_arrays(inp, Clpq, PScov_sim_Inv, best_fits, HILC=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, Ncomps, Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
    PScov_sim_Inv: (Nbins, Nbins, Ncomps, Ncomps, Ncomps, Ncomps) ndarray
        containing inverse of power spectrum covariance matrix
    best_fits: (Ncomps, Ncomps, Nbins) ndarray
        containing best fit sympy expressions in terms of parameters
    HILC: Bool, True is using harmonic ILC pipeline, False if using needlet ILC pipeline

    RETURNS
    -------
    a_array: (Ncomps, Nsims) ndarray containing best fit parameters for each simulation

    '''
    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(a_vec, [(inp, sim, Clpq, PScov_sim_Inv, best_fits) for sim in range(inp.Nsims)])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, Ncomps)
    a_array = param_array.T 
    pipeline = 'HILC' if HILC else 'NILC'
    naming_str = get_naming_str(inp, pipeline)
    pickle.dump(a_array, open(f'{inp.output_dir}/posteriors/a_array_{naming_str}.p', 'wb'))
    print(f'created {inp.output_dir}/posteriors/a_array_{naming_str}.p', flush=True)
    print('Results from maximum likelihood estimation', flush=True)
    print('----------------------------------------------', flush=True)
    names = [f'A{comp}' for comp in inp.comps]
    samples_MC = MCSamples(samples=list(a_array), names = names, labels = names)
    for par in names:
        print(samples_MC.getInlineLatex(par,limit=1), flush=True)
    return a_array


###############################
### FISHER MATRIX FORECAST  ###
###############################

def Fisher_inversion(inp, Clpq, PScov_sim_Inv, best_fits):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, Ncomps, Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim2: index0 is total power in Clpq, other indices are power from each component
    PScov_sim_Inv: (Nbins, Nbins, Ncomps, Ncomps, Ncomps, Ncomps) ndarray 
        containing inverse of power spectrum covariance matrix
    best_fits: (Ncomps, Ncomps, Nbins) ndarray
        containing best fit sympy expressions in terms of parameters

    RETURNS
    -------
    a_std_arr: list of length Ncomps containing predicted standard deviations of each parameter
        found by computing the Fisher matrix and inverting
    '''

    Ncomps = len(inp.comps)
    Clpq_mean = np.mean(Clpq, axis=0)
    deriv_vec = np.zeros((Ncomps, Ncomps, Ncomps, inp.Nbins))
    
    for A in range(Ncomps):
        h = 0.0001
        pars_high, pars_low = np.ones(Ncomps), np.ones(Ncomps)
        pars_high[A] += h
        pars_low[A] -= h
        for p in range(Ncomps):
            for q in range(Ncomps):
                Clpq_high, Clpq_low = np.zeros(inp.Nbins), np.zeros(inp.Nbins)
                for b in range(inp.Nbins):
                    Clpq_high[b] = call_fit(pars_high, best_fits[p][q][b])*Clpq_mean[p,q,b]
                    Clpq_low[b] = call_fit(pars_low, best_fits[p][q][b])*Clpq_mean[p,q,b]
                deriv_vec[A,p,q] = (Clpq_high-Clpq_low)/(2*h)

    Fisher = np.einsum('Aijb,bcijkl,Bklc->AB', deriv_vec, PScov_sim_Inv, deriv_vec)
    final_cov = np.linalg.inv(Fisher)
    a_std_arr = [np.sqrt(final_cov[i,i]) for i in range(len(final_cov))]

    print('Results from inverting Fisher matrix', flush=True)
    print('----------------------------------------', flush=True)
    for c, comp in enumerate(inp.comps):
        print(f'A{comp} std dev: ', a_std_arr[c], flush=True)
    return a_std_arr


##############################################
########   MCMC WITH ONE SIMULATION  #########
##############################################

def pos_lnL(pars, f, inp, sim, Clpq, PScov_sim_Inv, best_fits): 
    '''
    Expression for positive log likelihood for one sim

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of parameters
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clpq: (Nsims, Ncomps, Ncomps, Nbins) ndarray containing power spectra of maps p and q
    PScov_sim_Inv: (Nbins, Nbins, Ncomps, Ncomps, Ncomps, Ncomps) ndarray
        containing inverse of power spectrum covariance matrix
    best_fits: (Ncomps, Ncomps, Nbins) ndarray
        containing best fit sympy expressions in terms of parameters

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
    Clpq: (Nsims, Ncomps, Ncomps, Nbins) ndarray containing power spectra of maps p and q
    PScov_sim_Inv: (Nbins, Nbins, Ncomps, Ncomps, Ncomps, Ncomps) ndarray
        containing inverse of power spectrum covariance matrix
    best_fits: (Ncomps, Ncomps, Nbins) ndarray
        containing best fit sympy expressions in terms of parameters
    sim: int, simulation number

    RETURNS
    -------
    None
    '''

    np.random.seed(0)
    ndim = len(inp.comps)
    nwalkers = 10
    p0 = np.random.random((nwalkers, ndim))*(1.2-0.8)+0.8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[ClpqA, inp, sim, Clpq, PScov_sim_Inv, best_fits])
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(state, 1000)
    samples = sampler.get_chain(flat=True) #dimensions (Ncomps, 1000*nwalkers)
    print('Results from MCMC', flush=True)
    print('------------------------------------', flush=True)
    names = [f'A{comp}' for comp in inp.comps]
    samples_MC = MCSamples(samples=samples, names = names, labels = names)
    for par in names:
        print(samples_MC.getInlineLatex(par,limit=1), flush=True)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), flush=True)
    return None


##############################################
## COMPARE RESULTS FROM DIFFERENT METHODS  ###
##############################################


def get_all_a_vec(inp, Clpq, HILC=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, Nscalings, 2**Ncomps, Ncomps, Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim1: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
              idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim2: indices correspond to different combinations of scaled and unscaled components
    HILC: Bool, True is using harmonic ILC pipeline, False if using needlet ILC pipeline

    RETURNS
    -------
    a_array: (Ncomps, Nsims) ndarray of containing best fit parameters for each simulation
    '''
    Ncomps = len(inp.comps)
    best_fits = get_parameter_dependence(inp, Clpq[:inp.Nsims_for_fits], HILC=HILC)
    comp_scalings = [list(i) for i in itertools.product([0, 1], repeat=Ncomps)]
    Clpq_unscaled = Clpq[:,0,comp_scalings.index([0]*Ncomps)]

    PScov_sim = get_PScov_sim(inp, Clpq_unscaled)
    PScov_sim_alt_Inv = scipy.linalg.inv(PScov_sim)
    PScov_sim_Inv = np.zeros((inp.Nbins, inp.Nbins, Ncomps, Ncomps, Ncomps, Ncomps), dtype=np.float32)
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for pq, (p,q) in enumerate(list(itertools.product(range(Ncomps), range(Ncomps)))):
                for rs, (r,s) in enumerate(list(itertools.product(range(Ncomps), range(Ncomps)))):
                    PScov_sim_Inv[b1,b2,p,q,r,s] = PScov_sim_alt_Inv[pq*inp.Nbins+b1, rs*inp.Nbins+b2]
    PScov_sim_Inv *= (inp.Nsims-(inp.Nbins*3)-2)/(inp.Nsims-1) #correction factor from https://arxiv.org/pdf/astro-ph/0608064.pdf

    a_array = get_MLE_arrays(inp, Clpq_unscaled, PScov_sim_Inv, best_fits, HILC=HILC)
    
    print(flush=True)
    Fisher_inversion(inp, Clpq_unscaled, PScov_sim_Inv, best_fits)

    print(flush=True)
    MCMC(inp, Clpq_unscaled, PScov_sim_Inv, best_fits, sim=0)
   
    return a_array