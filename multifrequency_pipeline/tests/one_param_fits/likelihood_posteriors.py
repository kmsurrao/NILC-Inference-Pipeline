import sys
sys.path.append('../..')
sys.path.append('../../../shared')
import numpy as np
import multiprocessing as mp
import emcee
import scipy
from scipy.optimize import minimize


##############################################
#####  POWER SPECTRUM COVARIANCE MATRIX  #####
##############################################

def get_PScov_sim(Clij):
    '''
    ARGUMENTS
    ---------
    Clij: (Nsims, Nbins) ndarray containing power spectrum of component
    
    RETURNS
    -------
    cov: (Nbins, Nbins) ndarray containing covariance matrix
    '''
    cov = np.cov(Clij.T)
    return cov


##############################################
#########      NUMERICAL MLE      ############
##############################################

def ClijA(A1, Clij):
    '''
    Model for theoretical spectra Clpq including amplitude parameters

    ARGUMENTS
    ---------
    USED BY MINIMIZER
    A1: float, scaling parameter for power spectrum of first component

    CONSTANT ARGS
    Clij: (Nbins, ) ndarray containing power spectrum of component

    RETURNS
    -------
    (Nbins, ) ndarray

    '''
    return A1*Clij


def neg_lnL(pars, f, inp, sim, Clij_all_sims, PScov_sim_Inv): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of amplitude parameters
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij_all_sims: (Nsims, Nbins) ndarray containing power spectrum of component for all sims
    PScov_sim_Inv: (Nbins, Nbins) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    negative log likelihood for one simulation, combined over multipoles 
    '''
    model = f(*pars, np.mean(Clij_all_sims, axis=0))
    Clijd = Clij_all_sims[sim]
    return np.sum([[1/2* \
        (model[l1]-Clijd[l1])*PScov_sim_Inv[l1,l2]*(model[l2]-Clijd[l2])
        for l1 in range(inp.Nbins)] for l2 in range(inp.Nbins)]) 
    


def acmb_atsz_numerical(inp, sim, Clij_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to A1, A2 for one sim using numerical minimization routine

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij_all_sims: (Nsims, Nbins) ndarray containing power spectrum of component for all sims
    PScov_sim_Inv: (Nbins, Nbins) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit A1, A2 (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start] #a1_start
        res = minimize(neg_lnL, x0 = start_array, args = (ClijA, inp, sim, Clij_all_sims, PScov_sim_Inv), method='Nelder-Mead', bounds=None) #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x


##############################################
#########       ANALYTIC MLE      ############
##############################################

def acmb_atsz_analytic(inp, sim, Clij_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to A1, A2 for one sim analytically 

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit A1 (float)

    INDEX MAPPING IN EINSUM
    -----------------------
    bin1-->l, bin2-->m

    '''
    Clijd = Clij_all_sims[sim]
    Clij = np.mean(Clij_all_sims, axis=0) #shape (Nbins, )
    F = np.einsum('l,lm,m->', Clij, PScov_sim_Inv, Clij)
    F_inv = 1/F
    return np.einsum(',l,lm,m->', F_inv, Clij, PScov_sim_Inv, Clijd)


##############################################
########  ARRAYS OF MLE ESTIMATES  ###########
##############################################

def get_MLE_arrays(inp, Clij_all_sims, PScov_sim_Inv, use_analytic=True):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij_all_sims: (Nsims, Nbins) ndarray containing power spectrum of component
    PScov_sim_Inv: (Nbins, Nbins) ndarray;
        contains inverse power spectrum covariance matrix in tensor form
    use_analytic: Bool, whether to use analytic MLEs for parameters. If False, compute them with numerical minimization routine

    RETURNS
    -------
    a1_array: array of length Nsims containing best fit A1 for each simulation
    '''

    func = acmb_atsz_analytic if use_analytic else acmb_atsz_numerical
    string = 'analytic' if use_analytic else 'numerical'
    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(func, [(inp, sim, Clij_all_sims, PScov_sim_Inv) for sim in range(len(Clij_all_sims))])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, )
    a1_array = param_array
    print(f'Results from maximum likelihood estimation using {string} MLEs', flush=True)
    print('---------------------------------------------------------------', flush=True)
    print(f'A1 = {np.mean(a1_array)} +/- {np.std(a1_array)}', flush=True)
    return a1_array



###############################
### FISHER MATRIX INVERSION ###
###############################


def Fisher_inversion(Clij, PScov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    Clij: (Nsims, Nbins) ndarray containing power spectrum of component
    PScov_sim_Inv: (Nbins, Nbins) ndarray;
        contains inverse power spectrum covariance matrix in tensor form

    RETURNS
    -------
    a1_std: predicted standard deviations of A1 found by computing the Fisher matrix and inverting
    '''

    Clij_mean = np.mean(Clij, axis=0)
    deriv_vec = Clij_mean
    Fisher = np.einsum('b,bc,c->', deriv_vec, PScov_sim_Inv, deriv_vec)
    final_cov = 1/Fisher
    a1_std = np.sqrt(final_cov)

    print('Results from inverting Fisher matrix', flush=True)
    print('----------------------------------------', flush=True)
    print('A1 std dev: ', a1_std, flush=True)
    return a1_std


##############################################
########   MCMC WITH ONE SIMULATION  #########
##############################################

def pos_lnL(pars, f, inp, sim, Clij_all_sims, PScov_sim_Inv): 
    '''
    Expression for positive log likelihood for one sim

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of amplitude parameters
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij_all_sims: (Nsims, Nbins) ndarray containing power spectrum of component
    PScov_sim_Inv: (Nbins, Nbins) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    log likelihood for one simulation, combined over multipoles 
    '''
    return -neg_lnL(pars, f, inp, sim, Clij_all_sims, PScov_sim_Inv)


def MCMC(inp, Clij_all_sims, PScov_sim_Inv, sim=0):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij_all_sims: (Nsims, Nbins) ndarray containing power spectrum of component
    PScov_sim_Inv: (Nbins, Nbins) ndarray;
        contains inverse power spectrum covariance matrix in tensor form
    sim: int, simulation number to use for MCMC

    RETURNS
    -------
    a1_std: predicted standard deviation of A1 found from MCMC
    '''

    np.random.seed(0)
    ndim = 1
    nwalkers = 10
    p0 = np.random.random((nwalkers, ndim))*(1.2-0.8)+0.8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[ClijA, inp, sim, Clij_all_sims, PScov_sim_Inv])
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(state, 1000)
    samples = sampler.get_chain() #dimensions (1000, nwalkers, Ncomps=2)
    
    a1_std = np.mean(np.array([np.std(samples[:,walker,0]) for walker in range(nwalkers)]))

    print('Results from MCMC', flush=True)
    print('------------------------------------', flush=True)
    print('A1 std dev: ', a1_std, flush=True)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), flush=True)
    return a1_std


############################################
######## ANALYTIC COVARIANCE OF MLE   ######
############################################

def cov_of_MLE_analytic(Clij_all_sims, PScov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    Clij_all_sims: (Nsims, Nbins) ndarray containing power spectrum of component
    PScov_sim_Inv: (Nbins, Nbins) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    inverse of Fisher matrix

    INDEX MAPPING IN EINSUM
    -----------------------
    bin1-->l, bin2-->m

    '''
    Clij = np.mean(Clij_all_sims, axis=0)
    F = np.einsum('l,lm,m->', Clij, PScov_sim_Inv, Clij)
    F_inv = 1/F
    print('Results from Analytic Covariance of MLEs', flush=True)
    print('------------------------------------', flush=True)
    print('A1 std dev: ', np.sqrt(F_inv), flush=True)
    return F_inv 


##############################################
## COMPARE RESULTS FROM DIFFERENT METHODS  ###
##############################################


def get_all_acmb_atsz(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij: (Nsims, Nbins) ndarray containing power spectrum of component

    RETURNS
    -------
    a1_array: array of length Nsims containing best fit A1 for each simulation
    '''

    PScov_sim = get_PScov_sim(Clij)
    PScov_sim_Inv = scipy.linalg.inv(PScov_sim)
    if not inp.use_Gaussian_cov:
        PScov_sim_Inv *= (inp.Nsims-(inp.Nbins*3)-2)/(inp.Nsims-1) #correction factor from https://arxiv.org/pdf/astro-ph/0608064.pdf

    Clij_all_sims = Clij

    a1_array = get_MLE_arrays(inp, Clij_all_sims, PScov_sim_Inv, use_analytic=True)
    print(flush=True)
    a1_array = get_MLE_arrays(inp, Clij_all_sims, PScov_sim_Inv, use_analytic=False)
    
    print(flush=True)
    Fisher_inversion(Clij, PScov_sim_Inv)

    print(flush=True)
    MCMC(inp, Clij_all_sims, PScov_sim_Inv, sim=0)

    print(flush=True)
    cov_of_MLE_analytic(Clij_all_sims, PScov_sim_Inv)

    return a1_array




