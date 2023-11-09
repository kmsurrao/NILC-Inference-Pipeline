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
    inp: Info object containing input paramter specifications
    Clij: (Nsims, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    
    RETURNS
    -------
    cov: (3*Nbins, 3*Nbins) ndarray containing covariance matrix Cov_{ij b1, kl b2}
        index as cov[(0-2 for Cl00 Cl01 Cl11)*Nbins + bin1, (0-2 for Cl00 Cl01 Cl11)*Nbins + bin2]
    '''
    Clij_tmp = Clij[:,0] #shape (Nsims, Nbins)
    cov = np.cov(Clij_tmp.T)
    return cov


##############################################
#########      NUMERICAL MLE      ############
##############################################

def ClijA(A1, A2, Clij):
    '''
    Model for theoretical spectra Clpq including amplitude parameters

    ARGUMENTS
    ---------
    USED BY MINIMIZER
    A1: float, scaling parameter for power spectrum of first component
    A2: float, scaling parameter for power spectrum of second component

    CONSTANT ARGS
    inp: Info object containing input parameter specifications
    Clij: (1+Ncomps, Nbins) ndarray containing contribution of components to Clij

    RETURNS
    -------
    (Nbins, ) ndarray

    '''
    return A1*Clij[1] + A2*Clij[2]


def neg_lnL(pars, f, inp, sim, Clij_all_sims, PScov_sim_Inv): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of amplitude parameters
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    negative log likelihood for one simulation, combined over multipoles 
    '''
    Clij = np.mean(Clij_all_sims, axis=0)
    model = f(*pars, Clij)
    Clijd = Clij_all_sims[sim,0]
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
    Clij_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit A1, A2 (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start, start] #a1_start, a2_start
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
    Clij_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit A1, A2 (floats)

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, bin1-->l, bin2-->m

    '''
    Clijd = Clij_all_sims[sim,0]
    Clij = np.mean(Clij_all_sims, axis=0)[1:]
    F = np.einsum('al,lm,bm->ab', Clij, PScov_sim_Inv, Clij)
    F_inv = np.linalg.inv(F)
    return np.einsum('ab,bl,lm,m->a', F_inv, Clij, PScov_sim_Inv, Clijd)


##############################################
########  ARRAYS OF MLE ESTIMATES  ###########
##############################################

def get_MLE_arrays(inp, Clij_all_sims, PScov_sim_Inv, use_analytic=True):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins) ndarray;
        contains inverse power spectrum covariance matrix in tensor form
    use_analytic: Bool, whether to use analytic MLEs for parameters. If False, compute them with numerical minimization routine

    RETURNS
    -------
    a1_array: array of length Nsims containing best fit A1 for each simulation
    a2_array: array of length Nsims containing best fit A2 for each simulation
    '''

    func = acmb_atsz_analytic if use_analytic else acmb_atsz_numerical
    string = 'analytic' if use_analytic else 'numerical'
    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(func, [(inp, sim, Clij_all_sims, PScov_sim_Inv) for sim in range(len(Clij_all_sims))])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, 3 for A1 A2)
    a1_array = param_array[:,0]
    a2_array = param_array[:,1]
    

    final_cov = np.cov(np.array([a1_array, a2_array]))
    print(f'Results from maximum likelihood estimation using {string} MLEs', flush=True)
    print('---------------------------------------------------------------', flush=True)
    print(f'A1 = {np.mean(a1_array)} +/- {np.sqrt(final_cov[0,0])}', flush=True)
    print(f'A2 = {np.mean(a2_array)} +/- {np.sqrt(final_cov[1,1])}', flush=True)

    return a1_array, a2_array



###############################
### FISHER MATRIX FORECAST  ###
###############################


def Fisher_inversion(inp, Clij, PScov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij: (Nsims, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    PScov_sim_Inv: (Nbins, Nbins) ndarray;
        contains inverse power spectrum covariance matrix in tensor form

    RETURNS
    -------
    a1_std, a2_std: predicted standard deviations of A1, A2
        found by computing the Fisher matrix and inverting
    '''

    Ncomps = 2
    Clij_mean = np.mean(Clij, axis=0)
    deriv_vec = np.zeros((Ncomps, inp.Nbins))
    for A in range(Ncomps):
        deriv_vec[A] = Clij_mean[1+A]
    Fisher = np.einsum('Ab,bc,Bc->AB', deriv_vec, PScov_sim_Inv, deriv_vec)
    final_cov = np.linalg.inv(Fisher)
    a1_std = np.sqrt(final_cov[0,0])
    a2_std = np.sqrt(final_cov[1,1])

    print('Results from inverting Fisher matrix', flush=True)
    print('----------------------------------------', flush=True)
    print('A1 std dev: ', a1_std, flush=True)
    print('A2 std dev: ', a2_std, flush=True)
    return a1_std, a2_std


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
    Clij_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
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
    Clij_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray;
        contains inverse power spectrum covariance matrix in tensor form
    sim: int, simulation number to use for MCMC

    RETURNS
    -------
    a1_std, a2_std: predicted standard deviations of A1, A2 found from MCMC
    '''

    np.random.seed(0)
    ndim = 2
    nwalkers = 10
    p0 = np.random.random((nwalkers, ndim))*(1.2-0.8)+0.8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[ClijA, inp, sim, Clij_all_sims, PScov_sim_Inv])
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(state, 1000)
    samples = sampler.get_chain() #dimensions (1000, nwalkers, Ncomps=2)
    
    a1_std = np.mean(np.array([np.std(samples[:,walker,0]) for walker in range(nwalkers)]))
    a2_std = np.mean(np.array([np.std(samples[:,walker,1]) for walker in range(nwalkers)]))

    print('Results from MCMC', flush=True)
    print('------------------------------------', flush=True)
    print('A1 std dev: ', a1_std, flush=True)
    print('A2 std dev: ', a2_std, flush=True)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), flush=True)
    return a1_std, a2_std


############################################
######## ANALYTIC COVARIANCE OF MLE   ######
############################################

def cov_of_MLE_analytic(Clij_all_sims, PScov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    Clij_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    inverse of Fisher matrix

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, bin1-->l, bin2-->m

    '''
    Clij_mean = np.mean(Clij_all_sims, axis=0)[1:]
    F = np.einsum('al,lm,bm->ab', Clij_mean, PScov_sim_Inv, Clij_mean)
    F_inv = np.linalg.inv(F)
    print('Results from Analytic Covariance of MLEs', flush=True)
    print('------------------------------------', flush=True)
    print('A1 std dev: ', np.sqrt(F_inv[0,0]), flush=True)
    print('A2 std dev: ', np.sqrt(F_inv[1,1]), flush=True)
    return F_inv 


##############################################
## COMPARE RESULTS FROM DIFFERENT METHODS  ###
##############################################


def get_all_acmb_atsz(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij: (Nsims, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j

    RETURNS
    -------
    a1_array: array of length Nsims containing best fit A1 for each simulation
    a2_array: array of length Nsims containing best fit A2 for each simulation
    '''

    PScov_sim = get_PScov_sim(Clij)
    PScov_sim_Inv = scipy.linalg.inv(PScov_sim)
    PScov_sim_Inv *= (inp.Nsims-(inp.Nbins*3)-2)/(inp.Nsims-1) #correction factor from https://arxiv.org/pdf/astro-ph/0608064.pdf

    Clij_all_sims = Clij

    a1_array, a2_array = get_MLE_arrays(inp, Clij_all_sims, PScov_sim_Inv, use_analytic=True)
    print(flush=True)
    a1_array, a2_array = get_MLE_arrays(inp, Clij_all_sims, PScov_sim_Inv, use_analytic=False)
    
    print(flush=True)
    Fisher_inversion(inp, Clij, PScov_sim_Inv)

    print(flush=True)
    MCMC(inp, Clij_all_sims, PScov_sim_Inv, sim=0)

    print(flush=True)
    cov_of_MLE_analytic(Clij_all_sims, PScov_sim_Inv)

    return a1_array, a2_array




