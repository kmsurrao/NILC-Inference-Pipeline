import numpy as np
import pickle
import emcee
import itertools
import scipy
from scipy.optimize import minimize
import multiprocessing as mp
from getdist import MCSamples
import sys
sys.path.append('../shared')
from utils import get_naming_str

##############################################
#####  POWER SPECTRUM COVARIANCE MATRIX  #####
##############################################

def get_PScov_sim(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clij: (Nsims, Nfreqs, Nfreqs, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    
    RETURNS
    -------
    cov: (Nfreqs**2*Nbins, Nfreqs**2*Nbins) ndarray containing covariance matrix Cov_{ij b1, kl b2}
        index as cov[(0 to Nfreqs**2) * Nbins + bin1, (0 to Nfreqs**2) * Nbins + bin2],
    '''
    Clij = Clij[:,:,:,0] #shape (Nsims, Nfreqs, Nfreqs, Nbins)
    Nfreqs = len(inp.freqs)
    Clij_tmp = np.zeros((Nfreqs**2, Clij.shape[0], Clij.shape[-1]), dtype=np.float32) #shape (Nfreqs**2, Nsims, Nbins)
    idx = 0
    for i in range(Nfreqs):
        for j in range(Nfreqs):
            Clij_tmp[idx] = Clij[:,i,j]
            idx += 1
    Clij_tmp = np.transpose(Clij_tmp, axes=(0,2,1)) #shape (Nfreqs**2, Nbins, Nsims)
    Clij_tmp = np.reshape(Clij_tmp, (inp.Nbins*Nfreqs**2, -1))
    cov = np.cov(Clij_tmp)
    return cov


##############################################
#########      NUMERICAL MLE      ############
##############################################

def ClijA(Clij, *pars):
    '''
    Model for theoretical spectra Clij including amplitude parameters

    ARGUMENTS
    ---------
    CONSTANT ARGS
    Clij: (Nfreqs, Nfreqs, 1+Ncomps, Nbins) ndarray containing contribution of components to Clij

    USED BY MINIMIZER
    Acomp1, etc.: float, scaling parameter for comp1, etc. power spectrum

    RETURNS
    -------
    (Nbins, Nfreqs, Nfreqs) ndarray containing theoretical model given some paramters Acomp1, etc.

    '''
    return np.einsum('a,ijab->ijb', np.array(pars), Clij[:,:,1:])


def neg_lnL(pars, f, Clij_all_sims, PScov_sim_Inv, sim=None): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of amplitude parameters
    Clij_all_sims: (Nsims, Nfreqs, Nfreqs, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, Nfreqs, Nfreqs, Nfreqs, Nfreqs) ndarray containing inverse of power spectrum covariance matrix
    sim: int, simulation number. If None, takes mean over all sims instead of using a single sim

    RETURNS
    -------
    neg_log_lkl: float, negative log likelihood for one simulation, combined over multipoles 

    INDEX MAPPING IN EINSUM
    -----------------------
    bin1, bin2 = b, c
    frequencies: i,j,k,l
    '''
    model = f(np.mean(Clij_all_sims, axis=0), *pars)
    if sim is None:
        Clijd = np.mean(Clij_all_sims[:,:,:,0], axis=0)
    else:
        Clijd = Clij_all_sims[sim, :, :, 0]
    neg_log_lkl = 1/2*np.einsum('ijb,bcijkl,klc->', model-Clijd, PScov_sim_Inv, model-Clijd)
    return neg_log_lkl


def a_vec_numerical(inp, sim, Clij_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acomp1, etc. for one sim using numerical minimization routine

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij_all_sims: (Nsims, Nfreqs, Nfreqs, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, Nfreqs, Nfreqs, Nfreqs, Nfreqs) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit Acomp1, etc. (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start]*len(inp.comps)
        res = minimize(neg_lnL, x0 = start_array, args = (ClijA, Clij_all_sims, PScov_sim_Inv, sim), method='Nelder-Mead', bounds=None) #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x


##############################################
#########       ANALYTIC MLE      ############
##############################################

def a_vec_analytic(inp, sim, Clij_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb, Atsz for one sim analytically 

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij_all_sims: (Nsims, Nfreqs, Nfreqs, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, Nfreqs, Nfreqs, Nfreqs, Nfreqs) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit Acomp1, etc. (floats)

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, bin1-->l, bin2-->m, frequencies: i,j,k,n

    '''
    Clijd = Clij_all_sims[sim,:,:,0]
    Clij = np.mean(Clij_all_sims, axis=0)[:,:,1:]
    F = np.einsum('ijal,lmijkn,knbm->ab', Clij, PScov_sim_Inv, Clij)
    F_inv = np.linalg.inv(F)
    return np.einsum('ab,ijbl,lmijkn,knm->a', F_inv, Clij, PScov_sim_Inv, Clijd)


##############################################
########  ARRAYS OF MLE ESTIMATES  ###########
##############################################

def get_MLE_arrays(inp, Clij_all_sims, PScov_sim_Inv, use_analytic=True):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij_all_sims: (Nsims, Nfreqs, Nfreqs, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, Nfreqs, Nfreqs, Nfreqs, Nfreqs) ndarray;
        contains inverse power spectrum covariance matrix in tensor form
    use_analytic: Bool, whether to use analytic MLEs for parameters. If False, compute them with numerical minimization routine

    RETURNS
    -------
    a_array: (Ncomps, Nsims) ndarray containing best fit parameters for each simulation
    '''

    func = a_vec_analytic if use_analytic else a_vec_numerical
    string = 'analytic' if use_analytic else 'numerical'
    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(func, [(inp, sim, Clij_all_sims, PScov_sim_Inv) for sim in range(len(Clij_all_sims))])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, Ncomps)
    a_array = param_array.T #shape (Ncomps, Nsims)
    if not use_analytic:
        naming_str = get_naming_str(inp, 'multifrequency')
        pickle.dump(a_array, open(f'{inp.output_dir}/posteriors/a_array_{naming_str}.p', 'wb'))
        print(f'created {inp.output_dir}/posteriors/a_array_{naming_str}.p', flush=True)
    print(f'Results from maximum likelihood estimation using {string} MLEs', flush=True)
    print('---------------------------------------------------------------', flush=True)
    names = [f'A{comp}' for comp in inp.comps]
    samples_MC = MCSamples(samples=list(a_array), names = names, labels = names)
    for par in names:
        print(samples_MC.getInlineLatex(par,limit=1), flush=True)

    return a_array



###############################
### FISHER MATRIX FORECAST  ###
###############################


def Fisher_inversion(inp, Clij, PScov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij: (Nsims, Nfreqs, Nfreqs, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    PScov_sim_Inv: (Nbins, Nbins, Nfreqs, Nfreqs, Nfreqs, Nfreqs) ndarray;
        contains inverse power spectrum covariance matrix in tensor form

    RETURNS
    -------
    a_std_arr: list of length Ncomps containing predicted standard deviations of each parameter
        found by computing the Fisher matrix and inverting
    '''

    Ncomps = len(inp.comps)
    Nfreqs = len(inp.freqs)
    Clij_mean = np.mean(Clij, axis=0)
    deriv_vec = np.zeros((Ncomps, Nfreqs, Nfreqs, inp.Nbins))
    for A in range(Ncomps):
        for i in range(Nfreqs):
            for j in range(Nfreqs):
                deriv_vec[A,i,j] = Clij_mean[i,j,1+A]
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

def pos_lnL(pars, f, Clij_all_sims, PScov_sim_Inv): 
    '''
    Expression for positive log likelihood for one sim

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of amplitude parameters
    Clij_all_sims: (Nsims, Nfreqs, Nfreqs, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, Nfreqs, Nfreqs, Nfreqs, Nfreqs) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    log likelihood for one simulation, combined over multipoles 
    '''
    return -neg_lnL(pars, f, Clij_all_sims, PScov_sim_Inv, sim=None)


def MCMC(inp, Clij_all_sims, PScov_sim_Inv, sim=0):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    Clij_all_sims: (Nsims, Nfreqs, Nfreqs, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, Nfreqs, Nfreqs, Nfreqs, Nfreqs) ndarray;
        contains inverse power spectrum covariance matrix in tensor form
    sim: int, simulation number to use for MCMC

    RETURNS
    -------
    None
    '''

    np.random.seed(0)
    ndim = len(inp.comps)
    nwalkers = 10
    p0 = np.random.random((nwalkers, ndim))*(1.2-0.8)+0.8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[ClijA, Clij_all_sims, PScov_sim_Inv])
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


def get_all_a_vec(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij: (Nsims, Nfreqs, Nfreqs, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j

    RETURNS
    -------
    a_array: (Ncomps, Nsims) ndarray containing best fit parameters for each simulation
    '''

    PScov_sim = get_PScov_sim(inp, Clij)
    PScov_sim_alt_Inv = scipy.linalg.inv(PScov_sim)
    assert np.allclose(np.matmul(PScov_sim, PScov_sim_alt_Inv), np.eye(len(PScov_sim)), rtol=1.e-3, atol=1.e-3), "PS covmat inversion failed"
    Nfreqs = len(inp.freqs)
    PScov_sim_Inv = np.zeros((inp.Nbins, inp.Nbins, Nfreqs, Nfreqs, Nfreqs, Nfreqs), dtype=np.float32)
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for ij, (i,j) in enumerate(list(itertools.product(range(Nfreqs), range(Nfreqs)))):
                for kl, (k,l) in enumerate(list(itertools.product(range(Nfreqs), range(Nfreqs)))):
                    PScov_sim_Inv[b1,b2,i,j,k,l] = PScov_sim_alt_Inv[ij*inp.Nbins+b1, kl*inp.Nbins+b2]
    PScov_sim_Inv *= (inp.Nsims-(inp.Nbins*Nfreqs**2)-2)/(inp.Nsims-1) #correction factor from https://arxiv.org/pdf/astro-ph/0608064.pdf

    a_array = get_MLE_arrays(inp, Clij, PScov_sim_Inv, use_analytic=True)
    print(flush=True)
    a_array = get_MLE_arrays(inp, Clij, PScov_sim_Inv, use_analytic=False)
    
    print(flush=True)
    Fisher_inversion(inp, Clij, PScov_sim_Inv)

    print(flush=True)
    MCMC(inp, Clij, PScov_sim_Inv, sim=0)
   
    return a_array

