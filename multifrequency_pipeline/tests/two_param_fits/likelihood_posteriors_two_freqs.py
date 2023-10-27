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

def get_PScov_sim(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    
    RETURNS
    -------
    cov: (3*Nbins, 3*Nbins) ndarray containing covariance matrix Cov_{ij b1, kl b2}
        index as cov[(0-2 for Cl00 Cl01 Cl11)*Nbins + bin1, (0-2 for Cl00 Cl01 Cl11)*Nbins + bin2]
    '''
    Clij_tmp = Clij[:,:,:,0] #shape (Nsims, Nfreqs=2, Nfreqs=2, Nbins)
    Clij_tmp = np.array([Clij_tmp[:,0,0], Clij_tmp[:,0,1], Clij_tmp[:,1,1]]) #shape (3, Nsims, Nbins)
    Clij_tmp = np.transpose(Clij_tmp, axes=(0,2,1)) #shape (3 for Cl00 Cl01 and Cl11, Nbins, Nsims)
    Clij_tmp = np.reshape(Clij_tmp, (inp.Nbins*3, -1))
    cov = np.cov(Clij_tmp)
    return cov


##############################################
#########      NUMERICAL MLE      ############
##############################################

def ClijA(A1, A2, inp, Clij00, Clij01, Clij10, Clij11):
    '''
    Model for theoretical spectra Clpq including amplitude parameters

    ARGUMENTS
    ---------
    USED BY MINIMIZER
    A1: float, scaling parameter for power spectrum of first component
    A2: float, scaling parameter for power spectrum of second component

    CONSTANT ARGS
    inp: Info object containing input parameter specifications
    Clij{i}{j}: (1+Ncomps, Nbins) ndarray containing contribution of components to Clij

    RETURNS
    -------
    (Nbins, 2, 2) ndarray, 
    index as array[bin;  0-2 or ij=00, 01, 11]

    '''

    Clij_with_A_00 = A1*Clij00[1] + A2*Clij00[2]
    Clij_with_A_01 = A1*Clij01[1] + A2*Clij01[2]
    Clij_with_A_10 = A1*Clij10[1] + A2*Clij10[2]
    Clij_with_A_11 = A1*Clij11[1] + A2*Clij11[2]
    return np.array([[[Clij_with_A_00[b], Clij_with_A_01[b]],[Clij_with_A_10[b], Clij_with_A_11[b]]] for b in range(inp.Nbins)])


def neg_lnL(pars, f, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of amplitude parameters
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    negative log likelihood for one simulation, combined over multipoles 
    '''
    Clij00 = np.mean(Clij00_all_sims, axis=0)
    Clij01 = np.mean(Clij01_all_sims, axis=0)
    Clij10 = np.mean(Clij10_all_sims, axis=0)
    Clij11 = np.mean(Clij11_all_sims, axis=0)
    model = f(*pars, inp, Clij00, Clij01, Clij10, Clij11)
    Clij00d = Clij00_all_sims[sim,0]
    Clij01d = Clij01_all_sims[sim,0]
    Clij11d = Clij11_all_sims[sim,0]
    return np.sum([[1/2* \
     ((model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l2,0,0]*(model[l2][0,0]-Clij00d[l2]) + (model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l2,0,1]*(model[l2][0,1]-Clij01d[l2]) + (model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l2,0,2]*(model[l2][1,1]-Clij11d[l2]) \
    + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l2,1,0]*(model[l2][0,0]-Clij00d[l2]) + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l2,1,1]*(model[l2][0,1]-Clij01d[l2]) + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l2,1,2]*(model[l2][1,1]-Clij11d[l2]) \
    + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l2,2,0]*(model[l2][0,0]-Clij00d[l2]) + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l2,2,1]*(model[l2][0,1]-Clij01d[l2]) + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l2,2,2]*(model[l2][1,1]-Clij11d[l2])) \
    for l1 in range(inp.Nbins)] for l2 in range(inp.Nbins)]) 
    


def acmb_atsz_numerical(inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to A1, A2 for one sim using numerical minimization routine

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit A1, A2 (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start, start] #a1_start, a2_start
        res = minimize(neg_lnL, x0 = start_array, args = (ClijA, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv), method='Nelder-Mead', bounds=None) #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x


##############################################
#########       ANALYTIC MLE      ############
##############################################

def acmb_atsz_analytic(inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to A1, A2 for one sim analytically 

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit A1, A2 (floats)

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, bin1-->l, bin2-->m, ij-->i, kl-->j

    '''
    Clijd = np.array([Clij00_all_sims[sim,0], Clij01_all_sims[sim,0], Clij11_all_sims[sim,0]]) #shape (3,Ncomps,Nbins)
    Clij00 = (np.mean(Clij00_all_sims, axis=0))[1:]
    Clij01 = (np.mean(Clij01_all_sims, axis=0))[1:]
    Clij11 = (np.mean(Clij11_all_sims, axis=0))[1:]
    Clij = np.array([Clij00, Clij01, Clij11]) #shape (3, Ncomps, Nbins)
    F = np.einsum('ial,lmij,jbm->ab', Clij, PScov_sim_Inv, Clij)
    F_inv = np.linalg.inv(F)
    return np.einsum('ab,ibl,lmij,jm->a', F_inv, Clij, PScov_sim_Inv, Clijd)


##############################################
########  ARRAYS OF MLE ESTIMATES  ###########
##############################################

def get_MLE_arrays(inp, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv, use_analytic=True):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray;
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
    param_array = pool.starmap(func, [(inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv) for sim in range(len(Clij00_all_sims))])
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
### FISHER MATRIX INVERSION ###
###############################


def Fisher_inversion(inp, Clij, PScov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray;
        contains inverse power spectrum covariance matrix in tensor form

    RETURNS
    -------
    a1_std, a2_std: predicted standard deviations of A1, A2
        found by computing the Fisher matrix and inverting
    '''

    Ncomps = 2
    Clij_mean = np.mean(Clij, axis=0)
    deriv_vec = np.zeros((Ncomps, 3, inp.Nbins))
    for A in range(Ncomps):
        for ij in range(3):
            if ij==0: i,j = 0,0
            elif ij==1: i,j = 0,1
            else: i,j = 1,1
            deriv_vec[A,ij] = Clij_mean[i,j,1+A]
    Fisher = np.einsum('Aib,bcij,Bjc->AB', deriv_vec, PScov_sim_Inv, deriv_vec)
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

def pos_lnL(pars, f, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv): 
    '''
    Expression for positive log likelihood for one sim

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of amplitude parameters
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    log likelihood for one simulation, combined over multipoles 
    '''
    return -neg_lnL(pars, f, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv)


def MCMC(inp, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv, sim=0):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
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
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[ClijA, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv])
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

def cov_of_MLE_analytic(Clij00_all_sims, Clij01_all_sims, Clij11_all_sims, PScov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    inverse of Fisher matrix

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, bin1-->l, bin2-->m, ij-->i, kl-->j

    '''
    Clij00_mean = np.mean(Clij00_all_sims, axis=0)
    Clij01_mean = np.mean(Clij01_all_sims, axis=0)
    Clij11_mean = np.mean(Clij11_all_sims, axis=0)
    Clij = np.array([Clij00_mean[1:], Clij01_mean[1:], Clij11_mean[1:]]) #shape (3,Ncomps,Nbins)
    F = np.einsum('ial,lmij,jbm->ab', Clij, PScov_sim_Inv, Clij)
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
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, 1+Ncomps, Nbins) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j

    RETURNS
    -------
    a1_array: array of length Nsims containing best fit A1 for each simulation
    a2_array: array of length Nsims containing best fit A2 for each simulation
    '''

    PScov_sim = get_PScov_sim(inp, Clij)
    PScov_sim_alt_Inv = scipy.linalg.inv(PScov_sim)
    PScov_sim_Inv = np.zeros((inp.Nbins, inp.Nbins, 3, 3))
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for i in range(3):
                for j in range(3):
                    PScov_sim_Inv[b1, b2, i, j] = PScov_sim_alt_Inv[i*inp.Nbins+b1, j*inp.Nbins+b2]
    if not inp.use_Gaussian_cov:
        PScov_sim_Inv *= (inp.Nsims-(inp.Nbins*3)-2)/(inp.Nsims-1) #correction factor from https://arxiv.org/pdf/astro-ph/0608064.pdf

    Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims = Clij[:,0,0], Clij[:,0,1], Clij[:,1,0], Clij[:,1,1]

    a1_array, a2_array = get_MLE_arrays(inp, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv, use_analytic=True)
    print(flush=True)
    a1_array, a2_array = get_MLE_arrays(inp, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv, use_analytic=False)
    
    print(flush=True)
    Fisher_inversion(inp, Clij, PScov_sim_Inv)

    print(flush=True)
    MCMC(inp, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv, sim=0)

    print(flush=True)
    cov_of_MLE_analytic(Clij00_all_sims, Clij01_all_sims, Clij11_all_sims, PScov_sim_Inv)

    return a1_array, a2_array




