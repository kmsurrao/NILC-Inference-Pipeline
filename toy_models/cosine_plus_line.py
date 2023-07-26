import numpy as np
import pickle
import emcee
import scipy
from scipy.optimize import minimize
import multiprocessing as mp

'''
Create realizations of 
f(x) = Acos(x) + Bx for 0 <= x <= 2pi with Gaussian noise added to each realization
'''

def get_realizations(Nsims, xvals):
    '''
    ARGUMENTS
    ---------
    Nsims: int, number of realizations to generate
    xvals: numpy array of x values over which to compute function

    RETURNS
    -------
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with index 1 being 0 for cos(x) contribution or 1 for x contribution

    '''
    cov_matrix = 1.*np.eye(2*len(xvals))
    samples = np.random.multivariate_normal(np.ones(2*len(xvals)), cov_matrix, size=Nsims)
    realizations = np.zeros((Nsims, 2, len(xvals)))
    for i, sample in enumerate(samples):
        A,B = sample[:len(sample)//2], sample[len(sample)//2:]
        realizations[i,0] = A*np.cos(xvals)
        realizations[i,1] = B*xvals
    return realizations


##############################################
##### COVARIANCE MATRIX OF REALIZATIONS ######
##############################################

def get_cov_sim(realizations):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution
    
    RETURNS
    -------
    cov: (len(xvals),len(xvals)) ndarray containing covariance matrix of realizations
    '''
    realizations = np.sum(realizations, axis=1)
    return np.cov(np.transpose(realizations))


##############################################
#########      NUMERICAL MLE      ############
##############################################

def theory_model(A, B, realization):
    '''
    Theory model for f(x)

    ARGUMENTS
    ---------
    USED BY MINIMIZER
    A,B: parameters to fit in f(x) = Acos(x) + Bx

    CONSTANT ARGS
    realization: (2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realization[0-1, x] with 0-1 for cos(x) contribution or x contribution
    
    RETURNS
    -------
    numpy array with same length as xvals, giving f(x) for a given realization

    '''
    return A*realization[0] + B*realization[1]


def neg_lnL(pars, f, sim, realizations, cov_sim_Inv): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb and Atsz
    sim: int, simulation number
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix

    RETURNS
    -------
    negative log likelihood for one simulation
    '''
    model = f(*pars, realizations[sim])
    data = np.sum(np.mean(realizations, axis=0), axis=0)
    return np.sum([[1/2*(model[x1]-data[x1])*cov_sim_Inv[x1,x2]*(model[x2]-data[x2]) for x1 in range(len(data))] for x2 in range(len(data))])
    


def AB_numerical(sim, realizations, cov_sim_Inv):
    '''
    Maximize likelihood with respect to A and B for one sim using numerical minimization routine

    ARGUMENTS
    ---------
    sim: int, simulation number
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix

    RETURNS
    -------
    best fit A and B (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start, start] #A_start, B_start
        res = minimize(neg_lnL, x0 = start_array, args = (theory_model, sim, realizations, cov_sim_Inv), method='Nelder-Mead', bounds=None) #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x


##############################################
#########       ANALYTIC MLE      ############
##############################################

def AB_analytic(sim, realizations, cov_sim_Inv):
    '''
    Maximize likelihood with respect to A and B for one sim analytically 

    ARGUMENTS
    ---------
    sim: int, simulation number
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix

    RETURNS
    -------
    best fit A and B (floats)

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, x1-->l, x2-->m

    '''
    realization = realizations[sim] #shape (2, len(xvals))
    data = np.sum(np.mean(realizations, axis=0), axis=0) #shape (len(xvals),)
    F = np.einsum('al,lm,bm->ab', realization, cov_sim_Inv, realization)
    F_inv = np.linalg.inv(F)
    return np.einsum('ab,bl,lm,m->a', F_inv, realization, cov_sim_Inv, data)


##############################################
########  ARRAYS OF MLE ESTIMATES  ###########
##############################################

def get_MLE_arrays(realizations, cov_sim_Inv, use_analytic=True):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix
    use_analytic: Bool, whether to use analytic MLEs for parameters. If False, compute them with numerical minimization routine

    RETURNS
    -------
    A_array: array of length Nsims containing best fit A for each simulation
    B_array: array of length Nsims containing best fit B for each simulation
    '''
    func = AB_analytic if use_analytic else AB_numerical
    string = 'analytic' if use_analytic else 'numerical'
    pool = mp.Pool(8)
    param_array = pool.starmap(func, [(sim, realizations, cov_sim_Inv) for sim in range(len(realizations))])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, 2 for A and B)
    A_array = param_array[:,0]
    B_array = param_array[:,1]
    final_cov = np.cov(np.array([A_array, B_array]))
    print(f'Results from maximum likelihood estimation using {string} MLEs', flush=True)
    print('---------------------------------------------------------------', flush=True)
    print(f'A = {np.mean(A_array)} +/- {np.sqrt(final_cov[0,0])}', flush=True)
    print(f'B = {np.mean(B_array)} +/- {np.sqrt(final_cov[1,1])}', flush=True)
    return A_array, B_array



###############################
### FISHER MATRIX INVERSION ###
###############################


def Fisher_inversion(realizations, cov_sim_Inv):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix

    RETURNS
    -------
    A_std, B_std: predicted standard deviations of A and B, found by computing and inverting Fisher matrix
    '''
    realizations_mean = np.mean(realizations, axis=0)
    deriv_vec = realizations_mean
    Fisher = np.einsum('Ab,bc,Bc->AB', deriv_vec, cov_sim_Inv, deriv_vec)
    final_cov = np.linalg.inv(Fisher)
    A_std = np.sqrt(final_cov[0,0])
    B_std = np.sqrt(final_cov[1,1])
    print('Results from inverting Fisher matrix', flush=True)
    print('----------------------------------------', flush=True)
    print('A std dev: ', A_std, flush=True)
    print('B std dev: ', B_std, flush=True)
    return A_std, B_std


##############################################
########   MCMC WITH ONE SIMULATION  #########
##############################################

def pos_lnL(pars, f, sim, realizations, cov_sim_Inv): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb and Atsz
    sim: int, simulation number
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix

    RETURNS
    -------
    positive log likelihood for one simulation
    '''
    return -neg_lnL(pars, f, sim, realizations, cov_sim_Inv)


def MCMC(realizations, cov_sim_Inv, sim=0):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix
    sim: int, simulation number to use for MCMC

    RETURNS
    -------
    A_std, B_std: predicted standard deviations of A and B found from MCMC
    '''

    np.random.seed(0)
    ndim = 2
    nwalkers = 10
    p0 = np.random.random((nwalkers, ndim))*(1.3-0.7)+0.7
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[theory_model, sim, realizations, cov_sim_Inv])
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(state, 1000)
    samples = sampler.get_chain() #dimensions (1000, nwalkers, Ncomps=4)

    A_std = np.mean(np.array([np.std(samples[:,walker,0]) for walker in range(nwalkers)]))
    B_std = np.mean(np.array([np.std(samples[:,walker,1]) for walker in range(nwalkers)]))

    print('Results from MCMC', flush=True)
    print('------------------------------------', flush=True)
    print('A std dev: ', A_std, flush=True)
    print('B std dev: ', B_std, flush=True)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), flush=True)
    return A_std, B_std


############################################
######## ANALYTIC COVARIANCE OF MLE   ######
############################################

def cov_of_MLE_analytic(realizations, cov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb, Atsz, Anoise90, Anoise150 for one sim analytically 

    ARGUMENTS
    ---------
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix

    RETURNS
    -------
    inverse of Fisher matrix

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, x1-->l, x2-->m

    '''
    realizations_mean = np.mean(realizations, axis=0) 
    F = np.einsum('al,lm,bm->ab', realizations_mean, cov_sim_Inv, realizations_mean)
    F_inv = np.linalg.inv(F)
    print('Results from Analytic Covariance of MLEs', flush=True)
    print('------------------------------------', flush=True)
    print('A std dev: ', np.sqrt(F_inv[0,0]), flush=True)
    print('B std dev: ', np.sqrt(F_inv[1,1]), flush=True)
    return F_inv


##############################################
## COMPARE RESULTS FROM DIFFERENT METHODS  ###
##############################################


def get_all_AB(realizations):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, 2, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
        index as realizations[sim, 0-1, x] with 0-1 for cos(x) contribution or x contribution

    RETURNS
    -------
    A_array: array of length Nsims containing best fit A for each simulation
    B_array: array of length Nsims containing best fit B for each simulation
    '''
    Nsims = len(realizations)
    num_xvals = (realizations.shape)[-1]
    cov_sim = get_cov_sim(realizations) #shape (len(xvals), len(xvals))
    cov_sim_Inv = scipy.linalg.inv(cov_sim)
    cov_sim_Inv *= (Nsims-num_xvals-2)/(Nsims-1) #correction factor from https://arxiv.org/pdf/astro-ph/0608064.pdf

    A_array, B_array = get_MLE_arrays(realizations, cov_sim_Inv, use_analytic=True)
    print(flush=True)
    A_array, B_array = get_MLE_arrays(realizations, cov_sim_Inv, use_analytic=False)
    
    print(flush=True)
    Fisher_inversion(realizations, cov_sim_Inv)

    print(flush=True)
    MCMC(realizations, cov_sim_Inv, sim=1)

    print(flush=True)
    cov_of_MLE_analytic(realizations, cov_sim_Inv)
   
    return A_array, B_array



def main():
    np.random.seed(0)
    Nsims = 500
    xvals = np.linspace(0, 2*np.pi, 25)
    realizations = get_realizations(Nsims, xvals)
    get_all_AB(realizations)
    return


if __name__ == '__main__':
    main()