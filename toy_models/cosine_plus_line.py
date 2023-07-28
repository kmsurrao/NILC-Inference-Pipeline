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
    cov_matrix = np.diag(1.+np.sqrt(xvals))
    samples = np.random.multivariate_normal(np.zeros(len(xvals)), cov_matrix, size=Nsims)
    realizations = np.zeros((Nsims, len(xvals)))
    for i, sample in enumerate(samples):
        realizations[i] = 1*np.cos(xvals) + 1*xvals + sample
    return realizations



##############################################
##### COVARIANCE MATRIX OF REALIZATIONS ######
##############################################

def get_cov_sim(realizations):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    
    RETURNS
    -------
    cov: (len(xvals),len(xvals)) ndarray containing covariance matrix of realizations
    '''
    return np.cov(np.transpose(realizations))


##############################################
#########      NUMERICAL MLE      ############
##############################################

def theory_model(A, B, xvals):
    '''
    Theory model for f(x)

    ARGUMENTS
    ---------
    USED BY MINIMIZER
    A,B: parameters to fit in f(x) = Acos(x) + Bx

    CONSTANT ARGS
    xvals: numpy array of x values over which to compute function
    
    RETURNS
    -------
    numpy array with same length as xvals, giving f(x) for a given realization

    '''
    return A*np.cos(xvals) + B*xvals


def neg_lnL(pars, f, sim, realizations, cov_sim_Inv, xvals): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb and Atsz
    sim: int, simulation number
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix
    xvals: numpy array of x values over which to compute function

    RETURNS
    -------
    negative log likelihood for one simulation
    '''
    model = f(*pars, xvals)
    data = realizations[sim]
    return np.sum([[1/2*(model[x1]-data[x1])*cov_sim_Inv[x1,x2]*(model[x2]-data[x2]) for x1 in range(len(data))] for x2 in range(len(data))])
    


def AB_numerical(sim, realizations, cov_sim_Inv, xvals):
    '''
    Maximize likelihood with respect to A and B for one sim using numerical minimization routine

    ARGUMENTS
    ---------
    sim: int, simulation number
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix
    xvals: numpy array of x values over which to compute function

    RETURNS
    -------
    best fit A and B (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start, start] #A_start, B_start
        res = minimize(neg_lnL, x0 = start_array, args = (theory_model, sim, realizations, cov_sim_Inv, xvals), method='Nelder-Mead', bounds=None) #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x


##############################################
#########       ANALYTIC MLE      ############
##############################################

def AB_analytic(sim, realizations, cov_sim_Inv, xvals):
    '''
    Maximize likelihood with respect to A and B for one sim analytically 

    ARGUMENTS
    ---------
    sim: int, simulation number
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix
    xvals: numpy array of x values over which to compute function

    RETURNS
    -------
    best fit A and B (floats)

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, x1-->l, x2-->m

    '''
    deriv = [(theory_model(1.001,1.,xvals)-theory_model(0.999,1.,xvals))/(1.001-0.999), (theory_model(1, 1.001,xvals)-theory_model(1, 0.999,xvals))/(1.001-0.999)]
    # template = np.mean(realizations, axis=0) #shape (len(xvals),)
    F = np.einsum('al,lm,bm->ab', deriv, cov_sim_Inv, deriv)
    F_inv = np.linalg.inv(F)
    return np.einsum('ab,al,lm,m->a', F_inv, deriv, cov_sim_Inv, realizations[sim])


##############################################
########  ARRAYS OF MLE ESTIMATES  ###########
##############################################

def get_MLE_arrays(realizations, cov_sim_Inv, xvals, use_analytic=True):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix
    xvals: numpy array of x values over which to compute function
    use_analytic: Bool, whether to use analytic MLEs for parameters. If False, compute them with numerical minimization routine

    RETURNS
    -------
    A_array: array of length Nsims containing best fit A for each simulation
    B_array: array of length Nsims containing best fit B for each simulation
    '''
    func = AB_analytic if use_analytic else AB_numerical
    string = 'analytic' if use_analytic else 'numerical'
    pool = mp.Pool(8)
    param_array = pool.starmap(func, [(sim, realizations, cov_sim_Inv, xvals) for sim in range(len(realizations))])
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


def Fisher_inversion(cov_sim_Inv, xvals):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix
    xvals: numpy array of x values over which to compute function

    RETURNS
    -------
    A_std, B_std: predicted standard deviations of A and B, found by computing and inverting Fisher matrix
    '''
    deriv_vec = [(theory_model(1.001,1.,xvals)-theory_model(0.999,1.,xvals))/(1.001-0.999), (theory_model(1, 1.001,xvals)-theory_model(1, 0.999,xvals))/(1.001-0.999)]
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

def pos_lnL(pars, f, sim, realizations, cov_sim_Inv, xvals): 
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
    return -neg_lnL(pars, f, sim, realizations, cov_sim_Inv, xvals)


def MCMC(realizations, cov_sim_Inv, xvals, sim=0):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix
    xvals: numpy array of x values over which to compute function
    sim: int, simulation number to use for MCMC

    RETURNS
    -------
    A_std, B_std: predicted standard deviations of A and B found from MCMC
    '''

    np.random.seed(0)
    ndim = 2
    nwalkers = 10
    p0 = np.random.random((nwalkers, ndim))*(1.3-0.7)+0.7
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[theory_model, sim, realizations, cov_sim_Inv, xvals])
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

def cov_of_MLE_analytic(cov_sim_Inv, xvals):
    '''
    Maximize likelihood with respect to Acmb, Atsz, Anoise90, Anoise150 for one sim analytically 

    ARGUMENTS
    ---------
    cov_sim_Inv: (len(xvals), len(xvals)) ndarray containing inverse of realizations covariance matrix
    xvals: numpy array of x values over which to compute function

    RETURNS
    -------
    inverse of Fisher matrix

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, x1-->l, x2-->m

    '''
    deriv_vec = [(theory_model(1.001,1.,xvals)-theory_model(0.999,1.,xvals))/(1.001-0.999), (theory_model(1, 1.001,xvals)-theory_model(1, 0.999,xvals))/(1.001-0.999)]
    F = np.einsum('al,lm,bm->ab', deriv_vec, cov_sim_Inv, deriv_vec)
    F_inv = np.linalg.inv(F)
    print('Results from Analytic Covariance of MLEs', flush=True)
    print('------------------------------------', flush=True)
    print('A std dev: ', np.sqrt(F_inv[0,0]), flush=True)
    print('B std dev: ', np.sqrt(F_inv[1,1]), flush=True)
    return F_inv


##############################################
## COMPARE RESULTS FROM DIFFERENT METHODS  ###
##############################################


def get_all_AB(realizations, xvals):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    xvals: numpy array of x values over which to compute function

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

    A_array, B_array = get_MLE_arrays(realizations, cov_sim_Inv, xvals, use_analytic=True)
    print(flush=True)
    A_array, B_array = get_MLE_arrays(realizations, cov_sim_Inv, xvals, use_analytic=False)
    
    print(flush=True)
    Fisher_inversion(cov_sim_Inv, xvals)

    print(flush=True)
    MCMC(realizations, cov_sim_Inv, xvals, sim=1)

    print(flush=True)
    cov_of_MLE_analytic(cov_sim_Inv, xvals)
   
    return A_array, B_array



def main():
    np.random.seed(0)
    Nsims = 500
    xvals = np.arange(30)
    realizations = get_realizations(Nsims, xvals)
    get_all_AB(realizations, xvals)
    return


if __name__ == '__main__':
    main()