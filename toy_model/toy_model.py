import numpy as np
import emcee
import scipy
from scipy.optimize import minimize
import multiprocessing as mp
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, SNLE, SNRE, prepare_for_sbi, simulate_for_sbi

#########################################################################################
##############               Create realizations of               #######################
##############                f(x) = Acos(x) + Bx                 #######################  
##############    with Gaussian noise added to each realization   ####################### 
##############          A and B have fiducial values of 1         #######################   
#########################################################################################

def get_realizations(Nsims, xvals, pars=None, add_noise=False, sample_variance=True):
    '''
    ARGUMENTS
    ---------
    Nsims: int, number of realizations to generate
    xvals: numpy array of x values over which to compute function
    pars: array-like of floats A and B, defaults to None
    add_noise: Bool, whether to add a separate Gaussian noise term
    sample_variance: Bool, whether to build in sample variance to the 
        cosine and linear terms
    Note that at least one of add_noise or sample_variance must be True to avoid
    singular matrices.


    RETURNS
    -------
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x) + noise

    '''
    if pars is not None:
        A,B = pars
    else:
        A,B = 1., 1.

    # Add noise
    if add_noise:
        cov_matrix_noise = np.diag(1.+1./np.sqrt(2*xvals+1))
        samples = np.random.multivariate_normal(np.zeros(len(xvals)), cov_matrix_noise, size=Nsims)
    if sample_variance:
        cov_matA = np.diag(3./np.sqrt(2*xvals+1))
        cov_matB = np.diag(3./np.sqrt(2*xvals+1))
        samples_A = np.random.multivariate_normal(np.zeros(len(xvals)), cov_matA, size=Nsims)
        samples_B = np.random.multivariate_normal(np.zeros(len(xvals)), cov_matB, size=Nsims)
    
    # Generate the realizations
    realizations = np.zeros((Nsims, len(xvals)))
    for i in range(Nsims):
        if sample_variance:
            realizations[i] = A*(np.cos(xvals)+samples_A[i]) + B*(xvals+samples_B[i])
        else:
            realizations[i] = A*np.cos(xvals) + B*xvals
        if add_noise:
            realizations[i] += samples[i]
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
    if realizations.shape[-1] == 1: #only one x value
        print(np.array([[np.var(realizations[:,0])]]))
        return np.array([[np.var(realizations[:,0])]])
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
    for start in [0.5, 1.0]:
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
    step = 0.001
    mean_A, mean_B = 1., 1.
    deriv = [(theory_model(mean_A+step,mean_B,xvals)-theory_model(mean_A-step,mean_B,xvals))/(2*step), (theory_model(mean_A, mean_B+step,xvals)-theory_model(mean_A, mean_B-step,xvals))/(2*step)]
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
### FISHER MATRIX FORECAST  ###
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
    step = 0.001
    mean_A, mean_B = 1., 1.
    deriv_vec = [(theory_model(mean_A+step,mean_B,xvals)-theory_model(mean_A-step,mean_B,xvals))/(2*step), (theory_model(mean_A, mean_B+step,xvals)-theory_model(mean_A, mean_B-step,xvals))/(2*step)]
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
######## LIKELIHOOD-FREE INFERENCE   #######
############################################

def get_prior(prior_half_widths=None):
    '''
    ARGUMENTS
    ---------
    prior_half_widths: list of 2 floats, half width of uniform prior to use for each parameter
        The parameter prior will be set to [1-prior_half_width, 1+prior_half_width]
    
    RETURNS
    -------
    prior on A and B to use for likelihood-free inference
    '''
    mean = 1.0
    if prior_half_widths is not None:
        step_A, step_B = prior_half_widths
    else:
        step_A = 1.3
        step_B = 1.0
    prior = utils.BoxUniform(low= torch.tensor([mean-step_A, mean-step_B]), high= torch.tensor([mean+step_A, mean+step_B]))
    return prior



def get_posterior_LFI(realizations, Nsims, xvals, method='SNPE', prior_half_widths=None):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    Nsims: int, number of realizations to generate
    xvals: numpy array of x values over which to compute function
    method: str, either 'SNPE', 'SNLE', or 'SNRE'
    prior_half_widths: list of 2 floats, half width of uniform prior to use for each parameter
        The parameter prior will be set to [1-prior_half_width, 1+prior_half_width]

    RETURNS
    -------
    samples: torch tensor of shape (4, Nsims) containing Acmb, Atsz, Anoise1, Anoise2 posteriors
    '''
    def simulator(pars):
        '''
        ARGUMENTS
        ---------
        pars: [A, B] parameters (floats)

        RETURNS
        -------
        realizations: (len(xvals),) torch tensor containing a random realization of f(x) = Acos(x) + B(x)
        
        '''
        realization = torch.tensor(get_realizations(1, xvals, pars=pars)[0])
        return realization


    prior = get_prior(prior_half_widths=prior_half_widths)
    observation = np.mean(realizations, axis=0)

    # simulator_, prior_ = prepare_for_sbi(simulator, prior)
    # if method == 'SNPE':
    #     inference = SNPE(prior=prior_)
    # elif method == 'SNLE':
    #     inference = SNLE(prior=prior_)
    # elif method == 'SNRE':
    #     inference = SNRE(prior=prior_)

    # num_rounds = 2
    # posteriors = []
    # proposal = prior_
    # for _ in range(num_rounds):
    #     theta, x = simulate_for_sbi(simulator_, proposal, num_simulations=2*Nsims//num_rounds, num_workers=8)
    #     density_estimator = inference.append_simulations(
    #                 theta, x, proposal=proposal).train()
    #     posterior = inference.build_posterior(density_estimator)
    #     posteriors.append(posterior)
    #     proposal = posterior.set_default_x(observation)
    # samples = posterior.sample((Nsims,), x=observation)



    posterior = infer(simulator, prior, method=method, num_simulations=Nsims, num_workers=8)
    samples = posterior.sample((Nsims,), x=observation)

    A_array, B_array = np.array(samples, dtype=np.float32).T
    print(f'Results from likelihood-free inference, method={method}', flush=True)
    print('---------------------------------------------------------------', flush=True)
    print(f'A = {np.mean(A_array)} +/- {np.std(A_array)}', flush=True)
    print(f'B = {np.mean(B_array)} +/- {np.std(B_array)}', flush=True)
    return samples



##############################################
## COMPARE RESULTS FROM DIFFERENT METHODS  ###
##############################################


def get_all_AB(realizations, xvals, method='SNPE'):
    '''
    ARGUMENTS
    ---------
    realizations: (Nsims, len(xvals)) ndarray containing random realizations of f(x) = Acos(x) + B(x)
    xvals: numpy array of x values over which to compute function
    method: str, method to use for likelihood-free inference, either 'SNPE', 'SNLE', or 'SNRE'


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
    get_posterior_LFI(realizations, Nsims, xvals, method=method, prior_half_widths=[3*np.std(A_array), 3*np.std(B_array)])
   
    return A_array, B_array



def main():
    np.random.seed(0)
    Nsims = 1000
    xvals = np.arange(30)
    realizations = get_realizations(Nsims, xvals)
    method = 'SNPE'
    get_all_AB(realizations, xvals, method=method)
    return


if __name__ == '__main__':
    main()