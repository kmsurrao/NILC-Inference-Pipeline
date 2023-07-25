import numpy as np
import pickle
import emcee
import scipy
from scipy.optimize import minimize
from scipy import stats
import multiprocessing as mp
import sys
sys.path.append('../shared')
from utils import tsz_spectral_response

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

    if inp.use_Gaussian_cov:

        cov = np.zeros((inp.Nbins, inp.Nbins, 3, 3))
        
        Clij = np.mean(Clij, axis=0) #dim (Nfreqs=2, Nfreqs=2, 1+Ncomps, Nbins)
        g1, g2 = tsz_spectral_response(inp.freqs) #tSZ spectral response at 90 and 150 GHz
        CC = Clij[0,0,1] #CMB
        T = Clij[0,0,2]/g1**2 #tSZ (in Compton-y)
        N1 = Clij[0,0,3] #noise 90 GHz
        N2 = Clij[1,1,4] #noise 150 GHz
        Clij = Clij[:,:,0]
        f = 1. #fraction of sky

        #get mean ell in each bin
        res = stats.binned_statistic(np.arange(inp.ellmax+1)[2:], np.arange(inp.ellmax+1)[2:], statistic='mean', bins=inp.Nbins)

        for bin in np.arange(inp.Nbins):
            Nmodes = f* ((np.floor(res[1][bin+1]))**2 - (np.ceil(res[1][bin])-1)**2)
            cov[bin, bin] = (1/Nmodes)*np.array([
                        [2*Clij[0, 0, bin]**2,
                            2*(CC[bin] + g1**2*T[bin])*Clij[0, 1, bin] + 2*N1[bin]*Clij[0, 1, bin],
                            2*Clij[0, 1, bin]**2], 
                        [2*(CC[bin] + g1**2*T[bin])*Clij[0, 1, bin] + 2*N1[bin]*Clij[0, 1, bin], 
                            Clij[0, 0, bin]*Clij[1, 1, bin] + Clij[0, 1, bin]**2,
                            2*(CC[bin] + g2**2*T[bin])*Clij[0, 1, bin] + 2*N2[bin]*Clij[0, 1, bin]], 
                        [2*Clij[0, 1, bin]**2, 
                            2*(CC[bin] + g2**2*T[bin])*Clij[0, 1, bin] + 2*N2[bin]*Clij[0, 1, bin],
                            2*Clij[1, 1, bin]**2]])
        PScov_sim_alt = np.zeros((3*inp.Nbins, 3*inp.Nbins))
        for b1 in range(inp.Nbins):
            for b2 in range(inp.Nbins):
                for ij in range(3):
                    for kl in range(3):
                        PScov_sim_alt[ij*inp.Nbins+b1, kl*inp.Nbins+b2] = cov[b1,b2,ij,kl]
        return PScov_sim_alt

    Clij_tmp = Clij[:,:,:,0] #shape (Nsims, Nfreqs=2, Nfreqs=2, Nbins)
    Clij_tmp = np.array([Clij_tmp[:,0,0], Clij_tmp[:,0,1], Clij_tmp[:,1,1]]) #shape (3, Nsims, Nbins)
    Clij_tmp = np.transpose(Clij_tmp, axes=(0,2,1)) #shape (3 for Cl00 Cl01 and Cl11, Nbins, Nsims)
    Clij_tmp = np.reshape(Clij_tmp, (inp.Nbins*3, -1))
    cov = np.cov(Clij_tmp)
    return cov


##############################################
#########      NUMERICAL MLE      ############
##############################################

def ClijA(Acmb, Atsz, Anoise1, Anoise2, inp, Clij00, Clij01, Clij10, Clij11):
    '''
    Model for theoretical spectra Clpq including Acmb and Atsz parameters

    ARGUMENTS
    ---------
    USED BY MINIMIZER
    Acmb: float, scaling parameter for CMB power spectrum
    Atsz: float, scaling parameter for tSZ power spectrum
    Anoise1: float, scaling parameter for 90 GHz noise power spectrum
    Anoise2: float, scaling parameter for 150 GHz noise power spectrum

    CONSTANT ARGS
    inp: Info object containing input parameter specifications
    Clij{i}{j}: (1+Ncomps, Nbins) ndarray containing contribution of components to Clij

    RETURNS
    -------
    (Nbins, 2, 2) ndarray, 
    index as array[bin;  0-2 or ij=00, 01, 11]

    '''

    Clij_with_A_00 = Acmb*Clij00[1] + Atsz*Clij00[2] + Anoise1*Clij00[3] + Anoise2*Clij00[4]
    Clij_with_A_01 = Acmb*Clij01[1] + Atsz*Clij01[2] + Anoise1*Clij01[3] + Anoise2*Clij01[4]
    Clij_with_A_10 = Acmb*Clij10[1] + Atsz*Clij10[2] + Anoise1*Clij10[3] + Anoise2*Clij10[4]
    Clij_with_A_11 = Acmb*Clij11[1] + Atsz*Clij11[2] + Anoise1*Clij11[3] + Anoise2*Clij11[4]
    return np.array([[[Clij_with_A_00[b], Clij_with_A_01[b]],[Clij_with_A_10[b], Clij_with_A_11[b]]] for b in range(inp.Nbins)])


def lnL(pars, f, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb and Atsz
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    negative log likelihood for one simulation, combined over multipoles 
    '''
    Clij00 = Clij00_all_sims[sim]
    Clij01 = Clij01_all_sims[sim]
    Clij10 = Clij10_all_sims[sim]
    Clij11 = Clij11_all_sims[sim]
    model = f(*pars, inp, Clij00, Clij01, Clij10, Clij11)
    Clij00d = np.mean(Clij00_all_sims[:,0], axis=0)
    Clij01d = np.mean(Clij01_all_sims[:,0], axis=0)
    Clij11d = np.mean(Clij11_all_sims[:,0], axis=0)
    #return np.sum([1/2*((model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l1,0,0]*(model[l1][0,0]-Clij00d[l1]) + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l1,1,1]*(model[l1][0,1]-Clij01d[l1]) + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l1,2,2]*(model[l1][1,1]-Clij11d[l1])) for l1 in range(inp.Nbins)]) #diagonal only
    return np.sum([[1/2* \
     ((model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l2,0,0]*(model[l2][0,0]-Clij00d[l2]) + (model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l2,0,1]*(model[l2][0,1]-Clij01d[l2]) + (model[l1][0,0]-Clij00d[l1])*PScov_sim_Inv[l1,l2,0,2]*(model[l2][1,1]-Clij11d[l2]) \
    + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l2,1,0]*(model[l2][0,0]-Clij00d[l2]) + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l2,1,1]*(model[l2][0,1]-Clij01d[l2]) + (model[l1][0,1]-Clij01d[l1])*PScov_sim_Inv[l1,l2,1,2]*(model[l2][1,1]-Clij11d[l2]) \
    + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l2,2,0]*(model[l2][0,0]-Clij00d[l2]) + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l2,2,1]*(model[l2][0,1]-Clij01d[l2]) + (model[l1][1,1]-Clij11d[l1])*PScov_sim_Inv[l1,l2,2,2]*(model[l2][1,1]-Clij11d[l2])) \
    for l1 in range(inp.Nbins)] for l2 in range(inp.Nbins)]) 
    


def acmb_atsz_numerical(inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb, Atsz, Anoise90, Anoise150 for one sim using numerical minimization routine

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit Acmb, Atsz, Anoise1, Anoise2 (floats)
    '''
    bounds = ((0.001, 1000), (0.001, 1000), (0.001, 1000), (0.001, 1000))
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start, start, start, start] #acmb_start, atsz_start, anoise1_start, anoise2_start
        res = minimize(lnL, x0 = start_array, args = (ClijA, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv), method='Nelder-Mead', bounds=None) #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x


##############################################
#########       ANALYTIC MLE      ############
##############################################

def acmb_atsz_analytic(inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb, Atsz, Anoise90, Anoise150 for one sim analytically 

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit Acmb, Atsz, Anoise1, Anoise2 (floats)

    INDEX MAPPING IN EINSUM
    -----------------------
    alpha --> a, beta --> b, bin1-->l, bin2-->m, ij-->i, kl-->j

    '''
    Clij = np.array([Clij00_all_sims[sim,1:], Clij01_all_sims[sim,1:], Clij11_all_sims[sim,1:]]) #shape (3,Ncomps,Nbins)
    Clij00d = (np.mean(Clij00_all_sims, axis=0))[0]
    Clij01d = (np.mean(Clij01_all_sims, axis=0))[0]
    Clij11d = (np.mean(Clij11_all_sims, axis=0))[0]
    Clijd = np.array([Clij00d, Clij01d, Clij11d]) #shape (3,Nbins)
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
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation
    '''

    func = acmb_atsz_analytic if use_analytic else acmb_atsz_numerical
    string = 'analytic' if use_analytic else 'numerical'
    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(func, [(inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv) for sim in range(len(Clij00_all_sims))])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, 4 for Acmb Atsz Anoise1 Anoise2)
    acmb_array = param_array[:,0]
    atsz_array = param_array[:,1]
    anoise1_array = param_array[:,2]
    anoise2_array = param_array[:,3]
    
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_template_fitting_{string}.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_template_fitting_{string}.p', 'wb'))
    pickle.dump(anoise1_array, open(f'{inp.output_dir}/anoise1_template_fitting_{string}.p', 'wb'))
    pickle.dump(anoise2_array, open(f'{inp.output_dir}/anoise2_template_fitting_{string}.p', 'wb'))
    if inp.verbose:
        print(f'created {inp.output_dir}/acmb_array_template_fitting_{string}.p and atsz and anoise1 and anoise2', flush=True)

    # #remove section below and uncomment section above
    # acmb_array = pickle.load(open(f'{inp.output_dir}/acmb_array_template_fitting.p', 'rb'))
    # atsz_array = pickle.load(open(f'{inp.output_dir}/atsz_array_template_fitting.p', 'rb'))
    # anoise1_array = pickle.load(open(f'{inp.output_dir}/anoise1_array_template_fitting.p', 'rb'))
    # anoise2_array = pickle.load(open(f'{inp.output_dir}/anoise2_array_template_fitting.p', 'rb'))

    final_cov = np.cov(np.array([acmb_array, atsz_array, anoise1_array, anoise2_array]))
    print(f'Results from maximum likelihood estimation using {string} MLEs', flush=True)
    print('---------------------------------------------------------------', flush=True)
    print(f'Acmb = {np.mean(acmb_array)} +/- {np.sqrt(final_cov[0,0])}', flush=True)
    print(f'Atsz = {np.mean(atsz_array)} +/- {np.sqrt(final_cov[1,1])}', flush=True)
    print(f'Anoise1 = {np.mean(anoise1_array)} +/- {np.sqrt(final_cov[2,2])}', flush=True)
    print(f'Anoise2 = {np.mean(anoise2_array)} +/- {np.sqrt(final_cov[3,3])}', flush=True)

    return acmb_array, atsz_array, anoise1_array, anoise2_array 



###################################################
### FISHER MATRIX INVERSION FOR ONE SIMULATION  ###
###################################################


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
    acmb_std, atsz_std, anoise1_std, anoise2_std: predicted standard deviations of Acmb, etc.
        found by computing the Fisher matrix and inverting
    '''

    Ncomps = 4
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

def pos_lnL(pars, f, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv): 
    '''
    Expression for positive log likelihood for one sim

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb and Atsz
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    log likelihood for one simulation, combined over multipoles 
    '''
    return -lnL(pars, f, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv)


##############################################
########   MCMC WITH ONE SIMULATION  #########
##############################################


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
    acmb_std, atsz_std, anoise1_std, anoise2_std: predicted standard deviations of Acmb, etc.
        found by computing the Fisher matrix and inverting
    '''

    np.random.seed(0)
    ndim = 4
    nwalkers = 10
    p0 = np.random.random((nwalkers, ndim))*(1.2-0.8)+0.8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, pos_lnL, args=[ClijA, inp, sim, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv])
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(state, 1000)
    samples = sampler.get_chain() #dimensions (1000, nwalkers, Ncomps=4)

    if inp.save_files:
        pickle.dump(samples, open(f'{inp.output_dir}/MCMC_chains_multifrequency.p', 'wb'))
        if inp.verbose:
            print(f'saved {inp.output_dir}/MCMC_chains_multifrequency.p', flush=True)
    
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


############################################################
########   COVARIANCE OF MLEs FOR ONE SIMULATION   #########
############################################################

def covs_of_MLE_analytic(Clij00_all_sims, Clij01_all_sims, Clij11_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb, Atsz, Anoise90, Anoise150 for one sim analytically 

    ARGUMENTS
    ---------
    Clij{i}{j}_all_sims: (Nsims, 1+N_comps, Nbins) ndarray containing contribution of components to Clij
    PScov_sim_Inv: (Nbins, Nbins, 3 for Cl00 Cl01 Cl11, 3 for Cl00 Cl01 Cl11) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit Acmb, Atsz, Anoise1, Anoise2 (floats)

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
    print('Acmb std dev: ', np.sqrt(F_inv[0,0]), flush=True)
    print('Atsz std dev: ', np.sqrt(F_inv[1,1]), flush=True)
    print('Anoise1 std dev: ', np.sqrt(F_inv[2,2]), flush=True)
    print('Anoise2 std dev: ', np.sqrt(F_inv[3,3]), flush=True)
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
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation
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

    acmb_array, atsz_array, anoise1_array, anoise2_array = get_MLE_arrays(inp, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv, use_analytic=True)
    print(flush=True)
    acmb_array, atsz_array, anoise1_array, anoise2_array = get_MLE_arrays(inp, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv, use_analytic=False)
    
    print(flush=True)
    Fisher_inversion(inp, Clij, PScov_sim_Inv)

    print(flush=True)
    MCMC(inp, Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims, PScov_sim_Inv, sim=0)

    print(flush=True)
    covs_of_MLE_analytic(Clij00_all_sims, Clij01_all_sims, Clij11_all_sims, PScov_sim_Inv)


   
    return acmb_array, atsz_array, anoise1_array, anoise2_array

