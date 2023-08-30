import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
import multiprocessing as mp



def get_PScov_sim(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, 1+Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim3: index0 is total power in Clpq, other indices are power from each component
    
    RETURNS
    -------
    cov: (3*Nbins, 3*Nbins) ndarray containing covariance matrix Cov_{pq,rs}
        index as cov[(0-2 for ClTT ClTy Clyy)*Nbins + bin1, (0-2 for ClTT ClTy Clyy)*Nbins + bin2]
    '''
    Clpq_tmp = Clpq[:,:,:,0,:]
    Clpq_tmp = np.array([Clpq_tmp[:,0,0], Clpq_tmp[:,0,1], Clpq_tmp[:,1,1]])
    Clpq_tmp = np.transpose(Clpq_tmp, axes=(0,2,1)) #shape (3 for ClTT, ClTy, Clyy, Nbins, Nsims)
    Clpq_tmp = np.reshape(Clpq_tmp, (inp.Nbins*3, -1))
    cov = np.cov(Clpq_tmp)
    return cov



def ClpqA(Acmb, Atsz, Anoise1, Anoise2, inp, ClTT, ClTy, ClyT, Clyy):
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
    Cl{p}{q}: (1+Ncomps, Nbins) ndarray containing contribution of components to Clpq

    RETURNS
    -------
    theory_model: (Nbins, 2, 2) ndarray for ClTT, ClTy, ClyT, and Clyy in terms of A_y and A_z parameters

    '''
    theory_model = np.zeros((2, 2, inp.Nbins))
    A_vec = [Acmb, Atsz, Anoise1, Anoise2]

    for p,q in [(0,0), (0,1), (1,0), (1,1)]:
        if p==0 and q==0: 
            Clpq_here = ClTT
        elif p==0 and q==1:
            Clpq_here = ClTy
        elif p==1 and q==0:
            Clpq_here = ClyT
        elif p==1 and q==1:
            Clpq_here = Clyy
        for y in range(1, len(A_vec)+1):
            theory_model[p,q] += A_vec[y-1]*Clpq_here[y]

    theory_model = np.transpose(theory_model, axes=(2,0,1))
    return theory_model



def neg_lnL(pars, f, inp, sim, ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims, PScov_sim_Inv): 
    '''
    Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

    ARGUMENTS
    ---------
    pars: parameters to function f (not manually inputted but used by minimizer)
    f: function that returns theory model in terms of Acmb, Atsz, Anoise1, and Anoise2
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Cl{p}{q}_all_sims: (Nsims, 1+Ncomps, Nbins) ndarray containing contribution of components to Clpq
    PScov_sim_Inv: (Nbins, Nbins, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing inverse of power spectrum covariance matrix


    RETURNS
    -------
    negative log likelihood for one simulation, combined over multipoles 
    '''
    ClTT = np.mean(ClTT_all_sims, axis=0)
    ClTy = np.mean(ClTy_all_sims, axis=0)
    ClyT = np.mean(ClyT_all_sims, axis=0)
    Clyy = np.mean(Clyy_all_sims, axis=0)
    model = f(*pars, inp, ClTT, ClTy, ClyT, Clyy)
    ClTTd = ClTT_all_sims[sim,0]
    ClTyd = ClTy_all_sims[sim,0]
    Clyyd = Clyy_all_sims[sim,0]
    return np.sum([[1/2* \
         ((model[l1][0,0]-ClTTd[l1])*PScov_sim_Inv[l1,l2,0,0]*(model[l2][0,0]-ClTTd[l2]) + (model[l1][0,0]-ClTTd[l1])*PScov_sim_Inv[l1,l2,0,1]*(model[l2][0,1]-ClTyd[l2]) + (model[l1][0,0]-ClTTd[l1])*PScov_sim_Inv[l1,l2,0,2]*(model[l2][1,1]-Clyyd[l2]) \
        + (model[l1][0,1]-ClTyd[l1])*PScov_sim_Inv[l1,l2,1,0]*(model[l2][0,0]-ClTTd[l2]) + (model[l1][0,1]-ClTyd[l1])*PScov_sim_Inv[l1,l2,1,1]*(model[l2][0,1]-ClTyd[l2]) + (model[l1][0,1]-ClTyd[l1])*PScov_sim_Inv[l1,l2,1,2]*(model[l2][1,1]-Clyyd[l2]) \
        + (model[l1][1,1]-Clyyd[l1])*PScov_sim_Inv[l1,l2,2,0]*(model[l2][0,0]-ClTTd[l2]) + (model[l1][1,1]-Clyyd[l1])*PScov_sim_Inv[l1,l2,2,1]*(model[l2][0,1]-ClTyd[l2]) + (model[l1][1,1]-Clyyd[l1])*PScov_sim_Inv[l1,l2,2,2]*(model[l2][1,1]-Clyyd[l2])) \
    for l1 in range(inp.Nbins)] for l2 in range(inp.Nbins)]) 


def acmb_atsz(inp, sim, ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb and Atsz for one sim

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    Cl{p}{q}_all_sims: (Nsims, 1+Ncomps, Nbins) ndarray containing contribution of components to Clpq
    PScov_sim_Inv: (Nbins, Nbins, 3 for ClTT ClTy Clyy, 3 for ClTT ClTy Clyy) ndarray containing inverse of power spectrum covariance matrix

    RETURNS
    -------
    best fit Acmb, Atsz, Anoise1, Anoise2 (floats)
    '''
    all_res = []
    for start in [0.5, 1.0, 1.5]:
        start_array = [start, start, start, start] #acmb_start, atsz_start, anoise1_start, anoise2_start
        res = minimize(neg_lnL, x0 = start_array, args = (ClpqA, inp, sim, ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims, PScov_sim_Inv), method='Nelder-Mead') #default method is BFGS
        all_res.append(res)
    return (min(all_res, key=lambda res:res.fun)).x



def get_all_acmb_atsz(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (N_preserved_comps=2, N_preserved_comps=2, 1+Ncomps, Nbins) ndarray 
        containing binned auto- and cross-spectra of harmonic ILC maps p and q
        dim2: index0 is total power in Clpq, other indices are power from each component

    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    anoise1_array: array of length Nsims containing best fit Anoise1 for each simulation
    anoise2_array: array of length Nsims containing best fit Anoise2 for each simulation

    '''
    

    PScov_sim = get_PScov_sim(inp, Clpq)
    PScov_sim_alt_Inv = scipy.linalg.inv(PScov_sim)
    PScov_sim_Inv = np.zeros((inp.Nbins, inp.Nbins, 3, 3))
    for b1 in range(inp.Nbins):
        for b2 in range(inp.Nbins):
            for i in range(3):
                for j in range(3):
                    PScov_sim_Inv[b1, b2, i, j] = PScov_sim_alt_Inv[i*inp.Nbins+b1, j*inp.Nbins+b2]
    PScov_sim_Inv *= (inp.Nsims-(inp.Nbins*3)-2)/(inp.Nsims-1) #correction factor from https://arxiv.org/pdf/astro-ph/0608064.pdf

    ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims = Clpq[:,0,0], Clpq[:,0,1], Clpq[:,1,0], Clpq[:,1,1]

    pool = mp.Pool(inp.num_parallel)
    param_array = pool.starmap(acmb_atsz, [(inp, sim, ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims, PScov_sim_Inv) for sim in range(inp.Nsims)])
    pool.close()
    param_array = np.asarray(param_array, dtype=np.float32) #shape (Nsims, 4 for Acmb Atsz Anoise1 Anoise2)
    acmb_array = param_array[:,0]
    atsz_array = param_array[:,1]
    anoise1_array = param_array[:,2]
    anoise2_array = param_array[:,3]
    
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_array_hilc.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_array_hilc.p', 'wb'))
    pickle.dump(anoise1_array, open(f'{inp.output_dir}/anoise1_array_hilc.p', 'wb'))
    pickle.dump(anoise2_array, open(f'{inp.output_dir}/anoise2_array_hilc.p', 'wb'))
    if inp.verbose:
        print(f'created {inp.output_dir}/acmb_array_hilc.p, atsz_array_hilc.p, anoise1_array_hilc.p, anoise2_array_hilc.p', flush=True)
    print('Results from maximum likelihood estimation', flush=True)
    print('----------------------------------------------', flush=True)
    print(f'Acmb = {np.mean(acmb_array)} +/- {np.std(acmb_array)}', flush=True)
    print(f'Atsz = {np.mean(atsz_array)} +/- {np.std(atsz_array)}', flush=True)
    print(f'Anoise1 = {np.mean(anoise1_array)} +/- {np.std(anoise1_array)}', flush=True)
    print(f'Anoise2 = {np.mean(anoise2_array)} +/- {np.std(anoise2_array)}', flush=True)

    return acmb_array, atsz_array, anoise1_array, anoise2_array