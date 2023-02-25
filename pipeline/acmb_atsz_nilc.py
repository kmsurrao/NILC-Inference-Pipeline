import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
from scipy import ndimage

def get_data_arrays(Clpq):
    '''
    Get arrays of ClTT, ClTy, and Clyy for each simulation

    ARGUMENTS
    ---------
    Clpq: (Nsims, N_preserved_comps, N_preserved_comps, N_comps=3, 4, ellmax+1) ndarray,
        contains contributions from each component to the power spectrum of NILC maps
        with preserved components p and q,
        index as Clpq[sim, p,q,z,reMASTERed term,l]
    
    RETURNS
    -------
    ClTTd: (Nsims, N_comps, ellmax+1) ndarray, contains contributions to CMB preserved NILC power spectrum
        index as ClTT[sim][0-2 for CMB, tSZ, or noise component][l]
    ClTyd: (Nsims, N_comps, ellmax+1) ndarray, contains contributions to CMB preserved NILC and tSZ preserved NILC cross-spectrum
        index as ClTy[sim][0-2 for CMB, tSZ, or noise component][l]
    Clyyd: (Nsims, N_comps, ellmax+1) ndarray, contains contributions to tSZ preserved NILC power spectrum
        index as Clyy[sim][0-2 for CMB, tSZ, or noise component][l]
    '''
    sum_reMASTERed_terms = np.sum(Clpq, axis=4)
    ClTTd = sum_reMASTERed_terms[:,0,0,:,:]
    ClTyd = sum_reMASTERed_terms[:,0,1,:,:]
    Clyyd = sum_reMASTERed_terms[:,1,1,:,:]
    return ClTTd, ClTyd, Clyyd

def get_theory_arrays(ClTTd, ClTyd, Clyyd):
    '''
    Get theory arrays by averaging over sims for data arrays

    ARGUMENTS
    ---------
    ClTTd: (Nsims, N_comps, ellmax+1) ndarray, contains contributions to CMB preserved NILC power spectrum
        index as ClTT[sim][0-2 for CMB, tSZ, or noise component][l]
    ClTyd: (Nsims, N_comps, ellmax+1) ndarray, contains contributions to CMB preserved NILC and tSZ preserved NILC cross-spectrum
        index as ClTy[sim][0-2 for CMB, tSZ, or noise component][l]
    Clyyd: (Nsims, N_comps, ellmax+1) ndarray, contains contributions to tSZ preserved NILC power spectrum
        index as Clyy[sim][0-2 for CMB, tSZ, or noise component][l]
    
    RETURNS
    -------
    ClTT: (N_comps, ellmax+1) ndarray, contains contributions to CMB preserved NILC power spectrum
        index as ClTT[0-2 for CMB, tSZ, or noise component][l]
    ClTy: (N_comps, ellmax+1) ndarray, contains contributions to CMB preserved NILC and tSZ preserved NILC cross-spectrum
        index as ClTy[0-2 for CMB, tSZ, or noise component][l]
    Clyy: (N_comps, ellmax+1) ndarray, contains contributions to tSZ preserved NILC power spectrum
        index as Clyy[0-2 for CMB, tSZ, or noise component][l]
    '''
    ClTT = np.mean(ClTTd, axis=0)
    ClTy = np.mean(ClTyd, axis=0)
    Clyy = np.mean(Clyyd, axis=0)
    return ClTT, ClTy, Clyy


def get_PScov_sim(inp, ClTTd, ClTyd, Clyyd):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    ClTTd: (Nsims, N_comps, ellmax+1) ndarray, contains contributions to CMB preserved NILC power spectrum
        index as ClTT[sim][0-2 for CMB, tSZ, or noise component][l]
    ClTyd: (Nsims, N_comps, ellmax+1) ndarray, contains contributions to CMB preserved NILC and tSZ preserved NILC cross-spectrum
        index as ClTy[sim][0-2 for CMB, tSZ, or noise component][l]
    Clyyd: (Nsims, N_comps, ellmax+1) ndarray, contains contributions to tSZ preserved NILC power spectrum
        index as Clyy[sim][0-2 for CMB, tSZ, or noise component][l]
    

    RETURNS
    -------
    cov: (ellmax+1,3,3) ndarray containing covariance matrix Cov_{pq,rs}
        index as cov[l, 0-2 for ClTT ClTy Clyy, 0-2 for ClTT ClTy Clyy]
    '''
    Clpqd_array = np.array([np.sum(ClTTd,axis=1), np.sum(ClTyd,axis=1), np.sum(Clyyd,axis=1)]) #shape (3 for ClTT ClTy Clyy, Nsims, ellmax+1)
    Clpqd_array = np.transpose(Clpqd_array, axes=(2,0,1)) #shape (ellmax+1, 3 for ClTT ClTy and Clyy, Nsims)
    cov = np.array([np.cov(Clpqd_array[l]) for l in range(inp.ellmax+1)]) #shape (ellmax+1,3,3)
    return cov





def get_all_acmb_atsz(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specificaitons 
    Clpq: (Nsims, N_preserved_comps, N_preserved_comps, N_comps=3, 4, ellmax+1) ndarray,
        contains contributions from each component to the power spectrum of NILC maps
        with preserved components p and q,
        index as Clpq[sim, p,q,z,reMASTERed term,l]

    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation

    '''

    def ClpqA(Acmb, Atsz):
        '''
        Model for theoretical spectra Clpq including Acmb and Atsz parameters

        ARGUMENTS
        ---------
        Acmb: float, scaling parameter for CMB power spectrum
        Atsz: float, scaling parameter for tSZ power spectrum

        RETURNS
        -------
        (ellmax+1, 2, 2) ndarray, 
        index as array[l, 0-1 for T or y, 0-1 for T or y]

        '''
        return np.array([[[Acmb*ClTT[0][l] + Atsz*ClTT[1][l] + ClTT[2][l], Acmb*ClTy[0][l] + Atsz*ClTy[1][l] + ClTy[2][l]], 
            [Acmb*ClTy[0][l] + Atsz*ClTy[1][l] + ClTy[2][l], Acmb*Clyy[0][l] + Atsz*Clyy[1][l] + Clyy[2][l]]] for l in range(inp.ellmax+1)])


    def lnL(pars, f, sim, inp): 
        '''
        Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)

        ARGUMENTS
        ---------
        pars: parameters to function f (not manually inputted but used by minimizer)
        f: function that returns theory model in terms of Acmb and Atsz
        sim: int, simulation number
        inp: Info object containing input parameter specifications

        RETURNS
        -------
        negative log likelihood for one simulation, combined over multipoles 
        '''
        model = f(*pars)
        ClTTd = np.sum(ClTTd[sim], axis=0)
        ClTyd = np.sum(ClTyd[sim], axis=0)
        Clyyd = np.sum(Clyyd[sim], axis=0)
        return np.sum([1/2* \
        ((model[l][0,0]-ClTTd[l])*PScov_sim_Inv[l][0,0]*(model[l][0,0]-ClTTd[l]) + (model[l][0,0]-ClTTd[l])*PScov_sim_Inv[l][0,1]*(model[l][0,1]-ClTyd[l]) + (model[l][0,0]-ClTTd[l])*PScov_sim_Inv[l][0,2]*(model[l][1,1]-Clyyd[l]) \
        + (model[l][0,1]-ClTyd[l])*PScov_sim_Inv[l][1,0]*(model[l][0,0]-ClTTd[l]) + (model[l][0,1]-ClTyd[l])*PScov_sim_Inv[l][1,1]*(model[l][0,1]-ClTyd[l]) + (model[l][0,1]-ClTyd[l])*PScov_sim_Inv[l][1,2]*(model[l][1,1]-Clyyd[l]) \
        + (model[l][1,1]-Clyyd[l])*PScov_sim_Inv[l][2,0]*(model[l][0,0]-ClTTd[l]) + (model[l][1,1]-Clyyd[l])*PScov_sim_Inv[l][2,1]*(model[l][0,1]-ClTyd[l]) + (model[l][1,1]-Clyyd[l])*PScov_sim_Inv[l][2,2]*(model[l][1,1]-Clyyd[l])) \
        for l in range(inp.ellmax+1)]) 

    def acmb_atsz(sim):
        '''
        Maximize likelihood with respect to Acmb and Atsz for one sim

        ARGUMENTS
        ---------
        sim: int, simulation number

        RETURNS
        -------
        best fit Acmb, Atsz (floats)
        '''
        acmb_start = 1.0
        atsz_start = 1.0
        res = minimize(lnL, x0 = [acmb_start, atsz_start], args = (ClpqA, sim, inp), method='Nelder-Mead') #default method is BFGS
        return res.x #acmb, atsz
    
    ClTTd, ClTyd, Clyyd = get_data_arrays(Clpq)
    ClTT, ClTy, Clyy = get_theory_arrays(ClTTd, ClTyd, Clyyd)
    PScov_sim = get_PScov_sim(inp, ClTTd, ClTyd, Clyyd)
    PScov_sim_Inv = np.array([scipy.linalg.inv(PScov_sim[l]) for l in range(inp.ellmax+1)])

    acmb_array = np.ones(inp.Nsims, dtype=np.float32)
    atsz_array = np.ones(inp.Nsims, dtype=np.float32)
    for sim in range(inp.Nsims):
        acmb, atsz = acmb_atsz(sim)
        acmb_array[sim] = acmb
        atsz_array[sim] = atsz
    
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_array.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_array.p', 'wb'))
    if inp.verbose:
        print(f'created {inp.output_dir}/acmb_array.p and {inp.output_dir}/atsz_array.p', flush=True)
   
    return acmb_array, atsz_array




def get_var(P, edges, scaling):
    '''
    ARGUMENTS
    ---------
    P: ndarray of marginalized probabilities 
    edges: (nbins+1,) ndarray containing edges of the bins
    scaling: float, (maximum of edges) - (minimum of edges)

    RETURNS
    -------
    lower: float, lower bound of parameter (68% confidence)
    upper: float, upper bound of parameter (68% confidence)
    mean: float, mean of parameter

    Note: chi^2 for 1 sigma and 1 dof is 1.00
    '''
    P = -0.5*np.log(P) #convert probability to chi^2
    min, idx_min = np.amin(P), np.argmin(P)
    lower_idx, upper_idx = None, None
    for i, elt in enumerate(P):
        if i < idx_min:
            if abs(elt-min) <= 1.0 and lower_idx is None: #gets smallest value within 68% confidence interval
                lower_idx = i 
        elif i > idx_min:
            if abs(elt-min) <= 1.0: #gets largest value within 68% confidence interval
                upper_idx = i
    lower = scaling*lower_idx/len(P)+np.amin(edges)
    upper = scaling*upper_idx/len(P)+np.amin(edges)
    mean = scaling*idx_min/len(P)+np.amin(edges)
    return lower, upper, mean

def get_parameter_cov_matrix(acmb_array, atsz_array, nbins=100, smoothing_factor=0.065):
    '''
    ARGUMENTS
    ---------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation
    nbins: int, number of bins in each dimension for histogram of Acmb and Atsz values 
    smoothing_factor: float, nbins*smoothing_factor is standard deviation of Gaussian kernel for smoothing histogram

    RETURNS
    -------
    lower_acmb: float, lower bound of Acmb (68% confidence)
    upper_acmb: float, upper bound of Acmb (68% confidence)
    mean_acmb: float, mean value of Acmb
    lower_atsz: float, lower bound of Atsz (68% confidence)
    upper_atsz: float, upper bound of Atsz (68% confidence)
    mean_atsz: float, mean value of Atsz
    '''
    hist, xedges, yedges = np.histogram2d(acmb_array, atsz_array, bins=[nbins, nbins])
    hist = hist/np.sum(hist)
    hist = ndimage.gaussian_filter(hist, nbins*smoothing_factor) #smooth hist
    scaling = [np.amax(xedges)-np.amin(xedges), np.amax(yedges)-np.amin(yedges)]
    lower_acmb, upper_acmb, mean_acmb = get_var(np.sum(hist, axis=1), xedges, scaling[0])
    lower_atsz, upper_atsz, mean_atsz = get_var(np.sum(hist, axis=0), yedges, scaling[1])
    return lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz
    



