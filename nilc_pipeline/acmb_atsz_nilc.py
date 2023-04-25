import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
from scipy import ndimage



def get_PScov_sim(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, N_comps=3, N_comps=3, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra
    
    RETURNS
    -------
    cov: (ellmax+1,3,3) ndarray containing covariance matrix Cov_{pq,rs}
        index as cov[l, 0-2 for ClTT ClTy Clyy, 0-2 for ClTT ClTy Clyy]
    '''
    Clpq_tmp = np.sum(Clpq, axis=(3,4))
    Clpq_tmp = np.array([Clpq_tmp[:,0,0], Clpq_tmp[:,0,1], Clpq_tmp[:,1,1]])
    Clpq_tmp = np.transpose(Clpq_tmp, axes=(2,0,1)) #shape (ellmax+1, 3 for ClTT, ClTy, Clyy, Nsims)
    cov = np.array([np.cov(Clpq_tmp[l]) for l in range(inp.ellmax+1)]) #shape (ellmax+1,3,3)
    assert cov.shape == (inp.ellmax+1, 3, 3), f"covariance shape is {cov.shape} but should be ({inp.ellmax+1},3,3)"
    return cov


def get_all_acmb_atsz(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clpq: (Nsims, N_preserved_comps=2, N_preserved_comps=2, N_comps=3, N_comps=3, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra

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
        ClTT_with_A = Acmb*ClTT[0,0]               + np.sqrt(Acmb*Atsz)*ClTT[0,1] + np.sqrt(Acmb)*ClTT[0,2] \
                    + np.sqrt(Acmb*Atsz)*ClTT[1,0] + Atsz*ClTT[1,1]               + np.sqrt(Atsz)*ClTT[1,2] \
                    + np.sqrt(Acmb)*ClTT[2,0]      + np.sqrt(Atsz)*ClTT[2,1]      + ClTT[2,2]
        
        ClTy_with_A = Acmb*ClTy[0,0]               + np.sqrt(Acmb*Atsz)*ClTy[0,1] + np.sqrt(Acmb)*ClTy[0,2] \
                    + np.sqrt(Acmb*Atsz)*ClTy[1,0] + Atsz*ClTy[1,1]               + np.sqrt(Atsz)*ClTy[1,2] \
                    + np.sqrt(Acmb)*ClTy[2,0]      + np.sqrt(Atsz)*ClTy[2,1]      + ClTy[2,2]
        
        ClyT_with_A = Acmb*ClyT[0,0]               + np.sqrt(Acmb*Atsz)*ClyT[0,1] + np.sqrt(Acmb)*ClyT[0,2] \
                    + np.sqrt(Acmb*Atsz)*ClyT[1,0] + Atsz*ClyT[1,1]               + np.sqrt(Atsz)*ClyT[1,2] \
                    + np.sqrt(Acmb)*ClyT[2,0]      + np.sqrt(Atsz)*ClyT[2,1]      + ClyT[2,2]
        
        Clyy_with_A = Acmb*Clyy[0,0]               + np.sqrt(Acmb*Atsz)*Clyy[0,1] + np.sqrt(Acmb)*Clyy[0,2] \
                    + np.sqrt(Acmb*Atsz)*Clyy[1,0] + Atsz*Clyy[1,1]               + np.sqrt(Atsz)*Clyy[1,2] \
                    + np.sqrt(Acmb)*Clyy[2,0]      + np.sqrt(Atsz)*Clyy[2,1]      + Clyy[2,2]
        
        return np.array([[[ClTT_with_A, ClTy_with_A], [ClyT_with_A, Clyy_with_A]] for l in range(inp.ellmax+1)])


    def lnL(pars, f, inp): 
        '''
        Expression for log likelihood for one sim (actually equal to negative lnL since we have to minimize)
        Let Clpqd be the data spectra obtained by averaging over all the theory spectra from each sim

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
        ClTTd = np.mean(np.sum(ClTT_all_sims, axis=(3,4)), axis=0)
        ClTyd = np.mean(np.sum(ClTy_all_sims, axis=(3,4)), axis=0)
        Clyyd = np.mean(np.sum(Clyy_all_sims, axis=(3,4)), axis=0)
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
        res = minimize(lnL, x0 = [acmb_start, atsz_start], args = (ClpqA, inp), method='Nelder-Mead') #default method is BFGS
        return res.x #acmb, atsz
    
    PScov_sim = get_PScov_sim(inp, Clpq)
    # PScov_sim_Inv = np.array([scipy.linalg.inv(PScov_sim[l]) for l in range(inp.ellmax+1)])
    #remove chunk below and uncomment above
    PScov_sim_Inv = np.zeros((inp.ellmax+1, 3, 3))
    for l in range(inp.ellmax+1):
        try:
            inv = scipy.linalg.inv(PScov_sim[l])
        except Exception:
            print('l: ', l)
            inv = np.eye(3,3)
        PScov_sim_Inv[l] = inv

    ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims = Clpq[:,0,0], Clpq[:,0,1], Clpq[:,1,0], Clpq[:,1,1]

    acmb_array = np.ones(inp.Nsims, dtype=np.float32)
    atsz_array = np.ones(inp.Nsims, dtype=np.float32)
    for sim in range(inp.Nsims):
        ClTT, ClTy, ClyT, Clyy = ClTT_all_sims[sim], ClTy_all_sims[sim], ClyT_all_sims[sim], Clyy_all_sims[sim]
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
    



