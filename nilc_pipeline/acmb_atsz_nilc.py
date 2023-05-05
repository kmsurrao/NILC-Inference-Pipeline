import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
from scipy import ndimage
from fits import func_to_fit, call_fit, get_parameter_dependence



def get_PScov_sim(inp, Clpq_scaling1):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq_scaling1: (Nsims, N_preserved_comps=2, N_preserved_comps=2, N_comps=3, N_comps=3, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra
    
    RETURNS
    -------
    cov: (ellmax+1,3,3) ndarray containing covariance matrix Cov_{pq,rs}
        index as cov[l, 0-2 for ClTT ClTy Clyy, 0-2 for ClTT ClTy Clyy]
    '''
    Clpq_tmp = np.sum(Clpq_scaling1, axis=(3,4))
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
    Clpq: (Nsims, N_amps, N_amps, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra

    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation

    '''

    def ClpqA(Acmb, Atsz, Anoise1, Anoise2):
        '''
        Model for theoretical spectra Clpq including Acmb, Atsz, and Anoise parameters

        ARGUMENTS
        ---------
        Acmb: float, scaling parameter for CMB power spectrum
        Atsz: float, scaling parameter for tSZ power spectrum
        Anoise1: float, scaling parameter for 90 GHz noise power spectrum
        Anoise2: float, scaling parameter for 150 GHz noise power spectrum

        RETURNS
        -------
        theory_model: (ellmax+1, 2, 2) ndarray for ClTT, ClTy, ClyT, and Clyy in terms of A_y and A_z parameters

        '''
        theory_model = np.zeros((inp.ellmax+1, 2, 2))

        for l in range(inp.ellmax+1):
            for p,q in [(0,0), (0,1), (1,0), (1,1)]:

                if p==0 and q==0: 
                    best_fits_here, Clpq_here = best_fits[0,0], ClTT
                elif p==0 and q==1:
                    best_fits_here, Clpq_here = best_fits[0,1], ClTy
                elif p==1 and q==0:
                    best_fits_here, Clpq_here = best_fits[1,0], ClyT
                elif p==1 and q==1:
                    best_fits_here, Clpq_here = best_fits[1,1], Clyy

                theory_model[l,p,q] = \
                call_fit([Acmb,Acmb], best_fits_here[0,0,l])*Clpq_here[0,0]      + call_fit([Acmb,Atsz], best_fits_here[0,1,l])*Clpq_here[0,1]      + call_fit([Acmb,Anoise1], best_fits_here[0,2,l])*Clpq_here[0,2]       + call_fit([Acmb,Anoise2], best_fits_here[0,3,l])*Clpq_here[0,3]\
                + call_fit([Atsz,Acmb], best_fits_here[1,0,l])*Clpq_here[1,0]      + call_fit([Atsz,Atsz], best_fits_here[1,1,l])*Clpq_here[1,1]      + call_fit([Atsz,Anoise1], best_fits_here[1,2,l])*Clpq_here[1,2]       + call_fit([Atsz,Anoise2], best_fits_here[1,3,l])*Clpq_here[1,3] \
                + call_fit([Anoise1,Acmb], best_fits_here[2,0,l])*Clpq_here[2,0]   + call_fit([Anoise1,Atsz], best_fits_here[2,1,l])*Clpq_here[2,1]   + call_fit([Anoise1,Anoise1], best_fits_here[2,2,l])*Clpq_here[2,2]    + call_fit([Anoise1,Anoise2], best_fits_here[2,3,l])*Clpq_here[2,3] \
                + call_fit([Anoise2,Acmb], best_fits_here[3,0,l])*Clpq_here[3,0]   + call_fit([Anoise2,Atsz], best_fits_here[3,1,l])*Clpq_here[3,1]   + call_fit([Anoise2,Anoise1], best_fits_here[3,2,l])*Clpq_here[3,2]    + call_fit([Anoise2,Anoise2], best_fits_here[3,3,l])*Clpq_here[3,3]
    
        return theory_model



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
        ClTTd = np.mean(np.sum(ClTT_all_sims, axis=(1,2)), axis=0)
        ClTyd = np.mean(np.sum(ClTy_all_sims, axis=(1,2)), axis=0)
        Clyyd = np.mean(np.sum(Clyy_all_sims, axis=(1,2)), axis=0)
        return np.sum([1/2* \
            ((model[l][0,0]-ClTTd[l])*PScov_sim_Inv[l][0,0]*(model[l][0,0]-ClTTd[l]) + (model[l][0,0]-ClTTd[l])*PScov_sim_Inv[l][0,1]*(model[l][0,1]-ClTyd[l]) + (model[l][0,0]-ClTTd[l])*PScov_sim_Inv[l][0,2]*(model[l][1,1]-Clyyd[l]) \
        + (model[l][0,1]-ClTyd[l])*PScov_sim_Inv[l][1,0]*(model[l][0,0]-ClTTd[l]) + (model[l][0,1]-ClTyd[l])*PScov_sim_Inv[l][1,1]*(model[l][0,1]-ClTyd[l]) + (model[l][0,1]-ClTyd[l])*PScov_sim_Inv[l][1,2]*(model[l][1,1]-Clyyd[l]) \
        + (model[l][1,1]-Clyyd[l])*PScov_sim_Inv[l][2,0]*(model[l][0,0]-ClTTd[l]) + (model[l][1,1]-Clyyd[l])*PScov_sim_Inv[l][2,1]*(model[l][0,1]-ClTyd[l]) + (model[l][1,1]-Clyyd[l])*PScov_sim_Inv[l][2,2]*(model[l][1,1]-Clyyd[l])) \
        for l in range(2, inp.ellmax+1)]) 

    def acmb_atsz():
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
        anoise1_start = 1.0
        anoise2_start = 1.0
        bounds = ((0.0, None), (0.0, None), (0.0, None), (0.0, None))
        res = minimize(lnL, x0 = [acmb_start, atsz_start, anoise1_start, anoise2_start], args = (ClpqA, inp), method='Nelder-Mead', bounds=bounds) #default method is BFGS
        return res.x #acmb, atsz, anoise1, anoise2
    
    best_fits = get_parameter_dependence(inp, Clpq) #(N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 9)
    Clpq_scaling1 = Clpq[:,0,0]

    PScov_sim = get_PScov_sim(inp, Clpq_scaling1)
    PScov_sim_Inv = np.array([scipy.linalg.inv(PScov_sim[l]) for l in range(inp.ellmax+1)])

    ClTT_all_sims, ClTy_all_sims, ClyT_all_sims, Clyy_all_sims = Clpq_scaling1[:,0,0], Clpq_scaling1[:,0,1], Clpq_scaling1[:,1,0], Clpq_scaling1[:,1,1]

    acmb_array = np.ones(inp.Nsims, dtype=np.float32)
    atsz_array = np.ones(inp.Nsims, dtype=np.float32)
    anoise1_array = np.ones(inp.Nsims, dtype=np.float32)
    anoise2_array = np.ones(inp.Nsims, dtype=np.float32)
    for sim in range(inp.Nsims):
        ClTT, ClTy, ClyT, Clyy = ClTT_all_sims[sim], ClTy_all_sims[sim], ClyT_all_sims[sim], Clyy_all_sims[sim]
        acmb, atsz, anoise1, anoise2 = acmb_atsz()
        acmb_array[sim] = acmb
        atsz_array[sim] = atsz
        anoise1_array[sim] = anoise1
        anoise2_array[sim] = anoise2
    
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_array_nilc.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_array_nilc.p', 'wb'))
    pickle.dump(anoise1_array, open(f'{inp.output_dir}/anoise1_array_nilc.p', 'wb'))
    pickle.dump(anoise2_array, open(f'{inp.output_dir}/anoise2_array_nilc.p', 'wb'))
    if inp.verbose:
        print(f'created {inp.output_dir}/acmb_array_nilc.p, atsz_array_nilc.p, anoise1_array_nilc.p, anoise2_array_nilc.p', flush=True)
   
    return acmb_array, atsz_array, anoise1_array, anoise2_array




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
    



