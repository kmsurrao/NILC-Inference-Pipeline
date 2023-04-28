import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
from scipy import ndimage



def get_PScov_sim(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=3, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j
    
    RETURNS
    -------
    cov: (ellmax+1,3,3) ndarray containing covariance matrix Cov_{ij,kl}
        index as cov[l, freq1, freq2]
    '''
    Clij_tmp = np.sum(Clij, axis=3)
    Clij_tmp = np.array([Clij_tmp[:,0,0], Clij_tmp[:,0,1], Clij_tmp[:,1,1]])
    Clij_tmp = np.transpose(Clij_tmp, axes=(2,0,1)) #shape (ellmax+1, 3 for Cl00 Cl01 and Cl11, Nsims)
    cov = np.array([np.cov(Clij_tmp[l]) for l in range(inp.ellmax+1)]) #shape (ellmax+1,3,3)
    assert cov.shape == (inp.ellmax+1, 3, 3), f"covariance shape is {cov.shape} but should be ({inp.ellmax+1},3,3)"
    return cov


def get_all_acmb_atsz(inp, Clij):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications 
    Clij: (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=3, ellmax+1) ndarray 
        containing contributions of each component to the 
        auto- and cross- spectra of freq maps at freqs i and j

    RETURNS
    -------
    acmb_array: array of length Nsims containing best fit Acmb for each simulation
    atsz_array: array of length Nsims containing best fit Atsz for each simulation

    '''

    def ClijA(Acmb, Atsz, Anoise=1.0):
        '''
        Model for theoretical spectra Clpq including Acmb and Atsz parameters

        ARGUMENTS
        ---------
        Acmb: float, scaling parameter for CMB power spectrum
        Atsz: float, scaling parameter for tSZ power spectrum

        RETURNS
        -------
        (ellmax+1, 2, 2) ndarray, 
        index as array[l;  0-2 or ij=00, 01, 11]

        '''

        Clij_with_A_00 = Acmb*Clij00[0] + Atsz*Clij00[1] + Anoise*Clij00[2]
        Clij_with_A_01 = Acmb*Clij01[0] + Atsz*Clij01[1] + Anoise*Clij01[2]
        Clij_with_A_10 = Acmb*Clij10[0] + Atsz*Clij10[1] + Anoise*Clij10[2]
        Clij_with_A_11 = Acmb*Clij11[0] + Atsz*Clij11[1] + Anoise*Clij11[2]
        return np.array([[[Clij_with_A_00[l], Clij_with_A_01[l]],[Clij_with_A_10[l], Clij_with_A_11[l]]] for l in range(inp.ellmax+1)])


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
        Clij00d = np.mean(np.sum(Clij00_all_sims, axis=1), axis=0)
        Clij01d = np.mean(np.sum(Clij01_all_sims, axis=1), axis=0)
        Clij11d = np.mean(np.sum(Clij11_all_sims, axis=1), axis=0)
        assert Clij00d.shape == (inp.ellmax+1,), f"Clij00d.shape is {Clij00d.shape}, should be ({inp.ellmax+1},)"
        return np.sum([1/2* \
         ((model[l][0,0]-Clij00d[l])*PScov_sim_Inv[l][0,0]*(model[l][0,0]-Clij00d[l]) + (model[l][0,0]-Clij00d[l])*PScov_sim_Inv[l][0,1]*(model[l][0,1]-Clij01d[l]) + (model[l][0,0]-Clij00d[l])*PScov_sim_Inv[l][0,2]*(model[l][1,1]-Clij11d[l]) \
        + (model[l][0,1]-Clij01d[l])*PScov_sim_Inv[l][1,0]*(model[l][0,0]-Clij00d[l]) + (model[l][0,1]-Clij01d[l])*PScov_sim_Inv[l][1,1]*(model[l][0,1]-Clij01d[l]) + (model[l][0,1]-Clij01d[l])*PScov_sim_Inv[l][1,2]*(model[l][1,1]-Clij11d[l]) \
        + (model[l][1,1]-Clij11d[l])*PScov_sim_Inv[l][2,0]*(model[l][0,0]-Clij00d[l]) + (model[l][1,1]-Clij11d[l])*PScov_sim_Inv[l][2,1]*(model[l][0,1]-Clij01d[l]) + (model[l][1,1]-Clij11d[l])*PScov_sim_Inv[l][2,2]*(model[l][1,1]-Clij11d[l])) \
        for l in range(2, inp.ellmax+1)]) 

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
        anoise_start = 1.0
        res = minimize(lnL, x0 = [acmb_start, atsz_start, anoise_start], args = (ClijA, inp), method='Nelder-Mead') #default method is BFGS
        return res.x #acmb, atsz, anoise
    
    PScov_sim = get_PScov_sim(inp, Clij)
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

    Clij00_all_sims, Clij01_all_sims, Clij10_all_sims, Clij11_all_sims = Clij[:,0,0], Clij[:,0,1], Clij[:,1,0], Clij[:,1,1]

    acmb_array = np.ones(inp.Nsims, dtype=np.float32)
    atsz_array = np.ones(inp.Nsims, dtype=np.float32)
    anoise_array = np.ones(inp.Nsims, dtype=np.float32)
    for sim in range(inp.Nsims):
        Clij00, Clij01, Clij10, Clij11 = Clij00_all_sims[sim], Clij01_all_sims[sim], Clij10_all_sims[sim], Clij11_all_sims[sim]
        acmb, atsz, anoise = acmb_atsz(sim)
        acmb_array[sim] = acmb
        atsz_array[sim] = atsz
        anoise_array[sim] = anoise
    
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_array_template_fitting.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_array_template_fitting.p', 'wb'))
    pickle.dump(anoise_array, open(f'{inp.output_dir}/anoise_array_template_fitting.p', 'wb'))
    if inp.verbose:
        print(f'created {inp.output_dir}/acmb_array_template_fitting.p and atsz_array_template_fitting.p and anoise_array_template_fitting.p', flush=True)
   
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
    



