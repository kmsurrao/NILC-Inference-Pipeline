import numpy as np
import pickle
from data_spectra import tsz_spectral_response


def load_Clpq():
    '''
    load data spectra arrays 
    index like ClTTd_array[sim][0-2 for Acmb, Atsz, or noise component][ell]
    '''
    ClTTd_array = pickle.load(open('power_spectra/clTT.p', 'rb'))
    ClTyd_array = pickle.load(open('power_spectra/clTy.p', 'rb'))
    Clyyd_array = pickle.load(open('power_spectra/clyy.p', 'rb'))
    return ClTTd_array, ClTyd_array, Clyyd_array

def get_theory_arrays(ClTTd_array, ClTyd_array, Clyyd_array):
    '''
    find theory arrays by averaging over sims for data arrays
    index like ClTT[0-2 for Acmb, Atsz, or noise component][ell]
    '''
    ClTT = np.mean(ClTTd_array, axis=0)
    ClTy = np.mean(ClTyd_array, axis=0)
    Clyy = np.mean(Clyyd_array, axis=0)
    return ClTT, ClTy, Clyy


def get_PScov_sim(ellmax, ClTTd_array, ClTyd_array, Clyyd_array):
    '''
    each column is one sim
    row 0 is ClTT, row 1 is ClTy, row2 is Clyy
    ClTTd_array has dimensions (Nsims, 3, ell)
    '''
    def cov_helper(array):
        #array starts with dim (Nsims, 3, ell)
        array = np.sum(array, axis=1) #now has dim (Nsims, ells)
        array  = np.transpose(array) #now has dim (ells, Nsims)
        return array
    cov = np.zeros((ellmax+1, 3, 3,))
    ClTTd_array = cov_helper(ClTTd_array)
    ClTyd_array = cov_helper(ClTyd_array)
    Clyyd_array = cov_helper(Clyyd_array)
    Clpqd_array = np.array([[ClTTd_array[l], ClTyd_array[l], Clyyd_array[l]] for l in range(ellmax+1)]) #dim (ells,3,Nsims)
    cov = np.array([np.cov(Clpqd_array[l]) for l in range(ellmax+1)]) #dim (ells,3,3)
    return cov


def ClpqA(Acmb, Atsz):
    '''
    model for theoretical spectra Clpq including Acmb and Atsz parameters
    '''
    ClTT, ClTy, Clyy = get_theory_arrays(load_Clpq())
    ellmax = len(ClTT[0])-1
    return np.array([[[Acmb*ClTT[0][l] + Atsz*ClTT[1][l] + ClTT[2][l], Acmb*ClTy[0][l] + Atsz*ClTy[1][l] + ClTy[2][l]], [Acmb*ClTy[0][l] + Atsz*ClTy[1][l] + ClTy[2][l], Acmb*Clyy[0][l] + Atsz*Clyy[1][l] + Clyy[2][l]]] for l in range(ellmax+1)])


def lnL(pars, f, sim, ellmax, ClTTd_array, ClTyd_array, Clyyd_array, PScov_sim_Inv): 
    '''
    Write expression for likelihood for one sim
    (Actually equal to negative lnL since we have to minimize)
    '''
    model = f(*pars)
    ClTTd = ClTTd_array[sim]
    ClTyd = ClTyd_array[sim]
    Clyyd = Clyyd_array[sim]
    return np.sum([1/2* \
    ((model[l][0,0]-ClTTd[l])*PScov_sim_Inv[l][0,0]*(model[l][0,0]-ClTTd[l]) + (model[l][0,0]-ClTTd[l])*PScov_sim_Inv[l][0,1]*(model[l][0,1]-ClTyd[l]) + (model[l][0,0]-ClTTd[l])*PScov_sim_Inv[l][0,2]*(model[l][1,1]-Clyyd[l]) \
    + (model[l][0,1]-ClTyd[l])*PScov_sim_Inv[l][1,0]*(model[l][0,0]-ClTTd[l]) + (model[l][0,1]-ClTyd[l])*PScov_sim_Inv[l][1,1]*(model[l][0,1]-ClTyd[l]) + (model[l][0,1]-ClTyd[l])*PScov_sim_Inv[l][1,2]*(model[l][1,1]-Clyyd[l]) \
    + (model[l][1,1]-Clyyd[l])*PScov_sim_Inv[l][2,0]*(model[l][0,0]-ClTTd[l]) + (model[l][1,1]-Clyyd[l])*PScov_sim_Inv[l][2,1]*(model[l][0,1]-ClTyd[l]) + (model[l][1,1]-Clyyd[l])*PScov_sim_Inv[l][2,2]*(model[l][1,1]-Clyyd[l])) \
    for l in range(ellmax+1)]) 

def acmb_atsz(sim, ellmax, ClTTd_array, ClTyd_array, Clyyd_array, PScov_sim_Inv):
    '''
    Maximize likelihood with respect to Acmb and Atsz for one sim
    '''
    acmb_start = 1.0
    atsz_start = 1.0
    res = minimize(lnL, x0 = [acmb_start, atsz_start], args = (ClpqA, sim, ellmax, ClTTd_array, ClTyd_array, Clyyd_array, PScov_sim_Inv), method='Nelder-Mead') #default method is BFGS
    return res.x #acmb, atsz


def get_all_acmb_atsz(Nsims, ellmax, verbose):
    ClTTd_array, ClTyd_array, Clyyd_array = load_Clpq()
    ClTT, ClTy, Clyy = get_theory_arrays(ClTTd_array, ClTyd_array, Clyyd_array)
    PScov_sim = get_PScov_sim(ellmax, ClTTd_array, ClTyd_array, Clyyd_array)
    PScov_sim_Inv = np.array([scipy.linalg.inv(PScov_sim[l]) for l in range(ellmax+1)])
    acmb_array = []
    atsz_array = []
    for i in range(Nsims):
        acmb, atsz = acmb_atsz(i, ellmax, ClTTd_array, ClTyd_array, Clyyd_array, PScov_sim_Inv)
        acmb_array.append(acmb)
        atsz_array.append(atsz)
    pickle.dump(acmb_array, open('acmb_array.p', 'wb'))
    pickle.dump(atsz_array, open('atsz_array.p', 'wb'))
    if verbose:
        print('created acmb_array.p and atsz_array.p')
    return acmb_array, atsz_array

def get_var(P, edges, scaling):
    P = -0.5*np.log(P) #convert probability to chi^2
    min, idx_min = min(P), np.argmin(P)
    lower_idx, upper_idx = None, None
    for i, elt in enumerate(P):
        if i < idx_min:
            if elt-min <= 1.0 and lower_idx is None:
                lower_idx = i 
        elif i > idx_min:
            if elt-min <= 1.0:
                upper_idx = i
    lower = scaling*lower_idx/len(P)+min(edges)
    upper = scaling*upper_idx/len(P)+min(edges)
    mean = scaling*idx_min/len(P)+min(edges)
    return lower, upper, mean

def get_parameter_cov_matrix(Nsims, ellmax, verbose, nbins=1000, smoothing_factor=0.065):
    acmb, atsz = get_all_acmb_atsz(Nsims, ellmax, verbose)
    hist, xedges, yedges = np.histogram2d(acmb, atsz, bins=[nbins, nbins])
    hist = hist/np.sum(hist)
    hist = ndimage.gaussian_filter(hist, nbins*smoothing_factor) #smooth hist
    scaling = [max(xedges)-min(xedges), max(yedges)-min(yedges)]
    lower_acmb, upper_acmb, mean_acmb = get_var(np.sum(hist, axis=1), xedges, scaling[0])
    lower_atsz, upper_atsz, mean_atsz = get_var(np.sum(hist, axis=0), yedges, scaling[1])
    return lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz
    



