import numpy as np
import pickle
import scipy
from scipy.optimize import minimize
from scipy import ndimage
from data_spectra import tsz_spectral_response


def load_Clpq():
    '''
    load data spectra arrays 
    index like ClTTd_array[sim][0-2 for Acmb, Atsz, or noise component][ell]
    '''
    data_spectra = pickle.load(open('data_spectra.p', 'rb'))
    ClTTd_array = data_spectra[:,0,:,:]
    ClTyd_array = data_spectra[:,1,:,:]
    Clyyd_array = data_spectra[:,2,:,:]
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

def get_tot_data_spectra(array):
    #array starts with dim (Nsims, 3, ell), where 3 is for CMB, tSZ, and noise components
    new_array = np.sum(array, axis=1) #now has dim (Nsims, ells)
    new_array  = np.transpose(new_array) #now has dim (ells, Nsims)
    return new_array

def get_PScov_sim(ellmax, ClTTd_array, ClTyd_array, Clyyd_array):
    '''
    each column is one sim
    row 0 is ClTT, row 1 is ClTy, row2 is Clyy
    original ClTTd_array has dimensions (Nsims, 3, ell)
    after get_tot_data_spectra, ClTTd_array has dimensions (ell, Nsims)
    '''

    cov = np.zeros((ellmax+1, 3, 3,))
    ClTTd_array = get_tot_data_spectra(ClTTd_array)
    ClTyd_array = get_tot_data_spectra(ClTyd_array)
    Clyyd_array = get_tot_data_spectra(Clyyd_array)
    Clpqd_array = [[ClTTd_array[l], ClTyd_array[l], Clyyd_array[l]] for l in range(ellmax+1)] #dim (ells, 3 for ClTT ClTy and Clyy, Nsims)
    cov = np.array([np.cov(Clpqd_array[l]) for l in range(ellmax+1)]) #dim (ells,3,3)
    return cov


def ClpqA(Acmb, Atsz):
    '''
    model for theoretical spectra Clpq including Acmb and Atsz parameters
    '''
    ClTTd_array, ClTyd_array, Clyyd_array = load_Clpq()
    ClTT, ClTy, Clyy = get_theory_arrays(ClTTd_array, ClTyd_array, Clyyd_array)
    ellmax = len(ClTT[0])-1
    return np.array([[[Acmb*ClTT[0][l] + Atsz*ClTT[1][l] + ClTT[2][l], Acmb*ClTy[0][l] + Atsz*ClTy[1][l] + ClTy[2][l]], [Acmb*ClTy[0][l] + Atsz*ClTy[1][l] + ClTy[2][l], Acmb*Clyy[0][l] + Atsz*Clyy[1][l] + Clyy[2][l]]] for l in range(ellmax+1)])


def lnL(pars, f, sim, ellmax, ClTTd_array, ClTyd_array, Clyyd_array, PScov_sim_Inv): 
    '''
    Write expression for likelihood for one sim
    (Actually equal to negative lnL since we have to minimize)
    ClTTd_array has dim (Nsims, 3, ells)
    '''
    model = f(*pars)
    ClTTd = np.sum(ClTTd_array[sim], axis=0)
    ClTyd = np.sum(ClTyd_array[sim], axis=0)
    Clyyd = np.sum(Clyyd_array[sim], axis=0)
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
    # PScov_sim_Inv = np.array([scipy.linalg.inv(PScov_sim[l]) for l in range(ellmax+1)])
    #replace chunk below with line above
    PScov_sim_Inv = np.zeros((ellmax+1,3,3))
    for l in range(ellmax+1):
        try:
            PScov_sim_Inv[l] = scipy.linalg.inv(PScov_sim[l])
        except np.linalg.LinAlgError:
            print(l)
            print(PScov_sim_Inv[l])
            PScov_sim_Inv[l] = np.identity(3)
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
    min, idx_min = np.amin(P), np.argmin(P)
    lower_idx, upper_idx = None, None
    for i, elt in enumerate(P):
        if i < idx_min:
            if elt-min <= 1.0 and lower_idx is None:
                lower_idx = i 
        elif i > idx_min:
            if elt-min <= 1.0:
                upper_idx = i
    lower = scaling*lower_idx/len(P)+np.amin(edges)
    upper = scaling*upper_idx/len(P)+np.amin(edges)
    mean = scaling*idx_min/len(P)+np.amin(edges)
    return lower, upper, mean

def get_parameter_cov_matrix(Nsims, ellmax, verbose, nbins=100, smoothing_factor=0.065):
    acmb, atsz = get_all_acmb_atsz(Nsims, ellmax, verbose)
    hist, xedges, yedges = np.histogram2d(acmb, atsz, bins=[nbins, nbins])
    hist = hist/np.sum(hist)
    hist = ndimage.gaussian_filter(hist, nbins*smoothing_factor) #smooth hist
    scaling = [np.amax(xedges)-np.amin(xedges), np.amax(yedges)-np.amin(yedges)]
    lower_acmb, upper_acmb, mean_acmb = get_var(np.sum(hist, axis=1), xedges, scaling[0])
    lower_atsz, upper_atsz, mean_atsz = get_var(np.sum(hist, axis=0), yedges, scaling[1])
    return lower_acmb, upper_acmb, mean_acmb, lower_atsz, upper_atsz, mean_atsz
    



