import numpy as np
from scipy.optimize import curve_fit
import pickle

def fit_func(A, n):
    '''
    ARGUMENTS
    ---------
    A: independent variable--Acmb, Atsz, Anoise1, or Anoise2
    n: float, best fit exponent for power law scaling
    '''
    return A**n

def call_fit(A_vec, n_vec):
    '''
    ARGUMENTS
    ---------
    A_vec: list of [Acmb, Atsz, Anoise1, Anoise2] independent variables
    n_vec: list of floats [ncmb, ntsz, nnoise1, nnoise2] giving best fit exponents for power law scaling for each A_z

    RETURNS
    -------
    Acmb**ncmb * Atsz**ntsz * Anoise1**nnoise1 * Anoise2*nnoise2
    '''
    Acmb, Atsz, Anoise1, Anoise2 = A_vec
    ncmb, ntsz, nnoise1, nnoise2 = n_vec
    return fit_func(Acmb, ncmb) * fit_func(Atsz, ntsz) * fit_func(Anoise1, nnoise1) * fit_func(Anoise2, nnoise2)

def get_parameter_dependence(inp, Clpq, scale_factor):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, N_comps+1, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra
    scale_factor: float, multiplicative scaling factor used to determine parameter dependence
    
    RETURNS
    -------
    best_fits: (N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 4) array
        containing best fits to Acmb, Atsz, Anoise1, Anoise2

    '''
    N_preserved_comps = 2
    N_comps = 4
    x_vals = [scale_factor**2] #square needed since each comp scaled at map level and want parameter fit at power spectrum level

    Clpq_mean = np.mean(Clpq, axis=0)

    best_fits = np.zeros((N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 4)) #4 is for 4 exponent params
    for p in range(N_preserved_comps):
        for q in range(N_preserved_comps):
            for y in range(N_comps):
                for z in range(N_comps):
                    for ell in range(inp.ellmax+1):
                        for s in range(N_comps):
                            best_fits[s,p,q,y,z,ell] = curve_fit(fit_func, x_vals, Clpq_mean[s,p,q,y,z,ell]/Clpq_mean[N_comps,p,q,y,z,ell])[0][0]
    
    if inp.save_files:
        pickle.dump(best_fits, open(f'{inp.output_dir}/data_vecs/best_fits.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/best_fits.p')
    
    return best_fits
