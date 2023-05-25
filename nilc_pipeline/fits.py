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
    n_vec: list of floats [ncmb_low, ntsz_low, nnoise1_low, nnoise2_low, ncmb_high, ntsz_high, nnoise1_high, nnoise2_high] 
        giving best fit exponents for power law scaling for each A_z, with "low" used for A_z < 1.0 and "high" for A_z > 1.0

    RETURNS
    -------
    Acmb**ncmb * Atsz**ntsz * Anoise1**nnoise1 * Anoise2*nnoise2, with low or high for each n determined based on whether
        A is < 1.0 or > 1.0
    '''
    N_comps = 4
    Acmb, Atsz, Anoise1, Anoise2 = A_vec
    ncmb = n_vec[0] if Acmb < 1.0 else n_vec[N_comps]
    ntsz = n_vec[1] if Atsz < 1.0 else n_vec[1+N_comps]
    nnoise1 = n_vec[2] if Anoise1 < 1.0 else n_vec[2+N_comps]
    nnoise2 = n_vec[3] if Anoise2 < 1.0 else n_vec[3+N_comps]
    return fit_func(Acmb, ncmb) * fit_func(Atsz, ntsz) * fit_func(Anoise1, nnoise1) * fit_func(Anoise2, nnoise2)

def get_parameter_dependence(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, N_comps+1, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra
    
    RETURNS
    -------
    best_fits: (N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 2*N_comps) ndarray
        containing best fits to Acmb, Atsz, Anoise1, Anoise2 for low and high scalings
        2*N_comps is for exponent params, N_comps for scaled low and N_comps for scaled high

    '''
    N_preserved_comps = 2
    N_comps = 4
    Clpq_mean = np.mean(Clpq, axis=0)

    best_fits = np.zeros((N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 2*N_comps))
    for p in range(N_preserved_comps):
        for q in range(N_preserved_comps):
            for y in range(N_comps):
                for z in range(N_comps):
                    for ell in range(inp.ellmax+1):
                        for s in range(2*N_comps):
                            x_vals = [(inp.scaling_factors[s>=N_comps])**2] #square needed since each comp scaled at map level and want parameter fit at power spectrum level
                            best_fits[p,q,y,z,ell,s] = curve_fit(fit_func, x_vals, [Clpq_mean[s,p,q,y,z,ell]/Clpq_mean[2*N_comps,p,q,y,z,ell]])[0][0]
    
    if inp.save_files:
        pickle.dump(best_fits, open(f'{inp.output_dir}/data_vecs/best_fits.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/best_fits.p', flush=True)
    
    return best_fits
