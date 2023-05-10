import numpy as np
from scipy.optimize import curve_fit
import pickle

def fit_func(A, n):
    '''
    ARGUMENTS
    ---------
    A: independent variable--A_CMB, A_tSZ, A_noise1, or A_noise2
    n: float, best fit exponent for power law scaling
    '''
    return A**n

def call_fit(A_vec, n_vec):
    '''
    ARGUMENTS
    ---------
    A_vec: list of [A_y, A_z] independent variables--A_CMB, A_tSZ, A_noise1, or A_noise2
    n_vec: list of floats [ny, nz] giving best fit exponents for power law scaling for A_y and A_z

    RETURNS
    -------
    A_y**ny * A_z**nz
    '''
    A_y, A_z = A_vec
    ny, nz = n_vec
    return fit_func(A_y, ny) * fit_func(A_z, nz)

def get_parameter_dependence(inp, Clpq, scale_factor):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, 2, 2, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra
    scale_factor: float, multiplicative scaling factor used to determine parameter dependence
    
    RETURNS
    -------
    best_fits: (N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 9) array
        containing best fits to A_y and A_z

    '''
    N_preserved_comps = 2
    N_comps = 4
    x_vals = [1., scale_factor]

    Clpq_mean = np.transpose( np.mean(Clpq, axis=0), axes=(2,3,4,5,6,0,1) )

    best_fits = np.zeros((N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 2)) #2 is for 2 exponent params
    for p in range(N_preserved_comps):
        for q in range(N_preserved_comps):
            for y in range(N_comps):
                for z in range(N_comps):
                    for ell in range(inp.ellmax+1):
                        best_fits[p,q,y,z,ell,0] = curve_fit(fit_func, x_vals, Clpq_mean[p,q,y,z,ell,:,0])
                        best_fits[p,q,y,z,ell,1] = curve_fit(fit_func, x_vals, Clpq_mean[p,q,y,z,ell,0,:])
    
    if inp.save_files:
        pickle.dump(best_fits, open(f'{inp.output_dir}/data_vecs/best_fits.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/best_fits.p')
    
    return best_fits
