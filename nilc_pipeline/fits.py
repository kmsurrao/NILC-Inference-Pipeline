import numpy as np
from scipy.optimize import curve_fit
import pickle

def func_to_fit(A_vec, *coeffs):
    Ay, Az = A_vec
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeffs
    return c1*Ay**3 + c2*Az**3 + c3*Ay**2*Az + c4*Ay*Az**2 + \
            c5*Ay**2 + c6*Az**2 + c7*Ay*Az + c8*Ay + c9*Az

def call_fit(A_vec, coeffs):
    Ay, Az = A_vec
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeffs
    return c1*Ay**3 + c2*Az**3 + c3*Ay**2*Az + c4*Ay*Az**2 + \
            c5*Ay**2 + c6*Az**2 + c7*Ay*Az + c8*Ay + c9*Az


def get_parameter_dependence(inp, Clpq):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, N_amps, N_amps, N_preserved_comps=2, N_preserved_comps=2, N_comps=3, N_comps=3, ellmax+1) ndarray 
        containing propagation of each pair of component maps
        to NILC map auto- and cross-spectra
    
    RETURNS
    -------
    best_fits: (N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 9) array
        containing best fits to A_y and A_z

    '''
    N_preserved_comps = 2
    N_comps = 4
    scalings = [1,10,50,100]

    Clpq_mean = np.transpose( np.mean(Clpq, axis=0), axes=(2,3,4,5,6,0,1) )

    x_vals = [[], []]
    for s1 in range(len(scalings)):
        for s2 in range(len(scalings)):
            x_vals[0].append(scalings[s1])
            x_vals[1].append(scalings[s2])

    best_fits = np.zeros((N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.ellmax+1, 9))
    for p in range(N_preserved_comps):
        for q in range(N_preserved_comps):
            for y in range(N_comps):
                for z in range(N_comps):
                    for ell in range(inp.ellmax+1):
                        best_fits[p,q,y,z,ell] = curve_fit(func_to_fit, x_vals, Clpq_mean[p,q,y,z,ell], p0=np.zeros(9))
    
    if inp.save_files:
        pickle.dump(best_fits, open(f'{inp.output_dir}/data_vecs/best_fits.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/best_fits.p')
    
    return best_fits
