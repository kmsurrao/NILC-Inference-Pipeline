import numpy as np
import pickle
from pysr import PySRRegressor 
from utils import get_scalings

def symbolic_regression(inp, x_vals, y_vals):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    x_vals: list of points, where each point contains value of Acmb, Atsz
    y_vals: Clpq evaluated at scaled points / Clpq evaluated at all unscaled points

    RETURNS
    -------
    sympy expression of best fit model 
    '''
    model = PySRRegressor(
        niterations = 50,  # < Increase me for better results
        ncyclesperiteration = 1000,
        progress = False, 
        maxsize = 12,
        binary_operators = ["*", "+", "-", "/"],
        unary_operators = ["exp", "square", "cube", "inv(x) = 1/x"],
        extra_sympy_mappings = {"inv": lambda x: 1 / x},
        loss = "loss(prediction, target) = (prediction - target)^2",
        verbosity = 0,
        temp_equation_file = True,
        tempdir = inp.output_dir, 
        delete_tempfiles = True
    )
    model.fit(x_vals, y_vals)
    return model.sympy()


def call_fit(A_vec, expr):
    '''
    ARGUMENTS
    ---------
    A_vec: list of [Acmb, Atsz] independent variables
    expr: sympy expression of best fit involving parameters Acmb, Atsz,
        which map to x0, x1, respectively 

    RETURNS
    -------
    numerical evaluation of expr at the point given by A_vec
    '''
    return expr.subs('x0', A_vec[0]).subs('x1', A_vec[1])


def get_parameter_dependence(inp, Clpq, HILC=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims_for_fits, Nscalings, 2, 2, N_preserved_comps=2, N_preserved_comps=2, Nbins) ndarray 
        containing HILC/NILC map auto- and cross-spectra with different component scalings. 
        dim1: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
            idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim2: 0 for unscaled CMB, 1 for scaled CMB
        dim3: 0 for unscaled ftSZ, 1 for scaled ftSZ
    HILC: Bool, set to True if computing paramter dependence for harmonic ILC, False if for needlet ILC
    
    RETURNS
    -------
    best_fits: (N_preserved_comps, N_preserved_comps, Nbins) list
        containing sympy expressions with best fits to Acmb, Atsz

    '''
    print('\nRunning symbolic regression to get parameter dependence. This may take some time.', flush=True)

    N_preserved_comps = 2
    N_comps = 2
    Clpq_mean = np.mean(Clpq[:inp.Nsims_for_fits], axis=0)

    best_fits = np.zeros((N_preserved_comps, N_preserved_comps, inp.Nbins)).tolist() #need list to store sympy expressions
    scalings = get_scalings(inp)
    for p,q in [(0,0), (0,1), (1,0), (1,1)]:
        for bin in range(inp.Nbins):
            x_vals, y_vals = [], []
            for s in scalings:
                scaling_factor = (inp.scaling_factors[s[0]])**2
                x = np.ones(N_comps)
                x[np.array(s[1:])==1] = scaling_factor
                x_vals.append(x)
                y_vals.append(Clpq_mean[s[0],s[1],s[2],p,q,bin]/Clpq_mean[0,0,0,p,q,bin])
            best_fits[p][q][bin] = symbolic_regression(inp, x_vals, y_vals)
            if inp.verbose: print(f'estimated parameter dependence for p,q,bin={p},{q},{bin}', flush=True)

    if inp.save_files:
        if HILC:
            filename = f'{inp.output_dir}/data_vecs/best_fits_HILC.p'
        else:
            filename = f'{inp.output_dir}/data_vecs/best_fits_NILC.p'
        pickle.dump(best_fits, open(filename, 'wb'), protocol=4)
        print(f'saved {filename}', flush=True)
    
    return best_fits
