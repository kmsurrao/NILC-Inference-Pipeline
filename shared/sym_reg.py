import numpy as np
import pickle
from pysr import PySRRegressor 
import itertools
from utils import get_scalings, get_naming_str, sublist_idx

def symbolic_regression(inp, x_vals, y_vals):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    x_vals: list of points, where each point contains value of Acomp1, Acomp2, etc.
    y_vals: Clpq evaluated at scaled points / Clpq evaluated at all unscaled points

    RETURNS
    -------
    sympy expression of best fit model 
    '''
    model = PySRRegressor(
        niterations = 50,  # < Increase me for better results
        ncycles_per_iteration = 1000,
        progress = False, 
        maxsize = 12,
        binary_operators = ["*", "+", "-", "/"],
        unary_operators = ["exp", "square", "cube", "inv(x) = 1/x"],
        extra_sympy_mappings = {"inv": lambda x: 1 / x},
        elementwise_loss = "loss(prediction, target) = (prediction - target)^2",
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
    A_vec: list of [Acomp1, Acomp2, etc.] independent variables
    expr: sympy expression of best fit involving parameters Acomp1, Acomp2, etc.
        which map to x0, x1, etc., respectively 

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
    Clpq: (Nsims_for_fits, Nscalings, 2**Ncomps, Ncomps, Ncomps, Nbins) ndarray 
        containing HILC/NILC map auto- and cross-spectra with different component scalings. 
        dim1: idx0 if "scaled" means maps are scaled according to scaling factor 0 from input, 
            idx1 if "scaled" means maps are scaled according to scaling factor 1 from input, etc. up to idx Nscalings
        dim2: indices correspond to different combinations of scaled and unscaled components
    HILC: Bool, set to True if computing paramter dependence for harmonic ILC, False if for needlet ILC
    
    RETURNS
    -------
    best_fits: (Ncomps, Ncomps, Nbins) list
        containing sympy expressions with best fits in terms of parameters

    '''
    print('\nRunning symbolic regression to get parameter dependence. This may take some time.', flush=True)

    Ncomps = len(inp.comps)
    Clpq_mean = np.mean(Clpq[:inp.Nsims_for_fits], axis=0)
    comp_scalings = [list(i) for i in itertools.product([0, 1], repeat=Ncomps)]

    best_fits = np.zeros((Ncomps, Ncomps, inp.Nbins)).tolist() #need list to store sympy expressions
    scalings = get_scalings(inp)
    for p in range(Ncomps):
        for q in range(Ncomps):
            for bin in range(inp.Nbins):
                x_vals, y_vals = [], []
                for s in scalings:
                    scaling_factor = (inp.scaling_factors[s[0]])**2
                    x = np.ones(Ncomps)
                    x[np.array(s[1:])==1] = scaling_factor
                    x_vals.append(x)
                    y_vals.append(Clpq_mean[s[0], sublist_idx(comp_scalings, s[1:]), p, q, bin]/Clpq_mean[0,0,p,q,bin])
                best_fits[p][q][bin] = symbolic_regression(inp, x_vals, y_vals)
                if inp.verbose: print(f'estimated parameter dependence for p,q,bin={p},{q},{bin}', flush=True)

    if inp.save_files:
        pipeline = 'HILC' if HILC else 'NILC'
        naming_str = get_naming_str(inp, pipeline)
        filename = f'{inp.output_dir}/data_vecs/best_fits_{naming_str}.p'
        pickle.dump(best_fits, open(filename, 'wb'), protocol=4)
        print(f'saved {filename}', flush=True)
    
    return best_fits
