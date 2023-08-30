import numpy as np
import pickle
import itertools
import subprocess
from pysr import PySRRegressor 

def symbolic_regression(x_vals, y_vals):
    '''
    ARGUMENTS
    ---------
    x_vals: list of points, where each point contains value of Acmb, Atsz, Anoise1, Anoise2
    y_vals: Clpq evaluated at scaled points / Clpq evaluated at all unscaled points

    RETURNS
    -------
    sympy expression of best fit model 
    '''
    model = PySRRegressor(
        niterations = 50,  # < Increase me for better results
        ncyclesperiteration = 750,
        progress = False, 
        maxsize = 12,
        binary_operators = ["*", "+", "-", "/"],
        unary_operators = ["exp", "square", "cube", "inv(x) = 1/x"],
        extra_sympy_mappings = {"inv": lambda x: 1 / x},
        loss = "loss(prediction, target) = (prediction - target)^2",
    )
    model.fit(x_vals, y_vals)
    return model.sympy()


def call_fit(A_vec, expr):
    '''
    ARGUMENTS
    ---------
    A_vec: list of [Acmb, Atsz, Anoise1, Anoise2] independent variables
    expr: sympy expression of best fit involving parameters Acmb, Atsz, Anoise1, Anoise2,
        which map to x0, x1, x2, and x3, respectively 

    RETURNS
    -------
    numerical evaluation of expr at the point given by A_vec
    '''
    return expr.subs('x0', A_vec[0]).subs('x1', A_vec[1]).subs('x2', A_vec[2]).subs('x3', A_vec[3])


def get_parameter_dependence(inp, Clpq, env):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input paramter specifications
    Clpq: (Nsims, 2, 2, 2, 2, 2, N_preserved_comps=2, N_preserved_comps=2, N_comps=4, N_comps=4, Nbins) ndarray 
        containing propagation of each pair of component maps to NILC map auto- and cross-spectra. 
        dim0: 0 if "scaled" means maps are scaled down, 1 if "scaled" means maps are scaled up
        dim1: 0 for unscaled CMB, 1 for scaled CMB
        dim2: 0 for unscaled ftSZ, 1 for scaled ftSZ
        dim3: 0 for unscaled noise90, 1 for scaled noise90
        dim4: 0 for unscaled noise150, 1 for scaled noise150
        Note: for sim >= Nsims_for_fits, results are meaningless except for scaling 00000 (all unscaled)
    env: environment object
    
    RETURNS
    -------
    best_fits: (N_preserved_comps, N_preserved_comps, N_comps, N_comps, Nbins) list
        containing sympy expressions with best fits to Acmb, Atsz, Anoise1, Anoise2

    '''
    N_preserved_comps = 2
    N_comps = 4
    Clpq_mean = np.mean(Clpq[:inp.Nsims_for_fits], axis=0)

    best_fits = np.zeros((N_preserved_comps, N_preserved_comps, N_comps, N_comps, inp.Nbins)).tolist()
    scalings = [list(i) for i in itertools.product([0, 1], repeat=5)]
    for p,q in [(0,0), (0,1), (1,1)]:
        for y in range(N_comps):
            for z in range(N_comps):
                for bin in range(inp.Nbins):
                    x_vals, y_vals = [], []
                    for s in scalings:
                        scaling_factor = inp.scaling_factors[0]**2 if s[0] == 0 else inp.scaling_factors[1]**2
                        x = np.ones(N_comps)
                        x[np.array(s[1:])==1] = scaling_factor
                        x_vals.append(x)
                        y_vals.append(Clpq_mean[s[0],s[1],s[2],s[3],s[4],p,q,y,z,bin]/Clpq_mean[0,0,0,0,0,p,q,y,z,bin])
                    best_fits[p][q][y][z][bin] = symbolic_regression(x_vals, y_vals)
                    best_fits[q][p][z][y][bin] = best_fits[p][q][y][z][bin]
                    subprocess.call(f'rm -f hall_of_fame*', shell=True, env=env)
                    if inp.verbose: print(f'estimated parameter dependence for p,q,y,z,bin={p},{q},{y},{z},{bin}', flush=True)

    
    if inp.save_files:
        pickle.dump(best_fits, open(f'{inp.output_dir}/data_vecs/best_fits.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.output_dir}/data_vecs/best_fits.p', flush=True)
    
    return best_fits
