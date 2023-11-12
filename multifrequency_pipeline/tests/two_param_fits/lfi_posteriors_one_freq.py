import torch
from sbi import utils as utils
import numpy as np
import sys
sys.path.append('../..')
sys.path.append('../../../shared')
import multifrequency_data_vecs
import sbi_utils

def get_prior(prior_half_widths):
    '''
    ARGUMENTS
    ---------
    prior_half_widths: 1D array-like of size 2 containing half the width of uniform prior
        on each parameter. The prior will be set to [1-prior_half_width, 1+prior_half_width].

    RETURNS
    -------
    prior on A1, A2
    '''
    num_dim = 2
    mean_tensor = torch.ones(num_dim)
    prior = utils.BoxUniform(low=mean_tensor-torch.tensor(prior_half_widths) , high=mean_tensor+torch.tensor(prior_half_widths))
    return prior



def get_posterior(inp, prior_half_widths, observation_all_sims):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    prior_half_widths: 1D array-like of size 2 containing half the width of uniform prior
        on each parameter. The prior will be set to [1-prior_half_width, 1+prior_half_width].
    observation_all_sims: ndarray of shape (Nsims, (1+Ncomps)=3, Nbins) containing Clij vector

    RETURNS
    -------
    samples: torch tensor of shape (Nsims, 2) containing A1, A2 posteriors
    
    '''

    prior = get_prior(prior_half_widths)
    observation_all_sims = observation_all_sims[:,0,:]
    mean_vec = np.mean(observation_all_sims, axis=0)
    observation = torch.ones(inp.Nbins)

    def simulator(pars):
        '''
        ARGUMENTS
        ---------
        pars: [A1, A2] parameters (floats)

        RETURNS
        -------
        data_vec: torch tensor of shape (Nfreqs*Nfreqs*Nbins, ) containing outputs from simulation
        
        '''
        data_vec = multifrequency_data_vecs.get_data_vectors(inp, sim=None, pars=pars)[0,0,0,:] # shape (Nbins,)
        data_vec = torch.tensor(data_vec/mean_vec)
        return data_vec
    
    samples = sbi_utils.flexible_single_round_SNPE(inp, prior, simulator, observation, density_estimator='maf')
    a1_array, a2_array = np.array(samples, dtype=np.float32).T
    
    print('1D marginalized posteriors from likelihood-free inference', flush=True)
    print('---------------------------------------------------------', flush=True)
    print(f'A1 = {np.mean(a1_array)} +/- {np.std(a1_array)}', flush=True)
    print(f'A2 = {np.mean(a2_array)} +/- {np.std(a2_array)}', flush=True)

    return samples
