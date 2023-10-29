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
    prior_half_widths: 1D array-like of size 1 containing half the width of uniform prior
        on each parameter. The prior will be set to [1-prior_half_width, 1+prior_half_width].

    RETURNS
    -------
    prior on A1
    '''
    num_dim = 1
    mean_tensor = torch.ones(num_dim)
    prior = utils.BoxUniform(low=mean_tensor-torch.tensor(prior_half_widths) , high=mean_tensor+torch.tensor(prior_half_widths))
    return prior



def get_posterior(inp, prior_half_widths, observation_all_sims, comp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    prior_half_widths: 1D array-like of size 2 containing half the width of uniform prior
        on each parameter. The prior will be set to [1-prior_half_width, 1+prior_half_width].
    observation_all_sims: ndarray of shape (Nsims, Nbins) containing Clij vector
    comp: str, component used, must be either 'CMB', 'tSZ', 'noise1', or 'noise2'

    RETURNS
    -------
    samples: torch tensor of shape (Nsims, 1) containing A1 posterior
    
    '''
    assert comp in {'CMB', 'tSZ', 'noise1', 'noise2'}, 'comp must be either CMB, tSZ, noise1, or noise2'
    prior = get_prior(prior_half_widths)
    mean_vec = np.mean(observation_all_sims, axis=0)
    observation = torch.tensor(mean_vec)

    def simulator(pars):
        '''
        ARGUMENTS
        ---------
        pars: [A1] parameter (float)

        RETURNS
        -------
        data_vec: torch tensor of shape (Nfreqs*Nfreqs*Nbins, ) containing outputs from simulation
        
        '''
        new_pars = pars
        if comp == 'CMB':
            new_pars = torch.cat([new_pars, torch.tensor([0,0,0])])
        elif comp == 'tSZ':
            new_pars = torch.cat([torch.tensor([0]), new_pars, torch.tensor([0,0])])
        elif comp == 'noise1':
            new_pars = torch.cat([torch.tensor([0,0]), new_pars, torch.tensor([0])])
        elif comp == 'noise2':
            new_pars = torch.cat([torch.tensor([0,0,0]), new_pars])
        data_vec = multifrequency_data_vecs.get_data_vectors(inp, sim=None, pars=new_pars)[0,0,0,:] # shape (Nbins,)
        data_vec = torch.tensor(data_vec)
        return data_vec
    
    samples = sbi_utils.basic_single_round_SNPE(inp, prior, simulator, observation)
    a1_array = np.array(samples, dtype=np.float32).T
    
    print('1D marginalized posteriors from likelihood-free inference', flush=True)
    print('---------------------------------------------------------', flush=True)
    print(f'A1 = {np.mean(a1_array)} +/- {np.std(a1_array)}', flush=True)

    return samples
