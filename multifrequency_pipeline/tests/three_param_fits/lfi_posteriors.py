import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import numpy as np
import sys
sys.path.append('../..')
sys.path.append('../../../shared')
import multifrequency_data_vecs

def get_prior(prior_half_widths):
    '''
    ARGUMENTS
    ---------
    prior_half_widths: 1D array-like of size 3 containing half the width of uniform prior
        on each parameter. The prior will be set to [1-prior_half_width, 1+prior_half_width].

    RETURNS
    -------
    prior on A1, A2, A3
    '''
    num_dim = 3
    mean_tensor = torch.ones(num_dim)
    prior = utils.BoxUniform(low=mean_tensor-torch.tensor(prior_half_widths) , high=mean_tensor+torch.tensor(prior_half_widths))
    return prior



def get_posterior(inp, prior_half_widths, observation_all_sims, omit_tsz=False, omit_cmb=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    prior_half_widths: 1D array-like of size 3 containing half the width of uniform prior
        on each parameter. The prior will be set to [1-prior_half_width, 1+prior_half_width].
    observation_all_sims: ndarray of shape (Nsims, 2, 2, (1+Ncomps)=4, Nbins) containing Clij vector
    omit_tsz: Bool, whether to omit tSZ from the model
    omit_cmb: Bool, whether to omit CMB from the model

    RETURNS
    -------
    samples: torch tensor of shape (Nsims, 4) containing Acmb, Atsz, Anoise1, Anoise2 posteriors
    
    '''

    prior = get_prior(prior_half_widths)
    observation_all_sims = observation_all_sims[:,:,:,0,:]
    observation_all_sims = np.array([observation_all_sims[:,0,0], observation_all_sims[:,0,1], observation_all_sims[:,1,1]]) #shape (3,Nsims,Nbins)
    observation_all_sims = np.transpose(observation_all_sims, axes=(1,0,2)).reshape((-1, 3*inp.Nbins))
    mean_vec = np.mean(observation_all_sims, axis=0)
    observation = torch.ones(3*inp.Nbins)

    def simulator(pars):
        '''
        ARGUMENTS
        ---------
        pars: [A1, A2, A3] parameters (floats)

        RETURNS
        -------
        data_vec: torch tensor of shape (Nfreqs*Nfreqs*Nbins, ) containing outputs from simulation
        
        '''
        new_pars = pars
        if omit_tsz:
            new_pars = torch.cat([new_pars[:1], torch.tensor([0]), new_pars[1:]])
        elif omit_cmb:
            new_pars = torch.cat([torch.tensor([0]), new_pars])
        data_vec = multifrequency_data_vecs.get_data_vectors(inp, sim=None, pars=new_pars)[:,:,0,:] # shape (Nfreqs, Nfreqs, Nbins)
        data_vec = np.array([data_vec[0,0], data_vec[0,1], data_vec[1,1]]).flatten()
        data_vec = torch.tensor(data_vec/mean_vec)
        return data_vec
    

    posterior = infer(simulator, prior, method="SNPE", num_simulations=inp.Nsims, num_workers=inp.num_parallel)
    samples = posterior.sample((inp.Nsims,), x=observation)
    a1_array, a2_array, a3_array = np.array(samples, dtype=np.float32).T
    
    print('1D marginalized posteriors from likelihood-free inference', flush=True)
    print('---------------------------------------------------------', flush=True)
    print(f'A1 = {np.mean(a1_array)} +/- {np.std(a1_array)}', flush=True)
    print(f'A2 = {np.mean(a2_array)} +/- {np.std(a2_array)}', flush=True)
    print(f'A3 = {np.mean(a3_array)} +/- {np.std(a3_array)}', flush=True)

    return samples
