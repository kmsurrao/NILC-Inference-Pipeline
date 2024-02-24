import torch
from utils import get_naming_str
from sbi import utils as utils
import multiprocessing as mp
import numpy as np
import pickle
import tqdm
import itertools
from getdist import MCSamples
import sys
sys.path.append('../multifrequency_pipeline')
sys.path.append('../harmonic_ILC_pipeline')
sys.path.append('../needlet_ILC_pipeline')
import multifrequency_data_vecs
import hilc_analytic
import nilc_data_vecs
import sbi_utils
import hyperparam_sweep

def get_prior(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    prior on Acmb, Atsz to use for likelihood-free inference
    '''
    num_dim = 2
    mean_tensor = torch.ones(num_dim)
    prior = utils.BoxUniform(low=mean_tensor-torch.tensor(inp.prior_half_widths) , high=mean_tensor+torch.tensor(inp.prior_half_widths))
    return prior


def get_observation(inp, pipeline, env):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    pipeline: str, either 'multifrequency', 'HILC', or 'NILC'
    env: environment object, only needed if pipeline=='NILC

    RETURNS
    -------
    data_vec: ndarray containing outputs from simulation
            Clpq of shape (Nsims, N_preserved_comps, N_preserved_comps, Nbins) if HILC or NILC
            Clij of shape (Nsims, Nfreqs, Nfreqs, Nbins) if multifrequency
    
    '''
    sims_for_obs = min(inp.Nsims, 1000)

    if pipeline == 'HILC':

        fname = 'Clpq'

        pool = mp.Pool(inp.num_parallel)
        args = [(inp, sim) for sim in range(sims_for_obs)]
        print(f'Running {sims_for_obs} simulations of frequency-frequency power spectra as part of observation vector calculation...', flush=True)
        Clij = list(tqdm.tqdm(pool.imap(hilc_analytic.get_freq_power_spec_star, args), total=sims_for_obs))
        pool.close()
        Clij = np.asarray(Clij, dtype=np.float32)

        pool = mp.Pool(inp.num_parallel)
        inp.Clij_theory = np.mean(Clij, axis=0)
        args = [(inp, Clij[sim]) for sim in range(sims_for_obs)]
        print(f'Running {sims_for_obs} simulations to average together for observation vector...', flush=True)
        Clpq = list(tqdm.tqdm(pool.imap(hilc_analytic.get_data_vecs_star, args), total=sims_for_obs))
        pool.close()
        data_vec = np.asarray(Clpq, dtype=np.float32)[:,:,:,0,:] # shape (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins)

    else:
        pool = mp.Pool(inp.num_parallel)
        print(f'Running {sims_for_obs} simulations to average together for observation vector...', flush=True)
        if pipeline == 'multifrequency':
            func = multifrequency_data_vecs.get_data_vectors_star
            args = [(inp, sim) for sim in range(sims_for_obs)]

        elif pipeline == 'NILC':
            func = nilc_data_vecs.get_data_vectors_star
            args = [(inp, env, sim) for sim in range(sims_for_obs)]
        
        data_vec = list(tqdm.tqdm(pool.imap(func, args), total=sims_for_obs))
        pool.close()

        if pipeline == 'NILC':
            fname = 'Clpq'
            data_vec = np.asarray(data_vec, dtype=np.float32) # shape (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins)
        else:
            fname = 'Clij'
            data_vec = np.asarray(data_vec, dtype=np.float32)[:,:,:,0,:] # shape (Nsims, Nfreqs, Nfreqs, Nbins)
    
    naming_str = get_naming_str(inp, pipeline)
    pickle.dump(data_vec, open(f'{inp.output_dir}/data_vecs/{fname}_{naming_str}.p', 'wb'), protocol=4)
    print(f'\nsaved {inp.output_dir}/data_vecs/{fname}_{naming_str}.p', flush=True)
    return data_vec



def get_posterior(inp, pipeline, env):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    pipeline: str, either 'multifrequency', 'HILC', or 'NILC'
    env: environment object, only needed if pipeline=='NILC'

    RETURNS
    -------
    samples: torch tensor of shape (Nsims, 2) containing Acmb, Atsz posteriors
    
    '''

    assert pipeline in {'multifrequency', 'HILC', 'NILC'}, "pipeline must be either 'multifrequency', 'HILC', or 'NILC'"
    prior = get_prior(inp)

    observation_all_sims = get_observation(inp, pipeline, env)
    N = observation_all_sims.shape[1]
    observation_all_sims = np.array([observation_all_sims[:,i,j] for (i,j) in list(itertools.product(range(N), range(N)))])
    observation_all_sims = np.transpose(observation_all_sims, axes=(1,0,2)).reshape((-1, len(observation_all_sims)*inp.Nbins))
    mean_vec = np.mean(observation_all_sims, axis=0)
    std_dev_vec = np.std(observation_all_sims, axis=0)
    observation = np.zeros_like(mean_vec)

    def simulator(pars):
        '''
        ARGUMENTS
        ---------
        pars: [Acmb, Atsz] parameters (floats)

        RETURNS
        -------
        data_vec: torch tensor containing outputs from simulation
            Clpq of shape (N_preserved_comps*N_preserved_comps*Nbins,) if HILC or NILC
            Clij of shape (Nfreqs*Nfreqs*Nbins, ) if multifrequency
        
        '''
        if pipeline == 'multifrequency':
            data_vec = multifrequency_data_vecs.get_data_vectors(inp, sim=None, pars=pars)[:,:,0,:] # shape (Nfreqs, Nfreqs, Nbins)
        elif pipeline == 'HILC':
            Clij = hilc_analytic.get_freq_power_spec(inp, sim=None, pars=pars) # shape (Nfreqs, Nfreqs, 1+Ncomps, ellmax+1)
            data_vec = hilc_analytic.get_data_vecs(inp, Clij)[:,:,0,:] # shape (N_preserved_comps=2, N_preserved_comps=2, Nbins)
        elif pipeline == 'NILC':
            data_vec = nilc_data_vecs.get_data_vectors(inp, env, sim=None, pars=pars) # shape (N_preserved_comps=2, N_preserved_comps=2, Nbins)
        data_vec = np.array([data_vec[i,j] for (i,j) in list(itertools.product(range(N), range(N)))]).flatten()
        data_vec = torch.tensor((data_vec-mean_vec)/std_dev_vec)
        return data_vec
    
    if inp.tune_hyperparameters:
        samples, mean_stds, error_of_stds = hyperparam_sweep.run_sweep(inp, prior, simulator, observation, pipeline)
        for i, par in enumerate(['Acmb', 'Atsz']):
            print(f'mean of {par} posterior standard deviations over top 25% of sweeps: ', mean_stds[i], flush=True)
            print(f'standard deviation of {par} posterior standard deviations ("error of errors") over top 25% of sweeps: ', error_of_stds[i], flush=True)
    else:
        samples = sbi_utils.flexible_single_round_SNPE(inp, prior, simulator, observation,
                                                    learning_rate=inp.learning_rate, 
                                                    stop_after_epochs=inp.stop_after_epochs,
                                                    clip_max_norm=inp.clip_max_norm,
                                                    num_transforms=inp.num_transforms,
                                                    hidden_features=inp.hidden_features)
    acmb_array, atsz_array = np.array(samples, dtype=np.float32).T
    
    naming_str = get_naming_str(inp, pipeline)
    pickle.dump(acmb_array, open(f'{inp.output_dir}/posteriors/acmb_array_{naming_str}.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/posteriors/atsz_array_{naming_str}.p', 'wb'))
    print(f'\nsaved {inp.output_dir}/posteriors/acmb_array_{naming_str}.p and likewise for atsz')

    print('Results from Likelihood-Free Inference', flush=True)
    print('----------------------------------------------', flush=True)
    names = ['Acmb', 'Atsz']
    samples_MC = MCSamples(samples=[acmb_array, atsz_array], names = names, labels = names)
    for par in ['Acmb', 'Atsz']:
        print(samples_MC.getInlineLatex(par,limit=1), flush=True)

    return samples
