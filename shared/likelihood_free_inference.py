import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import multiprocessing as mp
import numpy as np
import pickle
import sys
sys.path.append('../multifrequency_pipeline')
sys.path.append('../harmonic_ILC_pipeline')
sys.path.append('../needlet_ILC_pipeline')
import multifrequency_data_vecs
import hilc_analytic
import nilc_data_vecs

def get_prior(inp):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications

    RETURNS
    -------
    prior on Acmb, Atsz, Anoise1, Anoise2 to use for likelihood-free inference
    '''
    num_dim = 4
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
    if pipeline == 'HILC':

        fname = 'Clpq_HILC.p'

        pool = mp.Pool(inp.num_parallel)
        Clij = pool.starmap(hilc_analytic.get_freq_power_spec, [(sim, inp) for sim in range(inp.Nsims)])
        pool.close()
        Clij = np.asarray(Clij, dtype=np.float32)

        pool = mp.Pool(inp.num_parallel)
        inp.Clij_theory = np.mean(Clij, axis=0)
        Clpq = pool.starmap(hilc_analytic.get_data_vecs, [(inp, Clij[sim]) for sim in range(inp.Nsims)])
        pool.close()
        data_vec = np.asarray(Clpq, dtype=np.float32)[:,:,:,0,:] # shape (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins)

    else:
        pool = mp.Pool(inp.num_parallel)
        if pipeline == 'multifrequency':
            func = multifrequency_data_vecs.get_data_vectors
            args = [(sim, inp) for sim in range(inp.Nsims)]

        elif pipeline == 'NILC':
            func = nilc_data_vecs.get_data_vectors
            args = [(sim, inp, env) for sim in range(inp.Nsims)]
        
        data_vec = pool.starmap(func, args)
        pool.close()

        if pipeline == 'NILC':
            fname = 'Clpq_NILC.p'
            data_vec = np.asarray(data_vec, dtype=np.float32) # shape (Nsims, N_preserved_comps=2, N_preserved_comps=2, Nbins)
        else:
            fname = 'Clij.p'
            data_vec = np.asarray(data_vec, dtype=np.float32)[:,:,:,0,:] # shape (Nsims, Nfreqs, Nfreqs, Nbins)
    
    pickle.dump(data_vec, open(f'{inp.output_dir}/data_vecs/{fname}', 'wb'), protocol=4)
    return data_vec



def get_posterior(inp, pipeline, env):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    pipeline: str, either 'multifrequency', 'HILC', or 'NILC'
    env: environment object, only needed if pipeline=='NILC

    RETURNS
    -------
    samples: torch tensor of shape (Nsims, 4) containing Acmb, Atsz, Anoise1, Anoise2 posteriors
    
    '''

    assert pipeline in {'multifrequency', 'HILC', 'NILC'}, "pipeline must be either 'multifrequency', 'HILC', or 'NILC'"
    prior = get_prior(inp)
    sim = 0

    def simulator(pars):
        '''
        ARGUMENTS
        ---------
        pars: [Acmb, Atsz, Anoise1, Anoise2] parameters (floats)

        RETURNS
        -------
        data_vec: torch tensor containing outputs from simulation
            Clpq of shape (N_preserved_comps*N_preserved_comps*Nbins,) if HILC or NILC
            Clij of shape (Nfreqs*Nfreqs*Nbins, ) if multifrequency
        
        '''
        nonlocal sim
        if pipeline == 'multifrequency':
            data_vec = multifrequency_data_vecs.get_data_vectors(sim, inp, pars=pars)[:,:,0,:] # shape (Nfreqs, Nfreqs, Nbins)
        elif pipeline == 'HILC':
            Clij = hilc_analytic.get_freq_power_spec(sim, inp, pars=pars) # shape (Nfreqs=, Nfreqs, 1+Ncomps, ellmax+1)
            data_vec = hilc_analytic.get_data_vecs(inp, Clij)[:,:,0,:] # shape (N_preserved_comps=2, N_preserved_comps=2, Nbins)
        elif pipeline == 'NILC':
            data_vec = nilc_data_vecs.get_data_vectors(sim, inp, env, pars=pars) # shape (N_preserved_comps=2, N_preserved_comps=2, Nbins)
        sim += 1
        data_vec = torch.tensor(np.array([data_vec[0,0], data_vec[0,1], data_vec[1,1]]).flatten())
        return data_vec
    
    posterior = infer(simulator, prior, method="SNPE", num_simulations=inp.Nsims, num_workers=inp.num_parallel)
    observation_all_sims = get_observation(inp, pipeline, env)
    #observation_all_sims = pickle.load(open(f'{inp.output_dir}/data_vecs/Clij.p', 'rb'))[:,:,:,0,:] #remove and uncomment above
    mean_observation = np.mean(observation_all_sims, axis=0)
    observation = torch.tensor(np.array([mean_observation[0,0], mean_observation[0,1], mean_observation[1,1]]).flatten())
    samples = posterior.sample((inp.Nsims,), x=observation)
    acmb_array, atsz_array, anoise1_array, anoise2_array = np.array(samples, dtype=np.float32).T
    pickle.dump(acmb_array, open(f'{inp.output_dir}/acmb_array_{pipeline}.p', 'wb'))
    pickle.dump(atsz_array, open(f'{inp.output_dir}/atsz_array_{pipeline}.p', 'wb'))
    pickle.dump(anoise1_array, open(f'{inp.output_dir}/anoise1_array_{pipeline}.p', 'wb'))
    pickle.dump(anoise2_array, open(f'{inp.output_dir}/anoise2_array_{pipeline}.p', 'wb'))
    print(f'saved {inp.output_dir}/acmb_array_{pipeline}.p and likewise for atsz, anoise1, anoise2')
    return samples
