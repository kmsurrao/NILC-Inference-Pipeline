import wandb
import torch
import heapq
import sbi_utils

def build_sweep_config(inp, pipeline):
    '''
    Build hyperparameter sweep configuration

    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    pipeline: str, pipeline being run, one of 'multifrequency', 'HILC', or 'NILC'

    RETURNS
    -------
    sweep_config: dictionary containing information about hyperparameter sweep

    '''

    # name for sweep
    name = f'{pipeline}_'
    if 'tsz' in inp.comps:
        tsz_idx = inp.comps.index('tsz')
        gaussian_str = 'gaussiantsz_' if inp.use_Gaussian[tsz_idx] else 'nongaussiantsz_'
        name += gaussian_str
    if pipeline == 'HILC':
        wts_str = 'weightsonce_' if inp.compute_weights_once else 'weightsvary_'
        name += wts_str
    if inp.Nsims % 1000 == 0:
        sims_str = f'{inp.Nsims//1000}ksims_'
    else:
        sims_str = f'{int(inp.Nsims)}sims'
    name += sims_str
    if 'tsz' in inp.comps:
        name += f'_tsz_amp{int(inp.amp_factors[tsz_idx])}'
    name += f'_{len(inp.freqs)}freqs'
    if pipeline == 'NILC':
        name += f'_{inp.Nscales}scales'

    # choose random configurations of hyperparameters
    sweep_config = {
        'method': 'random',
        'name': name
    }

    # goal of hyperparameter sweep
    metric = {
        'name': 'best_validation_log_prob',
        'goal': 'maximize'   
    }
    sweep_config['metric'] = metric

    # parameters for hyperparameter sweep
    parameters_dict = {
        'stop_after_epochs': {
            'distribution': 'int_uniform',
            'min': 20,
            'max': 60
        },
        'learning_rate':{
            'distribution': 'uniform',
            'min': 1.e-4,
            'max': 7.e-4,
        },
        'clip_max_norm': {
            'distribution': 'uniform',
            'min': 3.0,
            'max': 7.0
        },
        'num_transforms': {
            'distribution': 'int_uniform',
            'min': 3,
            'max': 10
        },
        'hidden_features': {
            'distribution': 'int_uniform',
            'min': 35,
            'max': 65
        }
    }
    sweep_config['parameters'] = parameters_dict
    return sweep_config


def run_sweep(inp, prior, simulator, observation, pipeline):
    '''                                                                                                                                            
    ARGUMENTS                                                                                                                                      
    ---------                                                                                                                                      
    inp: Info object containing input parameter specifications                                                                                     
    prior: prior on parameters to use for likelihood-free inference                                                                                
        (for example, sbi.utils.BoxUniform or torch tensor such as Gaussian)                                                                       
    simulator: function that generates simulations of the data vector                                                                              
    observation: torch tensor, contains "observation" of data vector  
    pipeline: str, pipeline being run, one of 'multifrequency', 'HILC', or 'NILC'                                                                             
                                                                                                                                                   
    RETURNS                                                                                                                                        
    -------                                                                                                                                        
    final_samples: (k*Nsims, Ndim) torch tensor containing samples drawn from posterior                                                          
        of the trained network that resulted in the highest validation log probability.
        Here, k is Nsweeps//4
    mean_stds: list of length Ncomps containing mean of amplitude parameter standard deviations
        obtained from k highest hyperparameter sweeps
    error_of_stds: list of length Ncomps containing standard deviation of amplitude parameter
        standard deviations ("erorr of errors") obtained from k highest hyperparameter sweeps                                                           
    '''
    sweep_results = []
    theta, x = sbi_utils.generate_samples(inp, prior, simulator)
    config = build_sweep_config(inp, pipeline)

    def run_one_sweep_iter():
        '''                                                                                                                                        
        Runs single round NPE with one set of hyperparameters                                                                                      
        '''
        wandb.init()
        nonlocal sweep_results
        samples, best_val_log_prob = sbi_utils.train_network(theta, x, inp, prior, observation,
                                    learning_rate=wandb.config.learning_rate, stop_after_epochs=wandb.config.stop_after_epochs,
                                    clip_max_norm=wandb.config.clip_max_norm, num_transforms=wandb.config.num_transforms,
                                    hidden_features=wandb.config.hidden_features)
        wandb.log({'best_validation_log_prob': best_val_log_prob})
        a_arrays = torch.transpose(samples, 0, 1)
        wandb_log_dict = {}
        for a, a_array in enumerate(a_arrays):
            wandb_log_dict[f'A{inp.comps[a]}_std'] = torch.std(a_array)
        wandb.log(wandb_log_dict)
        sweep_results.append((best_val_log_prob, samples))
        return
    
    # run the wandb tuning
    sweep_id = wandb.sweep(sweep=config, project=inp.wandb_project_name)
    wandb.agent(sweep_id, function=run_one_sweep_iter, project=inp.wandb_project_name, count=inp.Nsweeps)

    # take the top 25% of results (in terms of validation log probability)
    nsweeps_to_use = inp.Nsweeps//4
    if nsweeps_to_use < 1:
        nsweeps_to_use = 1
    best_sweep_results = heapq.nlargest(nsweeps_to_use, sweep_results, key=lambda x: x[0])
    best_samples = [s[1] for s in best_sweep_results]
    std_devs = torch.zeros(len(inp.comps), inp.Nsweeps//4)
    for s, sample in enumerate(best_samples):
        a_arrays = torch.transpose(sample, 0, 1)
        for a, a_array in enumerate(a_arrays):
            std_devs[a, s] = torch.std(a_array)

    # get final samples, average std dev over top 25% of sweeps, and std dev of std devs of 25% of sweeps
    final_samples = torch.cat(best_samples, 0)
    mean_stds = [torch.mean(std_devs[a]) for a in range(len(inp.comps))]
    error_of_stds = [torch.std(std_devs[a]) for a in range(len(inp.comps))]

    return final_samples, mean_stds, error_of_stds