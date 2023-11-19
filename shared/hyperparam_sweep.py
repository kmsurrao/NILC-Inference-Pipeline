import wandb
import numpy as np
import sbi_utils

def build_sweep_config():
    '''
    Build hyperparameter sweep configuration

    RETURNS
    -------
    sweep_config: dictionary containing information about hyperparameter sweep

    '''

    # choose random configurations of hyperparameters
    sweep_config = {
        'method': 'random'
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
            'max': 5.e-4,
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


def run_sweep(inp, prior, simulator, observation):
    '''                                                                                                                                            
    ARGUMENTS                                                                                                                                      
    ---------                                                                                                                                      
    inp: Info object containing input parameter specifications                                                                                     
    prior: prior on parameters to use for likelihood-free inference                                                                                
        (for example, sbi.utils.BoxUniform or torch tensor such as Gaussian)                                                                       
    simulator: function that generates simulations of the data vector                                                                              
    observation: torch tensor, contains "observation" of data vector                                                                               
                                                                                                                                                   
    RETURNS                                                                                                                                        
    -------                                                                                                                                        
    curr_best_samples: (Nsims, Ndim) torch tensor containing samples drawn from posterior                                                          
        of the trained network that resulted in the highest validation log probability                                                             
    '''
    curr_best_val_log_prob = 0
    curr_best_samples = None
    theta, x = sbi_utils.generate_samples(inp, prior, simulator)
    config = build_sweep_config()

    def run_one_sweep_iter():
        '''                                                                                                                                        
        Runs single round NPE with one set of hyperparameters                                                                                      
        '''
        wandb.init()
        nonlocal curr_best_val_log_prob
        nonlocal curr_best_samples
        samples, best_val_log_prob = sbi_utils.train_network(theta, x, inp, prior, observation,
                                    learning_rate=wandb.config.learning_rate, stop_after_epochs=wandb.config.stop_after_epochs,
                                    clip_max_norm=wandb.config.clip_max_norm, num_transforms=wandb.config.num_transforms,
                                    hidden_features=wandb.config.hidden_features)
        wandb.log({'best_validation_log_prob': best_val_log_prob})
        acmb_array, atsz_array = np.array(samples, dtype=np.float32).T
        wandb.log({'Acmb_std': np.std(acmb_array), 'Atsz_std': np.std(atsz_array)})
        if best_val_log_prob > curr_best_val_log_prob:
            curr_best_val_log_prob = best_val_log_prob
            curr_best_samples = samples
        return
    
    sweep_id = wandb.sweep(sweep=config, project=inp.wandb_project_name)
    wandb.agent(sweep_id, function=run_one_sweep_iter, project=inp.wandb_project_name, count=10)
    return curr_best_samples