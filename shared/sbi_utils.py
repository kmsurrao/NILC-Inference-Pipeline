from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

def multi_round_SNPE(inp, prior, simulator, observation, 
                    learning_rate=2.e-4, stop_after_epochs=40, clip_max_norm=5.0, 
                    num_transforms=5, hidden_features=50, num_rounds=2, 
                    sample_with_mcmc=False, sweep=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    prior: prior on parameters to use for likelihood-free inference
        (for example, sbi.utils.BoxUniform or torch tensor such as Gaussian)
    simulator: function that generates simulations of the data vector
    observation: torch tensor, contains "observation" of data vector
    learning_rate: float, learning rate
    stop_after_epochs: int, number of epochs to wait for improvement on the
                validation set before terminating training
    clip_max_norm: float, value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
    num_transforms: int, number of transforms when a flow is used. 
        Only relevant if density estimator is a normalizing flow (i.e. currently 
        either a maf or a nsf). Ignored if density estimator is a mdn or made.
    hidden_features: int, number of hidden features
    num_rounds: int, number of rounds of SNPE
    sample_with_mcmc: Bool, if True, samples with MCMC. If False, uses rejection sampling.
    sweep: Bool, whether running as part of hyperparameter sweep

    RETURNS
    -------
    samples: (Nsims, Ndim) torch tensor containing samples drawn from posterior
    best_val_log_prob: float, best validation log probability (only returned if sweep is True)
    '''
    simulator, prior = prepare_for_sbi(simulator, prior)
    density_estimator_build_fun = build_density_estimator_network(hidden_features=hidden_features, 
                                                                  num_transforms=num_transforms)
    inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)
    posteriors = []
    proposal = prior
    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=inp.Nsims//num_rounds, num_workers=inp.num_parallel)
        inference_append_sims = inference.append_simulations(theta, x, proposal=proposal)
        density_estimator = inference_append_sims.train(learning_rate=learning_rate, 
                                                        stop_after_epochs=stop_after_epochs, 
                                                        clip_max_norm=clip_max_norm)
        best_val_log_prob = inference_append_sims._summary["best_validation_log_prob"]
        if sample_with_mcmc:
            posterior = inference.build_posterior(density_estimator, sample_with='mcmc')
        else:
            posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(observation)
    samples = posterior.sample((inp.Nsims,), x=observation)
    if sweep:
        return samples, best_val_log_prob
    return samples


def basic_single_round_SNPE(inp, prior, simulator, observation):
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
    samples: (Nsims, Ndim) torch tensor containing samples drawn from posterior
    '''
    posterior = infer(simulator, prior, method="SNPE", num_simulations=inp.Nsims, num_workers=inp.num_parallel)
    samples = posterior.sample((inp.Nsims,), x=observation)
    return samples


def flexible_single_round_SNPE(inp, prior, simulator, observation, 
                                learning_rate=2.e-4, stop_after_epochs=40, clip_max_norm=5.0, 
                                num_transforms=5, hidden_features=50, sweep=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    prior: prior on parameters to use for likelihood-free inference
        (for example, sbi.utils.BoxUniform or torch tensor such as Gaussian)
    simulator: function that generates simulations of the data vector
    observation: torch tensor, contains "observation" of data vector
    learning_rate: float, learning rate
    stop_after_epochs: int, number of epochs to wait for improvement on the
                validation set before terminating training
    clip_max_norm: float, value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
    num_transforms: int, number of transforms when a flow is used. 
        Only relevant if density estimator is a normalizing flow (i.e. currently 
        either a maf or a nsf). Ignored if density estimator is a mdn or made.
    hidden_features: int, number of hidden features
    sweep: Bool, whether running as part of hyperparameter sweep
    best_val_log_prob: float, best validation log probability (only returned if sweep is True)

    RETURNS
    -------
    samples: (Nsims, Ndim) torch tensor containing samples drawn from posterior
    '''
    simulator, prior = prepare_for_sbi(simulator, prior)
    density_estimator_build_fun = build_density_estimator_network(hidden_features=hidden_features, 
                                                                  num_transforms=num_transforms)
    inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=inp.Nsims, num_workers=inp.num_parallel)
    inference_append_sims = inference.append_simulations(theta, x)
    density_estimator = inference_append_sims.train(learning_rate=learning_rate, 
                                                    stop_after_epochs=stop_after_epochs, 
                                                    clip_max_norm=clip_max_norm)
    best_val_log_prob = inference_append_sims._summary["best_validation_log_prob"]
    posterior = inference.build_posterior(density_estimator)
    samples = posterior.sample((inp.Nsims,), x=observation)
    if sweep:
        return samples, best_val_log_prob
    return samples


def build_density_estimator_network(model="maf", hidden_features=50, num_transforms=5):
    '''
    ARGUMENTS
    ---------
    model: str, The type of density estimator that will be created. 
        One of [mdn, made, maf, maf_rqs, nsf].
    hidden_features: int, number of hidden features
    num_transforms: int, number of transforms when a flow is used. 
        Only relevant if density estimator is a normalizing flow (i.e. currently 
        either a maf or a nsf). Ignored if density estimator is a mdn or made.

    RETURNS
    -------
    density_estimator_build_fun: function that builds a density estimator for learning the posterior.
        This function will usually be used for SNPE. The returned function is to be passed to the 
        inference class when using the flexible interface.
    '''
    density_estimator_build_fun = posterior_nn(
        z_score_theta='independent', z_score_x='independent',
        model=model, hidden_features=hidden_features, num_transforms=num_transforms)
    return density_estimator_build_fun