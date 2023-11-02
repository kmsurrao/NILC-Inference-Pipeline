from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

def multi_round_SNPE(inp, prior, simulator, observation, density_estimator='maf', sample_with_mcmc=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    prior: prior on parameters to use for likelihood-free inference
        (for example, sbi.utils.BoxUniform or torch tensor such as Gaussian)
    simulator: function that generates simulations of the data vector
    observation: torch tensor, contains "observation" of data vector
    density estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
    sample_with_mcmc: Bool, if True, samples with MCMC. If False, uses rejection sampling.

    RETURNS
    -------
    samples: (Nsims, Ndim) torch tensor containing samples drawn from posterior
    '''
    num_rounds = 2
    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE(prior=prior, density_estimator=density_estimator)
    posteriors = []
    proposal = prior
    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=inp.Nsims//num_rounds, num_workers=inp.num_parallel)
        density_estimator = inference.append_simulations(
                    theta, x, proposal=proposal).train()
        if sample_with_mcmc:
            posterior = inference.build_posterior(density_estimator, sample_with='mcmc')
        else:
            posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(observation)
    samples = posterior.sample((inp.Nsims,), x=observation)
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


def flexible_single_round_SNPE(inp, prior, simulator, observation, density_estimator='maf'):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    prior: prior on parameters to use for likelihood-free inference
        (for example, sbi.utils.BoxUniform or torch tensor such as Gaussian)
    simulator: function that generates simulations of the data vector
    observation: torch tensor, contains "observation" of data vector
    density estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.

    RETURNS
    -------
    samples: (Nsims, Ndim) torch tensor containing samples drawn from posterior
    '''
    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE(prior=prior, density_estimator=density_estimator)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=inp.Nsims, num_workers=inp.num_parallel)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    samples = posterior.sample((inp.Nsims,), x=observation)
    return samples


def build_density_estimator_network(model="nsf", hidden_features=50, num_transforms=5, num_bins=10, num_components=10):
    '''
    ARGUMENTS
    ---------
    model: str, The type of density estimator that will be created. 
        One of [mdn, made, maf, maf_rqs, nsf].
    hidden_features: int, number of hidden features
    num_transforms: int, number of transforms when a flow is used. 
        Only relevant if density estimator is a normalizing flow (i.e. currently 
        either a maf or a nsf). Ignored if density estimator is a mdn or made.
    num_bins: int, number of bins used for the splines in nsf. Ignored if density estimator not nsf.

    RETURNS
    -------
    density_estimator_build_fun: function that builds a density estimator for learning the posterior.
        This function will usually be used for SNPE. The returned function is to be passed to the 
        inference class when using the flexible interface.
    '''
    density_estimator_build_fun = posterior_nn(
        z_score_theta='independent', z_score_x='independent',
        model=model, hidden_features=hidden_features, num_transforms=num_transforms,
        num_bins=num_bins, num_components=num_components
        )
    return density_estimator_build_fun