# NILC-Inference-Pipeline
This repository contains pipelines for calculating parameter posteriors obtained via needlet internal linear combination (NILC) based inference ([needlet_ILC_pipeline](needlet_ILC_pipeline/)), harmonic internal linear combination (HILC) based inference ([harmonic_ILC_pipeline](harmonic_ILC_pipeline/)), and multifrequency power spectrum template-fitting ([multifrequency_pipeline](multifrequency_pipeline/)) based inference. The [shared](shared/) folder contains utilities and functions that are shared across various pipelines. The code currently assumes a sky model comprising only the CMB, tSZ effect (which can be amplified), and noise at two frequency channels, but the model can be extended easily. In particular, the code computes posteriors on amplitude parameters $A_{\rm CMB}$ and $A_{\rm {tSZ}}$ on the CMB and tSZ (potentially artificially amplified) power spectra, respectively. The sky model at frequencies $i$ and $j$ is given by  

$C_\ell^{ij} = A_{\rm CMB}C_\ell^{\rm CMB} + g_i g_j A_{\rm tSZ} C_\ell^{\rm tSZ} + \mathrm{noise}$, 

where $g_i$ and $g_j$ are the tSZ spectral response at frequencies $i$ and $j$, respectively.

For each pipeline, there are options for two types of inference: inference based on a Gaussian likelihood, and likelihood-free inference. When using the Gaussian likelihood, to ensure reliability of results with a given number of simulations, posteriors are computed using three methods: maximum likelihood estimation (MLE), Fisher matrix forecasting, and Markov chain Monte Carlo (MCMC). For likelihood-free inference, the default network used is a masked autoregressive flow, but this network can be changed, tuned, or custom-designed in [sbi_utils.py](shared/sbi_utils.py).

## Requirements and Set-up
 - Requires a clone of the pyilc repository (https://github.com/jcolinhill/pyilc). 
 - Requires a CMB map .fits file in Kelvin (lensed alm can be downloaded from WebSky at https://mocks.cita.utoronto.ca/data/websky/v0.0/). 
 - Requires Nsims (the number of simulations being run) tSZ maps in .fits file format in units of Kelvin, which can be generated via halosky (https://github.com/marcelo-alvarez/halosky) on NERSC.  

## Running
To run any of the main pipelines, one first needs to modify the appropriate yaml files. There are several examples for various set-ups. See the subsections below for details. For all variants of the needlet ILC pipeline and for some variants of the harmonic ILC pipeline (specifically, when using symbolic regression to estimate parameter dependence in likelihood-based inference), there is a large amount of I/O from running the program. This is handled by the python tempfile module. Temporary files and directories will be created in the directory specificed as "output_dir" in the yaml files. It is highly recommended to run on an HPC cluster and to set the output_dir parameter in the yaml files to be a directory in a *SCRATCH* space.  

When using likelihood-free inference, one has the option to tune hyperparameters with wandb using yaml files containing "with_tuning", or one can manually set hyperparemters (or use default ones that have been optimized for specific problem set-ups) using yaml files containing "no_tuning". If using "with_tuning", one first needs to create a wandb account, create a new project, and configure the login information before running the program. See details here (steps 1 and 2): https://docs.wandb.ai/quickstart. The project name should then be entered in the yaml file.       

### To run the parameter inference pipeline using needlet ILC power spectra:  
Modify [gaussian_likelihood.yaml](needlet_ILC_pipeline/example_yaml_files/gaussian_likelihood.yaml) or [lfi.yaml](needlet_ILC_pipeline/example_yaml_files/lfi.yaml) in [needlet_ILC_pipeline/example_yaml_files](needlet_ILC_pipeline/example_yaml_files) before running the needlet ILC pipeline to use an explicit Gaussian likelihood (with parameter dependence determined via symbolic regression) or likelihood-free inference, respectively. Then,  
```cd needlet_ILC_pipeline```   
```python main.py --config=example_yaml_files/[FILENAME]```       

### To run the parameter inference pipeline using multifrequency power spectra:  
Modify [gaussian_likelihood.yaml](multifrequency_pipeline/example_yaml_files/gaussian_likelihood.yaml), [lfi_no_tuning.yaml](multifrequency_pipeline/example_yaml_files/lfi_no_tuning.yaml), or [lfi_with_tuning.yaml](multifrequency_pipeline/example_yaml_files/lfi_with_tuning.yaml) in [multifrequency_pipeline/example_yaml_files](multifrequency_pipeline/example_yaml_files) before running the multifrequency pipeline to use an explicit Gaussian likelihood, likelihood-free inference with fixed hyperparameters, or likelihood-free inference with hyperparameter tuning, respectively. Then,    
```cd multifrequency_pipeline```       
```python main.py --config=example_yaml_files/[FILENAME]```  

### To run the parameter inference pipeline using harmonic ILC power spectra: 
Modify a yaml file in [harmonic_ILC_pipeline/example_yaml_files](harmonic_ILC_pipeline/example_yaml_files) before running the harmonic ILC pipeline. There are several variants.  To compute the harmonic ILC weights once from some template power spectra and apply those same weights to every realization, use a file with "weights_once", and to compute weights separately for every realization, use a file with "weights_vary". Use [weights_once_analytic.yaml](harmonic_ILC_pipeline/example_yaml_files/weights_once_analytic.yaml), [weights_once_SR.yaml](harmonic_ILC_pipeline/example_yaml_files/weights_once_SR.yaml), [weights_once_LFI_no_tuning.yaml](harmonic_ILC_pipeline/example_yaml_files/weights_once_LFI_no_tuning.yaml), or [weights_once_LFI_with_tuning.yaml](harmonic_ILC_pipeline/example_yaml_files/weights_once_LFI_with_tuning.yaml) to compute the weights once and then use a Gaussian likelihood with analytic parameter dependence, a Gaussian likelihood with parameter dependence determined via symbolic regression (SR), likelihood-free inference (LFI) with fixed hyperparameters, or LFI with hyperparameter tuning, respectively. Similarly, use [weights_vary_SR.yaml](harmonic_ILC_pipeline/example_yaml_files/weights_vary_SR.yaml), [weights_vary_LFI_no_tuning.yaml](harmonic_ILC_pipeline/example_yaml_files/weights_vary_LFI_no_tuning.yaml), or [weights_vary_LFI_with_tuning.yaml](harmonic_ILC_pipeline/example_yaml_files/weights_vary_LFI_with_tuning.yaml) to compute the weights separately for each simulation and then use a Gaussian likelihood (with parameter dependence determined via symbolic regression), likelihood-free inference (LFI) with fixed hyperparameters, or LFI with hyperparameter tuning, respectively. (Note that there is no option for analytic parameter dependence when the weights are varied for each simulation since the weights become nontrivial functions of the parameters in that case.) Then,   
```cd harmonic_ILC_pipeline```       
```python main.py --config=example_yaml_files/[FILENAME]```  

### Other Tools
The [plotting_notebooks](plotting_notebooks/) folder contains example Jupyter notebooks for producing plots after running the above pipelines. The [toy_model](toy_model/) folder runs inference using various methods (numerical and analytic MLE, Fisher matrix, MCMC, and likelihood-free inference) to verify that the results are the same on a simple toy model $A \cos(x) + Bx$, where $A$ and $B$ are free parameters to fit, and Gaussian noise is added to each realization. There are also various tests in the individual subdirectories.  

## Dependencies
python >= 3.7   
pytorch  
sbi  
pyyaml   
healpy  
pysr  
getdist  
emcee  
tqdm  
wandb  

## Acknowledgments  
Portions of this code are adapted from pyilc and sbi.  