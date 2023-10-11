# NILC-Inference-Pipeline
This repository contains pipelines for calculating parameter covariance matrix elements obtained via needlet internal linear combination (NILC) based likelihoods ([needlet_ILC_pipeline](needlet_ILC_pipeline)), harmonic internal linear combination (HILC) based likelihoods ([harmonic_ILC_pipeline](harmonic_ILC_pipeline)), and multifrequency power spectrum template-fitting ([multifrequency_pipeline](multifrequency_pipeline)). The [shared](shared) folder contains utilities and functions that are shared across various pipelines. The code currently assumes a sky model comprising only the CMB, tSZ effect (which can be amplified), and noise at two frequency channels, but the model can be extended easily.

## Requirements and Set-up
 - Requires a clone of the pyilc repository (https://github.com/jcolinhill/pyilc). 
 - Requires a CMB map .fits file in Kelvin (lensed alm can be downloaded from WebSky at https://mocks.cita.utoronto.ca/data/websky/v0.0/). 
 - Requires Nsims tSZ maps in .fits file format in units of Kelvin, which can be generated via halosky (https://github.com/marcelo-alvarez/halosky).
 - Modify example.yaml or create a similar yaml file in the appropriate subdirectories, modifying paths to the above requirements and any additional specifications. See the specific files for details.

## Running
To run the parameter inference pipeline for a needlet ILC-based likelihood:  
```cd needlet_ILC_pipeline```   
```python main.py --config=example.yaml```       

To run the parameter inference pipeline for multifrequency power spectrum template-fitting:  
```cd multifrequency_pipeline```       
```python main.py --config=example.yaml```  

To run the parameter inference pipeline for a harmonic ILC-based likelihood:  
```cd harmonic_ILC_pipeline```       
```python main.py --config=example.yaml```  

The [jupyter_notebooks](jupyter_notebooks) folder contains example Jupyter notebooks for producing plots after running the above pipelines. The [toy_model](toy_model) folder runs inference using various methods (numerical and analytic MLE, Fisher matrix, MCMC, and likelihood-free inference) to verify that the results are the same on a simple toy model $A \cos(x) + Bx$, where $A$ and $B$ are free parameters to fit, and Gaussian noise is added to each realization.

## Recommendations
There is a large amount of I/O from running this program. It is highly recommended to run on an HPC cluster and to set the output_dir parameter in the yaml files to be an empty subdirectory in a SCRATCH space.

## Dependencies
python >= 3.7   
pytorch  
sbi  
pyyaml   
healpy  
pysr  
getdist  
emcee  

## Acknowledgments
Portions of this code are adapted from PolyBin (https://github.com/oliverphilcox/PolyBin), pyilc (https://github.com/jcolinhill/pyilc), and reMASTERed (https://github.com/kmsurrao/reMASTERed).

