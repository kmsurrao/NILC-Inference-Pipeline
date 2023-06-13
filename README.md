# NILC-Inference-Pipeline
Pipelines for calculating parameter covariance matrix elements obtained via needlet internal linear combination (NILC) and power spectrum template-fitting. Computation of analytic expressions for NILC power spectra. Currently assumes a model with only CMB, tSZ (which can be amplified), and noise, at two frequency channels, but the model can easily be extended.

## Dependencies
Python >= 3.6.
pyyaml
pywigxjpf
healpy

## Requirements and Set-up
Markup: * Clone the pyilc repository (https://github.com/jcolinhill/pyilc) and insert this   path in the yaml files. 
        * Requires a CMB map fits file in Kelvin (lensed alm can be downloaded from WebSky at https://mocks.cita.utoronto.ca/data/websky/v0.0/). 
        * Requires Nsims tSZ maps in fits file format in units of Kelvin, which can be generated via halosky (https://github.com/marcelo-alvarez/halosky).
        * Modify example.yaml or create a similar yaml file in the appropriate subdirectories.

## Running
To run parameter inference pipeline for NILC:
```cd nilc_pipeline```
```python main.py --config=example.yaml```

To run parameter inference pipeline for power spectrum template-fitting:
```cd template_fitting_pipeline```
```python main.py --config=example.yaml```

To check the analytic NILC power spectrum result:
```cd analytic_model```
```python main.py --config=example.yaml```

## Recommendations
There is a large amount of I/O from running this program. It is highly recommended to run on an HPC cluster and to set the output_dir parameter in the yaml files to be an empty subdirectory in a SCRATCH space. It is also recommended (though not required) to comment out calls to healpy mollview in pyilc/pyilc/wavelets.py.

