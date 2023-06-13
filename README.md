# NILC-Inference-Pipeline
Pipelines for calculating parameter covariance matrix elements obtained via needlet internal linear combination (NILC) based likelihoods, harmonic internal linear combination (HILC) based likelihoods, and power spectrum template-fitting. Computation of analytic expressions for NILC power spectra. Currently assumes a sky model comprising only the CMB, tSZ effect (which can be amplified), and noise, at two frequency channels, but the model can be extended easily.

## Requirements and Set-up
 - Requires a clone of the pyilc repository (https://github.com/jcolinhill/pyilc). 
 - Requires a CMB map .fits file in Kelvin (lensed alm can be downloaded from WebSky at https://mocks.cita.utoronto.ca/data/websky/v0.0/). 
 - Requires Nsims tSZ maps in .fits file format in units of Kelvin, which can be generated via halosky (https://github.com/marcelo-alvarez/halosky).
 - Modify example.yaml or create a similar yaml file in the appropriate subdirectories, modifying paths to the above requirements and any additional specifications. See the specific files for details.

## Running
To run the parameter inference pipeline for a NILC-based likelihood:  
```cd nilc_pipeline```   
```python main.py --config=example.yaml```       

To run the parameter inference pipeline for power spectrum template-fitting:  
```cd template_fitting_pipeline```       
```python main.py --config=example.yaml```    

To check the analytic NILC power spectrum result:  
```cd analytic_model```   
```python main.py --config=example.yaml```  

The notebooks folder contains example Jupyter notebooks for producing plots after running the above pipelines. The hilc_and_template_fitting contains Mathematica notebooks for analytic demonstrations of equality of the HILC and template-fitting methods, as well as python scripts for numerical demonstrations of equality between the methods when using data-split cross-spectra.

## Recommendations
There is a large amount of I/O from running this program. It is highly recommended to run on an HPC cluster and to set the output_dir parameter in the yaml files to be an empty subdirectory in a SCRATCH space. It is also recommended (though not required) to comment out calls to healpy mollview in pyilc/pyilc/wavelets.py.

## Dependencies
python >= 3.6   
pyyaml  
pywigxjpf  
healpy  

## Acknowledgments
Portions of this code are adapted from PolyBin (https://github.com/oliverphilcox/PolyBin) and reMASTERed (https://github.com/kmsurrao/reMASTERed).

