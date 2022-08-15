# NILC-Parameter-Pipeline
Pipeline for calculating parameter covariance matrix obtained via needlet internal linear combination (NILC).

## Installation 
Must have pyilc, modified to handle simulation numbers for input frequency maps. Modify inputs in example.yaml as well as yaml files for pyilc (pyilc/input/CMB_preserved.yml and pyilc/input/tSZ_preserved.yml). Also requires a file for CMB lensed alm, which can be downloaded from WebSky (https://mocks.cita.utoronto.ca/data/websky/v0.0/); a file containing a 3D array of wigner3j symbols; and Nsims tSZ maps in fits file format, which can be generated via halosky (https://github.com/marcelo-alvarez/halosky).

## Running
To run with default inputs in example.yaml:
python main.py 

Can optionally specifiy input yaml file:
python main.py [path to input yaml file]

## Code Structure

### main.py
Executes all modules and prints final estimates for Acmb and Atsz.

### input.py
Reads in input yaml file and stores information in the class Info.

### generate_maps.py
Function generate_freq_maps creates maps consisting of CMB, tSZ (possibly multiplied by an amplification factor), and noise at specified frequencies. Final frequency maps are stored in the scratch_path/maps/ directory. Returns power spectra of CMB, amplified tSZ, and noise.

### wt_map_spectra.py
Calculates power spectra of weight maps outputted by pyilc (must specify output directory for weight maps from pyilc as NILC-Parameter_Pipeline/wt_maps/). Weight map spectra stored in scratch_path/wt_maps. Index weight map spectra M as M[0-2][n][m][i][j], where n and m are needlet filter scales; i and j are frequency channels; and the first index is 0 for TT weight map spectra, 1 for Ty weight may spectra, and 2 for yy weight map spectra.

### nilc_power_spectrum_calc.py
Contains function for calculating propagation of a component's power spectrum to final power spectra of two NILC maps.

### data_spectra.py
Calculates propagation of all component power spectra to NILC ClTT, ClTy, and Clyy spectra. Saved in power_spectra/. Index as clTT[sim][component][ell] where sim goes from 0 to Nsims; component is 0 for CMB contribution, 1 for tSZ contribution, and 2 for noise contribution; and ell goes from 0 to ellmax.

### acmb_atsz_nilc.py
Finds best fit Acmb and Atsz for each simulation and computes final parameter covariance matrix.
