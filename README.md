# NILC-Parameter-Pipeline
Pipeline for calculating parameter covariance matrix obtained via needlet internal linear combination (NILC).

## Installation 
Must have pyilc, modified to handle simulation numbers for input frequency maps. Modify inputs in example.yaml as well as yaml files for pyilc (pyilc/input/CMB_preserved.yml and pyilc/input/tSZ_preserved.yml). Also requires a file for CMB lensed alm, which can be downloaded from WebSky (https://mocks.cita.utoronto.ca/data/websky/v0.0/); a file containing a 3D array of wigner3j symbols for m1=m2=m3=0; a file containing a 3D array of wigner3j symbols for symbols of the form (l1 l2 l2 0 -m2 m2); and Nsims tSZ maps in fits file format, which can be generated via halosky (https://github.com/marcelo-alvarez/halosky).

## Running
To run with default inputs in example.yaml:
python pipeline/main.py 

Can optionally specifiy input yaml file:
python pipeline/main.py [path to input yaml file]

## Code Structure

### Pipeline
Pipeline for determining Acmb and Atsz parameter covariance matrix via the NILC approach.

#### main.py
Executes all modules and prints final estimates for Acmb and Atsz.

#### input.py
Reads in input yaml file and stores information in the class Info.

#### generate_maps.py
Function generate_freq_maps creates maps consisting of CMB, tSZ (possibly multiplied by an amplification factor), and noise at specified frequencies. Final frequency maps are stored in the scratch_path/maps directory. Returns power spectra of CMB, amplified tSZ, and noise.

#### wt_map_spectra.py
Calculates power spectra of weight maps outputted by pyilc (must specify output directory for weight maps from pyilc as scratch_path/wt_maps/). Index weight map spectra M as M[0-2][n][m][i][j], where n and m are needlet filter scales; i and j are frequency channels; and the first index is 0 for TT weight map spectra, 1 for Ty weight may spectra, and 2 for yy weight map spectra. Also calculates cross-spectra of component maps and weight maps. Index these spectra W as W[p][i][n], where i indexes the frequency channel, n indexes the needlet filter scale, and p indexes the preserved component used to generate the weight maps (0 for preserved CMB and 1 for preserved tSZ). 

#### data_spectra.py
Calculates propagation of all component power spectra to NILC ClTT, ClTy, and Clyy spectra. Saved in scratch_path/power_spectra. Index as clTT[sim][component][ell] where sim goes from 0 to Nsims; component is 0 for CMB contribution, 1 for tSZ contribution, and 2 for noise contribution; and ell goes from 0 through ellmax.

#### acmb_atsz_nilc.py
Finds best fit Acmb and Atsz for each simulation and computes final parameter covariance matrix.

### compare_analytic_to_sim
Scripts for checking agreement between analytic and direct calculation for propagation of contaminant power spectra to a NILC map. Outputs saved under sim 101.

#### compare_contam_spectra_nilc_cross.py
Compares propagation of components to cross-spectrum of tSZ and CMB NILC maps via analytic approach and direct calculation.

#### compare_contam_spectra_preserved_cmb.py
Compares propagation of components to CMB NILC maps via analytic approach and direct calculation.

#### compare_contam_spectra_preserved_tsz.py
Compares propagation of components to tSZ NILC maps via analytic approach and direct calculation.


### test_analytic
Contains scripts for checking each step of derivation for analytic power spectrum propagation of contaminant to NILC map power spectra.

#### test_master.py
Tests the MASTER equation relating power spectrum of a masked map the the power spectrum of the unmasked map.

#### test_applying_master.py
Tests the MASTER equation using our actual maps and pyilc derived weight maps for the frequencies and scales of interest.

#### test_sum_freqs.py
Tests MASTER equation at each frequency pair and then summing over frequencies. Ignores initial needlet filter.

#### test_initial_filter.py
Tests step involving initial application of needlet filters and then summing over frequencies.

#### test_without_final_filter.py
Tests entire analytic equation but without the final needlet filtering step.

#### test_freqmap_wtmap_corr.py
Computes power spectra and correlations of component maps and weight maps.



### Other

#### nilc_power_spectrum_calc.py
Contains function for calculating propagation of a component's power spectrum to final power spectra of two NILC maps.

#### ell_contributions_heat_map.py
Produces grid image showing contributions at every ell2 and ell3 to the final power spectrum propagation to NILC map at some ell. Outputs saved under sim 102. 

#### generate_random_wt_maps.py
Used for testing purposes only. Generates random weight maps satisfying the signal preservation constraint. Outputs saved under sim 103.