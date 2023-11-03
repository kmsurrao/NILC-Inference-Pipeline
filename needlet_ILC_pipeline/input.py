import yaml
import os
import warnings

##########################
# simple function for opening the file
def read_dict_from_yaml(yaml_file):
    assert(yaml_file != None)
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config
##########################

##########################
"""
class that contains input info
"""
class Info(object):
    def __init__(self, input_file):
        self.input_file = input_file
        p = read_dict_from_yaml(self.input_file)

        self.Nsims = p['Nsims'] # number of simulations
        assert type(self.Nsims) is int and self.Nsims>=0, "Nsims"
        self.num_parallel = p['num_parallel']
        assert type(self.num_parallel) is int and self.num_parallel >= 1, "num_parallel"

        self.nside = p['nside']
        assert type(self.nside) is int and self.nside>0, "nside"
        self.ellmax = p['ellmax']
        assert type(self.ellmax) is int and self.ellmax>0, "ellmax"
        self.Nbins = p['Nbins']
        assert type(self.Nbins) is int and self.Nbins>0, "Nbins"
        self.freqs = p['freqs']
        self.Nscales = p['Nscales']
        assert type(self.Nscales) is int and self.Nscales > 0, "Nscales"
        self.GN_FWHM_arcmin = p['GN_FWHM_arcmin']
        assert len(self.GN_FWHM_arcmin) == self.Nscales - 1, "GN_FWHM_arcmin"
        assert all(FWHM_val > 0. for FWHM_val in self.GN_FWHM_arcmin), "GN_FWHM_arcmin"
        self.tsz_amp = p['tSZ_amp']
        assert self.tsz_amp >= 0, 'tSZ_amp'
        self.use_Gaussian_tSZ = p['use_Gaussian_tSZ']
        self.noise = p['noise']
        assert self.noise >= 0, 'noise'

        self.use_lfi = p['use_lfi']
        if self.use_lfi:
            assert 'prior_half_widths' in p, "prior_half_widths must be defined if use_lfi is True"
            self.prior_half_widths = p['prior_half_widths']
            assert len(self.prior_half_widths)==2, "prior_half_widths must have length 2 for Acmb, Atsz"
        if 'Nsims_for_fits' in p and not self.use_lfi:
            self.Nsims_for_fits = p['Nsims_for_fits'] # number of simulations for fitting f(Acmb, Aftsz)
            assert type(self.Nsims_for_fits) is int and 0 <= self.Nsims_for_fits <= self.Nsims, "Nsims_for_fits cannot be greater than Nsims"
        if 'scaling_factors' in p and not self.use_lfi:
            self.scaling_factors = p['scaling_factors']
            assert len(self.scaling_factors) >= 1, "Need at least one scaling factor"
            assert 1 not in self.scaling_factors, "Cannot use 1.0 as a scaling factor"

        self.pyilc_path = p['pyilc_path']
        assert type(self.pyilc_path) is str, "TypeError: pyilc_path"
        assert os.path.isdir(self.pyilc_path), "pyilc path does not exist"
        self.halosky_maps_path = p['halosky_maps_path']
        assert type(self.halosky_maps_path) is str, "TypeError: halosky_maps_path"
        assert os.path.isdir(self.halosky_maps_path), "halosky maps path does not exist"
        self.cmb_map_file = p['cmb_map_file']
        assert type(self.cmb_map_file) is str, "TypeError: cmb_map_file"
        assert os.path.isfile(self.cmb_map_file), "CMB map file does not exist"
        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str, "TypeError: output_dir"
        if os.path.isdir(self.output_dir):
            if os.listdir(self.output_dir): #output directory not empty
                warnings.warn("Output directory is not empty! For safety, make sure to use an output directory that only contains outputs from this pipeline (and other pipelines in the repo) since files will be written and deleted.") 
                
        self.verbose = p['verbose']
        self.save_files = p['save_files']


        
