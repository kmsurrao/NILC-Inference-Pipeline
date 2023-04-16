import yaml
import numpy as np

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
class that contains map info (and associated data), ILC specifications, etc., and handles input
"""
class Info(object):
    def __init__(self, input_file):
        self.input_file = input_file
        p = read_dict_from_yaml(self.input_file)

        self.num_parallel = p['num_parallel']
        assert self.num_parallel >= 1, "num_parallel"
        self.nside = p['nside']
        assert type(self.nside) is int and self.nside>0, "nside"
        self.ellmax = p['ellmax']
        assert type(self.ellmax) is int and self.ellmax>0, "ellmax"
        self.ell_sum_max = p['ell_sum_max']
        assert type(self.ell_sum_max) is int and self.ell_sum_max>0, "ell_sum_max"
        self.freqs = p['freqs']
        self.Nscales = p['Nscales']
        assert type(self.Nscales) is int and self.Nscales > 0, "Nscales"
        self.GN_FWHM_arcmin = p['GN_FWHM_arcmin']
        assert len(self.GN_FWHM_arcmin) == self.Nscales - 1, "GN_FWHM_arcmin"
        assert all(FWHM_val > 0. for FWHM_val in self.GN_FWHM_arcmin), "GN_FWHM_arcmin"
        self.tsz_amp = p['tSZ_amp']
        assert self.tsz_amp >= 0, 'tSZ_amp'
        self.noise = p['noise']
        assert self.noise >= 0, 'noise'

        self.pyilc_path = p['pyilc_path']
        assert type(self.pyilc_path) is str, "TypeError: pyilc_path"
        self.halosky_maps_path = p['halosky_maps_path']
        assert type(self.halosky_maps_path) is str, "TypeError: halosky_maps_path"
        self.cmb_map_file = p['cmb_map_file']
        assert type(self.cmb_map_file) is str, "TypeError: cmb_map_file"
        self.wigner_file = p['wigner_file']
        assert type(self.wigner_file) is str, "TypeError: wigner_file"
        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str, "TypeError: output_dir"

        self.verbose = p['verbose']
        self.remove_files = p['remove_files']
        self.save_files = p['save_files']


        