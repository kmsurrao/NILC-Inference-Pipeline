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
        self.Nsims = p['Nsims'] # number of simulations
        assert type(self.Nsims) is int and self.Nsims>=0, "Nsims"
        self.nside = p['nside']
        assert type(self.nside) is int and self.nside>0, "nside"
        self.ellmax = p['ellmax']
        assert type(self.ellmax) is int and self.ellmax>0, "ellmax"
        self.halosky_scripts_path = p['halosky_scripts_path']
        assert type(self.halosky_scripts_path) is str, "TypeError: halosky_scripts_path"
        self.cmb_alm_file = p['cmb_alm_file']
        assert type(self.cmb_alm_file) is str, "TypeError: cmb_alm_file"
        self.wigner_file = p['wigner_file']
        assert type(self.wigner_file) is str, "TypeError: wigner_file"
        self.freqs = p['freqs']
        self.Nscales = p['Nscales']
        assert type(self.Nscales) is int and self.Nscales > 0, "Nscales"
        self.tsz_amp = p['tSZ_amp']
        assert self.tsz_amp >= 0, 'tSZ_amp'
        self.pyilc_path = p['pyilc_path']
        assert type(self.pyilc_path) is str, "TypeError: pyilc_path"
        self.verbose = p['verbose']
        self.remove_files = p['remove_files']
        self.GN_FWHM_arcmin = np.asarray(p['GN_FWHM_arcmin'])
        assert len(self.GN_FWHM_arcmin) == self.Nscales - 1, "GN_FWHM_arcmin"
        assert all(FWHM_val > 0. for FWHM_val in self.GN_FWHM_arcmin), "GN_FWHM_arcmin"

        