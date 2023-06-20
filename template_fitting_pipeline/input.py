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
        self.num_parallel = p['num_parallel']
        assert type(self.num_parallel) is int and self.num_parallel>=1, "num_parallel"

        self.nside = p['nside']
        assert type(self.nside) is int and self.nside>0, "nside"
        self.ellmax = p['ellmax']
        assert type(self.ellmax) is int and self.ellmax>0, "ellmax"
        self.Nbins = p['Nbins']
        assert type(self.Nbins) is int and self.Nbins>0, "Nbins"
        self.tsz_amp = p['tSZ_amp']
        assert self.tsz_amp >= 0, 'tSZ_amp'
        self.noise = p['noise']
        assert self.noise >= 0, 'noise'
        self.freqs = p['freqs']
        self.use_Gaussian_cov = p['use_Gaussian_cov']
        self.use_Gaussian_tSZ = p['use_Gaussian_tSZ']


        self.halosky_maps_path = p['halosky_maps_path']
        assert type(self.halosky_maps_path) is str, "TypeError: halosky_maps_path"
        self.cmb_map_file = p['cmb_map_file']
        assert type(self.cmb_map_file) is str, "TypeError: cmb_map_file"
        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str, "TypeError: output_dir"

        self.verbose = p['verbose']
        self.save_files = p['save_files']


        