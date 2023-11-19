import yaml
import os
import warnings
import wandb

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

        self.Nsims = p['Nsims']
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
        self.delta_l = p['delta_l']
        self.omit_central_ell = p['omit_central_ell']
        self.use_Gaussian_tSZ = p['use_Gaussian_tSZ']

        self.compute_weights_once = p['compute_weights_once']
        self.use_lfi = p['use_lfi']
        if self.use_lfi:
            assert 'prior_half_widths' in p, "prior_half_widths must be defined if use_lfi is True"
            self.prior_half_widths = p['prior_half_widths']
            assert len(self.prior_half_widths)==2, "prior_half_widths must have length 2 for Acmb, Atsz"
        if not self.compute_weights_once:
            assert self.use_lfi or (('use_symbolic_regression' in p) and (p['use_symbolic_regression'] is True)), "use_symbolic_regression must be True if compute_weights_once is False"
        if 'use_symbolic_regression' in p and not self.use_lfi:
            self.use_symbolic_regression = p['use_symbolic_regression']
            if self.use_symbolic_regression:
                assert 'Nsims_for_fits' in p, "Nsims_for_fits must be defined if use_symbolic_regression is True"
                assert 'scaling_factors' in p, "scaling_factors must be defined if use_symbolic_regression is True"
        if 'Nsims_for_fits' in p:
            self.Nsims_for_fits = p['Nsims_for_fits']
            assert type(self.Nsims_for_fits) is int and 0 <= self.Nsims_for_fits <= self.Nsims, "Nsims_for_fits cannot be greater than Nsims"
        if 'scaling_factors' in p:
            self.scaling_factors = p['scaling_factors']
            assert len(self.scaling_factors) >= 1, "Need at least one scaling factor"
            assert 1 not in self.scaling_factors, "Cannot use 1.0 as a scaling factor"
        
        self.halosky_maps_path = p['halosky_maps_path']
        assert type(self.halosky_maps_path) is str, "TypeError: halosky_maps_path"
        assert os.path.isdir(self.halosky_maps_path), "halosky maps path does not exist"
        self.cmb_map_file = p['cmb_map_file']
        assert type(self.cmb_map_file) is str, "TypeError: cmb_map_file"
        assert os.path.isfile(self.cmb_map_file), "CMB map file does not exist"
        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str, "TypeError: output_dir"

        self.verbose = p['verbose']
        self.save_files = p['save_files']

        if not self.use_Gaussian_tSZ and not self.use_lfi:
            warnings.warn("You are using a Gaussian likelihood with a non-Gaussian tSZ component. For\
                          more accurate posteriors, switch use_lfi to True to use likelihood-free inference.")
            
        if self.use_lfi:
            self.tune_hyperparameters = p['tune_hyperparameters']
            if self.tune_hyperparameters:
                if 'wandb_project_name' in p:
                    self.wandb_project_name = p['wandb_project_name']
                else:
                    self.wandb_project_name = None
                try:
                    wandb.login()
                except Exception:
                    print('Could not log into wandb. See instructions in README for configuring your login before running the program.')
                    raise
            else:
                self.learning_rate = p['learning_rate']
                self.stop_after_epochs = p['stop_after_epochs']
                self.clip_max_norm = p['clip_max_norm']
                self.num_transforms = p['num_transforms']
                self.hidden_features = p['hidden_features']


        
