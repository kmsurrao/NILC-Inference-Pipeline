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
        self.Nscales = p['Nscales']
        assert type(self.Nscales) is int and self.Nscales > 0, "Nscales"
        self.GN_FWHM_arcmin = p['GN_FWHM_arcmin']
        assert len(self.GN_FWHM_arcmin) == self.Nscales - 1, "GN_FWHM_arcmin"
        assert all(FWHM_val > 0. for FWHM_val in self.GN_FWHM_arcmin), "GN_FWHM_arcmin"

        self.freqs = p['freqs']
        self.noise = p['noise']
        assert len(self.noise) == len(self.freqs), f"Must provide white noise levels for each of {len(self.freqs)} frequencies"
        for n in self.noise:
            assert n >= 0, 'White noise level must be nonnegative'
        
        self.comps = p['comps']
        assert len(self.freqs) >= len(self.comps), "Must have at least as many frequencies as sky components"
        assert (set(self.comps)).issubset({'cmb', 'tsz', 'cib'}), "Currently the only supported components are cmb, tsz, and cib"
        self.paths_to_comps = p['paths_to_comps']
        assert len(self.paths_to_comps) == len(self.comps), f"Need {len(self.comps)} paths to components"
        self.use_Gaussian = []
        for c, path in enumerate(self.paths_to_comps):
            assert os.path.isfile(path) or os.path.isdir(path), f"No such file or directory: {path}"
            if os.path.isfile(path):
                self.use_Gaussian.append(True)
            else:
                self.use_Gaussian.append(False)
                assert os.path.isfile(f'{path}/{self.comps[c]}_00000.fits'), f"{path} must contain files of the form {self.comps[c]}_xxxxx.fits"
                assert os.path.isfile(f'{path}/{self.comps[c]}_{self.Nsims:05d}.fits'), f"{path} must contain files of the form {self.comps[c]}_xxxxx.fits up to {self.comps[c]}_{self.Nsims:05d}.fits"
        self.amp_factors = p['amp_factors']
        for amp in self.amp_factors:
            assert amp > 0, 'amp_factors must all be > 0'

        self.use_lfi = p['use_lfi']
        if self.use_lfi:
            assert 'prior_half_widths' in p, "prior_half_widths must be defined if use_lfi is True"
            self.prior_half_widths = p['prior_half_widths']
            assert len(self.prior_half_widths)==len(self.comps), f"prior_half_widths must have same length as number of components ({len(self.comps)})"
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
        self.output_dir = p['output_dir']
        assert type(self.output_dir) is str, "TypeError: output_dir" 
                
        self.verbose = p['verbose']
        self.save_files = p['save_files']

        if False in self.use_Gaussian and not self.use_lfi:
            warnings.warn('You are using a Gaussian likelihood with potentially non-Gaussian components. For more accurate posteriors, switch use_lfi to True to use likelihood-free inference.')
        
        if self.use_lfi:
            self.tune_hyperparameters = p['tune_hyperparameters']
            if self.tune_hyperparameters:
                self.Nsweeps = p['Nsweeps']
                assert type(self.Nsweeps) is int and self.Nsweeps >= 1, "Nsweeps must be integer >= 1"
                if 'wandb_project_name' in p:
                    self.wandb_project_name = p['wandb_project_name']
                else:
                    self.wandb_project_name = None
                if 'wandb_api_key' in p:
                    try:
                        wandb.login(key=self.wandb_api_key)
                    except Exception:
                        try:
                            wandb.login()
                        except Exception:
                            print('Could not log into wandb. Either configure your login prior to running the program (see instructions in README), orspecify a valid API key in the wandb_api_key field of the yaml file.')
                            raise
                else:
                    try:
                        wandb.login()
                    except Exception:
                        print('Could not log into wandb. Either configure your login prior to running the program (see instructions in README), or specify a valid API key in the wandb_api_key field of the yaml file.')
                        raise
            else:
                self.learning_rate = p['learning_rate']
                self.stop_after_epochs = p['stop_after_epochs']
                self.clip_max_norm = p['clip_max_norm']
                self.num_transforms = p['num_transforms']
                self.hidden_features = p['hidden_features']


        
