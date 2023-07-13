import subprocess
import yaml
import os

def setup_pyilc(sim, inp, env, suppress_printing=False, scaling=None):
    '''
    Sets up yaml files for pyilc and runs the code

    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    env: environment object
    suppress_printing: Bool, whether to suppress outputs and errors from pyilc code itself
    scaling: None or list of length 5
            idx0: 0 if "scaled" means maps are scaled down, 1 if "scaled" means maps are scaled up
            idx1: 0 for unscaled CMB, 1 for scaled CMB
            idx2: 0 for unscaled ftSZ, 1 for scaled ftSZ
            idx3: 0 for unscaled noise90, 1 for scaled noise90
            idx4: 0 for unscaled noise150, 1 for scaled noise150

    RETURNS
    -------
    None
    '''

    #set up yaml files for pyilc
    pyilc_input_params = {}
    pyilc_input_params['output_dir'] = str(inp.output_dir) + "/pyilc_outputs/"
    if scaling is not None: 
        scaling_str = ''.join(str(e) for e in scaling)
        pyilc_input_params['output_dir'] += f"{scaling_str}/"
    pyilc_input_params['output_prefix'] = "sim" + str(sim)
    pyilc_input_params['save_weights'] = "yes"
    pyilc_input_params['ELLMAX'] = inp.ell_sum_max
    pyilc_input_params['N_scales'] = inp.Nscales
    pyilc_input_params['GN_FWHM_arcmin'] = [inp.GN_FWHM_arcmin[i] for i in range(len(inp.GN_FWHM_arcmin))]
    pyilc_input_params['taper_width'] = 0
    pyilc_input_params['N_freqs'] = len(inp.freqs)
    pyilc_input_params['freqs_delta_ghz'] = inp.freqs
    pyilc_input_params['N_side'] = inp.nside
    pyilc_input_params['wavelet_type'] = "GaussianNeedlets"
    pyilc_input_params['bandpass_type'] = "DeltaBandpasses"
    pyilc_input_params['beam_type'] = "Gaussians"
    pyilc_input_params['beam_FWHM_arcmin'] = [1.4, 1.4] #update with whatever beam FWHM was used in the sim map construction; note that ordering should always be from lowest-res to highest-res maps (here and in the lists of maps, freqs, etc above)    pyilc_input_params_preserved_cmb = {'ILC_preserved_comp': 'CMB'}
    pyilc_input_params['ILC_bias_tol'] = 0.01
    pyilc_input_params['N_deproj'] = 0
    pyilc_input_params['N_SED_params'] = 0
    pyilc_input_params['N_maps_xcorr'] = 0
    if scaling is None:
        pyilc_input_params['freq_map_files'] = [f'{inp.output_dir}/maps/sim{sim}_freq1.fits', f'{inp.output_dir}/maps/sim{sim}_freq2.fits']
    else:
        pyilc_input_params['freq_map_files'] = [f'{inp.output_dir}/maps/{scaling_str}/sim{sim}_freq1.fits', f'{inp.output_dir}/maps/{scaling_str}/sim{sim}_freq2.fits']
    pyilc_input_params_preserved_cmb = {'ILC_preserved_comp': 'CMB'}
    pyilc_input_params_preserved_tsz = {'ILC_preserved_comp': 'tSZ'}
    pyilc_input_params_preserved_cmb.update(pyilc_input_params)
    pyilc_input_params_preserved_tsz.update(pyilc_input_params)
    if scaling is None:
        CMB_yaml = f'{inp.output_dir}/pyilc_yaml_files/sim{sim}_CMB_preserved.yml'
        tSZ_yaml = f'{inp.output_dir}/pyilc_yaml_files/sim{sim}_tSZ_preserved.yml'
    else:
        CMB_yaml = f'{inp.output_dir}/pyilc_yaml_files/{scaling_str}/sim{sim}_CMB_preserved.yml'
        tSZ_yaml = f'{inp.output_dir}/pyilc_yaml_files/{scaling_str}/sim{sim}_tSZ_preserved.yml'
    with open(CMB_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_cmb, outfile, default_flow_style=None)
    with open(tSZ_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_tsz, outfile, default_flow_style=None)

    #run pyilc for preserved CMB and preserved tSZ
    stdout = subprocess.DEVNULL if suppress_printing else None
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {CMB_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component CMB, sim {sim}, scaling {scaling_str}', flush=True)
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {tSZ_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component tSZ, sim {sim}, scaling {scaling_str}', flush=True)

    #remove unncessary files
    subprocess.run('rm -f ' + pyilc_input_params['output_dir'] + f'sim{sim}_needletcoeff*', shell=True, env=env)
    subprocess.run('rm -f ' + pyilc_input_params['output_dir'] + f'sim{sim}*.pdf', shell=True, env=env)
    
    return

def weight_maps_exist(sim, inp, scaling=None):
    '''
    Sets up yaml files for pyilc and runs the code

    ARGUMENTS
    ---------
    sim: int, simulation number
    inp: Info object containing input parameter specifications
    scaling: None or list of length 5
            idx0: 0 if "scaled" means maps are scaled down, 1 if "scaled" means maps are scaled up
            idx1: 0 for unscaled CMB, 1 for scaled CMB
            idx2: 0 for unscaled ftSZ, 1 for scaled ftSZ
            idx3: 0 for unscaled noise90, 1 for scaled noise90
            idx4: 0 for unscaled noise150, 1 for scaled noise150

    RETURNS
    -------
    Bool, whether or not weight maps already exist
    '''
    
    for comp in ['CMB', 'tSZ']:
        for freq in range(len(inp.freqs)):
            for scale in range(inp.Nscales):
                if scaling is not None:
                    scaling_str = ''.join(str(e) for e in scaling)
                    if not os.path.exists(f"{inp.output_dir}/pyilc_outputs/{scaling_str}/sim{sim}weightmap_freq{freq}_scale{scale}_component_{comp}.fits"):
                        return False
                else:
                    if not os.path.exists(f"{inp.output_dir}/pyilc_outputs/sim{sim}weightmap_freq{freq}_scale{scale}_component_{comp}.fits"):
                        return False
    return True