import subprocess
import yaml

def setup_pyilc(sim, inp, env):

    #set up yaml files for pyilc
    pyilc_input_params = {}
    pyilc_input_params['output_dir'] = str(inp.output_dir) + "/pyilc_outputs/"
    pyilc_input_params['output_prefix'] = "sim" + str(sim)
    pyilc_input_params['save_weights'] = "yes"
    pyilc_input_params['ELLMAX'] = inp.ell_sum_max
    pyilc_input_params['N_scales'] = inp.Nscales
    pyilc_input_params['GN_FWHM_arcmin'] = [inp.GN_FWHM_arcmin[i] for i in range(len(inp.GN_FWHM_arcmin))]
    pyilc_input_params['taper_width'] = 2*(inp.ell_sum_max-inp.ellmax)
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
    pyilc_input_params['freq_map_files'] = [f'{inp.output_dir}/maps/sim{sim}_freq1.fits', f'{inp.output_dir}/maps/sim{sim}_freq2.fits']
    pyilc_input_params_preserved_cmb = {'ILC_preserved_comp': 'CMB'}
    pyilc_input_params_preserved_tsz = {'ILC_preserved_comp': 'tSZ'}
    pyilc_input_params_preserved_cmb.update(pyilc_input_params)
    pyilc_input_params_preserved_tsz.update(pyilc_input_params)
    with open(f'{inp.output_dir}/pyilc_yaml_files/CMB_preserved.yml', 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_cmb, outfile, default_flow_style=None)
    with open(f'{inp.output_dir}/pyilc_yaml_files/tSZ_preserved.yml', 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_tsz, outfile, default_flow_style=None)

    #run pyilc for preserved CMB and preserved tSZ
    # subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.output_dir}/pyilc_yaml_files/CMB_preserved.yml"], shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.output_dir}/pyilc_yaml_files/CMB_preserved.yml"], shell=True, env=env)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component CMB, sim {sim}', flush=True)
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {inp.output_dir}/pyilc_yaml_files/tSZ_preserved.yml"], shell=True, env=env)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component tSZ, sim {sim}', flush=True)
    if inp.remove_files: #don't need frequency maps anymore
        subprocess.call(f'rm {inp.output_dir}/maps/sim{sim}_freq1.fits {inp.output_dir}/maps/sim{sim}_freq2.fits', shell=True, env=env)
    
    return