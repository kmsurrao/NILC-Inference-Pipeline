import subprocess
import yaml
import os
import tempfile
import healpy as hp
import numpy as np

def setup_pyilc(sim, split, inp, env, map_tmpdir, suppress_printing=False, scaling=None, pars=None, cib=False):
    '''
    Sets up yaml files for pyilc and runs the code for needlet ILC

    ARGUMENTS
    ---------
    sim: int, simulation number
    split: int, split number (1 or 2)
    inp: Info object containing input parameter specifications
    env: environment object
    map_tmpdir: str, directory in which maps are saved
    suppress_printing: Bool, whether to suppress outputs and errors from pyilc code itself
    scaling: None or list of length 3
            idx0: takes on values from 0 to len(inp.scaling_factors)-1,
                  indicating by which scaling factor the input maps are scaled
            idx1: 0 for unscaled CMB, 1 for scaled CMB
            idx2: 0 for unscaled ftSZ, 1 for scaled ftSZ
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)
    cib: Bool, whether to build CIB NILC maps in addition to CMB and ftSZ

    RETURNS
    -------
    tmpdir: str, temporary directory in which pyilc files were stored
    '''

    #set up temporary directory
    tmpdir = tempfile.mkdtemp(dir=inp.output_dir)

    #set up yaml files for pyilc
    pyilc_input_params = {}
    pyilc_input_params['output_dir'] = tmpdir + '/'
    if scaling is not None: 
        scaling_str = ''.join(str(e) for e in scaling)
    else:
        scaling_str = ''
    pyilc_input_params['output_prefix'] = f"sim{sim}_split{split}"
    if pars is not None:
        pars_str = f'_pars{pars[0]:.3f}_{pars[1]:.3f}_'
        pyilc_input_params['output_prefix'] += pars_str
    else:
        pars_str = ''
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
    pyilc_input_params['beam_FWHM_arcmin'] = [1.4, 1.4] #update with whatever beam FWHM was used in the sim map construction; note that ordering should always be from lowest-res to highest-res maps (here and in the lists of maps, freqs, etc above)
    pyilc_input_params['ILC_bias_tol'] = 0.01
    pyilc_input_params['N_deproj'] = 0
    pyilc_input_params['N_SED_params'] = 0
    pyilc_input_params['N_maps_xcorr'] = 0
    pyilc_input_params['freq_map_files'] = [f'{map_tmpdir}/sim{sim}_freq1_split{split}{pars_str}.fits', f'{map_tmpdir}/sim{sim}_freq2_split{split}{pars_str}.fits']
    pyilc_input_params_preserved_cmb = {'ILC_preserved_comp': 'CMB'}
    pyilc_input_params_preserved_tsz = {'ILC_preserved_comp': 'tSZ'}
    pyilc_input_params_preserved_cmb.update(pyilc_input_params)
    pyilc_input_params_preserved_tsz.update(pyilc_input_params)
    if cib:
        pyilc_input_params_preserved_cib = {'ILC_preserved_comp': 'CIB'}
        pyilc_input_params_preserved_cib.update(pyilc_input_params)

    #dump yaml files
    CMB_yaml = f'{tmpdir}/sim{sim}_split{split}_CMB_preserved.yml'
    tSZ_yaml = f'{tmpdir}/sim{sim}_split{split}_tSZ_preserved.yml'
    with open(CMB_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_cmb, outfile, default_flow_style=None)
    with open(tSZ_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_tsz, outfile, default_flow_style=None)
    if cib:
        CIB_yaml = f'{tmpdir}/sim{sim}_split{split}_CIB_preserved.yml'
        with open(CIB_yaml, 'w') as outfile:
            yaml.dump(pyilc_input_params_preserved_cib, outfile, default_flow_style=None)

    #run pyilc for preserved CMB and preserved tSZ
    stdout = subprocess.DEVNULL if suppress_printing else None
    if scaling is not None:
        scaling_str = f', scaling {scaling_str}'
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {CMB_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component CMB, sim {sim}{scaling_str}, pars={pars}', flush=True)
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {tSZ_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
    if inp.verbose:
        print(f'generated NILC weight maps for preserved component tSZ, sim {sim}{scaling_str}, pars={pars}', flush=True)
    if cib:
        subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {CIB_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
        if inp.verbose:
            print(f'generated NILC weight maps for preserved component CIB, sim {sim}{scaling_str}, pars={pars}', flush=True)

    
    return tmpdir



def setup_pyilc_hilc(sim, split, inp, env, map_tmpdir, suppress_printing=False, scaling=None, pars=None):
    '''
    Sets up yaml files for pyilc and runs the code for harmonic ILC
    (This function isn't used in the main pipeline but can be useful to view HILC maps)

    ARGUMENTS
    ---------
    sim: int, simulation number
    split: int, split number (1 or 2)
    inp: Info object containing input parameter specifications
    env: environment object
    map_tmpdir: str, directory in which maps are saved
    suppress_printing: Bool, whether to suppress outputs and errors from pyilc code itself
    scaling: None or list of length 3
            idx0: takes on values from 0 to len(inp.scaling_factors)-1,
                  indicating by which scaling factor the input maps are scaled
            idx1: 0 for unscaled CMB, 1 for scaled CMB
            idx2: 0 for unscaled ftSZ, 1 for scaled ftSZ
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)

    RETURNS
    -------
    tmpdir: str, temporary directory in which pyilc files were stored
    '''

    #set up temporary directory
    tmpdir = tempfile.mkdtemp(dir=inp.output_dir)

    #set up yaml files for pyilc
    pyilc_input_params = {}
    pyilc_input_params['output_dir'] = tmpdir + '/'
    if scaling is not None: 
        scaling_str = ''.join(str(e) for e in scaling)
    else:
        scaling_str = ''
    pyilc_input_params['output_prefix'] = f"sim{sim}_split{split}"
    if pars is not None:
        pars_str = f'_pars{pars[0]:.3f}_{pars[1]:.3f}_'
        pyilc_input_params['output_prefix'] += pars_str
    else:
        pars_str = ''
    pyilc_input_params['save_weights'] = "no"
    pyilc_input_params['ELLMAX'] = inp.ell_sum_max     
    pyilc_input_params['taper_width'] = 0
    pyilc_input_params['N_freqs'] = len(inp.freqs)
    pyilc_input_params['freqs_delta_ghz'] = inp.freqs
    pyilc_input_params['N_side'] = inp.nside
    pyilc_input_params['wavelet_type'] = 'TopHatHarmonic'
    pyilc_input_params['BinSize'] = 10
    pyilc_input_params['bandpass_type'] = "DeltaBandpasses"
    pyilc_input_params['beam_type'] = "Gaussians"
    pyilc_input_params['beam_FWHM_arcmin'] = [1.4, 1.4] #update with whatever beam FWHM was used in the sim map construction; note that ordering should always be from lowest-res to highest-res maps (here and in the lists of maps, freqs, etc above)
    pyilc_input_params['ILC_bias_tol'] = 0.01
    pyilc_input_params['N_deproj'] = 0
    pyilc_input_params['N_SED_params'] = 0
    pyilc_input_params['N_maps_xcorr'] = 0
    pyilc_input_params['freq_map_files'] = [f'{map_tmpdir}/sim{sim}_freq1_split{split}{pars_str}.fits', f'{map_tmpdir}/sim{sim}_freq2_split{split}{pars_str}.fits']
    pyilc_input_params_preserved_cmb = {'ILC_preserved_comp': 'CMB'}
    pyilc_input_params_preserved_tsz = {'ILC_preserved_comp': 'tSZ'}
    pyilc_input_params_preserved_cmb.update(pyilc_input_params)
    pyilc_input_params_preserved_tsz.update(pyilc_input_params)

    #dump yaml files
    CMB_yaml = f'{tmpdir}/sim{sim}_split{split}_CMB_preserved.yml'
    tSZ_yaml = f'{tmpdir}/sim{sim}_split{split}_tSZ_preserved.yml'
    with open(CMB_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_cmb, outfile, default_flow_style=None)
    with open(tSZ_yaml, 'w') as outfile:
        yaml.dump(pyilc_input_params_preserved_tsz, outfile, default_flow_style=None)

    #run pyilc for preserved CMB and preserved tSZ
    stdout = subprocess.DEVNULL if suppress_printing else None
    if scaling is not None:
        scaling_str = f', scaling {scaling_str}'
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {CMB_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
    if inp.verbose:
        print(f'generated HILC weight maps for preserved component CMB, sim {sim}{scaling_str}, pars={pars}', flush=True)
    subprocess.run([f"python {inp.pyilc_path}/pyilc/main.py {tSZ_yaml}"], shell=True, env=env, stdout=stdout, stderr=stdout)
    if inp.verbose:
        print(f'generated HILC weight maps for preserved component tSZ, sim {sim}{scaling_str}, pars={pars}', flush=True)
    
    return tmpdir


def weight_maps_exist(sim, split, inp, tmpdir, pars=None):
    '''
    Checks whether all weight maps for a given simulation and scaling already exist

    ARGUMENTS
    ---------
    sim: int, simulation number
    split: int, split number (1 or 2)
    inp: Info object containing input parameter specifications
    tmpdir: str, temporary directory in which weight maps were placed
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)

    RETURNS
    -------
    Bool, whether or not weight maps already exist
    '''
    
    if pars is not None:
        pars_str = f'_pars{pars[0]:.3f}_{pars[1]:.3f}_'
    else:
        pars_str = ''
    
    for comp in ['CMB', 'tSZ']:
        for freq in range(len(inp.freqs)):
            for scale in range(inp.Nscales):
                if not os.path.exists(f"{tmpdir}/sim{sim}_split{split}{pars_str}weightmap_freq{freq}_scale{scale}_component_{comp}.fits"):
                    return False
    return True


def load_wt_maps(inp, sim, split, tmpdir, pars=None, cib=False):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    split: int, split number (1 or 2)
    tmpdir: str, temporary directory in which pyilc outputs were placed
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)
    cib: Bool, whether to load CIB weight maps

    RETURNS
    --------
    CMB_wt_maps: (Nscales, Nfreqs=2, npix (variable for each scale and freq)) nested list,
                contains NILC weight maps for preserved CMB
    tSZ_wt_maps: (Nscales, Nfreqs=2, npix (variable for each scale and freq)) nested list,
                contains NILC weight maps for preserved tSZ

    '''
    CMB_wt_maps = np.zeros((inp.Nscales, len(inp.freqs), 12*inp.nside**2))
    tSZ_wt_maps = np.zeros((inp.Nscales, len(inp.freqs), 12*inp.nside**2))
    if cib:
       CIB_wt_maps = np.zeros((inp.Nscales, len(inp.freqs), 12*inp.nside**2)) 
       comps = ['CMB', 'tSZ', 'CIB']
    else:
        comps = ['CMB', 'tSZ']
    for comp in comps:
        for scale in range(inp.Nscales):
            for freq in range(2):
                if pars is not None:
                    pars_str = f'_pars{pars[0]:.3f}_{pars[1]:.3f}_'
                else:
                    pars_str = ''
                wt_map_path = f'{tmpdir}/sim{sim}_split{split}{pars_str}weightmap_freq{freq}_scale{scale}_component_{comp}.fits'
                wt_map = hp.read_map(wt_map_path)
                wt_map = hp.ud_grade(wt_map, inp.nside)
                if comp=='CMB':
                    CMB_wt_maps[scale][freq] = wt_map*10**(-6) #since pyilc outputs CMB map in uK
                elif comp=='tSZ':
                    tSZ_wt_maps[scale][freq] = wt_map
                else:
                    CIB_wt_maps[scale][freq] = wt_map
    if cib:
        return CMB_wt_maps, tSZ_wt_maps, CIB_wt_maps
    return CMB_wt_maps, tSZ_wt_maps