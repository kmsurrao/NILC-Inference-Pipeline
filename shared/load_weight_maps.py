import healpy as hp

def load_wt_maps(inp, sim, split, band_limit=False, scaling=None, pars=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    split: int, split number (1 or 2)
    band_limit: Bool, whether or not to remove all power in weight maps above ellmax
    scaling: None or list of length 5
            idx0: 0 if "scaled" means maps are scaled down, 1 if "scaled" means maps are scaled up
            idx1: 0 for unscaled CMB, 1 for scaled CMB
            idx2: 0 for unscaled ftSZ, 1 for scaled ftSZ
    pars: array of floats [Acmb, Atsz] (if not provided, all assumed to be 1)

    RETURNS
    --------
    CMB_wt_maps: (Nscales, Nfreqs=2, npix (variable for each scale and freq)) nested list,
                contains NILC weight maps for preserved CMB
    tSZ_wt_maps: (Nscales, Nfreqs=2, npix (variable for each scale and freq)) nested list,
                contains NILC weight maps for preserved tSZ

    '''
    CMB_wt_maps = [[[],[]] for i in range(inp.Nscales)]
    tSZ_wt_maps = [[[],[]] for i in range(inp.Nscales)]
    for comp in ['CMB', 'tSZ']:
        for scale in range(inp.Nscales):
            for freq in range(2):
                if pars is not None:
                    pars_str = f'_pars{pars[0]}_{pars[1]}_'
                else:
                    pars_str = ''
                if scaling is None:
                    wt_map_path = f'{inp.output_dir}/pyilc_outputs/sim{sim}_split{split}{pars_str}weightmap_freq{freq}_scale{scale}_component_{comp}.fits'
                else:
                    scaling_str = ''.join(str(e) for e in scaling)
                    wt_map_path = f'{inp.output_dir}/pyilc_outputs/{scaling_str}/sim{sim}_split{split}{pars_str}weightmap_freq{freq}_scale{scale}_component_{comp}.fits'
                wt_map = hp.read_map(wt_map_path)
                wt_map = hp.ud_grade(wt_map, inp.nside)
                if band_limit:
                    l_arr, m_arr = hp.Alm.getlm(3*inp.nside-1)
                    wlm = hp.map2alm(wt_map)
                    wlm = wlm*(l_arr<=inp.ellmax)
                    wt_map = hp.alm2map(wlm, nside=inp.nside)
                if comp=='CMB':
                    CMB_wt_maps[scale][freq] = wt_map*10**(-6) #since pyilc outputs CMB map in uK
                else:
                    tSZ_wt_maps[scale][freq] = wt_map
    return CMB_wt_maps, tSZ_wt_maps