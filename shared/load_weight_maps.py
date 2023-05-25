import healpy as hp

def load_wt_maps(inp, sim, band_limit=False, scaling=None):
    '''
    ARGUMENTS
    ---------
    inp: Info object containing input parameter specifications
    sim: int, simulation number
    band_limit: Bool, whether or not to remove all power in weight maps above ellmax
    scaling: None or list of [scaling_amplitude, scaled component]

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
                if not scaling:
                    wt_map_path = f'{inp.output_dir}/pyilc_outputs/unscaled/sim{sim}weightmap_freq{freq}_scale{scale}_component_{comp}.fits'
                else:
                    scale_factor, scaled_comp = scaling
                    scaling_type = 'low' if scale_factor < 1.0 else 'high'
                    wt_map_path = f'{inp.output_dir}/pyilc_outputs/scaled_{scaling_type}_{scaled_comp}/sim{sim}weightmap_freq{freq}_scale{scale}_component_{comp}.fits'
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