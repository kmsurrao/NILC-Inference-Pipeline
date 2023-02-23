import healpy as hp

def load_wt_maps(inp, sim):
    CMB_wt_maps = [[[],[]] for i in range(inp.Nscales)]
    tSZ_wt_maps = [[[],[]] for i in range(inp.Nscales)]
    for comp in ['CMB', 'tSZ']:
        for scale in range(inp.Nscales):
            for freq in range(2):
                wt_map_path = f'{inp.output_dir}/pyilc_outputs/sim{sim}weightmap_freq{freq}_scale{scale}_component_{comp}.fits'
                wt_map = hp.read_map(wt_map_path)
                if comp=='CMB':
                    CMB_wt_maps[scale][freq] = wt_map*10**(-6) #since pyilc outputs CMB map in uK
                else:
                    tSZ_wt_maps[scale][freq] = wt_map
    return CMB_wt_maps, tSZ_wt_maps