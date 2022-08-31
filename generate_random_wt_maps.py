import sys
import numpy as np
import healpy as hp
from input import Info
import warnings
hp.disable_warnings()



def generate_random_weight_maps(sim, inp, g, comp, check_summation_constraint=True):
    '''
    uses divide by total method
    '''
    Nbands_NILC = inp.Nscales
    Nfreqs = len(inp.freqs)
    NILC_weights_Nside = inp.nside
    NILC_weights_Npix = 12*NILC_weights_Nside**2
    NILC_weights = np.zeros((Nbands_NILC,Nfreqs,NILC_weights_Npix))
    NILC_weights = np.random.rand(Nbands_NILC, Nfreqs, NILC_weights_Npix)
    tot = np.einsum('nip,i->np', NILC_weights, g)
    inv_tot = 1/tot
    NILC_weights = np.einsum('nip,np->nip', NILC_weights, inv_tot)
    if check_summation_constraint:
        sums = np.einsum('nip,i->np',NILC_weights,g)
        max_diff = np.amax(np.absolute(sums-1.))
        print('maximum diff: ', max_diff)
        print(sums)
        print(NILC_weights)
    for n in range(Nbands_NILC):
        for i in range(Nfreqs):
            hp.write_map(inp.scratch_path + f'/wt_maps/{comp}/{sim}_weightmap_freq{i}_scale{n}_component_{comp}.fits', NILC_weights[n][i], overwrite=True)
            if inp.verbose:
                print('saved ' + inp.scratch_path + f'/wt_maps/{comp}/{sim}_weightmap_freq{i}_scale{n}_component_{comp}.fits', flush=True)
    return NILC_weights


def generate_random_weight_maps_dirichlet(sim, inp, g, comp, check_summation_constraint=True):
    '''
    uses Dirichlet distribution method
    '''
    Nbands_NILC = inp.Nscales
    Nfreqs = len(inp.freqs)
    NILC_weights_Nside = inp.nside
    NILC_weights_Npix = 12*NILC_weights_Nside**2
    NILC_weights = np.zeros((Nbands_NILC,Nfreqs,NILC_weights_Npix))
    for n in range(Nbands_NILC):
        for p in range(NILC_weights_Npix):
            weights = np.random.dirichlet(np.array([1,1,1,1,1,1,1,1,1])/1,size=1)
            # weights = np.random.dirichlet(np.array([1,2,3,4,5,6,7,8,9]),size=1)
            for i in range(Nfreqs):
                NILC_weights[n][i][p]=weights[0][i]/g[i]
    if check_summation_constraint:
        sums = np.einsum('nip,i->np',NILC_weights,g)
        max_diff = np.amax(np.absolute(sums-1.))
        print('maximum diff: ', max_diff)
    for n in range(Nbands_NILC):
        for i in range(Nfreqs):
            hp.write_map(inp.scratch_path + f'/wt_maps/{comp}/{sim}_weightmap_freq{i}_scale{n}_component_{comp}.fits', NILC_weights[n][i], overwrite=True)
            if inp.verbose:
                print('saved ' + inp.scratch_path + f'/wt_maps/{comp}/{sim}_weightmap_freq{i}_scale{n}_component_{comp}.fits', flush=True)
    return NILC_weights


# main input file containing most specifications
try:
    input_file = (sys.argv)[1]
except IndexError:
    input_file = 'example.yaml'

# read in the input file and set up relevant info object
inp = Info(input_file)

#set sim number to 101 (to not conflict with runs from main.py)
#use sim 103 to use random weight maps. comment out pyilc line
sim = 103

#write files with random weight maps to scratch_path
a = np.ones(len(inp.freqs))
comp = 'CMB'
generate_random_weight_maps(sim, inp, a, comp, check_summation_constraint=True)