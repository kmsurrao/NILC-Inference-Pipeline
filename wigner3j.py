import py3nj
import pickle
import numpy as np

def get_wigner3j_zero_m(inp, save=False):
    if inp.wigner_file:
        wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
        if inp.verbose:
            print('loaded wigner-3j for m1,m2,m3=0', flush=True)
    else:
        num_l1s, num_l2s, num_l3s = inp.ellmax+1, inp.ellmax+1, inp.ellmax+1
        l1_to_compute = np.repeat(np.arange(num_l1s), num_l2s*num_l3s)
        l2_to_compute = np.tile( np.repeat(np.arange(num_l2s), num_l3s), num_l1s)
        l3_to_compute = np.tile(np.arange(num_l3s), num_l1s*num_l2s)
        m_to_compute = np.zeros(num_l1s*num_l2s*num_l3s, dtype='int32')
        wigner = py3nj.wigner3j(2*l1_to_compute, 2*l2_to_compute, 2*l3_to_compute, 2*m_to_compute, 2*m_to_compute, 2*m_to_compute)
        wigner = np.reshape(wigner, (num_l1s, num_l2s, num_l3s)) #index as wigner[l1][l2][l3]
        if save:
            pickle.dump(wigner, open(f'{inp.scratch_path}/wigner3j_zero_m.p', 'wb'), protocol=4)
            if inp.verbose:
                print(f'saved {inp.scratch_path}/wigner3j_zero_m.p', flush=True)
        if inp.verbose:
            print('calculated wigner-3j for m1,m2,m3=0', flush=True)
    return wigner


def get_wigner3j_nonzero_m(inp, save=False):
    '''
    used for wigner-3j symbols of the form 
    l1 l2 l2
    0 -m2 m2
    '''
    if inp.wigner_nonzero_m_file:
        wigner = pickle.load(open(inp.wigner_nonzero_m_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :2*inp.ellmax+1]
        if inp.verbose:
            print('loaded wigner-3j for possibly nonzero m2, m3', flush=True)
    else:
        print('started', flush=True)
        num_l1s = inp.ellmax+1
        num_l2s = inp.ellmax+1
        num_m2s = 2*inp.ellmax+1
        l1_to_compute = np.repeat(np.arange(num_l1s), num_l2s*num_m2s)
        l2_to_compute = np.tile( np.repeat(np.arange(num_l2s), num_m2s), num_l1s)
        m2_to_compute = np.tile(np.arange(-inp.ellmax, inp.ellmax+1), num_l1s*num_l2s)
        m1_to_compute = np.zeros(num_l1s*num_l2s*num_m2s, dtype='int32')
        print('got ells and ms to compute', flush=True)
        wigner = py3nj.wigner3j(2*l1_to_compute, 2*l2_to_compute, 2*l2_to_compute, 2*m1_to_compute, -2*m2_to_compute, 2*m2_to_compute, ignore_invalid=True)
        print('computed wigner', flush=True)
        wigner = np.array(wigner, dtype='float32')
        wigner = np.reshape(wigner, (num_l1s, num_l2s, num_m2s))
        print('saved as float32', flush=True)
        if save:
            pickle.dump(wigner, open(f'{inp.scratch_path}/wigner3j_nonzero_m.p', 'wb'), protocol=4)
            if inp.verbose:
                print(f'saved {inp.scratch_path}/wigner3j_nonzero_m.p', flush=True)
        if inp.verbose:
            print('calculated wigner-3j for possibly nonzero m2, m3', flush=True)
    return wigner