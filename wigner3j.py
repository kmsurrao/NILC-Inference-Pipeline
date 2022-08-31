import py3nj
import pickle
import numpy as np
import sys
from input import Info
import multiprocessing as mp

def get_wigner3j_zero_m_vectorized(inp, save=False):
    '''
    used for wigner-3j symbols of the form 
    l1 l2 l3
    0  0  0

    right now this verison requires too much memory for ellmax=1000
    '''
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


def get_wigner3j_nonzero_m_vectorized(inp, save=False):
    '''
    used for wigner-3j symbols of the form 
    l1 l2 l2
    0 -m2 m2

    right now this verison requires too much memory for ellmax=1000
    '''
    if inp.wigner_nonzero_m_file:
        wigner = pickle.load(open(inp.wigner_nonzero_m_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :2*inp.ellmax+1]
        if inp.verbose:
            print('loaded wigner-3j for possibly nonzero m2, m3', flush=True)
    else:
        num_l1s = inp.ellmax+1
        num_l2s = inp.ellmax+1
        num_m2s = 2*inp.ellmax+1
        l1_to_compute = np.repeat(np.arange(num_l1s), num_l2s*num_m2s)
        l2_to_compute = np.tile( np.repeat(np.arange(num_l2s), num_m2s), num_l1s)
        m2_to_compute = np.tile(np.arange(-inp.ellmax, inp.ellmax+1), num_l1s*num_l2s)
        m1_to_compute = np.zeros(num_l1s*num_l2s*num_m2s, dtype='int32')
        wigner = py3nj.wigner3j(2*l1_to_compute, 2*l2_to_compute, 2*l2_to_compute, 2*m1_to_compute, -2*m2_to_compute, 2*m2_to_compute, ignore_invalid=True)
        wigner = np.array(wigner, dtype='float32')
        wigner = np.reshape(wigner, (num_l1s, num_l2s, num_m2s))
        if save:
            pickle.dump(wigner, open(f'{inp.scratch_path}/wigner3j_nonzero_m.p', 'wb'), protocol=4)
            if inp.verbose:
                print(f'saved {inp.scratch_path}/wigner3j_nonzero_m.p', flush=True)
        if inp.verbose:
            print('calculated wigner-3j for possibly nonzero m2, m3', flush=True)
    return wigner

def get_wigner3j_zero_m_one_ell(l1, inp, save=False):
    '''
    used for wigner-3j symbols of the form 
    l1 l2 l3
    0  0  0
    '''
    wigner = np.zeros((inp.ellmax+1, inp.ellmax+1), dtype='float32')
    for l2 in range(inp.ellmax+1):
        l1_to_compute = np.array([l1]*(inp.ellmax+1))
        l2_to_compute = np.array([l2]*(inp.ellmax+1))
        l3_to_compute = np.arange(0,inp.ellmax+1)
        m_to_compute = np.zeros(inp.ellmax+1, dtype='int32')
        symbols = py3nj.wigner3j(2*l1_to_compute, 2*l2_to_compute, 2*l3_to_compute, 2*m_to_compute, 2*m_to_compute, 2*m_to_compute)
        symbols = symbols.astype('float32')
        wigner[l2] = symbols
    if save:
        pickle.dump(wigner, open(f'{inp.scratch_path}/wigner3j_zero_m/{l1}.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.scratch_path}/wigner3j_zero_m/{l1}.p', flush=True)
    return wigner


def get_wigner3j_nonzero_m_one_ell(l1, inp, save=False):
    '''
    used for wigner-3j symbols of the form 
    l1 l2 l2
    0 -m2 m2
    '''
    wigner = np.zeros((inp.ellmax+1, 2*inp.ellmax+1), dtype='float32')
    for l2 in range(inp.ellmax+1):
        l1_to_compute = np.array([l1]*(2*l2+1))
        l2_to_compute = np.array([l2]*(2*l2+1))
        m2_to_compute = np.arange(-l2,l2+1)
        if l2==0:
            m2_to_compute = np.array([0])
        symbols = py3nj.wigner3j(2*l1_to_compute, 2*l2_to_compute, 2*l2_to_compute, np.zeros(2*l2+1, dtype='int32'), -2*m2_to_compute, 2*m2_to_compute)
        symbols = symbols.astype('float32')
        zero_idx = inp.ellmax
        wigner[l2][zero_idx-l2:zero_idx+l2+1] = symbols
    if save:
        pickle.dump(wigner, open(f'{inp.scratch_path}/wigner3j_nonzero_m/{l1}.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.scratch_path}/wigner3j_nonzero_m/{l1}.p', flush=True)
    return wigner

def get_wigner3j_zero_m(inp, save=False):
    '''
    used for wigner-3j symbols of the form 
    l1 l2 l3
    0  0  0
    '''
    if inp.wigner_file:
        wigner = pickle.load(open(inp.wigner_file, 'rb'))[:inp.ellmax+1, :inp.ellmax+1, :inp.ellmax+1]
        if inp.verbose:
            print('loaded wigner-3j for m1,m2,m3=0', flush=True)
    else:
        pool = mp.Pool(32)
        pool.starmap(get_wigner3j_zero_m_one_ell, [(l1, inp, True) for l1 in range(0,inp.ellmax+1)])
        pool.close()
            all_wigner = pickle.load(open(f'{inp.scratch_path}/wigner3j_zero_m/0.p', 'rb'))
        for l1 in range(1,inp.ellmax+1):
            new = pickle.load(open(f'{inp.scratch_path}/wigner3j_zero_m/{l1}.p', 'rb'))
            all_wigner = np.concatenate((all_wigner, new), axis=0)
        pickle.dump(all_wigner, open(f'{inp.scratch_path}/wigner3j_zero_m/all_wigner3j_zero_m.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.scratch_path}/wigner3j_zero_m/all_wigner3j_zero_m.p')
    return wigner

def get_wigner3j_nonzero_m(l1, inp, save=False):
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
        pool = mp.Pool(32)
        pool.starmap(get_wigner3j_nonzero_m_one_ell, [(l1, inp, True) for l1 in range(0,inp.ellmax+1)])
        pool.close()
            all_wigner = pickle.load(open(f'{inp.scratch_path}/wigner3j_nonzero_m/0.p', 'rb'))
        for l1 in range(1,inp.ellmax+1):
            new = pickle.load(open(f'{inp.scratch_path}/wigner3j_nonzero_m/{l1}.p', 'rb'))
            all_wigner = np.concatenate((all_wigner, new), axis=0)
        pickle.dump(all_wigner, open(f'{inp.scratch_path}/wigner3j_nonzero_m/all_wigner3j_nonzero_m.p', 'wb'), protocol=4)
        if inp.verbose:
            print(f'saved {inp.scratch_path}/wigner3j_nonzero_m/all_wigner3j_nonzero_m.p')
    return wigner



if __name__ == '__main__':
    # main input file containing most specifications 
    try:
        input_file = (sys.argv)[1]
    except IndexError:
        input_file = 'example.yaml'

    # read in the input file and set up relevant info object
    inp = Info(input_file)

    # # get wigner-3j symbols for nonzero m
    # pool = mp.Pool(32)
    # pool.starmap(get_wigner3j_nonzero_m, [(l1, inp, True) for l1 in range(841,1001)])
    # pool.close()

    #combine all wigner-3j symbols for nonzero m
    all_wigner = pickle.load(open(f'{inp.scratch_path}/wigner3j_nonzero_m/0.p', 'rb'))
    for l1 in range(1,inp.ellmax+1):
        if l1%50==0:
            print('l1: ', l1, flush=True)
        new = pickle.load(open(f'{inp.scratch_path}/wigner3j_nonzero_m/{l1}.p', 'rb'))
        all_wigner = np.concatenate((all_wigner, new), axis=0)
    pickle.dump(all_wigner, open(f'{inp.scratch_path}/wigner3j_nonzero_m/all_wigner3j_nonzero_m.p', 'wb'), protocol=4)
    if inp.verbose:
        print(f'saved {inp.scratch_path}/wigner3j_nonzero_m/all_wigner3j_nonzero_m.p')



