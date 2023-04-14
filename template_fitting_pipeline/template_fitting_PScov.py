import numpy as np
import healpy as hp
import scipy
from scipy import linalg
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from classy import Class


Nsims = 1000
bw = 1
amplification = 1000
Td_array = pickle.load(open( 'tSZ_1000Nsims_ellmax3002.p' , "rb"))*amplification*10**12
CCd_array = pickle.load(open( 'CMB_1000Nsims_ellmax3002.p' , "rb"))
Nd_array = pickle.load(open( 'Noise_1000Nsims_ellmax3002.p' , "rb"))


def bin(array, bw):
   '''
   PARAMETERS
   array: list of data to bin
   bw: int, bin width

   RETURNS
   binned array of length len(array)/bw
   '''
   if bw==1:
      return np.array(array)
   binned = []
   for i in range(len(array)//bw):
      tot = 0.
      for j in range(bw):
         tot += array[i*bw+j]
      binned.append(tot/bw)
   return np.array(binned)


T_cmb = 2.726
h = 6.62607004*10**(-34)
kb = 1.38064852*10**(-23)

ell_min = 2
orig_ell_max = 3002
orig_ells = np.arange(orig_ell_max)
f = 1. #fsky
def tsz_spectral_response(freq): #input frequency in GHz
    x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
    return T_cmb*(x*1/np.tanh(x/2)-4) #was factor of tcmb microkelvin before
g90, g150 = tsz_spectral_response(90), tsz_spectral_response(150)
g1, g2 = g90, g150


#tSZ theoretical power spectrum
Td_avg = np.mean(Td_array, axis=0)
T = Td_avg

#get theoretical CMB power spectrum from Class
LambdaCDM = Class()
LambdaCDM.set({'omega_b':0.0223828,'omega_cdm':0.1201075,'h':0.67810,'A_s':2.100549e-09,'n_s':0.9660499,'tau_reio':0.05430842, 'l_max_scalars':10000})
LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0})
LambdaCDM.compute()
cls = LambdaCDM.lensed_cl(orig_ell_max-1)
CC = cls['tt']*(T_cmb*10**6)**2 #Class gives CC in K, represents temp as DeltaT/T
CC = np.array(CC)

#theoretical noise spectrum
theta_fwhm = (1.4/60.)*(np.pi/180.)
sigma = theta_fwhm/np.sqrt(8.*np.log(2.))
W = (1/60.)*(np.pi/180.)
noise_cl = []
for ell in orig_ells:
    noise_cl.append(W**2*np.exp(ell*(ell+1)*sigma**2))
N1 = np.array(noise_cl) #theoretical noise 1
N2 = np.array(noise_cl) #theoretical noise 2

all_Clijd = np.zeros((Nsims, (orig_ell_max-ell_min)//bw, 3))

#bin the theoretical maps
T = bin(T[ell_min:], bw) #binned theoretical y map
CC = bin(CC[ell_min:], bw) #binned theoretical CMB map
N1 = bin(N1[ell_min:], bw) #binned theoretical noise 1
N2 = bin(N2[ell_min:], bw) #binned theoretical noise 2

#define new ell_max and array of ells
ell_max = (orig_ell_max-ell_min)//bw
ells = np.arange(ell_max)

'''
Define matrix of auto- and cross-frequency power spectra at some multipole ell.
Here, CC = C_ell,CMB , N1 = N_ell,{freq1} , N2 = N_ell,{freq2} ,
g1 = tSZ spectral function at freq1, g2 = tSZ spectral function at freq2,
T = C_ell**yy (tSZ power spectrum in frequency-independent Compton-y units).
'''
Clij = np.array([[[CC[l] + N1[l] + g1**2*T[l], CC[l] + g1*g2*T[l]], [CC[l] + g1*g2*T[l], CC[l] + N2[l] + g2**2*T[l]]] for l in range(ell_max)])
ClijInv = np.array([linalg.inv(Clij[l]) for l in range(ell_max)])


'''
First, construct the covariance matrix of the auto- and cross-frequency power spectra.
Explicitly include 150x90 terms in power spectrum covariance matrix, i.e., it is a 4x4 matrix.
'''
PScovFull = np.array([(1/(f*(2*l+1)*bw))*np.array([[2*Clij[l][0, 0]**2,
2*(CC[l] + g1**2*T[l])*Clij[l][0, 1] + 2*N1[l]*Clij[l][0, 1],
2*(CC[l] + g1**2*T[l])*Clij[l][0, 1] + 2*N1[l]*Clij[l][0, 1],
2*Clij[l][0, 1]**2], [2*(CC[l] + g1**2*T[l])*Clij[l][0, 1] +
 2*N1[l]*Clij[l][0, 1], Clij[l][0, 0]*Clij[l][1, 1] + Clij[l][0, 1]**2,
Clij[l][0, 0]*Clij[l][1, 1] + Clij[l][0, 1]**2,
2*(CC[l] + g2**2*T[l])*Clij[l][0, 1] +
 2*N2[l]*Clij[l][0, 1]], [2*(CC[l] + g1**2*T[l])*Clij[l][0, 1] +
 2*N1[l]*Clij[l][0, 1], Clij[l][0, 0]*Clij[l][1, 1] + Clij[l][0, 1]**2,
Clij[l][0, 0]*Clij[l][1, 1] + Clij[l][0, 1]**2,
2*(CC[l] + g2**2*T[l])*Clij[l][0, 1] + 2*N2[l]*Clij[l][0, 1]], [2*
 Clij[l][0, 1]**2, 2*(CC[l] + g2**2*T[l])*Clij[l][0, 1] + 2*N2[l]*Clij[l][0, 1],
2*(CC[l] + g2**2*T[l])*Clij[l][0, 1] + 2*N2[l]*Clij[l][0, 1],
2*Clij[l][1, 1]**2]]) for l in range(ell_max)])
#check eigenvalues of this matrix, if one very small, close to singular


'''
Begin multifrequency template fitting approach. Find inverse of power spectrum covariance matrix Clij^-1.
'''
PScov = np.array([[[PScovFull[l][0, 0], PScovFull[l][0, 1], PScovFull[l][0, 3]],
[PScovFull[l][1, 0], PScovFull[l][1, 1], PScovFull[l][1, 3]],
[PScovFull[l][3, 0], PScovFull[l][3, 1], PScovFull[l][3, 3]]] for l in range(ell_max)])

PScovInv = np.array([linalg.inv(PScov[l]) for l in range(ell_max)]) #check inverse dotted with self for sims

print('OLD-------------------------------------------------------------------------------------------')
print('PScov: ', PScov)


for i in range(Nsims):

   #simulated maps (tSZ, CMB, and noise)
   Td = Td_array[i] #y map from data
   CCd = CCd_array[i]
   N1d = Nd_array[i]
   N2d = Nd_array[i]

   #bin the maps from data
   Td = bin(Td[ell_min:], bw) #binned y-map spectrum from data
   CCd = bin(CCd[ell_min:], bw) #binned CC from data
   N1d = bin(N1d[ell_min:], bw) #binned noise 1 from data
   N2d = bin(N2d[ell_min:], bw) #binned noise 2 from data

   #Define auto and cross spectra from data
   Clijd = np.array([[[CCd[l] + N1d[l] + g1**2*Td[l], CCd[l] + g1*g2*Td[l]], [CCd[l] + g1*g2*Td[l], CCd[l] + N2d[l] + g2**2*Td[l]]] for l in range(ell_max)])

   #Add Clijd values to large Clijd array for this sim
   Clijd_vec= np.array([[Clijd[l][0][0], Clijd[l][0][1], Clijd[l][1][1]] for l in range(ell_max)])
   all_Clijd[i] = Clijd_vec

'''
Calculate frequency-frequency covariance matrix directly from sims
'''
print('NEW-------------------------------------------------------------------------------------------')

#original all_Clijd dimensions are Nsims, orig_ell-ell_min, 3
all_Clijd = all_Clijd.transpose((1, 2, 0)) #new dimensions are orig_ell-ell_min, 3, Nsims

# #manually calculated cov
# PScov = np.array([1/(Nsims-1)*(all_Clijd[l]-np.mean(all_Clijd[l], axis=1)[:,None]) @ (all_Clijd[l]-np.mean(all_Clijd[l], axis=1)[:,None]).T for l in range(ell_max)])

PScov = np.array([np.cov(all_Clijd[l]) for l in range(ell_max)]) #ells x 3 x 3

print('PScov from sims: ', PScov)
inv = np.array([linalg.inv(PScov[l]) for l in range(ell_max)])
id = np.array([PScov[l] @ inv[l] for l in range(ell_max)])
where_to_save = f'/moto/hill/users/kms2320/PScov_{Nsims}Nsims_amplification{amplification}.p'
pickle.dump(PScov, open(where_to_save, "wb"))
print('saved ' + where_to_save)
