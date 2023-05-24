## must be run after template_fitting_pipeline/main.py with save_files=True

import numpy as np
import pickle
import argparse
import sys
sys.path.append('../../shared')
sys.path.append('..')
from input import Info
from utils import tsz_spectral_response

def main():

   # main input file containing most specifications 
   parser = argparse.ArgumentParser(description="Analytic covariance from template-fitting approach.")
   parser.add_argument("--config", default="../stampede.yaml")
   args = parser.parse_args()
   input_file = args.config

   # read in the input file and set up relevant info object
   inp = Info(input_file)
   ells = np.arange(inp.ellmax+1)

   Clij = pickle.load(open(f'{inp.output_dir}/data_vecs/Clij.p', 'rb')) #dim (Nsims, Nfreqs=2, Nfreqs=2, Ncomps=4, ellmax+1)
   Clij_mean = np.mean(Clij, axis=0) #dim (Nfreqs=2, Nfreqs=2, Ncomps=4, ellmax+1)

   g1, g2 = tsz_spectral_response(inp.freqs) #tSZ spectral response at 90 and 150 GHz

   CC = Clij_mean[0,0,0] #CMB
   T = Clij_mean[0,0,1]/g1**2 #tSZ (in Compton-y)
   N1 = Clij_mean[0,0,2] #noise 90 GHz
   N2 = Clij_mean[1,1,3] #noise 150 GHz

   Nmodes = inp.ellmax+1




   #use analytic expressions derived from Mathematica notebook
   #with Gaussian power spectrum covariance matrix

   CMB_var = (2* (CC* (g1 - g2)**2 + g2**2*N1 + g1**2*N2)**2 *(g1**2*N2*T +
      CC* (N1 + N2 + (g1 - g2)**2*T) + N1* (N2 + g2**2*T)))/(CC**2 *(g1 -
      g2)**2 *Nmodes *(2*g2**2*N1**2 + g1**2*N1*N2 + 2*g1*g2*N1*N2 +
      g2**2*N1*N2 + 2*g1**2*N2**2 + (g1 - g2)**2 *(g2**2*N1 + g1**2*N2)*T +
      CC*(g1 - g2)**2 *(N1 + N2 + (g1 - g2)**2 *T)))
   CMB_tSZ_covar = -((2 *(g2*N1 + g1*N2)**2 *(g1**2*N2*T + CC
      *(N1 + N2 + (g1 - g2)**2*T) + N1 *(N2 + g2**2*T)))/
      (CC *(g1 - g2)**2*Nmodes*T *(2*g2**2*N1**2 + g1**2*N1*N2 + 2*g1*g2*N1*N2 +
      g2**2*N1*N2 + 2*g1**2*N2**2 + (g1 - g2)**2 *(g2**2*N1 + g1**2*N2)*T +
      CC*(g1 - g2)**2 *(N1 + N2 + (g1 - g2)**2*T))))
   tSZ_var = (2 *(N1 + N2 + (g1 - g2)**2*T)**2 *(g1**2*N2*T +
   CC *(N1 + N2 + (g1 - g2)**2*T) + N1 *(N2 + g2**2*T)))/((g1 -
   g2)**2 *Nmodes *T**2 *(2*g2**2*N1**2 + g1**2*N1*N2 + 2*g1*g2*N1*N2 +
   g2**2*N1*N2 + 2*g1**2*N2**2 + (g1 - g2)**2 *(g2**2*N1 + g1**2*N2)*T +
   CC*(g1 - g2)**2 *(N1 + N2 + (g1 - g2)**2 *T)))

   full_covar = np.array([CMB_var, CMB_tSZ_covar, tSZ_var])
   if inp.save_files:
      pickle.dump(full_covar, open(f'{inp.output_dir}/template_fiting_analytic_covar_gaussian.p', 'wb'))
      if inp.verbose:
         print(f'saved {inp.output_dir}/template_fiting_analytic_covar_gaussian.p')





   #use analytic expressions derived from Mathematica notebook but with power spectrum
   #covariance matrix measured directly from sims

   '''
   First, construct the covariance matrix of the auto- and cross-frequency power spectra.
   Explicitly include 150x90 terms in power spectrum covariance matrix, i.e., it is a 4x4 matrix.
   '''
   Clij_tmp = np.sum(Clij, axis=3)
   Clij_tmp = np.array([Clij_tmp[:,0,0], Clij_tmp[:,0,1], Clij_tmp[:,1,0], Clij_tmp[:,1,1]])
   Clij_tmp = np.transpose(Clij_tmp, axes=(2,0,1)) #shape (ellmax+1, 4 for Cl00 Cl01 Cl11 and Cl11, Nsims)
   PScovFull = np.array([np.cov(Clij_tmp[l]) for l in range(inp.ellmax+1)]) #shape (ellmax+1,4,4)

   
   '''
   We are really representing a four-index tensor, each of whose indices can take 2 values, as a
   4x4 matrix -- both objects contain 16 elements. To make the summation in Eq. 87 straightforward, let's
   just construct the four-index tensor explicitly.
   '''
   PScovFullTensor = np.zeros((inp.ellmax+1, 2,2,2,2))
   PScovFullTensor[:, 0, 0, 0, 0] = PScovFull[:, 0, 0]
   PScovFullTensor[:, 0, 0, 0, 1] = PScovFull[:, 0, 1]
   PScovFullTensor[:, 0, 0, 1, 0] = PScovFull[:, 0, 2]
   PScovFullTensor[:, 0, 0, 1, 1] = PScovFull[:, 0, 3]
   PScovFullTensor[:, 0, 1, 0, 0] = PScovFull[:, 1, 0]
   PScovFullTensor[:, 0, 1, 0, 1] = PScovFull[:, 1, 1]
   PScovFullTensor[:, 0, 1, 1, 0] = PScovFull[:, 1, 2]
   PScovFullTensor[:, 0, 1, 1, 1] = PScovFull[:, 1, 3]
   PScovFullTensor[:, 1, 0, 0, 0] = PScovFull[:, 2, 0]
   PScovFullTensor[:, 1, 0, 0, 1] = PScovFull[:, 2, 1]
   PScovFullTensor[:, 1, 0, 1, 0] = PScovFull[:, 2, 2]
   PScovFullTensor[:, 1, 0, 1, 1] = PScovFull[:, 2, 3]
   PScovFullTensor[:, 1, 1, 0, 0] = PScovFull[:, 3, 0]
   PScovFullTensor[:, 1, 1, 0, 1] = PScovFull[:, 3, 1]
   PScovFullTensor[:, 1, 1, 1, 0] = PScovFull[:, 3, 2]
   PScovFullTensor[:, 1, 1, 1, 1] = PScovFull[:, 3, 3]

   '''
   Begin multifrequency template fitting approach. Find inverse of power spectrum covariance matrix Clij^-1.
   '''
   PScov = [[[PScovFull[l, 0, 0], PScovFull[l, 0, 1], PScovFull[l, 0, 3]],
             [PScovFull[l, 1, 0], PScovFull[l, 1, 1], PScovFull[l, 1, 3]],
             [PScovFull[l, 3, 0], PScovFull[l, 3, 1], PScovFull[l, 3, 3]]] 
            for l in range(inp.ellmax+1)]
   PScovInv = np.array([np.linalg.inv(PScov[l]) for l in range(inp.ellmax+1)])

   '''
   Find K_ {alpha,beta} from eq.65 by summing over each term of PScovInv once.
   '''
   KCMBCMB = np.array([CC[l]**2*(\
      PScovInv[l, 0, 0] + PScovInv[l, 0, 1] + PScovInv[l, 0, 2] + \
      PScovInv[l, 1, 0] + PScovInv[l, 1, 1] + PScovInv[l, 1, 2] + \
      PScovInv[l, 2, 0] + PScovInv[l, 2, 1] + PScovInv[l, 2, 2]) \
      for l in range(inp.ellmax+1)])
   KCMBtSZ = np.array([CC[l]*T[l]*(\
      PScovInv[l, 0, 0]*g1**2 + PScovInv[l, 0, 1]*g1*g2 + PScovInv[l, 0, 2]*g2**2 + \
      PScovInv[l, 1, 0]*g1**2 + PScovInv[l, 1, 1]*g1*g2 + PScovInv[l, 1, 2]*g2**2 + \
      PScovInv[l, 2, 0]*g1**2 + PScovInv[l, 2, 1]*g1*g2 + PScovInv[l, 2, 2]*g2**2) \
      for l in range(inp.ellmax+1)])
   KtSZtSZ = np.array([T[l]**2*(
      g1**2*PScovInv[l, 0, 0]*g1**2 + g1**2*PScovInv[l, 0, 1]*g1*g2 + g1**2*PScovInv[l, 0, 2]*g2**2 + \
      g1*g2*PScovInv[l, 1, 0]*g1**2 + g1*g2*PScovInv[l, 1, 1]*g1*g2 + g1*g2*PScovInv[l, 1, 2]*g2**2 + \
      g2**2*PScovInv[l, 2, 0]*g1**2 + g2**2*PScovInv[l, 2, 1]*g1*g2 + g2**2*PScovInv[l, 2, 2]*g2**2) \
      for l in range(inp.ellmax+1)])
   K = np.array([[[KCMBCMB[l], KCMBtSZ[l]], [KCMBtSZ[l], KtSZtSZ[l]]] for l in range(inp.ellmax+1)])

   '''
   Find final covariance matrix from template fitting approach using eq.69
   '''
   FinalCovTemplateFitting = np.array([np.linalg.inv(K[l]) for l in range(inp.ellmax+1)])

   full_covar = np.array([FinalCovTemplateFitting[:,0,0], FinalCovTemplateFitting[:,0,1], FinalCovTemplateFitting[:,1,1]])
   if inp.save_files:
      pickle.dump(full_covar, open(f'{inp.output_dir}/template_fiting_analytic_covar_sims.p', 'wb'))
      if inp.verbose:
         print(f'saved {inp.output_dir}/template_fiting_analytic_covar_sims.p')



if __name__=='__main__':
    main()