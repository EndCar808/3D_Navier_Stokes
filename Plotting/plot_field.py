import numpy as np 
import h5py as h5
import sys
import getopt
import matplotlib as mpl
if mpl.__version__ > '2':    
       mpl.rcParams['text.usetex'] = True
       mpl.rcParams['font.family'] = 'serif'
       mpl.rcParams['font.serif']  = 'Computer Modern Roman'
from matplotlib import pyplot as plt


if __name__ == "__main__":

       # Construct input file path
       data_dir = "/home/ecarroll/PhD/3D_Navier_Stokes/Data/Stats/RESULTS_NAVIER_AB4_N[512][512][512]_T[260-660]_[10-51-12]_[ORNU_N512_contd-Stats]"
       input_file_path = data_dir + "/HDF_Global_FOURIER.h5"
       out_dir = data_dir

       N = 512

       with h5.File(input_file_path, 'r') as f:
              t = "0100"
              w_hat = f["Timestep_{}".format(t)]["W_hat"][:, :, :, :]

              w_x = np.fft.irfftn(w_hat[:, :, :, 0]) * (512**3)
              w_y = np.fft.irfftn(w_hat[:, :, :, 1]) * (512**3)
              w_z = np.fft.irfftn(w_hat[:, :, :, 2]) * (512**3)
       # Compute Vorticity Intensity
       slice_indx = 100

       w_amp = np.sqrt(w_x[slice_indx, :, :]**2 + w_y[slice_indx, :, :]**2 + w_z[slice_indx, :, :]**2)

       plt.figure()
       plt.imshow(w_amp)
       plt.colorbar()
       plt.savefig(out_dir + "/Vort_Amp.png")
       plt.close()

       dy = 2.0 * np.pi / N
       y = np.arange(0.0, 2.0 * np.pi, dy)
       dz = 2.0 * np.pi / N
       z = np.arange(0.0, 2.0 * np.pi, dz)

       Y, Z = np.meshgrid(y, z)


       fig = plt.figure(figsize =(14, 9))
       ax = plt.axes(projection ='3d')
       ax.plot_surface(Y, Z, w_amp)
       ax.set_axis_off()
       plt.savefig(out_dir + "/Vort_Amp_Surface.png")
       plt.close()