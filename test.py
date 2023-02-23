import numpy as np 
import h5py as h5
from numba import njit
from matplotlib import pyplot as plt
import scipy as scp
import numpy.fft as fft
import pyfftw.interfaces.numpy_fft as fftw
import pyfftw
# from mpi4py import MPI

@njit
def compute_long_incr(long_inc, u, Nx, Ny, Nz, incrmnts):

	num_incs = incrmnts.shape[0]

	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				for r_indx, r in enumerate(incrmnts):
					long_inc[r_indx, i, j, k] += u[(i + r) % Nx, j, k, 0] - u[i, j, k, 0] 
					long_inc[r_indx, i, j, k] += u[i, (j + r) % Ny, k, 1] - u[i, j, k, 1] 
					long_inc[r_indx, i, j, k] += u[i, j, (k + r) % Nz, 2] - u[i, j, k, 2]

	return long_inc

@njit
def get_u_hat(w_hat, u_hat, kx, ky, kz):

	Nx = kx.shape[0]
	Ny = ky.shape[0]
	Nz = kz.shape[0]
	Nzf = Nz//2 + 1

	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nzf):
				if kx[i] == 0 and ky[j] == 0 and kz[k] == 0:
					continue
				else:
					k_fac = kx[i]**2 + ky[j]**2 + kz[k]**2

					u_hat[i, j, k, 0] = np.complex(0.0, 1.0) * (1.0 / k_fac) * (ky[j] * w_hat[i, j, k, 2] - kz[k] * w_hat[i, j, k, 1])
					u_hat[i, j, k, 1] = np.complex(0.0, 1.0) * (1.0 / k_fac) * (kz[k] * w_hat[i, j, k, 0] - kx[i] * w_hat[i, j, k, 2])
					u_hat[i, j, k, 2] = np.complex(0.0, 1.0) * (1.0 / k_fac) * (kx[i] * w_hat[i, j, k, 1] - ky[j] * w_hat[i, j, k, 0])

	return u_hat



def my_fftn(u, fu, for_trans_func):
    """ FFT of u in three directions . """
    fu[:] = for_trans_func(u, axes = (0, 1, 2))
    
    return fu

def my_ifftn(fu, u, back_trans_func):
    """ Inverse FFT of fu in three
    directions ."""
    u[:] = back_trans_func(fu, axes = (0, 1, 2))
    
    return u

def for_fft(u, u_h):
	for i in range (3):
	    u_h[i] = my_fftn(u[i], u_h[i], fft.rfftn)
	return u_h

def inv_fft(u_h, u):
	for i in range (3):
	    u[i] = my_ifftn(u_h[i], u[i], fft.irfftn)
	return u

Nx = 256
Ny = 256
Nz = 256
Nzf = Nx//2 + 1
X = np.mgrid[:Nx, :Ny, :Nz].astype('float64')*2.0 * np.pi/Nx
U     = pyfftw.empty_aligned((3, Nx, Ny, Nz), dtype = 'float64')
U_inv = pyfftw.empty_aligned((3, Nx, Ny, Nz), dtype = 'float64')
U_hat = pyfftw.empty_aligned((3, Nx, Ny, Nzf), dtype = 'complex128')


# fft_py  = pyfftw.FFTW(U, U_hat)
# ifft_py = pyfftw.FFTW(U_hat, U, direction='FFTW_BACKARD')

U[0] = np.sin(X[0]) * np.cos(X[1]) * np.cos(X[2])
U[1] = -np.cos(X[0]) * np.sin(X[1]) * np.cos(X[2])
U[2] = 0

utest = U.copy()


U_hat = for_fft(U, U_hat)
U_inv = inv_fft(U_hat, U_inv)
# ifft_u = ifft_py()
# U_inv = ifft_u.copy
err_x = np.linalg.norm(utest[0] - U_inv[0])
err_y = np.linalg.norm(utest[1] - U_inv[1])
err_z = np.linalg.norm(utest[2] - U_inv[2])
print(err_x, err_y, err_z)


with h5.File("./Data/Stats/Stats_HDF_Data_TAG[TestAlt-22].h5", 'r') as f:
	print("Reading C Data")
	u_incr_hist = f["LongitudinalVelIncrements_BinCounts"][:, :]
	u_incr_ranges = f["LongitudinalVelIncrements_BinRanges"][:, :]


	for r in range(u_incr_hist.shape[0]):
		
		print("Plotting C Data {}".format(r))

		hist_c, bin_edges = u_incr_hist[r, :], u_incr_ranges[r, :] 
		bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
		bin_wdt     = bin_edges[1] - bin_edges[0]
		pdf = hist_c / (np.sum(hist_c) * bin_wdt)

		plt.figure()
		plt.plot(bin_centres, pdf, label=r"$r = {}$".format(r))
		plt.yscale('log')
		plt.ylabel(r"PDF")
		plt.legend()
		plt.xlabel(r"$\delta_{\parallel}\mathbf{u}$")
		plt.savefig("./Data/Stats/PDF_c_r{}.png".format(r))
		plt.close()

	plt.figure()
	for r in range(u_incr_hist.shape[0]):
		
		print("Plotting C Data Again {}".format(r))

		hist_c, bin_edges = u_incr_hist[r, :], u_incr_ranges[r, :] 
		bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
		bin_wdt     = bin_edges[1] - bin_edges[0]
		pdf = hist_c / (np.sum(hist_c) * bin_wdt)
		std = np.sqrt(np.sum(pdf * bin_centres**2 * bin_wdt))
		bin_wdt /= std
		bin_centres /= std
		pdf *= std
		plt.plot(bin_centres, pdf, label=r"$r = {}$".format(r))
	plt.yscale('log')
	plt.ylabel(r"PDF")
	plt.legend()
	plt.xlabel(r"$\delta_{\parallel}\mathbf{u} / \langle \delta_{\parallel}\mathbf{u} ^2\rangle ^{1/2}$")
	plt.savefig("./Data/Stats/PDF_c_COMBINED.png".format(r))
	plt.close()




Nx = 256
Ny = 256
Nz = 256
Nzf = Nx//2 + 1
Dim = 3
num_t = 22

w_hat = pyfftw.empty_aligned((Nx, Ny, Nzf, 3), dtype='complex128')
w_hat.flat[:] = 0. + 0.*1j
u_hat = pyfftw.empty_aligned((Nx, Ny, Nzf, 3), dtype='complex128')
u_hat.flat[:] = 0. + 0.*1j
w = pyfftw.empty_aligned((Nx, Ny, Nz, 3), dtype='float64')
w.flat[:] = 0.
u = pyfftw.empty_aligned((Nx, Ny, Nz, 3), dtype='float64')
u.flat[:] = 0.


incrmnts = np.asarray([1, 2, 4, 16, Nx//2])
u_incr_data = np.zeros((num_t, len(incrmnts), Nx, Ny, Nz))

num_threads = 1
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = num_threads
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

w_to_wh = [0] * Dim
wh_to_w = [0] * Dim
u_to_uh = [0] * Dim
uh_to_u = [0] * Dim
for i in range(Dim):
	w_to_wh[i] = pyfftw.FFTW(w[:, :, :, i],  w_hat[:, :, :, i], axes = (-3, -2, -1), threads = num_threads)
	wh_to_w[i] = pyfftw.FFTW(w_hat[:, :, :, i],  w[:, :, :, i], axes = (-3, -2, -1), threads = num_threads, direction='FFTW_BACKWARD')
	u_to_uh[i] = pyfftw.FFTW(u[:, :, :, i],  u_hat[:, :, :, i], axes = (-3, -2, -1), threads = num_threads)
	uh_to_u[i] = pyfftw.FFTW(u_hat[:, :, :, i],  u[:, :, :, i], axes = (-3, -2, -1), threads = num_threads, direction='FFTW_BACKWARD')


with h5.File("/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/Dublin_Results/RESULTS_3D/RESULTS_NAVIER_AB4_N[256][256][256]_T[0-20]_[00-20-27]_[ORNU_H256]/HDF_Global_FOURIER.h5", 'r') as f:
	for t, l in enumerate(list(f.keys())):
		print("Reading Tstep: {}".format(t))
		if t == 0:
			kx = f[l]["kx"][:]
			ky = f[l]["ky"][:]
			kz = f[l]["kz"][:]

			w_hat_real = f[l]["W_hat"][:, :, :, :]['real'].tolist()
			w_hat_imag = f[l]["W_hat"][:, :, :, :]['imag'].tolist()

			w_hat[:, :, :, :] = np.asarray(w_hat_real) + np.asarray(w_hat_imag) * np.complex(0.0, 1.0)

			# for i in range(Dim):
			# 	wh_to_w[i].execute()
			# 	# w[:, :, :, i] = my_ifftn(w_hat[:, :, :, i], w[:, :, :, i], fft.irfftn) * (Nx * Ny * Nz)
			
			# for i in range(10):
			# 	for j in range(10):
			# 		for k in range(10):
			# 			print("{},{},{} -w- x: {:1.8f}\ty: {:1.8f}\tz: {:1.8f}".format(i, j, k, w[i, j, k, 0], w[i, j, k, 1], w[i, j, k, 2])) 
			

			# for i in range(10):
			# 	for j in range(10):
			# 		for k in range(10):
			# 			print("{},{},{} -what- x: {:1.8f} {:1.8f}\ty: {:1.8f} {:1.8f}\tz: {:1.8f} {:1.8f}".format(i, j, k, np.real(w_hat_data[i, j, k, 0]), np.imag(w_hat_data[i, j, k, 0]), np.real(w_hat_data[i, j, k, 1]), np.imag(w_hat_data[i, j, k, 1]), np.real(w_hat_data[i, j, k, 2]), np.imag(w_hat_data[i, j, k, 2])))
			# print(type(w_hat_data[i, j, k, 0]))
			
			u_hat[:, :, :, :] = get_u_hat(w_hat[:, :, :, :], u_hat[:, :, :, :], kx, ky, kz)
			
			# for i in range(10):
			# 	for j in range(10):
			# 		for k in range(10):
			# 			print("{},{},{} -uhat- x: {:1.8f} {:1.8f}\ty: {:1.8f} {:1.8f}\tz: {:1.8f} {:1.8f}".format(i, j, k, np.real(u_hat[i, j, k, 0]), np.imag(u_hat[i, j, k, 0]), np.real(u_hat[i, j, k, 1]), np.imag(u_hat[i, j, k, 1]), np.real(u_hat[i, j, k, 2]), np.imag(u_hat[i, j, k, 2])))
			# # Transform back to real space
			for i in range(Dim):
				uh_to_u[i].execute()
				# my_ifftn(u_hat[:, :, :, i], u[:, :, :, i], fftw.irfftn)
			
			# for i in range(10):
			# 	for j in range(10):
			# 		for k in range(10):
			# 			print("{},{},{} -u- x: {:1.8f}\ty: {:1.8f}\tz: {:1.8f}".format(i, j, k, u[i, j, k, 0], u[i, j, k, 1], u[i, j, k, 2]))

			
			# print(u, u.shape)
			
			u_incr_data[t, :, :, :, :] = compute_long_incr(u_incr_data[t, :, :, :, :], u, Nx, Ny, Nz, incrmnts)

			# print(u_incr_data[t, :, :, :, :])

print()
for r in range(len(incrmnts)):
	
	print("Plotting {}".format(r))

	hist_c, bin_edges = np.histogram(u_incr_data[0, r, :, :, :], bins = 1000)
	bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
	bin_wdt     = bin_edges[1] - bin_edges[0]
	pdf = hist_c / (np.sum(hist_c) * bin_wdt)

	plt.figure()
	plt.plot(bin_centres, pdf, label=r"$r = {}$".format(r))
	plt.yscale('log')
	plt.ylabel(r"PDF")
	plt.xlabel(r"$\delta_{\parallel}\mathbf{u}$")
	plt.savefig("./Data/Stats/PDF_py_r{}.png".format(r))
	plt.close()


plt.figure()
for r in range(len(incrmnts)):
	
	print("Plotting {}".format(r))

	hist_c, bin_edges = np.histogram(u_incr_data[0, r, :, :, :], bins = 1000)
	bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
	bin_wdt     = bin_edges[1] - bin_edges[0]
	pdf = hist_c / (np.sum(hist_c) * bin_wdt)
	std = np.sqrt(np.sum(pdf * bin_centres**2 * bin_wdt))
	bin_wdt /= std
	bin_centres /= std
	pdf *= std
	plt.plot(bin_centres, pdf, label=r"$r = {}$".format(r))
plt.yscale('log')
plt.ylabel(r"PDF")
plt.xlabel(r"$\delta_{\parallel}\mathbf{u}$")
plt.savefig("./Data/Stats/PDF_py_COMBINED.png".format(r))
plt.close()

# with h5.File("TestData.h5", "w") as f:
# 	for t in range(num_t):
# 		f.create_dataset("Timestep_{:4d}/W_hat".format(t), data = w_hat_data[t, :, :, :, :])