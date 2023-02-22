import numpy as np 
import h5py as h5
from numba import njit
from matplotlib import pyplot as plt
import scipy as scp

@njit
def compute_long_incr(u_x, u_y, u_z, Nx, Ny, Nz, incrmnts):

	num_incs = incrmnts.shape[0]
	long_inc = np.zeros((num_incs, Nx, Ny, Nz))

	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				for r_indx, r in enumerate(incrmnts):
					long_inc[r_indx, i, j, k] += u_x[(i + r) % Nx, j, k] - u_x[i, j, k] 
					long_inc[r_indx, i, j, k] += u_y[i, (j + r) % Ny, k] - u_y[i, j, k] 
					long_inc[r_indx, i, j, k] += u_z[i, j, (k + r) % Nz] - u_z[i, j, k]

	return long_inc

@njit
def get_u_hat(w_hat, kx, ky, kz):

	Nx = kx.shape[0]
	Ny = ky.shape[0]
	Nz = kz.shape[0]
	Nzf = Nz//2 + 1

	u_hat = np.empty_like(w_hat)

	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nzf):
				if kx[i] == 0 and ky[j] == 0 and kz[k] == 0:
					continue
				else:
					k_fac = kx[i]**2 + ky[j]**2 + kz[k]**2

					u_hat[i, j, k, 0] = 1j * (1.0 / k_fac) * (ky[j] * w_hat[i, j, k, 2] - kz[k] * w_hat[i, j, k, 1])
					u_hat[i, j, k, 1] = 1j * (1.0 / k_fac) * (kz[k] * w_hat[i, j, k, 0] - kx[i] * w_hat[i, j, k, 2])
					u_hat[i, j, k, 2] = 1j * (1.0 / k_fac) * (kx[i] * w_hat[i, j, k, 1] - ky[j] * w_hat[i, j, k, 0])

	return u_hat






with h5.File("./Data/Stats/Stats_HDF_Data_TAG[Test-0].h5", 'r') as f:
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




Nx = 256
Ny = 256
Nz = 256
Nzf = Nx//2 + 1
Dim = 3
num_t = 22
w_hat_data = np.zeros((Nx, Ny, Nzf, 3)) * 1j
# w_hat_data = np.zeros((num_t, Nx, Ny, Nzf, 3)) * 1j
incrmnts = np.asarray([1, 2, 4, 16, Nx//2])
u_incr_data = np.zeros((num_t, len(incrmnts), Nx, Ny, Nz))
with h5.File("/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/Dublin_Results/RESULTS_3D/RESULTS_NAVIER_AB4_N[256][256][256]_T[0-20]_[00-20-27]_[ORNU_H256]/HDF_Global_FOURIER.h5", 'r') as f:
	for t, l in enumerate(list(f.keys())):
		print("Reading Tstep: {}".format(t))
		if t == 0:
			kx = f[l]["kx"][:]
			ky = f[l]["ky"][:]
			kz = f[l]["kz"][:]

			w_hat_real = f[l]["W_hat"][:, :, :, :]['real'].tolist()
			w_hat_imag = f[l]["W_hat"][:, :, :, :]['imag'].tolist()

			w_hat_data[:, :, :, :] = np.asarray(w_hat_real) + np.asarray(w_hat_imag) * 1j

			u_hat = get_u_hat(w_hat_data[:, :, :, :], kx, ky, kz)
			u_x = np.fft.irfft(u_hat[:, :, :, 0], s) 
			u_y = np.fft.irfft(u_hat[:, :, :, 1], s) 
			u_z = np.fft.irfft(u_hat[:, :, :, 2], s) 
			print(u_x -8.12480283e49)
			# u_incr_data[t, :, :, :, :] = compute_long_incr(u_x, u_y, u_z, Nx, Ny, Nz, incrmnts)

			# print(u_incr_data[t, :, :, :, :])

# print()
# for r in range(len(incrmnts)):
	
# 	print("Plotting {}".format(r))

# 	hist_c, bin_edges = np.histogram(u_incr_data[0, r, :, :, :], bins = 100)
# 	bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
# 	bin_wdt     = bin_edges[1] - bin_edges[0]
# 	pdf = hist_c / (np.sum(hist_c) * bin_wdt)

# 	plt.figure()
# 	plt.plot(bin_centres, pdf, label=r"$r = {}$".format(r))
# 	plt.yscale('log')
# 	plt.ylabel(r"PDF")
# 	plt.xlabel(r"$\delta_{\parallel}\mathbf{u}$")
# 	plt.savefig("./Data/Stats/PDF_py_r{}.png".format(r))
# 	plt.close()



# with h5.File("TestData.h5", "w") as f:
# 	for t in range(num_t):
# 		f.create_dataset("Timestep_{:4d}/W_hat".format(t), data = w_hat_data[t, :, :, :, :])