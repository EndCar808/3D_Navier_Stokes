import numpy as np 
import h5py as h5


Nx = 256
Ny = 256
Nzf = 128
Dim = 3
num_t = 22
w_hat_data = np.zeros((num_t, Nx, Ny, Nzf, 3)) * 1j

with h5.File("/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/Dublin_Results/RESULTS_3D/RESULTS_NAVIER_AB4_N[256][256][256]_T[0-20]_[00-20-27]_[ORNU_H256]/HDF_Global_FOURIER.h5", 'r') as f:
	for t, l in enumerate(list(f.keys())):
		w_hat = f[l]["W_hat"][:, :, :, :]
		print(t)
		for i in range(Nx):
			for j in range(Ny):
				for k in range(Nzf):
					for d in range(Dim):
						w_hat_data[t, i, j, k, d] = np.complex(w_hat[i, j, k, d][0], w_hat[i, j, k, d][1])




with h5.File("TestData.h5", "w") as f:
	for t in range(num_t):
		f.create_dataset("Timestep_{:4.0d}/W_hat".format(t), data = w_hat_data[t, :, :, :, :])