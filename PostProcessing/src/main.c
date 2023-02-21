/**
* @file main.c 
* @author Enda Carroll
* @date Sept 2021
* @brief Main file for post processing solver data from the 2D Navier stokes psuedospectral solver
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_func.h"
#include "post.h"
// #include "utils.h"
// #include "str_func.h"
// ---------------------------------------------------------------------
//  Global Variables Declarations
// ---------------------------------------------------------------------
// Define the global points that will be pointed to the global structs
system_vars_struct*       sys_vars;
runtime_data_struct*      run_data;
HDF_file_info_struct*    file_info;
stats_data_struct*      stats_data;
// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// --------------------------------
	//  Create Global Stucts
	// --------------------------------
	// Create instances of global variables structs
	runtime_data_struct runtime_data;
	system_vars_struct   system_vars;
	HDF_file_info_struct   HDF_file_info;
	stats_data_struct statistics_data;

	// Point the global pointers to these structs
	run_data   = &runtime_data;
	sys_vars   = &system_vars;
	file_info  = &HDF_file_info;
	stats_data = &statistics_data;

	printf("\n\ngcc version: %d.%d.%d\n\n", __GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__);

	
	OpenInputAndInitialize();


	// Initialize variables
	int Nx = sys_vars->N[0];
	int Ny = sys_vars->N[1];
	int Nz = sys_vars->N[2];
	int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	sys_vars->Max_Incr = Nx / 2;
	int tmp1, tmp2, indx;

	double Lx = 2.0 * M_PI;
	double Ly = 2.0 * M_PI;
	double Lz = 2.0 * M_PI;
	sys_vars->dx = Lx / Nx;
	sys_vars->dy = Ly / Ny;
	sys_vars->dz = Lz / Nz;

	AllocateMemory(sys_vars->N);

	InitializeFFTWPlans(sys_vars->N);
	
	printf("Nx: %d Lx: %lf dx: %lf\nNy: %d Ly: %lf dy: %lf\nNz: %d Lz: %lf dz: %lf\n\n", Nx, Lx, sys_vars->dx, Ny, Ly, sys_vars->dy, Nz, Lz, sys_vars->dz);

	// for (int i = 0; i < sys_vars->N[0]; ++i) {
	// 	if (i < Nz_Fourier) {
	// 		printf("kx[%d]: %d \t ky[%d]: %d \t kz[%d]: %d\n", i, run_data->k[0][i], i, run_data->k[1][i], i, run_data->k[2][i]);
	// 	}
	// 	else {
	// 		printf("kx[%d]: %d \t ky[%d]: %d \n", i, run_data->k[0][i], i, run_data->k[1][i]);
	// 	}
	// }
	
	fftw_complex tmp = 0.0 + 0.0 * I;
	double tmp_r = 0.0;

	for (int snap = 0; snap < sys_vars->num_snaps; ++snap) { 
		
		// Read In Data
		ReadInData(snap);

		for (int i = 0; i < Nx; ++i) {
			tmp1 = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				tmp2 = Nz_Fourier * (tmp1 + j);
				for (int k = 0; k < Nz_Fourier; ++k) {
					indx = tmp2 + k;

					tmp += run_data->u_hat_tmp[SYS_DIM * indx + 0];
					tmp += run_data->u_hat_tmp[SYS_DIM * indx + 1];
					tmp += run_data->u_hat_tmp[SYS_DIM * indx + 2];
					tmp_r += run_data->u[SYS_DIM * indx + 0];
					tmp_r += run_data->u[SYS_DIM * indx + 1];
					tmp_r += run_data->u[SYS_DIM * indx + 2];
				}
			}
		}
		printf("Snap: %d\t\twx: %1.16lf %1.16lf i\t\t %1.16lf\n", snap, creal(tmp), cimag(tmp), tmp_r);
	}


	return 0;
}
/**
 * Wrapper function for initializing FFTW plans
 * @param N Array containing the size of the dimensions of the system
 */
void InitializeFFTWPlans(const long int* N) {

	// Initialize variables
	const int N_batch[SYS_DIM] = {N[0], N[1], N[2]};

	// Initialize Batch Fourier Transforms
	sys_vars->fftw_3d_dft_batch_c2r = fftw_plan_many_dft_c2r(SYS_DIM, N_batch, SYS_DIM, run_data->u_hat_tmp, NULL, SYS_DIM, 1, run_data->u, NULL, SYS_DIM, 1, FFTW_MEASURE);
	if (sys_vars->fftw_3d_dft_batch_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch C2R FFTW Plan\n-->> Exiting!!!\n");
		exit(1);
	}

	// Initialize Batch Fourier Transforms
	sys_vars->fftw_3d_dft_batch_r2c = fftw_plan_many_dft_r2c(SYS_DIM, N_batch, SYS_DIM, run_data->u, NULL, SYS_DIM, 1, run_data->u_hat, NULL, SYS_DIM, 1, FFTW_MEASURE);
	if (sys_vars->fftw_3d_dft_batch_r2c == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch R2C FFTW Plan\n-->> Exiting!!!\n");
		exit(1);
	}
}
/**
 * Wrapper function used to allocate the nescessary data objects
 * @param N Array containing the dimensions of the system
 */
void AllocateMemory(const long int* N) {

	// Initialize variables
	int tmp1, tmp2_r, tmp2_f, indx_r, indx_f;
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Nz = N[2];
	const long int Nz_Fourier = Nz / 2 + 1;

	// --------------------------------
	//  Allocate Field Data
	// --------------------------------
	// Allocate current Fourier vorticity
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nz_Fourier * SYS_DIM);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Vorticity");
		exit(1);
	}
	run_data->w_hat_tmp = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nz_Fourier * SYS_DIM);
	if (run_data->w_hat_tmp == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Vorticity");
		exit(1);
	}

	// Allocate current Fourier velocity
	run_data->u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nz_Fourier * SYS_DIM);
	if (run_data->u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}
	run_data->u_hat_tmp = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nz_Fourier * SYS_DIM);
	if (run_data->u_hat_tmp == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Velocity");
		exit(1);
	}

	// Allocate current Real vorticity
	run_data->w = (double* )fftw_malloc(sizeof(double) * Nx * Ny * Nz * SYS_DIM);
	if (run_data->w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Vorticity");
		exit(1);
	}
	run_data->u = (double* )fftw_malloc(sizeof(double) * Nx * Ny * Nz * SYS_DIM);
	if (run_data->u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity");
		exit(1);
	}

	// Initialize arrays
	for (int i = 0; i < Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2_r = Nz * (tmp1 + j);
			tmp2_f = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz; ++k) {
				indx_r = tmp2_r + k;
				indx_f = tmp2_f + k;
				for (int s = 0; s < SYS_DIM; ++s) {
					if (k < Nz_Fourier) {
						run_data->w_hat[SYS_DIM * indx_f + s]     = 0.0 + 0.0 * I;
						run_data->w_hat_tmp[SYS_DIM * indx_f + s] = 0.0 + 0.0 * I;
						run_data->u_hat[SYS_DIM * indx_f + s]     = 0.0 + 0.0 * I;
						run_data->u_hat_tmp[SYS_DIM * indx_f + s] = 0.0 + 0.0 * I;
					}
					run_data->w[SYS_DIM * indx_r + s] = 0.0;
					run_data->u[SYS_DIM * indx_r + s] = 0.0;
				}
			}
		}
	}
}

/**
 * Wrapper function to free memory and close any objects before exiting
 */
void FreeMemoryAndCleanUp(void) {

	// --------------------------------
	//  Free memory
	// --------------------------------
	fftw_free(run_data->w_hat);
	fftw_free(run_data->w_hat_tmp);
	fftw_free(run_data->u_hat_tmp);
	fftw_free(run_data->w);
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);

	// --------------------------------
	//  Free FFTW Plans
	// --------------------------------
	// Destroy FFTW plans
	fftw_destroy_plan(sys_vars->fftw_3d_dft_batch_c2r);
	fftw_destroy_plan(sys_vars->fftw_3d_dft_batch_r2c);

}

void str_func_test(int Nx, int Ny, int Nz, int num_thread) {

	int Max_Incr = Nx / 2;
	double increment;
	int tmp1, tmp2, indx;


	omp_set_num_threads(num_thread);
	int grain_size = NUM_POW * Max_Incr / num_thread;

	double Lx = 1.0;
	double Ly = 1.0;
	double Lz = 1.0;
	double dx = Lx / Nx;
	double dy = Ly / Ny;
	double dz = Lz / Nz;


	// Allocate test memory
	double* data = (double* )fftw_malloc(sizeof(double) * Nx * Ny * Nz * SYS_DIM);
	double* str_func_par[NUM_POW];
	double* str_func_ser[NUM_POW];

	// Initialize test memory
	for (int i = 0; i < Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz * (tmp1 + j);
			for (int k = 0; k < Nz; ++k) {
				indx = tmp2 + k;
					data[SYS_DIM * indx + 0] = i * dx;
					data[SYS_DIM * indx + 1] = j * dy;
					data[SYS_DIM * indx + 2] = k * dz;
			}
		}
	}
	
	for (int p = 1; p <= NUM_POW; ++p) {
		str_func_ser[p - 1] = (double* )fftw_malloc(sizeof(double) * Max_Incr);
		str_func_par[p - 1] = (double* )fftw_malloc(sizeof(double) * Max_Incr);
		for (int r = 0; r < Max_Incr; ++r){
			str_func_par[p - 1][r] = 0.0;
			str_func_ser[p - 1][r] = 0.0;
		}
	}

	


	printf("\n\nN: %d\t Max r: %d\t Num_pow: %d\tThreads: %d Grainsize: %d\n", Nx, Max_Incr, NUM_POW, num_thread, grain_size);


	// Get parllel time
	double start = omp_get_wtime();
	
	// // Loop over powers
	// #pragma omp parallel num_threads(num_thread) shared(data, str_func_par) private(tmp1, tmp2, indx)
	// {
	// 	#pragma omp single 
	// 	{
	// 		#pragma omp taskloop reduction(+:increment) grainsize(grain_size) collapse(2)
			for (int p = 1; p <= NUM_POW; ++p) {
				// Loop over increments
				for (int r = 1; r <= Max_Incr; ++r) {
					// Initialize increment
					increment = 0.0;

					// Loop over space
					// #pragma omp taskloop reduction(+:increment) grainsize(Nx * Ny * Nz / num_thread) collapse(3)
					for (int i = 0; i < Nx; ++i) {
						for (int j = 0; j < Ny; ++j) {
							for (int k = 0; k < Nz; ++k) {
								tmp1 = i * Ny;
								tmp2 = Nz * (tmp1 + j);
								indx = tmp2 + k;

								// Compute increments
								increment += pow(data[SYS_DIM * (Nz * (i * Ny + j) + ((k + r) % Nz)) + 0]  - data[SYS_DIM * indx + 0], p);
								increment += pow(data[SYS_DIM * (Nz * (i * Ny + ((j + r) % Ny)) + k) + 1] - data[SYS_DIM * indx + 1], p);
								increment += pow(data[SYS_DIM * (Nz * (((i + r) % Nx) * Ny + j) + k) + 2]  - data[SYS_DIM * indx + 2], p);
							}
						}
					}

					// Update structure function
					str_func_par[p - 1][r - 1] = increment; 
				}
			}
	// 	} 
	// }

	// Get parallel time
	double par_time = omp_get_wtime() - start;
	double ser_time = 0.0;
	printf("\n\nTimes: %lfs (ser) %lfs (par)\t\tSpeed Up: %lf\n", ser_time, par_time, ser_time / par_time);

	double err = 0.0;
	for (int p = 1; p <= NUM_POW; ++p) {
		for (int r = 1; r <= Max_Incr; ++r) {
			err += str_func_par[p - 1][r - 1];
		}
	}
	printf("str_func[%d][%d]: %1.16lf\n", 0, 0, err);


	for (int p = 0; p < NUM_POW; ++p) {
		fftw_free(str_func_ser[p]);
		fftw_free(str_func_par[p]);
	}
	fftw_free(data);
}