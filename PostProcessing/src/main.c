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
#include "stats.h"
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

	AllocateStatsObjects();


	
	printf("Nx: %d Lx: %lf dx: %lf\nNy: %d Ly: %lf dy: %lf\nNz: %d Lz: %lf dz: %lf\n\n", Nx, Lx, sys_vars->dx, Ny, Ly, sys_vars->dy, Nz, Lz, sys_vars->dz);

	Precompute();

	for (int snap = 0; snap < sys_vars->num_snaps; ++snap) { 

		printf("Post Step: %d/%ld\n", snap + 1, sys_vars->num_snaps);
		
		// Read In Data
		ReadInData(snap);

		
	}

	FreeStatsObjects();

	FreeMemoryAndCleanUp();

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