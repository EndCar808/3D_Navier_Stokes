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
#include "utils.h"
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

	// --------------------------------
	//  Begin Timing
	// --------------------------------
	// Initialize timing counter
	clock_t main_begin = omp_get_wtime();

	// --------------------------------
	//  Get Command Line Arguements
	// --------------------------------
	if ((GetCMLArgs(argc, argv)) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line aguments, check utils.c file for details\n");
		exit(1);
	}

	// --------------------------------
	//  Initialize Thread Info
	// --------------------------------
	// Set the number of threads and get thread IDs
	omp_set_num_threads(sys_vars->num_threads);
	#pragma omp parallel
	{
		// Get thread id
		sys_vars->thread_id = omp_get_thread_num();

		// Plot the number of threads
		if (!(sys_vars->thread_id)) {
			printf("\nOMP Threads Active: "CYAN"%d"RESET"\n", sys_vars->num_threads);
		}
	}

	// Initialize and set threads for fftw plans
	fftw_init_threads();

	// Set the number of threads for FFTW and print to screen
	fftw_plan_with_nthreads(sys_vars->num_fftw_threads);
	printf("\nFFTW Threads: "CYAN"%d"RESET"\n", sys_vars->num_threads);


	// --------------------------------
	// Open Input File 
	// --------------------------------
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



	// --------------------------------
	// Allocate and Initialize Memory 
	// --------------------------------
	AllocateMemory(sys_vars->N);

	InitializeFFTWPlans(sys_vars->N);

	AllocateStatsObjects();
	// --------------------------------
	// Loop Through Data To Precompute 
	// --------------------------------
	Precompute();

	// Initialize quatities
	int file_indx = 0;
	int skip = 1;
	int write = 100;
	for (int snap = 0; snap < sys_vars->num_snaps; snap+=skip) { 

		// Print Update to screen
		printf("Post Step: %d/%ld\n", snap + 1, sys_vars->num_snaps);
		
		// --------------------------------
		// Read In Data
		// --------------------------------
		ReadInData(snap);

		// --------------------------------
		// Compute Stats
		// --------------------------------
		ComputeStats(snap);

		// --------------------------------
		//  Write State Periodically
		// --------------------------------
		// Write Data
		if (snap % write == 0) {
			WriteStatsData(snap, file_indx);
			file_indx++;
		}
	}

	// --------------------------------
	//  Write Final State
	// --------------------------------
	// Write final state
	printf("\n\nFinal Write to file...\n");
	WriteStatsData(sys_vars->num_snaps, sys_vars->num_snaps);


	// --------------------------------
	//  Free Memory
	// --------------------------------
	// Free stats objects
	FreeStatsObjects();

	// Free other memory
	FreeMemoryAndCleanUp();


	// --------------------------------
	//  Clean Up FFTW Objects
	// --------------------------------
	// Clean up FFTW Thread Info
	fftw_cleanup_threads();

	// Clean Up FFTW Plan Objects
	fftw_cleanup();

	// --------------------------------
	//  End Timing
	// --------------------------------
	// Finish timing
	clock_t main_end = omp_get_wtime();

	// Print time taken to screen
	PrintTime(main_begin, main_end);

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