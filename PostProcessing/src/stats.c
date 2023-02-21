/**
* @file stats.c  
* @author Enda Carroll
* @date Jun 2021
* @brief File containing the stats functions for the pseudospectral solver data
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h> 
#include <math.h>
#include <complex.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_func.h"

// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Performs a run over the data to precompute and quantities needed before performing
 * the proper run over the data
 */
void Precompute(void) {

	// Initialize variables
	int gsl_status;
	int tmp1, tmp2, indx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = Nz / 2 + 1;
	int r;
	double long_incr_x;
	double long_incr_y;
	double long_incr_z;
	int x_incr_indx, y_incr_indx, z_incr_indx;
	double std_u_incr;
	
	// --------------------------------
	// Loop Through Snapshots
	// --------------------------------
	for (int s = 0; s < sys_vars->num_snaps; ++s) {

		printf("Precomputation Step: %d/%ld\n", s + 1, sys_vars->num_snaps);
		
		// Read in snaps
		ReadInData(s);


		// Loop over real space
		for (int i = 0; i < Nx; ++i) {
			tmp1 = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				tmp2 = Nz * (tmp1 + j);
				for (int k = 0; k < Nz; ++k) {
					indx = tmp2 + k;

					// Compute velocity increments and update histograms
					for (int r_indx = 0; r_indx < NUM_INCR; ++r_indx) {
						// Get the current increment
						r = stats_data->increments[r_indx];

						//------------- Get the longitudinal Velocity increments
						x_incr_indx = (i + r) % Nx;
						y_incr_indx = (j + r) % Ny;
						z_incr_indx = (k + r) % Nz;
						long_incr_x = run_data->u[SYS_DIM * (Nz * (x_incr_indx * Ny + j) + k) + 0] - run_data->u[SYS_DIM * indx + 0];
						long_incr_y = run_data->u[SYS_DIM * (Nz * (i * Ny + y_incr_indx) + k) + 1] - run_data->u[SYS_DIM * indx + 1];
						long_incr_z = run_data->u[SYS_DIM * (Nz * (i * Ny + j) + z_incr_indx) + 2] - run_data->u[SYS_DIM * indx + 2];

						// Update the stats accumulators
						gsl_rstat_add(long_incr_x, stats_data->u_incr_stats[0][r_indx]);
						gsl_rstat_add(long_incr_y, stats_data->u_incr_stats[0][r_indx]);
						gsl_rstat_add(long_incr_z, stats_data->u_incr_stats[0][r_indx]);
					}
				}
			}
		}		
	}

	// --------------------------------
	// Initialize Increment Histograms
	// --------------------------------
	// Set the bin limits for the velocity increments
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			// Get the std of the incrments
			std_u_incr = gsl_rstat_sd(stats_data->u_incr_stats[i][j]);

			// Velocity increments
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->u_incr_hist[i][j], -BIN_LIM * std_u_incr, BIN_LIM * std_u_incr);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Increments");
				exit(1);
			}
		}
	}
}
/**
 * Wrapper function to allocate memory and initialzie stats objects
 */
void AllocateStatsObjects(void) {

	// Initialzie variables
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];
	const long int Nz = sys_vars->N[2];
	const long int Nz_Fourier = Nz / 2 + 1;
	sys_vars->Max_Incr = (int) (Ny / 2);
	int Max_Incr = sys_vars->Max_Incr;

	// Set up increments array
	stats_data->increments = (int* )fftw_malloc(sizeof(int) * NUM_INCR);
	int increment[NUM_INCR] = {1, 2, 4, 16, Max_Incr};
	memcpy(stats_data->increments, increment, sizeof(increment));


	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {			
			// Initialize the velocity incrment objects
			stats_data->u_incr_hist[i][j]  = gsl_histogram_alloc(N_BINS);
			stats_data->u_incr_stats[i][j] = gsl_rstat_alloc();
		}
	}
}
/**
 * Wrapper function to free all stats related objects
 */
void FreeStatsObjects(void) {

	// --------------------------------
	//  Free GSL objects
	// --------------------------------
	// Free histogram structs
	for (int i = 0; i < INCR_TYPES; ++i) {
		for (int j = 0; j < NUM_INCR; ++j) {
			gsl_histogram_free(stats_data->u_incr_hist[i][j]);
			gsl_rstat_free(stats_data->u_incr_stats[i][j]);
		}	
	}
}
/**
 * Function to test the parallel computation of structure functions
 * @param Nx         The dimension in the x direction
 * @param Ny         The dimension size in the y direction
 * @param Nz         The dimension size in the z direction
 * @param num_thread The number of OMP threads to use
 */
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
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
