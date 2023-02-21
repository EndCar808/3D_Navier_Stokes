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
#include <time.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_func.h"
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

	// Initialize variables
	sys_vars->N[0] = atoi(argv[1]);
	int Nx = sys_vars->N[0];
	sys_vars->N[1] = Nx;
	sys_vars->N[2] = Nx;
	int Ny = sys_vars->N[1];
	int Nz = sys_vars->N[2];
	sys_vars->Max_Incr = Nx / 2;
	sys_vars->num_threads = atoi(argv[2]);
	int tmp1, tmp2, indx;

	double Lx = 2.0 * M_PI;
	double Ly = 2.0 * M_PI;
	double Lz = 2.0 * M_PI;
	sys_vars->dx = Lx / Nx;
	sys_vars->dy = Ly / Ny;
	sys_vars->dz = Lz / Nz;


	
	OpenInputAndInitialize();

	
	printf("Nx: %d Lx: %lf dx: %lf\nNy: %d Ly: %lf dy: %lf\nNz: %d Lz: %lf dz: %lf\n\n", Nx, Lx, sys_vars->dx, Ny, Ly, sys_vars->dy, Nz, Lz, sys_vars->dz);

	
	return 0;
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