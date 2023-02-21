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
#include <complex.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <omp.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "data_types.h"
// #include "utils.h"
// #include "post_proc.h"
// ---------------------------------------------------------------------
//  Global Variables Declarations
// ---------------------------------------------------------------------
#define NUM_POW 6
#define NUM_INCR 2
#define SYS_DIM 3
// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	printf("\n\ngcc version: %d.%d.%d\n\n", __GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__);

	// Initialize variables
	int Nx = atoi(argv[1]);
	int Ny = Nx;
	int Nz = Ny;
	int Max_Incr = Nx / 2;
	int num_thread = atoi(argv[2]);
	double increment;
	int tmp1, tmp2, indx;
	int num_pow = NUM_POW;

	double Lx = 1.0;
	double Ly = 1.0;
	double Lz = 1.0;
	double dx = Lx / Nx;
	double dy = Ly / Ny;
	double dz = Lz / Nz;

	omp_set_num_threads(num_thread);
	int grain_size = NUM_POW * Max_Incr / num_thread;
	printf("\n\nN: %d\t Max r: %d\t Num_pow: %d\tThreads: %d Grainsize: %d\n", Nx, Max_Incr, num_pow, num_thread, grain_size);
	printf("Nx: %d Lx: %lf dx: %lf\nNy: %d Ly: %lf dy: %lf\nNz: %d Lz: %lf dz: %lf\n\n", Nx, Lx, dx, Ny, Ly, dy, Nz, Lz, dz);

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

	// Get parllel time
	double start = omp_get_wtime();
	
	// // Loop over powers
	#pragma omp parallel num_threads(num_thread) shared(data, str_func_par) private(tmp1, tmp2, indx)
	{
		#pragma omp single 
		{
			#pragma omp taskloop reduction(+:increment) grainsize(grain_size) collapse(2)
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
		} 
	}

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

	return 0;
}