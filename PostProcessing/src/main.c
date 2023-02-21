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

	// Initialize variables
	int Nx = atoi(argv[1]);
	int Ny = Nx;
	int Nz = Ny;
	int Max_Incr = Nx / 2;
	int num_threads = atoi(argv[2]);
	double increment;
	int tmp1, tmp2, indx;

	omp_set_num_threads(num_threads);
	printf("\n\nN: %d\t Max r: %d\tThreads: %d\n\n", Nx, Max_Incr, num_threads);

	// Allocate test memory
	double* data = (double* )fftw_malloc(sizeof(double) * Nx * Ny * Nz * SYS_DIM);
	double* str_func_par[NUM_POW];
	double* str_func_ser[NUM_POW];

	// Initialize test memory
	for (int i = 0; i < Nz; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nx * (tmp1 + j);
			for (int k = 0; k < Nx; ++k) {
				indx = tmp2 + k;
				for (int l = 0; l < SYS_DIM; ++l) {
					data[SYS_DIM * (Nx * (i * Ny + j) + k) + l] = 1.0;
				}
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

	printf("gcc version: %d.%d.%d\n", __GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__);
	
	// Loop over powers
	// #pragma omp parallel num_threads(num_threads) shared(data, str_func_ser) private(Nx, Ny, Nz, tmp1, tmp2, indx, Max_Incr)
	// {
	// 	#pragma omp single 
	// 	{
	// 		#pragma omp taskloop reduction(+:increment) 
			for (int p = 1; p <= NUM_POW; ++p) {
				// Loop over increments
				for (int r = 1; r <= Max_Incr; ++r) {
					// Initialize increment
					increment = 0.0;

					// Loop over space
					for (int i = 0; i < Nz; ++i) {
						tmp1 = i * Ny;
						for (int j = 0; j < Ny; ++j) {
							tmp2 = Nx * (tmp1 + j);
							for (int k = 0; k < Nx; ++k) {
								indx = tmp2 + k;

								// Compute increments
								increment += pow(data[SYS_DIM * (Nx * (i * Ny + j) + ((k + r) % Nx)) + 0]  - data[SYS_DIM * indx + 0], p);
								increment += pow(data[SYS_DIM * (Nx * (i * Ny + ((j + r) % Ny)) + Nx) + 1] - data[SYS_DIM * indx + 1], p);
								increment += pow(data[SYS_DIM * (Nx * (((i + r) % Nz) * Ny + j) + k) + 2]  - data[SYS_DIM * indx + 2], p);
							}
						}
					}

					// Update structure function
					str_func_ser[p - 1][r - 1] = increment; 
				}
			}
	// 	} 
	// }


	for (int p = 1; p <= NUM_POW; ++p) {
		for (int r = 1; r <= Max_Incr; ++r) {
			printf("str_func[%d][%d]: %lf\n", p, r, str_func_ser[p - 1][r - 1]);
		}
	}

	for (int p = 0; p < NUM_POW; ++p) {
		fftw_free(str_func_ser[p]);
		fftw_free(str_func_par[p]);
	}
	fftw_free(data);

	return 0;
}