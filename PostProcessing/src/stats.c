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
#include <sys/stat.h>
#include <sys/types.h>
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
	int r;
	double long_incr_x;
	double long_incr_y;
	double long_incr_z;
	int x_incr_indx, y_incr_indx, z_incr_indx;
	double std_u_incr;
	double max_u_incr, min_u_incr;


	// --------------------------------
	//  Begin Timing
	// --------------------------------
	// Initialize timing counter
	clock_t pre_begin = omp_get_wtime();

	
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
			max_u_incr = gsl_rstat_max(stats_data->u_incr_stats[i][j]);
			min_u_incr = gsl_rstat_min(stats_data->u_incr_stats[i][j]);

			// Velocity increments
			gsl_status = gsl_histogram_set_ranges_uniform(stats_data->u_incr_hist[i][j], min_u_incr - 0.05, max_u_incr + 0.05);
			if (gsl_status != 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set bin ranges for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Velocity Increments");
				exit(1);
			}
		}
	}


	// --------------------------------
	//  End Timing
	// --------------------------------
	// Finish timing
	clock_t pre_end = omp_get_wtime();

	// Print time of precompute to screen
	double time_spent = (double) pre_end - pre_begin;
	int hh = (int) time_spent / 3600;
    int mm = ((int )time_spent - hh * 3600) / 60;
    int ss = time_spent - (hh * 3600) - (mm * 60);
	printf("\n\nPrecomputing took: %lfsec: %d (h) %d (m) %d (s)\n\n", time_spent, hh, mm, ss);
}
/**
 * Function to compute the statistics of the velocity field
 * @param s Current snapshot index
 */
void ComputeStats(int s) {

	// Initialize variables
	int tmp1, tmp2;
	int indx;
	int gsl_status;
	const long int Ny = sys_vars->N[0];
	const long int Nx = sys_vars->N[1];
	const long int Nz = sys_vars->N[2];
	int r;
	double long_incr_x;
	double long_incr_y;
	double long_incr_z;
	int x_incr_indx, y_incr_indx, z_incr_indx;


	// --------------------------------
	// Update Histogram Counts
	// --------------------------------
	// Update histograms with the data from the current snapshot
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

			
					// Update the histograms
					gsl_status = gsl_histogram_increment(stats_data->u_incr_hist[0][r_indx], long_incr_x);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Velocity Increment x", s, gsl_status, long_incr_x);
						exit(1);
					}
					gsl_status = gsl_histogram_increment(stats_data->u_incr_hist[0][r_indx], long_incr_y);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Velocity Increment y", s, gsl_status, long_incr_y);
						exit(1);
					}
					gsl_status = gsl_histogram_increment(stats_data->u_incr_hist[0][r_indx], long_incr_z);
					if (gsl_status != 0) {
						fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to update bin count for ["CYAN"%s"RESET"] for Snap ["CYAN"%d"RESET"] -- GSL Exit Status [Err:"CYAN" %d"RESET" - Val:"CYAN" %lf"RESET"]\n-->> Exiting!!!\n", "Longitudinal Velocity Increment z", s, gsl_status, long_incr_z);
						exit(1);
					}
				}
			}
		}
	}		

}
/**
 * Function to Write the stats to file
 * @param snap Current snapshot index
 * @param indx Output file index
 */
void WriteStatsData(const int snap, int indx) {

	// Initialize variables
	herr_t status;
	char file_name[1024];
	struct stat st = {0};	// this is used to check whether the output directories exist or not.
	static const hsize_t Dims1D = 1;
	hsize_t dset_dims_1d[Dims1D];
	static const hsize_t Dims2D = 2;
	hsize_t dset_dims_2d[Dims2D];
	static const hsize_t Dims3D = 3;
    hsize_t dset_dims_3d[Dims3D];



    // --------------------------------
    //  Generate Output File Path
    // --------------------------------
    if (strcmp(file_info->output_dir, "NONE") != 0) {
    	// Construct pathh
    	strcpy(file_info->output_file_name, file_info->output_dir);
    	sprintf(file_name, "Stats_HDF_Data_TAG[%s-%d].h5", file_info->output_tag, indx);
    	strcat(file_info->output_file_name, file_name);

    	// Print output file path to screen
    	printf("Output File: "CYAN"%s"RESET"\n", file_info->output_file_name);	
    }
    else if ((strcmp(file_info->output_dir, "NONE") == 0) && (stat(file_info->input_dir, &st) == 0)) {
    	printf("\n["YELLOW"NOTE"RESET"] --- No Output directory provided. Using input directory instead \n");

    	// Construct path
    	strcpy(file_info->output_file_name, file_info->input_dir);
    	sprintf(file_name, "Stats_HDF_Data_TAG[%s-%d].h5", file_info->output_tag, indx);
    	strcat(file_info->output_file_name, file_name);

    	// Print output file path to screen
    	printf("Output File: "CYAN"%s"RESET"\n", file_info->output_file_name);
    }
    else if ((stat(file_info->input_dir, &st) == -1) && (stat(file_info->output_dir, &st) == -1)) {
    	fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Output folder not provided or doesn't exist. Please provide output folder - see utils.c: \n-->>Exiting....\n");
    	exit(1);
    }


    // --------------------------------
    //  Create Output File
    // --------------------------------
    file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_info->output_file_handle < 0) {
    	fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create HDF5 output file at: "CYAN"%s"RESET" at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting....\n", file_info->output_file_name, snap);
    	exit(1);
    }


    // -------------------------------
    // Write Increment Stats
    // -------------------------------
    // Create temporary array to hold the data for writing
    double vel_incr_stats[INCR_TYPES][NUM_INCR][6];
    for (int i = 0; i < INCR_TYPES; ++i) {
    	for (int j = 0; j < NUM_INCR; ++j) {
			// Velocity increment stats
			vel_incr_stats[i][j][0] = gsl_rstat_min(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][1] = gsl_rstat_max(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][2] = gsl_rstat_mean(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][3] = gsl_rstat_sd(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][4] = gsl_rstat_skew(stats_data->u_incr_stats[i][j]);
			vel_incr_stats[i][j][5] = gsl_rstat_kurtosis(stats_data->u_incr_stats[i][j]);
	    }
    }

    // Write the data
    dset_dims_3d[0] = INCR_TYPES;
   	dset_dims_3d[1] = NUM_INCR;
   	dset_dims_3d[2] = 6;
   	status = H5LTmake_dataset(file_info->output_file_handle, "VelocityIncrementStats", Dims3D, dset_dims_3d, H5T_NATIVE_DOUBLE, vel_incr_stats);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", "Velocity Increment Stats", snap);
        exit(1);
    }


    // -------------------------------
    // Write Increment Histogram
    // -------------------------------
    ///----------------------------------- Write the Velocity Increments
	// Allocate temporary memory to record the histogram data contiguously
    double* vel_inc_range  = (double*) fftw_malloc(sizeof(double) * NUM_INCR * (N_BINS + 1));
    double* vel_inc_counts = (double*) fftw_malloc(sizeof(double) * NUM_INCR * (N_BINS));

    //-------------- Write the longitudinal increments
   	for (int r = 0; r < NUM_INCR; ++r) {
   		for (int b = 0; b < N_BINS + 1; ++b) {
	   		vel_inc_range[r * (N_BINS + 1) + b] = stats_data->u_incr_hist[0][r]->range[b];
	   		if (b < N_BINS) {
	   			vel_inc_counts[r * (N_BINS) + b] = stats_data->u_incr_hist[0][r]->bin[b];	   			
	   		}
   		}
   	}
   	dset_dims_2d[0] = NUM_INCR;
   	dset_dims_2d[1] = N_BINS + 1;
   	status = H5LTmake_dataset(file_info->output_file_handle, "LongitudinalVelIncrements_BinRanges", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_inc_range);
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", "Longitudinal Velocity Increment PDF Bin Ranges", snap);
        exit(1);
    }		
	dset_dims_2d[1] = N_BINS;
	status = H5LTmake_dataset(file_info->output_file_handle, "LongitudinalVelIncrements_BinCounts", Dims2D, dset_dims_2d, H5T_NATIVE_DOUBLE, vel_inc_counts);	
	if (status < 0) {
        fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write ["CYAN"%s"RESET"] to file at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", "Longitudinal Velocity Increment PDF Bin Counts", snap);
        exit(1);
    }

    // Free temp memory
    fftw_free(vel_inc_range);
    fftw_free(vel_inc_counts);


    // --------------------------------
	//  Close HDF5 Identifiers
	// --------------------------------
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->output_file_name, snap);
		exit(1);		
	}
}
/**
 * Wrapper function to allocate memory and initialzie stats objects
 */
void AllocateStatsObjects(void) {

	// Initialzie variables
	sys_vars->Max_Incr = (int) (sys_vars->N[0] / 2);
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
