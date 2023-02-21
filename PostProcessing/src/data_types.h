/**
* @file data_types.h 
* @author Enda Carroll
* @date Feb 2023
* @brief file containing the main data types and global variables
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#ifndef __DATA_TYPES
#ifndef __HDF5_HDR
#include <hdf5.h>
#include <hdf5_hl.h>
#define __HDF5_HDR
#endif
#ifndef __FFTW3
#include <fftw3.h>
#define __FFTW3
#endif
#ifndef __OPENMP
#include <omp.h>
#define __OPENMP
#endif
#include <gsl/gsl_histogram.h> 
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_math.h>
// ---------------------------------------------------------------------
//  Compile Time Macros and Definitions
// ---------------------------------------------------------------------
#define checkError(x) ({int __val = (x); __val == -1 ? \
	({fprintf(stderr, "ERROR ("__FILE__":%d) -- %s\n", __LINE__, strerror(errno)); \
	exit(-1);-1;}) : __val; })

// For coloured printing to screen
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define RESET   "\x1b[0m"
// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
// These definitions define some of the solver parameters.
#define SYS_DIM 3 				// The system dimension i.e., 3D
// Statistics definitions
#define N_BINS 1000				// The number of histogram bins to use
#define NUM_INCR 5              // The number of increment length scales
#define INCR_TYPES 1 			// The number of increment directions i.e., longitudinal and transverse
#define BIN_LIM 40              // The limit of the bins for the velocity increments
#define NUM_POW 6               // The number of powers for the str function
// ---------------------------------------------------------------------
//  Global Struct Definitions
// ---------------------------------------------------------------------
// System variables struct
typedef struct system_vars_struct {
	char u0[64];						// String to indicate the initial condition to use
	char forcing[64];					// String to indicate what type of forcing is selected
	long int N[SYS_DIM];				// Array holding the no. of collocation pts in each dim
	fftw_plan fftw_3d_dft_r2c;			// FFTW plan to perform transform from Real to Fourier
	fftw_plan fftw_3d_dft_c2r;			// FFTW plan to perform transform from Fourier to Real
	fftw_plan fftw_3d_dft_batch_r2c;	// FFTW plan to perform a batch transform from Real to Fourier
	fftw_plan fftw_3d_dft_batch_c2r;	// FFTW plan to perform a batch transform from Real to Fourier
	long int num_snaps;					// Number of snapshots in the input data file
	long int kmax; 						// The largest dealiased wavenumber
	double t0;							// Intial time
	double T;							// Final time
	double t;							// Time variable
	double dt;							// Timestep
	double dx;							// Collocation point spaceing in the x direction
	double dy;							// Collocation point spacing in the y direction	
	double dz;							// Collocation point spacing in the z direction
	int num_threads;					// The number of OMP threads to use
	int thread_id;						// The ID of the OMP threads
	int Max_Incr;
} system_vars_struct;

// Runtime data struct
typedef struct runtime_data_struct {
	double* x[SYS_DIM];      // Array to hold collocation pts
	int* k[SYS_DIM];		 // Array to hold wavenumbers
	fftw_complex* w_hat;     // Fourier space vorticity
	fftw_complex* w_hat_tmp; // Temporary Fourier space vorticity
	fftw_complex* u_hat;     // Fourier space velocity
	fftw_complex* u_hat_tmp; // Fourier space velocity
	double* u_tmp; 			 // Temporary array to read & write in velocities
	double* w;				 // Real space vorticity
	double* u;				 // Real space velocity
	double* time;			 // Array to hold the simulation times
} runtime_data_struct;

// Post processing stats data struct
typedef struct stats_data_struct {
	gsl_rstat_workspace* grad_u_stats[SYS_DIM + 1];		  			// Workplace for the running stats for the gradients of velocity (both for each direction and combined)
	gsl_rstat_workspace* grad_w_stats[SYS_DIM + 1];		  			// Workplace for the running stats for the gradients of vorticity (both for each direction and combined)
	gsl_rstat_workspace* w_stats;									// Workplace for the running stats for the velocity (both for each direction and combined)
	gsl_rstat_workspace* u_stats[SYS_DIM + 1];						// Workplace for the running stats for the vorticity (both for each direction and combined)
	gsl_rstat_workspace* u_incr_stats[INCR_TYPES][NUM_INCR];		// Workplace for the running stats for the velocity increments
	gsl_rstat_workspace* w_incr_stats[INCR_TYPES][NUM_INCR];		// Workplace for the running stats for the vorticity increments
	gsl_histogram* w_hist;		 									// Histogram struct for the vorticity distribution
	gsl_histogram* u_hist;		  									// Histrogam struct for the velocity distribution
	gsl_histogram* u_grad_hist[SYS_DIM + 1];	 					// Array to hold the PDFs of the longitudinal and transverse velocity gradients 
	gsl_histogram* w_grad_hist[SYS_DIM + 1];	 					// Array to hold the PDFs of the longitudinal and transverse vorticity gradients 
	gsl_histogram* u_incr_hist[INCR_TYPES][NUM_INCR];				// Array to hold the PDFs of the longitudinal and transverse velocity increments for each increment
	gsl_histogram* w_incr_hist[INCR_TYPES][NUM_INCR];				// Array to hold the PDFs of the longitudinal and transverse vorticity increments for each increment
	int * increments; 												// Pointer to array to hold the array index increments
} stats_data_struct;

// HDF5 file info struct
typedef struct HDF_file_info_struct {
	char input_file_name[512];		// Array holding input file name
	char output_file_name[512];     // Output file name array
	char output_dir[512];			// Output directory
	char input_dir[512];			// Input directory
	char output_tag[64]; 			// Tag to be added to the output directory
	hid_t output_file_handle;		// File handle for the output file 
	hid_t input_file_handle;		// File handle for the input file 
	hid_t COMPLEX_DTYPE;			// Complex datatype handle
} HDF_file_info_struct;

// Complex datatype struct for HDF5
typedef struct complex_type_tmp {
	double re;   		// real part 
	double im;   		// imaginary part 
} complex_type_tmp;

// Declare the global variable pointers across all files
extern system_vars_struct *sys_vars; 		    // Global pointer to system parameters struct
extern runtime_data_struct *run_data; 			// Global pointer to system runtime variables struct 
extern HDF_file_info_struct *file_info; 		// Global pointer to system forcing variables struct 
extern stats_data_struct *stats_data;           // Globale pointer to the statistics struct

#define __DATA_TYPES
#endif
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------