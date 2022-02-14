/**
* @file hdf5_funcs.c  
* @author Enda Carroll
* @date Feb 2022
* @brief File containing HDF5 function wrappers for creating, opening, wrtining to and closing output file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "solver.h"
#include "data_types.h"
#include "hdf5_funcs.h"
#include "utils.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Wrapper function that creates the ouput directory, creates and opens the main output file using parallel access and the spectra file using normal serial access
 */
void CreateOutputFilesWriteICs(const long int* N, double dt) {

	// Initialize variabeles
	const long int Nx 		  = N[0];
	const long int Ny 		  = N[1];
	const long int Nz 		  = N[2];
	const long int Nz_Fourier = N[2] / 2 + 1;
	hid_t main_group_id;
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	hid_t spectra_group_id;
	#endif
	char group_name[128];
	herr_t status;
	hid_t plist_id;
	int tmp1, tmp2;
	int indx;

    #if (defined(__VORT_FOUR) || defined(__MODES)) && !defined(DEBUG)
	// Create compound datatype for the complex datasets
	file_info->COMPLEX_DTYPE = CreateComplexDatatype();
	#endif

	///////////////////////////
	/// Create & Open Files
	///////////////////////////
	// -----------------------------------
	// Create Output Directory and Path
	// -----------------------------------
	GetOutputDirPath();

	// ------------------------------------------
	// Create Parallel File PList for Main File
	// ------------------------------------------
	// Create proptery list for main file access and set to parallel I/O
	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	status   = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
	if (status < 0) {
		printf("\n["RED"ERROR"RESET"] --- Could not set parallel I/O access for HDF5 output file! \n-->>Exiting....\n");
		exit(1);
	}

	// ---------------------------------
	// Create the output files
	// ---------------------------------
	// Create the main output file
	file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
	if (file_info->output_file_handle < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create main HDF5 output file at: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->output_file_name);
		exit(1);
	}

	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (!sys_vars->rank){
		// Create the spectra output file
		file_info->spectra_file_handle = H5Fcreate(file_info->spectra_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"]  --- Could not create HDF5 spectra output file at: "CYAN"%s"RESET" \n-->>Exiting....\n", file_info->spectra_file_name);
			exit(1);
		}
	}
	#endif


	////////////////////////////////
	/// Write Initial Condtions
	////////////////////////////////
	// --------------------------------------
	// Create Group for Initial Conditions
	// --------------------------------------
	// Initialize Group Name
	sprintf(group_name, "/Iter_%05d", 0);
	
	// Create group for the current iteration data
	main_group_id = CreateGroup(file_info->output_file_handle, file_info->output_file_name, group_name, 0.0, dt, 0);
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (!sys_vars->rank) {
		spectra_group_id = CreateGroup(file_info->spectra_file_handle, file_info->spectra_file_name, group_name, 0.0, dt, 0);
	}
	#endif


	// --------------------------------------
	// Write Initial Conditions
	// --------------------------------------
	#if !defined(TRANSIENTS)
	// Create dimension arrays
	static const int d_set_rank2D = 2;
	hsize_t dset_dims2D[d_set_rank2D];        // array to hold dims of the dataset to be created
	hsize_t slab_dims2D[d_set_rank2D];	      // Array to hold the dimensions of the hyperslab
	hsize_t mem_space_dims2D[d_set_rank2D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding
	static const int d_set_rank3D = 3;
	hsize_t dset_dims3D[d_set_rank3D];        // array to hold dims of the dataset to be created
	hsize_t slab_dims3D[d_set_rank3D];	      // Array to hold the dimensions of the hyperslab
	hsize_t mem_space_dims3D[d_set_rank3D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding
	static const int d_set_rank4D = 4;
	hsize_t dset_dims4D[d_set_rank4D];        // array to hold dims of the dataset to be created
	hsize_t slab_dims4D[d_set_rank4D];	      // Array to hold the dimensions of the hyperslab
	hsize_t mem_space_dims4D[d_set_rank4D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding

	///---------------------------- Write Real Space Vorticity
	#if defined(__VORT_REAL)
	// Transform vorticity back to real space and normalize
	fftw_mpi_execute_dft_c2r(sys_vars->fftw_3d_dft_batch_c2r, run_data->w_hat, run_data->w);
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = (Nz + 2) * (tmp1 + j);
			for (int k = 0; k < Nz; ++k) {
				indx = tmp2 + k;

				// Normalize
				run_data->w[SYS_DIM * indx + 0] *= 1.0 / (double) (Nx * Ny * Nz);
				run_data->w[SYS_DIM * indx + 1] *= 1.0 / (double) (Nx * Ny * Nz);
				run_data->w[SYS_DIM * indx + 2] *= 1.0 / (double) (Nx * Ny * Nz);
			}			
		}
	}

	// Specify dataset dimensions
	dset_dims4D[0] 	  = Nx;
	dset_dims4D[1] 	  = Ny;
	dset_dims4D[2] 	  = Nz;
	dset_dims4D[3] 	  = SYS_DIM;
	slab_dims4D[0]      = sys_vars->local_Nx;
	slab_dims4D[1]      = Ny;
	slab_dims4D[2]      = Nz;
	slab_dims4D[3]      = SYS_DIM;
	mem_space_dims4D[0] = sys_vars->local_Nx;
	mem_space_dims4D[1] = Ny;
	mem_space_dims4D[2] = Nz + 2;
	mem_space_dims4D[3] = SYS_DIM;

	// Write the real space vorticity
	WriteGroupDataReal(0.0, 0, main_group_id, "w", H5T_NATIVE_DOUBLE, d_set_rank4D, dset_dims4D, slab_dims4D, mem_space_dims4D, sys_vars->local_Nx_start, run_data->w);
	#endif

	///---------------------------- Write Fourier Space Vorticity
	#if defined(__VORT_FOUR)
	// Create dimension arrays
	dset_dims4D[0] 	  = Nx;
	dset_dims4D[1] 	  = Ny;
	dset_dims4D[2] 	  = Nz_Fourier;
	dset_dims4D[3] 	  = SYS_DIM;
	slab_dims4D[0]      = sys_vars->local_Nx;
	slab_dims4D[1]      = Ny;
	slab_dims4D[2]      = Nz_Fourier;
	slab_dims4D[3]      = SYS_DIM;
	mem_space_dims4D[0] = sys_vars->local_Nx;
	mem_space_dims4D[1] = Ny;
	mem_space_dims4D[2] = Nz_Fourier;
	mem_space_dims4D[3] = SYS_DIM;

	// Write the real space vorticity
	WriteGroupDataFourier(0.0, 0, main_group_id, "w_hat", file_info->COMPLEX_DTYPE, d_set_rank4D, dset_dims4D, slab_dims4D, mem_space_dims4D, sys_vars->local_Nx_start, run_data->w_hat);
	#endif

	///-------------------------- Write the Fourier Space velocity 
	#if defined(__MODES)
	// Create dimension arrays
	dset_dims4D[0] 	    = Nx;
	dset_dims4D[1] 	    = Ny;
	dset_dims4D[2] 	    = Nz_Fourier;
	dset_dims4D[3]      = SYS_DIM;
	slab_dims4D[0] 	    = sys_vars->local_Nx;
	slab_dims4D[1] 	    = Ny;
	slab_dims4D[2] 	    = Nz_Fourier;
	slab_dims4D[3]      = SYS_DIM;
	mem_space_dims4D[0] = sys_vars->local_Nx;
	mem_space_dims4D[1] = Ny;
	mem_space_dims4D[2] = Nz_Fourier;
	mem_space_dims4D[3] = SYS_DIM;

	// Write the real space vorticity
	WriteGroupDataFourier(0.0, 0, main_group_id, "u_hat", file_info->COMPLEX_DTYPE, d_set_rank4D, dset_dims4D, slab_dims4D, mem_space_dims4D, sys_vars->local_Nx_start, run_data->u_hat);
	#endif

	///-------------------------- Write the Real Space velocity 
	#if defined(__REALSPACE)
	// Transform velocities back to real space and normalize
	fftw_mpi_execute_dft_c2r(sys_vars->fftw_3d_dft_batch_c2r, run_data->u_hat, run_data->u);
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = (Nz + 2) * (tmp1 + j);
			for (int k = 0; k < Nz; ++k) {
				indx = tmp2 + k;

				// Normalize
				run_data->u[SYS_DIM * indx + 0] *= 1.0 / (double) (Nx * Ny * Nz);
				run_data->u[SYS_DIM * indx + 1] *= 1.0 / (double) (Nx * Ny * Nz);
				run_data->u[SYS_DIM * indx + 2] *= 1.0 / (double) (Nx * Ny * Nz);
			}
		}
	}

	// Specify dataset dimensions
	dset_dims4D[0] 	  = Nx;
	dset_dims4D[1] 	  = Ny;
	dset_dims4D[2] 	  = Nz;
	dset_dims4D[3] 	  = SYS_DIM;
	slab_dims4D[0]      = sys_vars->local_Nx;
	slab_dims4D[1]      = Ny;
	slab_dims4D[2]      = Nz;
	slab_dims4D[3]      = SYS_DIM;
	mem_space_dims4D[0] = sys_vars->local_Nx;
	mem_space_dims4D[1] = Ny;
	mem_space_dims4D[2] = Nz + 2;
	mem_space_dims4D[3] = SYS_DIM;

	// Write the real space vorticity
	WriteGroupDataReal(0.0, 0, main_group_id, "u", H5T_NATIVE_DOUBLE, d_set_rank4D, dset_dims4D, slab_dims4D, mem_space_dims4D, sys_vars->local_Nx_start, run_data->u);
	#endif
	#endif

	// ------------------------------------
	// Close Identifiers - also close file
	// ------------------------------------
	status = H5Pclose(plist_id);
	status = H5Gclose(main_group_id);
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->output_file_name, 0, 0.0);
		exit(1);		
	}
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (!sys_vars->rank) {
		status = H5Gclose(spectra_group_id);
		status = H5Fclose(file_info->spectra_file_handle);
		if (status < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->spectra_file_name, 0, 0.0);
			exit(1);		
		}
	}
	#endif
}
/**
 * Function that creates the output file paths and directories
 */
void GetOutputDirPath(void) {

	// Initialize variables
	char sys_type[64];
	char solv_type[64];
	char model_type[64];
	char tmp_path[512];
	char file_data[512];  
	struct stat st = {0};	// this is used to check whether the output directories exist or not.

	// ----------------------------------
	// Check if Provided Directory Exists
	// ----------------------------------
	if (!sys_vars->rank) {
		// Check if output directory exists
		if (stat(file_info->output_dir, &st) == -1) {
			printf("\n["YELLOW"NOTE"RESET"] --- Provided Output directory doesn't exist, now creating it...\n");
			// If not then create it
			if ((mkdir(file_info->output_dir, 0700)) == -1) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create provided output directory ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->output_dir);
				exit(1);
			}
		}
	}

	////////////////////////////////////////////
	// Check if Output File Only is Requested
	////////////////////////////////////////////
	if (file_info->file_only) {
		// Update to screen that file only output option is selected
		printf("\n["YELLOW"NOTE"RESET"] --- File only output option selected...\n");
		
		// ----------------------------------
		// Get Simulation Details
		// ----------------------------------
		#if defined(__NAVIER)
		sprintf(sys_type, "%s", "NAV");
		#elif defined(__EULER)
		sprintf(sys_type, "%s", "EUL");
		#else
		sprintf(sys_type, "%s", "UKN");
		#endif
		#if defined(__RK4)
		sprintf(solv_type, "%s", "RK4");
		#elif defined(__RK5)
		sprintf(solv_type, "%s", "RK5");
		#elif defined(__DPRK5)
		sprintf(solv_type, "%s", "DP5");
		#else 
		sprintf(solv_type, "%s", "UKN");
		#endif
		#if defined(__PHASE_ONLY)
		sprintf(model_type, "%s", "PO");
		#else
		sprintf(model_type, "%s", "FULL");
		#endif

		// -------------------------------------
		// Get File Label from Simulation Data
		// -------------------------------------
		// Construct file label from simulation data
		sprintf(file_data, "_SIM[%s-%s-%s]_N[%ld,%ld]_T[%d-%d]_NU[%1.10lf]_CFL[%1.2lf]_u0[%s].h5", sys_type, solv_type, model_type, sys_vars->N[0], sys_vars->N[1], (int )sys_vars->t0, (int )sys_vars->T, sys_vars->NU, sys_vars->CFL_CONST, sys_vars->u0);

		// ----------------------------------
		// Construct File Paths
		// ---------------------------------- 
		// Construct main file path
		strcpy(tmp_path, file_info->output_dir);
		strcat(tmp_path, "Main_HDF_Data"); 
		strcpy(file_info->output_file_name, tmp_path); 
		strcat(file_info->output_file_name, file_data);
		if ( !(sys_vars->rank) ) {
			printf("\nMain Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);
		}

		#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
		if ( !(sys_vars->rank) ) {
			// Construct Spectra file path
			strcpy(tmp_path, file_info->output_dir);
			strcat(tmp_path, "Spectra_HDF_Data"); 
			strcpy(file_info->spectra_file_name, tmp_path); 
			strcat(file_info->spectra_file_name, file_data);
			printf("Spectra Output File: "CYAN"%s"RESET"\n\n", file_info->spectra_file_name);
		}	
		#endif
	}
	else {
		// ----------------------------------
		// Get Simulation Details
		// ----------------------------------
		#if defined(__NAVIER)
		sprintf(sys_type, "%s", "NAVIER");
		#elif defined(__EULER)
		sprintf(sys_type, "%s", "EULER");
		#else
		sprintf(sys_type, "%s", "SYS_UNKN");
		#endif
		#if defined(__RK4)
		sprintf(solv_type, "%s", "RK4");
		#elif defined(__RK5)
		sprintf(solv_type, "%s", "RK5");
		#elif defined(__DPRK5)
		sprintf(solv_type, "%s", "DP5");
		#else 
		sprintf(solv_type, "%s", "SOLV_UKN");
		#endif
		#if defined(__PHASE_ONLY)
		sprintf(model_type, "%s", "PHAEONLY");
		#else
		sprintf(model_type, "%s", "FULL");
		#endif

		// ----------------------------------
		// Construct Output folder
		// ----------------------------------
		// Construct file label from simulation data
		sprintf(file_data, "SIM_DATA_%s_%s_%s_N[%ld,%ld]_T[%d-%d]_NU[%1.10lf]_CFL[%1.2lf]_u0[%s]_TAG[%s]/", sys_type, solv_type, model_type, sys_vars->N[0], sys_vars->N[1], (int )sys_vars->t0, (int )sys_vars->T, sys_vars->NU, sys_vars->CFL_CONST, sys_vars->u0, file_info->output_tag);

		// ----------------------------------
		// Check Existence of Output Folder
		// ----------------------------------
		strcat(file_info->output_dir, file_data);
		if (!sys_vars->rank) {
			// Check if folder exists
			if (stat(file_info->output_dir, &st) == -1) {
				// If not create it
				if ((mkdir(file_info->output_dir, 0700)) == -1) {
					fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create folder for output files ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->output_dir);
					exit(1);
				}
			}
		}

		// ----------------------------------
		// Construct File Paths
		// ---------------------------------- 
		// Construct main file path
		strcpy(file_info->output_file_name, file_info->output_dir); 
		strcat(file_info->output_file_name, "Main_HDF_Data.h5");
		if ( !(sys_vars->rank) ) {
			printf("\nMain Output File: "CYAN"%s"RESET"\n\n", file_info->output_file_name);
		}

		#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
		if ( !(sys_vars->rank) ) {
			// Construct spectra file path
			strcpy(file_info->spectra_file_name, file_info->output_dir); 
			strcat(file_info->spectra_file_name, "Spectra_HDF_Data.h5");
			printf("Spectra Output File: "CYAN"%s"RESET"\n\n", file_info->spectra_file_name);
		}	
		#endif
	}

	// Make All process wait before opening output files later
	MPI_Barrier(MPI_COMM_WORLD);
}
/**
 * Wrapper function that writes the data to file by openining it, creating a group for the current iteration and writing the data under this group. The file is then closed again 
 * @param t     The current time of the simulation
 * @param dt    The current timestep being used
 * @param iters The current iteration
 */
void WriteDataToFile(double t, double dt, long int iters) {

	// Initialize Variables
	int tmp;
	int indx;
	char group_name[128];
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Ny_Fourier = sys_vars->N[1] / 2 + 1;
	herr_t status;
	hid_t main_group_id;
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	hid_t spectra_group_id;
	#endif
	hid_t plist_id;
	static const int d_set_rank2D = 2;
	hsize_t dset_dims2D[d_set_rank2D];        // array to hold dims of the dataset to be created
	hsize_t slab_dims2D[d_set_rank2D];	      // Array to hold the dimensions of the hyperslab
	hsize_t mem_space_dims2D[d_set_rank2D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding
	#if defined(__MODES) || defined(__REALSPACE)
	static const int d_set_rank3D = 3;
	hsize_t dset_dims3D[d_set_rank3D];        // array to hold dims of the dataset to be created
	hsize_t slab_dims3D[d_set_rank3D];	      // Array to hold the dimensions of the hyperslab
	hsize_t mem_space_dims3D[d_set_rank3D];   // Array to hold the dimensions of the memoray space - for real data this will be different to slab_dims due to 0 padding
	#endif

	// --------------------------------------
	// Check if files exist and Open/Create
	// --------------------------------------
	// Create property list for setting parallel I/O access properties for file
	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

	// Check if main file exists - open it if it does if not create it
	if (access(file_info->output_file_name, F_OK) != 0) {
		file_info->output_file_handle = H5Fcreate(file_info->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->output_file_name, iters, t);
			exit(1);
		}
	}
	else {
		// Open file with parallel I/O access properties
		file_info->output_file_handle = H5Fopen(file_info->output_file_name, H5F_ACC_RDWR, plist_id);
		if (file_info->output_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->output_file_name, iters, t);
			exit(1);
		}
	}
	H5Pclose(plist_id);

	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (!sys_vars->rank) {
		// Check if spectra file exists - open it if it does if not create it
		if (access(file_info->output_file_name, F_OK) != 0) {
			file_info->spectra_file_handle = H5Fcreate(file_info->spectra_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
			if (file_info->spectra_file_handle < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to create spectra file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->spectra_file_name, iters, t);
				exit(1);
			}
		}
		else {
			// Open file with parallel I/O access properties
			file_info->spectra_file_handle = H5Fopen(file_info->spectra_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
			if (file_info->spectra_file_handle < 0) {
				fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open spectra file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->spectra_file_name, iters, t);
				exit(1);
			}
		}
	}
	#endif


	// -------------------------------
	// Close identifiers and File
	// -------------------------------
	status = H5Gclose(main_group_id);
	status = H5Fclose(file_info->output_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->output_file_name, iters, t);
		exit(1);
	}
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	if (!sys_vars->rank) {
		status = H5Gclose(spectra_group_id);
		status = H5Fclose(file_info->spectra_file_handle);
		if (status < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close output file ["CYAN"%s"RESET"] at: Iter = ["CYAN"%ld"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", file_info->spectra_file_name, iters, t);
			exit(1);		
		}
	}
	#endif
}
/**
 * Function that creates a dataset in a created Group in the output file and writes the data to this dataset for Fourier Space arrays
 * @param group_id       The identifier of the Group for the current iteration to write the data to
 * @param dset_name      The name of the dataset to write
 * @param dtype          The datatype of the data being written
 * @param dset_dims      Array containg the dimensions of the dataset to create
 * @param slab_dims      Array containing the dimensions of the hyperslab to select
 * @param mem_space_dims Array containing the dimensions of the memory space that will be written to file
 * @param offset_Nx      The offset in the dataset that each process will write to
 * @param data           The data being written to file
 */
void WriteGroupDataFourier(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, fftw_complex* data) {

	// Initialize variables
	hid_t plist_id;
	hid_t dset_space;
	hid_t file_space;
	hid_t mem_space;
	const int Dims = dset_rank;
	hsize_t dims[Dims];          // array to hold dims of the dataset to be created
	hsize_t mem_dims[Dims];	     // Array to hold the dimensions of the memory space - this will be diferent to slab dims for real data due to zero
	hsize_t mem_offset[Dims];    // Array to hold the offset in eahc direction for the local hypslabs to write from
	hsize_t slabsize[Dims];      // Array holding the size of the hyperslab in each direction
	hsize_t dset_offset[Dims];   // Array containig the offset positions in the file for each process to write to
	hsize_t dset_slabsize[Dims]; // Array containing the size of the slabbed that is being written to in file	

	// -------------------------------
	// Create Dataset In Group
	// -------------------------------
	// Create the dataspace for the data set
	for (int i = 0; i < dset_rank; ++i) {
		dims[i] = dset_dims[i];
	}
	dset_space = H5Screate_simple(Dims, dims, NULL); 
	if (dset_space < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set dataspace for dataset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);
	}

	// Create the file space id for the dataset in the group
	file_space = H5Dcreate(group_id, dset_name, dtype, dset_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// -------------------------------
	// Select Hyperslab in Memory
	// -------------------------------
	// Setup for memory hyperslab selection dimensions
	for (int i = 0; i < dset_rank; ++i) {
		slabsize[i]   = slab_dims[i];
		mem_offset[i] = 0;
		mem_dims[i]   = mem_space_dims[i];
	}
	
	// Create the memory space for the hyperslabs for each process
	mem_space = H5Screate_simple(Dims, mem_dims, NULL);

	// Select local hyperslab from the memoryspace (slab size adjusted to ignore 0 padding) - local to each process
	if ((H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_offset, NULL, slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- unable to select local hyperslab for datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// -------------------------------
	// Select Hyperslab in File
	// -------------------------------
	// Setup for file hyperslab selection dimensions
	for (int i = 0; i < dset_rank; ++i) {
		dset_offset[i]   = 0;
		dset_slabsize[i] = slab_dims[i];
	}
	dset_offset[0]   = offset_Nx;

	// Select the hyperslab in the dataset on file to write to
	if ((H5Sselect_hyperslab(dset_space, H5S_SELECT_SET, dset_offset, NULL, dset_slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to select hyperslab in file for datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// --------------------------------------
	// Set Up Collective Write & Write Data
	// --------------------------------------
	// Set up Collective write property
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	// Write data to file
	if ((H5Dwrite(file_space, dtype, mem_space, dset_space, plist_id, data)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write data to datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// -------------------------------
	// Close identifiers
	// -------------------------------
	H5Pclose(plist_id);
	H5Dclose(file_space);
	H5Sclose(dset_space);
	H5Sclose(mem_space);
}
/**
 * Function that creates a dataset in a created Group in the output file and writes the data to this dataset for Real Space arrays
 * @param group_id       The identifier of the Group for the current iteration to write the data to
 * @param dset_name      The name of the dataset to write
 * @param dtype          The datatype of the data being written
 * @param dset_dims      Array containg the dimensions of the dataset to create
 * @param slab_dims      Array containing the dimensions of the hyperslab to select
 * @param mem_space_dims Array containing the dimensions of the memory space that will be written to file
 * @param offset_Nx      The offset in the dataset that each process will write to
 * @param data           The data being written to file
 */
void WriteGroupDataReal(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, double* data) {

	// Initialize variables
	hid_t plist_id;
	hid_t dset_space;
	hid_t file_space;
	hid_t mem_space;
	const int Dims = dset_rank;
	hsize_t dims[Dims];          // array to hold dims of the dataset to be created
	hsize_t mem_dims[Dims];	     // Array to hold the dimensions of the memory space - this will be diferent to slab dims for real data due to zero
	hsize_t mem_offset[Dims];    // Array to hold the offset in each direction for the local hypslabs to write from
	hsize_t slabsize[Dims];      // Array holding the size of the hyperslab in each direction
	hsize_t dset_offset[Dims];   // Array containig the offset positions in the file for each process to write to
	hsize_t dset_slabsize[Dims]; // Array containing the size of the slabs that is being written to in file	

	// -------------------------------
	// Create Dataset In Group
	// -------------------------------
	// Create the dataspace for the data set
	for (int i = 0; i < dset_rank; ++i) {
		dims[i] = dset_dims[i];
	}
	dset_space = H5Screate_simple(Dims, dims, NULL); 
	if (dset_space < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to set dataspace for dataset: ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);
	}	

	// Create the file space id for the dataset in the group
	file_space = H5Dcreate(group_id, dset_name, H5T_NATIVE_DOUBLE, dset_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// -------------------------------
	// Select Hyperslab in Memory
	// -------------------------------
	// Setup for memory hyperslab selection dimensions
	for (int i = 0; i < dset_rank; ++i) {
		slabsize[i]   = slab_dims[i];
		mem_offset[i] = 0;
		mem_dims[i]   = mem_space_dims[i];
	}
	
	// Create the memory space for the hyperslabs for each process - reset second dimension for hyperslab selection to ignore padding
	mem_space = H5Screate_simple(Dims, mem_dims, NULL);

	// Select local hyperslab from the memoryspace (slab size adjusted to ignore 0 padding) - local to each process
	if ((H5Sselect_hyperslab(mem_space, H5S_SELECT_SET, mem_offset, NULL, slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- unable to select local hyperslab for datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// -------------------------------
	// Select Hyperslab in File
	// -------------------------------
	// Set up file hyperslab selection dimensions
	for (int i = 0; i < dset_rank; ++i) {
		dset_offset[i]   = 0;
		dset_slabsize[i] = slab_dims[i];
	}
	dset_offset[0]   = offset_Nx;

	// Select the hyperslab in the dataset on file to write to
	if ((H5Sselect_hyperslab(dset_space, H5S_SELECT_SET, dset_offset, NULL, dset_slabsize, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to select hyperslab in file for datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// --------------------------------------
	// Set Up Collective Write & Write Data
	// --------------------------------------
	// Set up Collective write property
	plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	// Write data to file
	if ((H5Dwrite(file_space, dtype, mem_space, dset_space, plist_id, data)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to write data to datset ["CYAN"%s"RESET"] at: Iter = ["CYAN"%d"RESET"] t = ["CYAN"%lf"RESET"]\n-->> Exiting...\n", dset_name, iters, t);
		exit(1);		
	}

	// -------------------------------
	// Close identifiers
	// -------------------------------
	H5Pclose(plist_id);
	H5Dclose(file_space);
	H5Sclose(dset_space);
	H5Sclose(mem_space);
}
/**
 * Wrapper function used to create a Group for the current iteration in the HDF5 file 
 * @param  group_name The name of the group - will be the Iteration counter
 * @param  t          The current time in the simulation
 * @param  dt         The current timestep being used
 * @param  iters      The current iteration counter
 * @return            Returns a hid_t identifier for the created group
 */
hid_t CreateGroup(hid_t file_handle, char* filename, char* group_name, double t, double dt, long int iters) {

	// Initialize variables
	herr_t status;
	hid_t attr_id;
	hid_t group_id;
	hid_t attr_space;
	static const hsize_t attrank = 1;
	hsize_t attr_dims[attrank];

	// -------------------------------
	// Create the group
	// -------------------------------
	// Check if group exists
	if (H5Lexists(file_handle, group_name, H5P_DEFAULT)) {		
		// Open group if it already exists
		group_id = H5Gopen(file_handle, group_name, H5P_DEFAULT);
	}
	else {
		// If not create new group and add time data as attribute to Group
		group_id = H5Gcreate(file_handle, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);	

		// -------------------------------
		// Write Timedata as Attribute
		// -------------------------------
		// Create attribute datatspace
		attr_dims[0] = 1;
		attr_space   = H5Screate_simple(attrank, attr_dims, NULL); 	

		// Create attribute for current time in the integration
		attr_id = H5Acreate(group_id, "TimeValue", H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
		if ((H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &t)) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not write current time as attribute to group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
			exit(1);
		}
		status = H5Aclose(attr_id);
		if (status < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
			exit(1);
		}
		// Create attribute for the current timestep
		attr_id = H5Acreate(group_id, "TimeStep", H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
		if ((H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &dt)) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not write current timestep as attribute to group in file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
			exit(1);
		}


		// -------------------------------
		// Close the attribute identifiers
		// -------------------------------
		status = H5Aclose(attr_id);
		if (status < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
			exit(1);
		}
		status = H5Sclose(attr_space);
		if (status < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close attribute Idenfiers for file ["CYAN"%s"RESET"] at: t = ["CYAN"%lf"RESET"] Iter = ["CYAN"%ld"RESET"]!!\n-->> Exiting...\n", filename, t, iters);
			exit(1);
		}
	}

	return group_id;
}
/**
 * Function to create a HDF5 datatype for complex data
 */
hid_t CreateComplexDatatype(void) {

	// Declare HDF5 datatype variable
	hid_t dtype;

	// error handling var
	herr_t status;
	
	// Create complex struct
	struct complex_type_tmp cmplex;
	cmplex.re = 0.0;
	cmplex.im = 0.0;

	// create complex compound datatype
	dtype  = H5Tcreate(H5T_COMPOUND, sizeof(cmplex));

	// Insert the real part of the datatype
  	status = H5Tinsert(dtype, "r", offsetof(complex_type_tmp,re), H5T_NATIVE_DOUBLE);
  	if (status < 0) {
  		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert real part for the Complex Compound Datatype!!\nExiting...\n");
  		exit(1);
  	}

  	// Insert the imaginary part of the datatype
  	status = H5Tinsert(dtype, "i", offsetof(complex_type_tmp,im), H5T_NATIVE_DOUBLE);
  	if (status < 0) {
  		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert imaginary part for the Complex Compound Datatype! \n-->>Exiting...\n");
  		exit(1);
  	}

  	return dtype;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------