/**
* @file hdf5_funcs.c  
* @author Enda Carroll
* @date Sept 2021
* @brief File containing HDF5 function wrappers for creating, opening, wrtining to and closing input/output HDF5 file
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
#include "data_types.h"
#include "hdf5_func.h"
// #include "utils.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function to open the input file, read in simulation data and initialize some of the system parameters
 */
void OpenInputAndInitialize(void) {

	// Initialize variables
	hid_t dset;
	hid_t dspace;
	herr_t status;
	hsize_t Dims[SYS_DIM + 1];
	int snaps = 0;
	char group_string[64];

	// --------------------------------
	//  Create Complex Datatype
	// --------------------------------
	// Create compound datatype for the complex datasets
	file_info->COMPLEX_DTYPE = CreateComplexDatatype();
	

	strcpy(file_info->input_dir, "/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/Dublin_Results/RESULTS_3D/RESULTS_NAVIER_AB4_N[256][256][256]_T[0-20]_[00-20-27]_[ORNU_H256]/");
	// strcpy(file_info->input_dir, "/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/Dublin_Results/RESULTS_3D/RESULTS_NAVIER_AB4_N[256][256][256]_T[0-100]_[11-06-55]_[KOLO4_Re200_N256_contd]/");
	strcpy(file_info->output_dir, "./Data/Stats/");
	strcpy(file_info->output_tag, "TestAlt");



	// --------------------------------
	//  Get Input File Path
	// --------------------------------
	if (strcmp(file_info->input_dir, "NONE") != 0) {
		// If input folder construct input file path
		strcpy(file_info->input_file_name, file_info->input_dir);
		strcat(file_info->input_file_name, "HDF_Global_FOURIER.h5");

		// Print input file path to screen
		printf("\nInput File: "CYAN"%s"RESET"\n\n", file_info->input_file_name);
	}
	else {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- No input directory/file provided. Please provide input directory/file - see utils.c\n-->> Exiting...\n");
		exit(1);
	}
	

	// --------------------------------
	//  Open File
	// --------------------------------
	// Check if file exists
	if (access(file_info->input_file_name, F_OK) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Input file ["CYAN"%s"RESET"] does not exist\n-->> Exiting...\n", file_info->input_file_name);
		exit(1);
	}
	else {
		// Open file
		file_info->input_file_handle = H5Fopen(file_info->input_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->input_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open input file ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->input_file_name);
			exit(1);
		}
	}


	// --------------------------------
	//  Get Number of Snaps
	// --------------------------------
	// Count the number of snapshot groups in file
	printf("Checking Number of Snapshots ");
	for(int i = 0; i < (int) 1e6; ++i) {
		// Check for snap
		sprintf(group_string, "/Timestep_%04d", i);	
		if(H5Lexists(file_info->input_file_handle, group_string, H5P_DEFAULT) > 0 ) {
			snaps++;
		}
	}
	// Print total number of snaps to screen
	printf("\nTotal Snapshots: ["CYAN"%d"RESET"]\n\n", snaps);

	// Save the total number of snaps
	sys_vars->num_snaps = snaps;

	// --------------------------------
	//  Get System Dimensions
	// --------------------------------
	// Open dataset
	sprintf(group_string, "/Timestep_%04d/W_hat", 0);	
	if((dset = H5Dopen(file_info->input_file_handle, group_string , H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset ["CYAN"%s"RESET"]\n-->> Exiting...\n", group_string);
		exit(1);
	}

	// Get dataspace handle
	dspace = H5Dget_space(dset); 

	// Get dims from dataspace
	if(H5Sget_simple_extent_ndims(dspace) != SYS_DIM + 1) {
 	  fprintf(stderr, "\n["RED"ERROR"RESET"] --- Number of dimensions in HDF5 Datasets ["CYAN"%s"RESET"] is not 2!\n-->> Exiting...\n", group_string);
		exit(1);		 
	}
	if((H5Sget_simple_extent_dims(dspace, Dims, NULL)) < 0 ) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- unable to get data extents(dimensions) from HDF5 dataset ["CYAN"%s"RESET"]\n-->> Exiting...\n", group_string);
		exit(1);		
	}	

	// Record system dims
	sys_vars->N[0] = (long int)Dims[0];
	sys_vars->N[1] = (long int)Dims[1];
	sys_vars->N[2] = ((long int)Dims[2] - 1) * 2;

	// Close identifiers
	status = H5Dclose(dset);
	status = H5Sclose(dspace);
	

	// --------------------------------
	//  Read In/Initialize Space Arrays
	// --------------------------------
	// Allocate memory for real space
	run_data->x[0] = (double* )fftw_malloc(sizeof(double) * sys_vars->N[0]);
	run_data->x[1] = (double* )fftw_malloc(sizeof(double) * sys_vars->N[1]);
	run_data->x[2] = (double* )fftw_malloc(sizeof(double) * sys_vars->N[2]);
	if(run_data->x[0] == NULL || run_data->x[1] == NULL || run_data->x[2] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "collocation points");
		exit(1);
	}
	for (int i = 0; i < sys_vars->N[0]; ++i) {
		run_data->x[0][i] = 0.0 + i * (2.0 * M_PI / sys_vars->N[0]);
		run_data->x[1][i] = 0.0 + i * (2.0 * M_PI / sys_vars->N[0]);
		run_data->x[2][i] = 0.0 + i * (2.0 * M_PI / sys_vars->N[0]);
	}

	// Allocate memory for fourier space
	run_data->k[0] = (int* )fftw_malloc(sizeof(int) * sys_vars->N[0]);
	run_data->k[1] = (int* )fftw_malloc(sizeof(int) * (sys_vars->N[1]));
	run_data->k[2] = (int* )fftw_malloc(sizeof(int) * (sys_vars->N[2] / 2 + 1));
	if(run_data->k[0] == NULL || run_data->k[1] == NULL || run_data->k[2] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "wavenumber list");
		exit(1);
	}

	// Read in space arrays if they exist in input file if not intialize them
	if((H5Lexists(file_info->input_file_handle, "Timestep_0000/kx", H5P_DEFAULT) > 0) && (H5Lexists(file_info->input_file_handle, "Timestep_0000/ky", H5P_DEFAULT) > 0) && (H5Lexists(file_info->input_file_handle, "Timestep_0000/kz", H5P_DEFAULT) > 0)) {
		// Read in Fourier space arrays
		if(H5LTread_dataset(file_info->input_file_handle, "Timestep_0000/kx", H5T_NATIVE_INT, run_data->k[0]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "kx");
			exit(1);	
		}
		if(H5LTread_dataset(file_info->input_file_handle, "Timestep_0000/ky", H5T_NATIVE_INT, run_data->k[1]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "ky");
			exit(1);
		}
		if(H5LTread_dataset(file_info->input_file_handle, "Timestep_0000/kz", H5T_NATIVE_INT, run_data->k[2]) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "kz");
			exit(1);
		}
	}


	// --------------------------------
	//  Close HDF5 Identifiers
	// --------------------------------
	status = H5Fclose(file_info->input_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close input file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->input_file_name, "initial");
		exit(1);		
	}
}
/**
 * Function to read in the solver data for the current snapshot. If certain data does not exist but is needed it is computed here e.g. real space vorticity and velocity
 * @param snap_indx The index of the currrent snapshot
 */
void ReadInData(int snap_indx) {

	// Initialize variables
	int indx, tmp1, tmp2;
	const long int Nx 		  = sys_vars->N[0];
	const long int Ny 		  = sys_vars->N[1];
	const long int Nz 		  = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	char group_string_w[64];
	hid_t dset;
	herr_t status;
	double k_fac;

	// --------------------------------
	//  Open File
	// --------------------------------
	// Check if file exists
	if (access(file_info->input_file_name, F_OK) != 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Input file ["CYAN"%s"RESET"] does not exist\n-->> Exiting...\n", file_info->input_file_name);
		exit(1);
	}
	else {
		// Open file
		file_info->input_file_handle = H5Fopen(file_info->input_file_name, H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_info->input_file_handle < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open input file ["CYAN"%s"RESET"]\n-->> Exiting...\n", file_info->input_file_name);
			exit(1);
		}
	}

	// --------------------------------
	//  Read in Fourier Vorticity
	// --------------------------------
	// Open Fourier space vorticity
	sprintf(group_string_w, "/Timestep_%04d/W_hat", snap_indx);	
	if (H5Lexists(file_info->input_file_handle, group_string_w, H5P_DEFAULT) > 0 ) {
		// Open the dataset
		dset = H5Dopen (file_info->input_file_handle, group_string_w, H5P_DEFAULT);
		if (dset < 0 ) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to open dataset for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "W_hat", snap_indx);
			exit(1);		
		} 
	
		// Read in Fourier space vorticity
		if(H5LTread_dataset(file_info->input_file_handle, group_string_w, file_info->COMPLEX_DTYPE, run_data->w_hat_tmp) < 0) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to read in data for ["CYAN"%s"RESET"] at Snap = ["CYAN"%d"RESET"]\n-->> Exiting!!!\n", "W_hat", snap_indx);
			exit(1);	
		}
	}
	else {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to find ["CYAN"%s"RESET"] in file ["CYAN"%s"RESET"]. Please check input file\n-->> Exiting...\n", "W_hat", file_info->input_file_name);
		exit(1);
	}

	// --------------------------------
	//  Get Real Space Fields
	// --------------------------------
	// Compute the Fourier Velocity from the Vorticity
	for (int i = 0; i < Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Compute the Fourier velocity (store in temp array for transform)
				if (run_data->k[0][i] == 0 && run_data->k[1][j] == 0 && run_data->k[2][k] == 0) {
					// Get the zero mode
					run_data->u_hat_tmp[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
					run_data->u_hat_tmp[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
					run_data->u_hat_tmp[SYS_DIM * indx + 2] = 0.0 + 0.0 * I;
				}
				else {
					k_fac = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + run_data->k[2][k] * run_data->k[2][k]);

					// Get the fourier space velocity
					run_data->u_hat_tmp[SYS_DIM * indx + 0] = I * (1.0 / k_fac) * (run_data->k[1][j] * run_data->w_hat_tmp[SYS_DIM * indx + 2] - run_data->k[2][k] * run_data->w_hat_tmp[SYS_DIM * indx + 1]);
					run_data->u_hat_tmp[SYS_DIM * indx + 1] = I * (1.0 / k_fac) * (run_data->k[2][k] * run_data->w_hat_tmp[SYS_DIM * indx + 0] - run_data->k[0][i] * run_data->w_hat_tmp[SYS_DIM * indx + 2]);
					run_data->u_hat_tmp[SYS_DIM * indx + 2] = I * (1.0 / k_fac) * (run_data->k[0][i] * run_data->w_hat_tmp[SYS_DIM * indx + 1] - run_data->k[1][j] * run_data->w_hat_tmp[SYS_DIM * indx + 0]);	
				}

				// Write it to the array
				run_data->u_hat[SYS_DIM * indx + 0] = run_data->u_hat_tmp[SYS_DIM * indx + 0];
				run_data->u_hat[SYS_DIM * indx + 1] = run_data->u_hat_tmp[SYS_DIM * indx + 1];
				run_data->u_hat[SYS_DIM * indx + 2] = run_data->u_hat_tmp[SYS_DIM * indx + 2];
			}
		}
	}

	// Transform back to real space
	fftw_execute_dft_c2r(sys_vars->fftw_3d_dft_batch_c2r, run_data->u_hat_tmp, run_data->u);
	// fftw_execute_dft_c2r(sys_vars->fftw_3d_dft_batch_c2r, run_data->w_hat_tmp, run_data->w);

	// --------------------------------
	//  Close HDF5 Identifiers
	// --------------------------------
	status = H5Dclose(dset);
	status = H5Fclose(file_info->input_file_handle);
	if (status < 0) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to close input file ["CYAN"%s"RESET"] at: Snap = ["CYAN"%d"RESET"]\n-->> Exiting...\n", file_info->input_file_name, snap_indx);
		exit(1);		
	}
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
  	status = H5Tinsert(dtype, "real", offsetof(complex_type_tmp,re), H5T_NATIVE_DOUBLE);
  	if (status < 0) {
  		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert real part for the Complex Compound Datatype!!\nExiting...\n");
  		exit(1);
  	}

  	// Insert the imaginary part of the datatype
  	status = H5Tinsert(dtype, "imag", offsetof(complex_type_tmp,im), H5T_NATIVE_DOUBLE);
  	if (status < 0) {
  		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Could not insert imaginary part for the Complex Compound Datatype! \n-->>Exiting...\n");
  		exit(1);
  	}

  	return dtype;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------