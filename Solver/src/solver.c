/**
* @file solver.c 
* @author Enda Carroll
* @date Feb 2022
* @brief file containing the main functions used in the pseudopectral method
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "hdf5_funcs.h"
#include "utils.h"
#include "solver.h"

// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
// Define RK4 variables
#if defined(__RK4)
static const double RK4_C2 = 0.5, 	  RK4_A21 = 0.5, \
				  	RK4_C3 = 0.5,           					RK4_A32 = 0.5, \
				  	RK4_C4 = 1.0,                      									   RK4_A43 = 1.0, \
				              	 	  RK4_B1 = 1.0/6.0, 		RK4_B2  = 1.0/3.0, 		   RK4_B3  = 1.0/3.0, 		RK4_B4 = 1.0/6.0;
// Define RK5 Dormand Prince variables
#elif defined(__RK5) || defined(__DPRK5)
static const double RK5_C2 = 0.2, 	  RK5_A21 = 0.2, \
				  	RK5_C3 = 0.3,     RK5_A31 = 3.0/40.0,       RK5_A32 = 0.5, \
				  	RK5_C4 = 0.8,     RK5_A41 = 44.0/45.0,      RK5_A42 = -56.0/15.0,	   RK5_A43 = 32.0/9.0, \
				  	RK5_C5 = 8.0/9.0, RK5_A51 = 19372.0/6561.0, RK5_A52 = -25360.0/2187.0, RK5_A53 = 64448.0/6561.0, RK5_A54 = -212.0/729.0, \
				  	RK5_C6 = 1.0,     RK5_A61 = 9017.0/3168.0,  RK5_A62 = -355.0/33.0,     RK5_A63 = 46732.0/5247.0, RK5_A64 = 49.0/176.0,    RK5_A65 = -5103.0/18656.0, \
				  	RK5_C7 = 1.0,     RK5_A71 = 35.0/384.0,								   RK5_A73 = 500.0/1113.0,   RK5_A74 = 125.0/192.0,   RK5_A75 = -2187.0/6784.0,    RK5_A76 = 11.0/84.0, \
				              		  RK5_B1  = 35.0/384.0, 							   RK5_B3  = 500.0/1113.0,   RK5_B4  = 125.0/192.0,   RK5_B5  = -2187.0/6784.0,    RK5_B6  = 11.0/84.0, \
				              		  RK5_Bs1 = 5179.0/57600.0, 						   RK5_Bs3 = 7571.0/16695.0, RK5_Bs4 = 393.0/640.0,   RK_Bs5  = -92097.0/339200.0, RK5_Bs6 = 187.0/2100.0, RK5_Bs7 = 1.0/40.0;
#endif
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Main function that performs the pseudospectral solver
 */
void SpectralSolve(void) {

	// Initialize variables
	const long int N[SYS_DIM]      	= {sys_vars->N[0], sys_vars->N[1], sys_vars->N[2]};
	const long int NBatch[SYS_DIM] 	= {sys_vars->N[0], sys_vars->N[1], sys_vars->N[2] / 2 + 1};
	const long int NTBatch[SYS_DIM] = {sys_vars->N[0], sys_vars->N[1], sys_vars->N[2] / 2 + 1};

	// Initialize the Runge-Kutta struct
	struct RK_data_struct* RK_data;	   // Initialize pointer to a RK_data_struct
	struct RK_data_struct RK_data_tmp; // Initialize a RK_data_struct
	RK_data = &RK_data_tmp;		       // Point the ptr to this new RK_data_struct

	// -------------------------------
	// Allocate memory
	// -------------------------------
	AllocateMemory(NBatch, RK_data);

	// -------------------------------
	// FFTW Plans Setup
	// -------------------------------
	InitializeFFTWPlans(N, NTBatch);

	// -------------------------------
	// Initialize the System
	// -------------------------------
	// Initialize the collocation points and wavenumber space 
	InitializeSpaceVariables(run_data->x, run_data->k, N);
	PrintSpaceVariables(N);

	// Get initial conditions
	InitialConditions(run_data->w_hat, run_data->u, run_data->u_hat, N);

	// -------------------------------
	// Integration Variables
	// -------------------------------
	// Initialize integration variables
	double t0;
	double t;
	double dt;
	double T;
	long int trans_steps;

	// Get timestep and other integration variables
	// InitializeIntegrationVariables(&t0, &t, &dt, &T, &trans_steps);
	
	// -------------------------------
	// Create & Open Output File
	// -------------------------------
	// Inialize system measurables
	InitializeSystemMeasurables(RK_data);
	   
	// Create and open the output file - also write initial conditions to file
	CreateOutputFilesWriteICs(N, dt);

	// -------------------------------------------------
	// Print IC to Screen 
	// -------------------------------------------------
	// #if defined(__PRINT_SCREEN)
	// PrintUpdateToTerminal(0, t0, dt, T, 0);
	// #endif

	//////////////////////////////
	// Begin Integration
	//////////////////////////////
	t += dt;
	long int iters = 1;
	while(t <= T) {

		// -------------------------------	
		// Perform Integration Step
		// -------------------------------
		

		// -------------------------------
		// Write To File
		// -------------------------------


		// -------------------------------
		// Print Update To Screen
		// -------------------------------
		
		
		// -------------------------------
		// Update & System Check
		// -------------------------------
		// Update timestep & iteration counter
		iters++;
		t = iters * dt;
	}
	//////////////////////////////
	// End Integration
	//////////////////////////////
	

	// -------------------------------
	// Clean Up 
	// -------------------------------
	FreeMemory(RK_data);
}
/**
 * Function to initialize all the integration time variables
 * @param t0           The initial time of the simulation
 * @param t            The current time of the simulaiton
 * @param dt           The timestep
 * @param T            The final time of the simulation
 * @param trans_steps  The number of iterations to perform before saving to file begins
 */
void InitializeIntegrationVariables(double* t0, double* t, double* dt, double* T, long int* trans_steps) {
	
	// -------------------------------
	// Get the Timestep
	// -------------------------------
	#if defined(__ADAPTIVE_STEP)
	GetTimestep(&(sys_vars->dt));
	#endif

	// -------------------------------
	// Get Time variables
	// -------------------------------
	// Compute integration time variables
	(*t0) = sys_vars->t0;
	(*t ) = sys_vars->t0;
	(*dt) = sys_vars->dt;
	(*T ) = sys_vars->T;
	sys_vars->min_dt = 10;
	sys_vars->max_dt = MIN_STEP_SIZE;

	// -------------------------------
	// Integration Counters
	// -------------------------------
	// Number of time steps and saving steps
	sys_vars->num_t_steps = ((*T) - (*t0)) / (*dt);
	#if defined(TRANSIENTS)
	// Get the transient iterations
	(* trans_steps)       = (long int)(TRANS_FRAC * sys_vars->num_t_steps);
	sys_vars->trans_iters = (* trans_steps);

	// Get the number of steps to perform before printing to file -> allowing for a transient fraction of these to be ignored
	sys_vars->num_print_steps = (sys_vars->num_t_steps >= sys_vars->SAVE_EVERY ) ? (sys_vars->num_t_steps - sys_vars->trans_iters) / sys_vars->SAVE_EVERY : sys_vars->num_t_steps - sys_vars->trans_iters;	 
	if (!(sys_vars->rank)){
		printf("Total Iters: %ld\t Saving Iters: %ld\t Transient Steps: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps, sys_vars->trans_iters);
	}
	#else
	// Get the transient iterations
	(* trans_steps)       = 0;
	sys_vars->trans_iters = (* trans_steps);

	// Get the number of steps to perform before printing to file
	sys_vars->num_print_steps = (sys_vars->num_t_steps >= sys_vars->SAVE_EVERY ) ? sys_vars->num_t_steps / sys_vars->SAVE_EVERY + 1 : sys_vars->num_t_steps + 1; // plus one to include initial condition
	if (!(sys_vars->rank)){
		printf("Total Iters: %ld\t Saving Iters: %ld\n", sys_vars->num_t_steps, sys_vars->num_print_steps);
	}
	#endif

	// Variable to control how ofter to print to screen -> set it to half the saving to file steps
	sys_vars->print_every = (sys_vars->num_t_steps >= 10 ) ? (int)sys_vars->SAVE_EVERY : 1;
}
/**
 * Function to initializes and computes the system measurables and spectra of the initial conditions
 * @param RK_data The struct containing the Runge-Kutta arrays to compute the nonlinear term for the fluxes
 */
void InitializeSystemMeasurables(RK_data_struct* RK_data) {

	// Set the size of the arrays to twice the number of printing steps to account for extra steps due to adaptive stepping
	#if defined(__ADAPTIVE_STEP)
	sys_vars->num_print_steps = 2 * sys_vars->num_print_steps;
	#else
	sys_vars->num_print_steps = sys_vars->num_print_steps;
	#endif
	int print_steps = sys_vars->num_print_steps;

	// Get the size of the spectrum arrays
	#if defined(__ENST_SPECT) || defined(__ENRG_SPECT) || defined(__ENST_FLUX_SPECT) || defined(__ENRG_FLUX_SPECT)
	const long int Nx = sys_vars->N[0];
	const long int Ny = sys_vars->N[1];
	const long int Nz = sys_vars->N[2];

	sys_vars->n_spect = (int) sqrt(pow((double)Nx / 2.0, 2.0) + pow((double)Ny / 2.0, 2.0) + pow((double)Nz / 2.0, 2.0)) + 1;
	int n_spect = sys_vars->n_spect;
	#endif
		
	// ------------------------
	// Allocate Memory
	// ------------------------
	#if defined(__SYS_MEASURES)
	// Total Energy in the system
	run_data->tot_energy = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_energy == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Energy");
		exit(1);
	}	

	// Total Enstrophy
	run_data->tot_enstr = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_enstr == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Enstrophy");
		exit(1);
	}	

	// Total Palinstrophy
	run_data->tot_palin = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_palin == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Palinstrophy");
		exit(1);
	}	

	// Energy Dissipation Rate
	run_data->enrg_diss = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_diss == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Rate");
		exit(1);
	}	

	// Enstrophy Dissipation Rate
	run_data->enst_diss = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_diss == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Rate");
		exit(1);
	}	
	#endif
	#if defined(__ENST_SPECT )
	// Enstrophy Spectrum
	run_data->enst_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enst_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Spectrum");
		exit(1);
	}	
	#endif
	#if defined(__ENRG_SPECT )
	// Energy Spectrum
	run_data->enrg_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Spectrum");
		exit(1);
	}	
	#endif
	#if defined(__ENST_FLUX)
	// Enstrophy flux
	run_data->enst_flux_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_flux_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Subset");
		exit(1);
	}	

	// Enstrophy Dissipation Rate
	run_data->enst_diss_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enst_diss_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Dissipation Rate Subset");
		exit(1);
	}	
	#endif
	#if defined(__ENRG_FLUX)
	// Energy Flux
	run_data->enrg_flux_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_flux_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Flux Subset");
		exit(1);
	}	

	// Energy Dissipation Rate
	run_data->enrg_diss_sbst = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_diss_sbst == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Rate Subset");
		exit(1);
	}
	#endif
	// Time
	#if defined(__TIME)
	if (!(sys_vars->rank)){
		run_data->time = (double* )fftw_malloc(sizeof(double) * print_steps);
		if (run_data->time == NULL) {
			fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Time");
			exit(1);
		}	
	}
	#endif
	#if defined(__ENST_FLUX_SPECT )
	// Enstrophy Spectrum
	run_data->enst_flux_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enst_flux_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}	
	#endif
	#if defined(__ENRG_FLUX_SPECT )
	// Energy Spectrum
	run_data->enrg_flux_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_flux_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Flux Spectrum");
		exit(1);
	}	
	#endif

	// // ----------------------------
	// // Get Measurables of the ICs
	// // ----------------------------
	// #if defined(__SYS_MEASURES)
	// // Total Energy
	// run_data->tot_energy[0] = TotalEnergy();

	// // Total Enstrophy
	// run_data->tot_enstr[0] = TotalEnstrophy();

	// // Total Palinstrophy
	// run_data->tot_palin[0] = TotalPalinstrophy();

	// // Energy dissipation rate
	// run_data->enrg_diss[0] = EnergyDissipationRate();

	// // Enstrophy dissipation rate
	// run_data->enst_diss[0] = EnstrophyDissipationRate();
	// #endif
	// #if defined(__ENST_FLUX)
	// // Enstrophy Flux and dissipation from/to Subset of modes
	// EnstrophyFlux(&(run_data->enst_flux_sbst[0]), &(run_data->enst_diss_sbst[0]), RK_data);
	// #endif
	// #if defined(__ENRG_FLUX)
	// // Energy Flux and dissipation from/to a subset of modes
	// EnergyFlux(&(run_data->enrg_flux_sbst[0]), &(run_data->enrg_diss_sbst[0]), RK_data);
	// #endif
	// // Time
	// #if defined(__TIME)
	// if (!(sys_vars->rank)) {
	// 	run_data->time[0] = sys_vars->t0;
	// }
	// #endif

	// // ----------------------------
	// // Get Spectra of the ICs
	// // ----------------------------
	// // Call spectra functions
	// #if defined(__ENST_SPECT)
	// EnstrophySpectrum();
	// #endif
	// #if defined(__ENRG_SPECT)
	// EnergySpectrum();
	// #endif
	// #if defined(__ENRG_FLUX_SPECT)
	// EnergyFluxSpectrum(RK_data);
	// #endif
	// #if defined(__ENST_FLUX_SPECT)
	// EnstrophyFluxSpectrum(RK_data);
	// #endif
}
/**
 * Function to compute the initial condition for the integration
 * @param w_hat Fourier space vorticity
 * @param u     Real space velocities in batch layout - both u and v
 * @param u_hat Fourier space velocities in batch layout - both u_hat and v_hat
 * @param N     Array containing the dimensions of the system
 */
void InitialConditions(fftw_complex* w_hat, double* u, fftw_complex* u_hat, const long int* N) {

	// Initialize variables
	int tmp1, tmp2, indx;
	const long int Nx         = N[0];
	const long int Ny 		  = N[1];
	const long int Nz 		  = N[2];
	const long int Nz_Fourier = N[2] / 2 + 1; 

	// Initialize local variables 
	ptrdiff_t local_Nx = sys_vars->local_Nx;

    // ------------------------------------------------
    // Set Seed for RNG
    // ------------------------------------------------
    srand(123456789);

    if(!(strcmp(sys_vars->u0, "TAYLOR_GREEN"))) {
    	// ------------------------------------------------
    	// Taylor Green Initial Condition - Real Space
    	// ------------------------------------------------
    	for (int i = 0; i < local_Nx; ++i) {
    		tmp1 = i * Ny;
    		for (int j = 0; j < Ny; ++j) {
    			tmp2 = (Nz + 2) * (tmp1 + j);
    			for (int k = 0; k < Nz; ++k) {
    				indx = tmp2 + k;
    				
    				// Fill the velocities
    				u[SYS_DIM * indx + 0] = sin(KAPPA * run_data->x[0][i]) * cos(KAPPA * run_data->x[1][j]) * cos(KAPPA * run_data->x[2][k]);
    				u[SYS_DIM * indx + 1] = -cos(KAPPA * run_data->x[0][i]) * sin(KAPPA * run_data->x[1][j]) * cos(KAPPA * run_data->x[2][k]);			
    				u[SYS_DIM * indx + 2] = 0.0;			
    			}
    		}
    	}

    	// Transform velocities to Fourier space & dealias
    	fftw_mpi_execute_dft_r2c(sys_vars->fftw_3d_dft_batch_r2c, u, u_hat);
    	ApplyDealiasing(u_hat, SYS_DIM, N);
    }
    else if(!(strcmp(sys_vars->u0, "TESTING"))) {
    	// ------------------------------------------------
    	// Testing Initial Condition
    	// ------------------------------------------------
    }
    else {
		printf("\n["MAGENTA"WARNING"RESET"] --- No initial conditions specified\n---> Using random initial conditions...\n");
		// ---------------------------------------
		// Random Initial Conditions
		// ---------------------------------------
		for (int i = 0; i < local_Nx; ++i) {	
			tmp1 = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				tmp2 = Nz_Fourier * (tmp1 + j);
				for (int k = 0; k < Nz_Fourier; ++k) {
					indx = tmp2 + k;

					// Fill vorticity
					u_hat[SYS_DIM * indx + 0] = ((double)rand() / (double) RAND_MAX) * cexp(((double)rand() / (double) RAND_MAX) * 2.0 * M_PI * I);
					u_hat[SYS_DIM * indx + 1] = ((double)rand() / (double) RAND_MAX) * cexp(((double)rand() / (double) RAND_MAX) * 2.0 * M_PI * I);
					u_hat[SYS_DIM * indx + 2] = ((double)rand() / (double) RAND_MAX) * cexp(((double)rand() / (double) RAND_MAX) * 2.0 * M_PI * I);
				}
			}
		}		
   	}

   	// -------------------------------------------------
   	// Initialize the Dealiasing
   	// -------------------------------------------------
   	// ApplyDealiasing(w_hat, 1, N);
   	   
   	   
   	// -------------------------------------------------
   	// Initialize the Forcing
   	// -------------------------------------------------
   	// ApplyForcing(w_hat, N);
   	  
   	  
   	// -------------------------------------------------
   	// Initialize Taylor Green Vortex Soln 
   	// -------------------------------------------------
   	// If testing is enabled and TG initial condition selected -> compute TG solution for writing to file @ t = t0
   	// #if defined(TESTING)
   	// if(!(strcmp(sys_vars->u0, "TG_VEL")) || !(strcmp(sys_vars->u0, "TG_VORT"))) {
   	// 	TaylorGreenSoln(0.0, N);
   	// }
   	// #endif
}
/**
 * Function to apply the selected dealiasing filter to the input array. Can be Fourier vorticity or velocity
 * @param array    	The array containing the Fourier modes to dealiased
 * @param array_dim The extra array dimension -> will be 1 for scalar or 2 for vector
 * @param N        	Array containing the dimensions of the system
 */
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N) {

	// Initialize variables
	int tmp1, tmp2, indx;
	ptrdiff_t local_Nx        = sys_vars->local_Nx;
	const long int Nx         = N[0];
	const long int Ny         = N[1];
	const long int Nz         = N[2];
	const long int Nz_Fourier = Ny / 2 + 1;
	#if defined(__DEALIAS_HOU_LI)
	double hou_li_filter;
	#endif

	// --------------------------------------------
	// Apply Appropriate Filter 
	// --------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = (tmp2 + k);

				#if defined(__DEALIAS_23)
				if (sqrt((double) run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + run_data->k[2][k] * run_data->k[2][k]) > Nx / 3) {
					for (int l = 0; l < array_dim; ++l) {
						// Set dealised modes to 0
						array[array_dim * indx + l] = 0.0 + 0.0 * I;	
					}
				}
				else {
					for (int l = 0; l < array_dim; ++l) {
						// Apply DFT normaliztin to undealiased modes
						array[array_dim * indx + l] = array[indx + l];	
					}				
				}
				#elif __DEALIAS_HOU_LI
				// Compute Hou-Li filter
				hou_li_filter = exp(-36.0 * pow((sqrt(pow(run_data->k[0][i] / (Nx / 2), 2.0) + pow(run_data->k[1][j] / (Ny / 2), 2.0) + pow(run_data->k[2][k] / (Nz / 2), 2.0))), 36.0));

				for (int l = 0; l < array_dim; ++l) {
					// Apply filter and DFT normaliztion
					array[array_dim * indx + l] *= hou_li_filter;
				}
				#endif
			}
		}
	}
}	
/**
 * Function to initialize the Real space collocation points arrays and Fourier wavenumber arrays
 * 
 * @param x Array containing the collocation points in real space
 * @param k Array to contain the wavenumbers on both directions
 * @param N Array containging the dimensions of the system
 */
void InitializeSpaceVariables(double** x, int** k, const long int* N) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Nz = N[2];
	const long int Nz_Fourier = N[2] / 2 + 1;

	// Initialize local variables 
	ptrdiff_t local_Nx       = sys_vars->local_Nx;
	ptrdiff_t local_Nx_start = sys_vars->local_Nx_start;

	// Set the spatial increments
	sys_vars->dx = 2.0 * M_PI / (double )Nx;
	sys_vars->dy = 2.0 * M_PI / (double )Ny;
	sys_vars->dz = 2.0 * M_PI / (double )Nz;

	// -------------------------------
	// Fill the first dirction 
	// -------------------------------
	int j = 0;
	for (int i = 0; i < Nx; ++i) {
		if((i >= local_Nx_start) && ( i < local_Nx_start + local_Nx)) { // Ensure each process only writes to its local array slice
			x[0][j] = (double) i * sys_vars->dx;
			j++;
		}
	}
	j = 0;
	for (int i = 0; i < local_Nx; ++i) {
		if (local_Nx_start + i <= Nx / 2) {   // Set the first half of array to the positive k
			k[0][j] = local_Nx_start + i;
			j++;
		}
		else if (local_Nx_start + i > Nx / 2) { // Set the second half of array to the negative k
			k[0][j] = local_Nx_start + i - Nx;
			j++;
		}
	}

	// -------------------------------
	// Fill the second direction 
	// -------------------------------
	for (int i = 0; i < Ny; ++i) {
		if (i < Ny / 2 + 1) {
			k[1][i] = i;
		}
		if (i >= Ny / 2 + 1) {
			k[1][i] = -Ny + i;
		}
		x[1][i] = (double) i * sys_vars->dy;
	}

	// -------------------------------
	// Fill the third direction 
	// -------------------------------
	for (int i = 0; i < Nz; ++i) {
		if (i < Nz_Fourier) {
			k[2][i] = i;
		}
		x[2][i] = (double) i * sys_vars->dz;
	}

}
/**
 * Wrapper function used to allocate memory all the nessecary local and global system and integration arrays
 * @param NBatch  Array holding the dimensions of the Fourier space arrays
 * @param RK_data Pointer to struct containing the integration arrays
 */
void AllocateMemory(const long int* NBatch, RK_data_struct* RK_data) {

	// Initialize variables
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;	

	// -------------------------------
	// Get Local Array Sizes - FFTW 
	// -------------------------------
	//  Find the size of memory for the FFTW transforms - use these to allocate appropriate memory
	sys_vars->alloc_local       = fftw_mpi_local_size_3d(Nx, Ny, Nz_Fourier, MPI_COMM_WORLD, &(sys_vars->local_Nx), &(sys_vars->local_Nx_start));
	sys_vars->alloc_local_batch = fftw_mpi_local_size_many((int)SYS_DIM, NBatch, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, MPI_COMM_WORLD, &(sys_vars->local_Nx), &(sys_vars->local_Nx_start));
	if (sys_vars->local_Nx == 0) {
		printf("\n["MAGENTA"WARNING"RESET"] --- FFTW was unable to allocate local memory for each process -->> Code will run but will be slow\n");
	}

	// -------------------------------
	// Allocate Space Variables 
	// -------------------------------
	// Allocate the wavenumber arrays
	run_data->k[0] = (int* )fftw_malloc(sizeof(int) * sys_vars->local_Nx);  // kx
	run_data->k[1] = (int* )fftw_malloc(sizeof(int) * Ny);     			    // ky
	run_data->k[2] = (int* )fftw_malloc(sizeof(int) * Nz_Fourier);     		// kz
	if (run_data->k[0] == NULL || run_data->k[1] == NULL || run_data->k[2] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "wavenumber list");
		exit(1);
	}

	// Allocate the collocation points
	run_data->x[0] = (double* )fftw_malloc(sizeof(double) * sys_vars->local_Nx);  // x direction 
	run_data->x[1] = (double* )fftw_malloc(sizeof(double) * Ny);     			  // y direction
	run_data->x[2] = (double* )fftw_malloc(sizeof(double) * Nz);     			  // z direction
	if (run_data->x[0] == NULL || run_data->x[1] == NULL || run_data->x[2] == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "collocation points");
		exit(1);
	}

	// -------------------------------
	// Allocate System Variables 
	// -------------------------------
	// Allocate the Real and Fourier space vorticity
	run_data->w     = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (run_data->w == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Real Space Vorticity" );
		exit(1);
	}
	run_data->w_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (run_data->w_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Space Vorticity");
		exit(1);
	}

	// Allocate the Real and Fourier space velocities
	run_data->u     = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (run_data->u == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Real Space Velocities");
		exit(1);
	}
	run_data->u_hat = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (run_data->u_hat == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Fourier Space Velocities");
		exit(1);
	}

	// -------------------------------
	// Allocate Integration Variables 
	// -------------------------------
	// Runge-Kutta Integration arrays
	RK_data->RK1 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK1 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK1");
		exit(1);
	}
	RK_data->RK2 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK2 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK2");
		exit(1);
	}
	RK_data->RK3 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK3 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK3");
		exit(1);
	}
	RK_data->RK4 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK4 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK4");
		exit(1);
	}
	RK_data->RK_tmp = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * sys_vars->alloc_local_batch);
	if (RK_data->RK_tmp == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK_tmp");
		exit(1);
	}

	// -------------------------------
	// Initialize All Data 
	// -------------------------------
	int tmp2_r, tmp2_f, tmp1;
	int indx_r, indx_f;
	for (int i = 0; i < sys_vars->local_Nx; ++i) {
		tmp1 = i * Ny;
		run_data->k[0][i] = 0; 
		run_data->x[0][i] = 0.0; 
		for (int j = 0; j < Ny; ++j) {
			tmp2_f = Nz_Fourier * (tmp1 + j);
			tmp2_r = (Nz + 2) * (tmp1 + j);
			if (i == 0) {
				run_data->k[1][j] = 0; 
				run_data->x[1][j] = 0.0;
			}
			for (int k = 0; k < Nz; ++k) {
				indx_f = tmp2_f + k;
				indx_r = tmp2_r + k;
				if (i == 0 && j == 0) {
					if (k < Nz_Fourier) {
						run_data->k[2][k] = 0; 
					}
					run_data->x[2][k] = 0.0;
				}
				if (k < Nz_Fourier) {
					run_data->w_hat[SYS_DIM * indx_f + 0] = 0.0 + 0.0 * I;
					run_data->w_hat[SYS_DIM * indx_f + 1] = 0.0 + 0.0 * I;
					run_data->w_hat[SYS_DIM * indx_f + 2] = 0.0 + 0.0 * I;
					run_data->u_hat[SYS_DIM * indx_f + 0] = 0.0 + 0.0 * I;
					run_data->u_hat[SYS_DIM * indx_f + 1] = 0.0 + 0.0 * I;
					run_data->u_hat[SYS_DIM * indx_f + 2] = 0.0 + 0.0 * I;
				}
				run_data->w[SYS_DIM * indx_r + 0] = 0.0;
				run_data->w[SYS_DIM * indx_r + 1] = 0.0;
				run_data->w[SYS_DIM * indx_r + 2] = 0.0;
				run_data->u[SYS_DIM * indx_r + 0] = 0.0;
				run_data->u[SYS_DIM * indx_r + 1] = 0.0;
				run_data->u[SYS_DIM * indx_r + 2] = 0.0;
			}
		}
	}
}
/**
 * Wrapper function that initializes the FFTW plans using MPI
 * @param N Array containing the dimensions of the system
 */
void InitializeFFTWPlans(const long int* N, const long int* NTBatch) {

	// Initialize variables
	const long int Nx = N[0];
	const long int Ny = N[1];
	const long int Nz = N[2];

	// -----------------------------------
	// Initialize Plans for Vorticity 
	// -----------------------------------
	// Set up FFTW plans for normal transform - vorticity field
	sys_vars->fftw_3d_dft_r2c = fftw_mpi_plan_dft_r2c_3d(Nx, Ny, Nz, run_data->w, run_data->w_hat, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	sys_vars->fftw_3d_dft_c2r = fftw_mpi_plan_dft_c2r_3d(Nx, Ny, Nz, run_data->w_hat, run_data->w, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	if (sys_vars->fftw_3d_dft_r2c == NULL || sys_vars->fftw_3d_dft_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize basic FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}

	// -------------------------------------
	// Initialize Batch Plans for Velocity 
	// -------------------------------------
	// Set up FFTW plans for batch transform - velocity fields
	sys_vars->fftw_3d_dft_batch_r2c = fftw_mpi_plan_many_dft_r2c((int)SYS_DIM, N, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u, run_data->u_hat, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	sys_vars->fftw_3d_dft_batch_c2r = fftw_mpi_plan_many_dft_c2r((int)SYS_DIM, N, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u_hat, run_data->u, MPI_COMM_WORLD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	if (sys_vars->fftw_3d_dft_batch_r2c == NULL || sys_vars->fftw_3d_dft_batch_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize batch FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}

	// -----------------------------------------------
	// Initialize Transposed Batch Plans for Velocity 
	// -----------------------------------------------
	// Set up FFTW plans for batch transform - velocity fields
	sys_vars->fftw_3d_dft_trans_batch_r2c = fftw_mpi_plan_many_dft_r2c((int)SYS_DIM, NTBatch, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u, run_data->u_hat, MPI_COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT | FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	sys_vars->fftw_3d_dft_trans_batch_c2r = fftw_mpi_plan_many_dft_c2r((int)SYS_DIM, NTBatch, (ptrdiff_t) SYS_DIM, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK, run_data->u_hat, run_data->u, MPI_COMM_WORLD, FFTW_MPI_TRANSPOSED_IN | FFTW_MEASURE | FFTW_PRESERVE_INPUT);	
	if (sys_vars->fftw_3d_dft_trans_batch_r2c == NULL || sys_vars->fftw_3d_dft_trans_batch_c2r == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to initialize transposed batch FFTW Plans \n-->> Exiting!!!\n");
		exit(1);
	}
}
/**
 * Wrapper function that frees any memory dynamcially allocated in the programme
 * @param RK_data Pointer to a struct contaiing the integraiont arrays
 */
void FreeMemory(RK_data_struct* RK_data) {

	// ------------------------
	// Free memory 
	// ------------------------
	// Free space variables
	for (int i = 0; i < SYS_DIM; ++i) {
		fftw_free(run_data->x[i]);
		fftw_free(run_data->k[i]);
	}

	// Free system variables
	fftw_free(run_data->u);
	fftw_free(run_data->u_hat);
	fftw_free(run_data->w);
	fftw_free(run_data->w_hat);
	#if defined(__SYS_MEASURES)
	fftw_free(run_data->tot_energy);
	fftw_free(run_data->tot_enstr);
	fftw_free(run_data->tot_palin);
	fftw_free(run_data->enrg_diss);
	fftw_free(run_data->enst_diss);
	#endif
	#if defined(__ENST_FLUX)
	fftw_free(run_data->enst_flux_sbst);
	fftw_free(run_data->enst_diss_sbst);
	#endif
	#if defined(__ENRG_FLUX)
	fftw_free(run_data->enrg_flux_sbst);
	fftw_free(run_data->enrg_diss_sbst);
	#endif
	#if defined(__ENRG_SPECT)
	fftw_free(run_data->enrg_spect);
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	fftw_free(run_data->enrg_flux_spect);
	#endif
	#if defined(__ENST_SPECT)
	fftw_free(run_data->enst_spect);
	#endif
	#if defined(__ENST_FLUX_SPECT)
	fftw_free(run_data->enst_flux_spect);
	#endif
	#if defined(TESTING)
	fftw_free(run_data->tg_soln);
	#endif
	#if defined(__TIME)
	if (!(sys_vars->rank)){
		fftw_free(run_data->time);
	}
	#endif

	// Free integration arrays
	fftw_free(RK_data->RK1);
	fftw_free(RK_data->RK2);
	fftw_free(RK_data->RK3);
	fftw_free(RK_data->RK4);

	// ------------------------
	// Destroy FFTW plans 
	// ------------------------
	fftw_destroy_plan(sys_vars->fftw_3d_dft_r2c);
	fftw_destroy_plan(sys_vars->fftw_3d_dft_c2r);
	fftw_destroy_plan(sys_vars->fftw_3d_dft_batch_r2c);
	fftw_destroy_plan(sys_vars->fftw_3d_dft_batch_c2r);
	fftw_destroy_plan(sys_vars->fftw_3d_dft_trans_batch_r2c);
	fftw_destroy_plan(sys_vars->fftw_3d_dft_trans_batch_c2r);
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------