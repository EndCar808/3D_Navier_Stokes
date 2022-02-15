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
				              		  RK5_Bs1 = 5179.0/57600.0, 						   RK5_Bs3 = 7571.0/16695.0, RK5_Bs4 = 393.0/640.0,   RK5_Bs5 = -92097.0/339200.0, RK5_Bs6 = 187.0/2100.0, RK5_Bs7 = 1.0/40.0;
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
	InitialConditions(run_data->u_hat, run_data->u, N);
	// -------------------------------
	// Integration Variables
	// -------------------------------
	// Initialize integration variables
	double t0;
	double t;
	double dt;
	double T;
	long int trans_steps;
	#if defined(__DPRK5)
	int tries = 1;
	double dt_new;
	#endif

	// Get timestep and other integration variables
	InitializeIntegrationVariables(&t0, &t, &dt, &T, &trans_steps);
	
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
	#if defined(TRANSIENTS)
	long int save_data_indx = 0;
	#else
	long int save_data_indx = 1;
	#endif
	while(t <= T) {

		// -------------------------------	
		// Perform Integration Step
		// -------------------------------
		#if defined(__RK4)
		RK4Step(dt, N, sys_vars->local_Nx, RK_data);
		#elif defined(__RK5)
		RK5DPStep(dt, N, iters, sys_vars->local_Nx, RK_data);
		#elif defined(__DPRK5)
		while (tries < DP_MAX_TRY) {
			// Try a Dormand Prince step and compute the local error
			RK5DPStep(dt, N, iters, sys_vars->local_Nx, RK_data);

			// Compute the new timestep
			dt_new = dt * DPMin(DP_DELTA_MAX, DPMax(DP_DELTA_MIN, DP_DELTA * pow(1.0 / RK_data->DP_err, 0.2)));
			
			// If error is bad repeat with smaller timestep, else move on
			if (RK_data->DP_err <= 1.0) {
				RK_data->DP_fails++;
				tries++;
				dt = dt_new;
				continue;
			}
			else {
				dt = dt_new;
				break;
			}
		}
		#endif

		// -------------------------------
		// Write To File
		// -------------------------------
		if ((iters > trans_steps) && (iters % sys_vars->SAVE_EVERY == 0)) {
			#if defined(TESTING)
			TaylorGreenSoln(t, N);
			#endif

			// Record System Measurables
			RecordSystemMeasures(t, save_data_indx, RK_data);

			// Write the appropriate datasets to file
			WriteDataToFile(t, dt, save_data_indx);
			
			// Update saving data index
			save_data_indx++;
		}

		// -------------------------------
		// Print Update To Screen
		// -------------------------------
		#if defined(__PRINT_SCREEN)
		#if defined(TRANSIENTS)
		if (iters == trans_steps && !(sys_vars->rank)) {
			printf("\n\n...Transient Iterations Complete!\n\n");
		}
		#endif
		if (iters % sys_vars->SAVE_EVERY == 0) {
			// PrintUpdateToTerminal(iters, t, dt, T, save_data_indx - 1);
			if (!sys_vars->rank) printf("Iter: %ld\t dt: %1.16lf\tt: %lf\n", iters, dt, t);
		}
		#endif
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
 * Function to perform a single step of the RK5 or Dormand Prince scheme
 * @param dt       The current timestep of the system
 * @param N        Array containing the dimensions of the system
 * @param local_Nx Int indicating the local size of the first dimension of the arrays	
 * @param RK_data  Struct pointing the Integration variables: stages, tmp arrays, rhs and arrays needed for NonlinearRHS function
 */
#if defined(__RK5) || defined(__DPRK5)
void RK5DPStep(const double dt, const long int* N, const int iters, const ptrdiff_t local_Nx, RK_data_struct* RK_data) {

	// Initialize vairables
	int tmp1, tmp2;
	int indx;
	#if defined(__NAVIER)
	double k_sqr;
	double D_fac;
	#endif
	const long int Ny  		  = N[1];
	const long int Nz_Fourier = N[2] / 2 + 1;
	#if defined(__DPRK5)
	const long int Nx = N[0];
	double dp_ho_step_x, dp_ho_step_y, dp_ho_step_z;
	double err_sum_x, err_sum_y, err_sum_z;
	double err_denom_x, err_denom_y, err_denom_z;
	double err_x, err_y, err_z;
	#endif
	
	/////////////////////
	/// RK STAGES
	/////////////////////
	// ----------------------- Stage 1
	NonlinearRHSBatch(run_data->u_hat, RK_data->RK1, RK_data->curl, RK_data->vel, RK_data->vort);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Update temporary input for nonlinear term
				RK_data->RK_tmp[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + dt * RK5_A21 * RK_data->RK1[SYS_DIM * indx + 0];
				RK_data->RK_tmp[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + dt * RK5_A21 * RK_data->RK1[SYS_DIM * indx + 1];
				RK_data->RK_tmp[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + dt * RK5_A21 * RK_data->RK1[SYS_DIM * indx + 2];
			}
		}
	}
	// ----------------------- Stage 2
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK2, RK_data->curl, RK_data->vel, RK_data->vort);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Update temporary input for nonlinear term
				RK_data->RK_tmp[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + dt * RK5_A31 * RK_data->RK1[SYS_DIM * indx + 0] + dt * RK5_A32 * RK_data->RK2[SYS_DIM * indx + 0];
				RK_data->RK_tmp[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + dt * RK5_A31 * RK_data->RK1[SYS_DIM * indx + 1] + dt * RK5_A32 * RK_data->RK2[SYS_DIM * indx + 1];
				RK_data->RK_tmp[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + dt * RK5_A31 * RK_data->RK1[SYS_DIM * indx + 2] + dt * RK5_A32 * RK_data->RK2[SYS_DIM * indx + 2];
			}
		}
	}
	// ----------------------- Stage 3
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK3, RK_data->curl, RK_data->vel, RK_data->vort);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Update temporary input for nonlinear termiters, 
				RK_data->RK_tmp[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + dt * RK5_A41 * RK_data->RK1[SYS_DIM * indx + 0] + dt * RK5_A42 * RK_data->RK2[SYS_DIM * indx + 0] + dt * RK5_A43 * RK_data->RK3[SYS_DIM * indx + 0];
				RK_data->RK_tmp[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + dt * RK5_A41 * RK_data->RK1[SYS_DIM * indx + 1] + dt * RK5_A42 * RK_data->RK2[SYS_DIM * indx + 1] + dt * RK5_A43 * RK_data->RK3[SYS_DIM * indx + 1];
				RK_data->RK_tmp[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + dt * RK5_A41 * RK_data->RK1[SYS_DIM * indx + 2] + dt * RK5_A42 * RK_data->RK2[SYS_DIM * indx + 2] + dt * RK5_A43 * RK_data->RK3[SYS_DIM * indx + 2];
			}
		}
	}
	// ----------------------- Stage 4
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK4, RK_data->curl, RK_data->vel, RK_data->vort);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Update temporary input for nonlinear term
				RK_data->RK_tmp[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + dt * RK5_A51 * RK_data->RK1[SYS_DIM * indx + 0] + dt * RK5_A52 * RK_data->RK2[SYS_DIM * indx + 0] + dt * RK5_A53 * RK_data->RK3[SYS_DIM * indx + 0] + dt * RK5_A54 * RK_data->RK4[SYS_DIM * indx + 0];
				RK_data->RK_tmp[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + dt * RK5_A51 * RK_data->RK1[SYS_DIM * indx + 1] + dt * RK5_A52 * RK_data->RK2[SYS_DIM * indx + 1] + dt * RK5_A53 * RK_data->RK3[SYS_DIM * indx + 1] + dt * RK5_A54 * RK_data->RK4[SYS_DIM * indx + 1];
				RK_data->RK_tmp[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + dt * RK5_A51 * RK_data->RK1[SYS_DIM * indx + 2] + dt * RK5_A52 * RK_data->RK2[SYS_DIM * indx + 2] + dt * RK5_A53 * RK_data->RK3[SYS_DIM * indx + 2] + dt * RK5_A54 * RK_data->RK4[SYS_DIM * indx + 2];
			}
		}
	}
	// ----------------------- Stage 5
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK5, RK_data->curl, RK_data->vel, RK_data->vort);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Update temporary input for nonlinear term
				RK_data->RK_tmp[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + dt * RK5_A61 * RK_data->RK1[SYS_DIM * indx + 0] + dt * RK5_A62 * RK_data->RK2[SYS_DIM * indx + 0] + dt * RK5_A63 * RK_data->RK3[SYS_DIM * indx + 0] + dt * RK5_A64 * RK_data->RK4[SYS_DIM * indx + 0] + dt * RK5_A65 * RK_data->RK5[SYS_DIM * indx + 0];
				RK_data->RK_tmp[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + dt * RK5_A61 * RK_data->RK1[SYS_DIM * indx + 1] + dt * RK5_A62 * RK_data->RK2[SYS_DIM * indx + 1] + dt * RK5_A63 * RK_data->RK3[SYS_DIM * indx + 1] + dt * RK5_A64 * RK_data->RK4[SYS_DIM * indx + 1] + dt * RK5_A65 * RK_data->RK5[SYS_DIM * indx + 1];
				RK_data->RK_tmp[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + dt * RK5_A61 * RK_data->RK1[SYS_DIM * indx + 2] + dt * RK5_A62 * RK_data->RK2[SYS_DIM * indx + 2] + dt * RK5_A63 * RK_data->RK3[SYS_DIM * indx + 2] + dt * RK5_A64 * RK_data->RK4[SYS_DIM * indx + 2] + dt * RK5_A65 * RK_data->RK5[SYS_DIM * indx + 2];
			}
		}
	}
	// ----------------------- Stage 6
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK6, RK_data->curl, RK_data->vel, RK_data->vort);
	#if defined(__DPRK5)
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Update temporary input for nonlinear term
				RK_data->RK_tmp[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + dt * RK5_A71 * RK_data->RK1[SYS_DIM * indx + 0] + dt * RK5_A73 * RK_data->RK3[SYS_DIM * indx + 0] + dt * RK5_A74 * RK_data->RK4[SYS_DIM * indx + 0] + dt * RK5_A75 * RK_data->RK5[SYS_DIM * indx + 0] + dt * RK5_A76 * RK_data->RK6[SYS_DIM * indx + 0];
				RK_data->RK_tmp[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + dt * RK5_A71 * RK_data->RK1[SYS_DIM * indx + 1] + dt * RK5_A73 * RK_data->RK3[SYS_DIM * indx + 1] + dt * RK5_A74 * RK_data->RK4[SYS_DIM * indx + 1] + dt * RK5_A75 * RK_data->RK5[SYS_DIM * indx + 1] + dt * RK5_A76 * RK_data->RK6[SYS_DIM * indx + 1];
				RK_data->RK_tmp[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + dt * RK5_A71 * RK_data->RK1[SYS_DIM * indx + 2] + dt * RK5_A73 * RK_data->RK3[SYS_DIM * indx + 2] + dt * RK5_A74 * RK_data->RK4[SYS_DIM * indx + 2] + dt * RK5_A75 * RK_data->RK5[SYS_DIM * indx + 2] + dt * RK5_A76 * RK_data->RK6[SYS_DIM * indx + 2];
			}
		}
	}
	// ----------------------- Stage 7
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK7, RK_data->curl, RK_data->vel, RK_data->vort);
	#endif

	/////////////////////
	/// UPDATE STEP
	/////////////////////
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				#if defined(__EULER)
				// Update the Fourier space vorticity with the RHS
				run_data->w_hat[SYS_SIM * indx + 0] = run_data->u_hat[SYS_SIM * indx + 0] + (dt * (RK5_B1 * RK_data->RK1[SYS_SIM * indx + 0]) + dt * (RK5_B3 * RK_data->RK3[SYS_SIM * indx + 0]) + dt * (RK5_B4 * RK_data->RK4[SYS_SIM * indx + 0]) + dt * (RK5_B5 * RK_data->RK5[SYS_SIM * indx + 0]) + dt * (RK5_B6 * RK_data->RK6[SYS_SIM * indx + 0]));
				run_data->w_hat[SYS_SIM * indx + 1] = run_data->u_hat[SYS_SIM * indx + 1] + (dt * (RK5_B1 * RK_data->RK1[SYS_SIM * indx + 1]) + dt * (RK5_B3 * RK_data->RK3[SYS_SIM * indx + 1]) + dt * (RK5_B4 * RK_data->RK4[SYS_SIM * indx + 1]) + dt * (RK5_B5 * RK_data->RK5[SYS_SIM * indx + 1]) + dt * (RK5_B6 * RK_data->RK6[SYS_SIM * indx + 1]));
				run_data->w_hat[SYS_SIM * indx + 2] = run_data->u_hat[SYS_SIM * indx + 2] + (dt * (RK5_B1 * RK_data->RK1[SYS_SIM * indx + 2]) + dt * (RK5_B3 * RK_data->RK3[SYS_SIM * indx + 2]) + dt * (RK5_B4 * RK_data->RK4[SYS_SIM * indx + 2]) + dt * (RK5_B5 * RK_data->RK5[SYS_SIM * indx + 2]) + dt * (RK5_B6 * RK_data->RK6[SYS_SIM * indx + 2]));
				#elif defined(__NAVIER)
				// Compute the pre factors for the RK4CN update step
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + run_data->k[2][k] * run_data->k[2][k]);
				
				#if defined(HYPER_VISC)
				// Hyperviscosity
				D_fac = dt * (sys_vars->NU * pow(k_sqr, VIS_POW)); 
				#else 
				// No hyper viscosity or no ekman drag -> just normal viscosity
				D_fac = dt * (sys_vars->NU * k_sqr); 
				#endif

				// Complete the update step
				run_data->u_hat[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK5_B1 * RK_data->RK1[SYS_DIM * indx + 0] + RK5_B3 * RK_data->RK3[SYS_DIM * indx + 0] + RK5_B4 * RK_data->RK4[SYS_DIM * indx + 0] + RK5_B5 * RK_data->RK5[SYS_DIM * indx + 0] + RK5_B6 * RK_data->RK6[SYS_DIM * indx + 0]);
				run_data->u_hat[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK5_B1 * RK_data->RK1[SYS_DIM * indx + 1] + RK5_B3 * RK_data->RK3[SYS_DIM * indx + 1] + RK5_B4 * RK_data->RK4[SYS_DIM * indx + 1] + RK5_B5 * RK_data->RK5[SYS_DIM * indx + 1] + RK5_B6 * RK_data->RK6[SYS_DIM * indx + 1]);
				run_data->u_hat[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK5_B1 * RK_data->RK1[SYS_DIM * indx + 2] + RK5_B3 * RK_data->RK3[SYS_DIM * indx + 2] + RK5_B4 * RK_data->RK4[SYS_DIM * indx + 2] + RK5_B5 * RK_data->RK5[SYS_DIM * indx + 2] + RK5_B6 * RK_data->RK6[SYS_DIM * indx + 2]);
				#endif
				
				#if defined(__DPRK5)
				if (iters > 1) {
					// Get the higher order update step
					dp_ho_step_x = run_data->u_hat[SYS_DIM * indx + 0] + (dt * (RK5_Bs1 * RK_data->RK1[SYS_DIM * indx + 0]) + dt * (RK5_Bs3 * RK_data->RK3[SYS_DIM * indx + 0]) + dt * (RK5_Bs4 * RK_data->RK4[SYS_DIM * indx + 0]) + dt * (RK5_Bs5 * RK_data->RK5[SYS_DIM * indx + 0]) + dt * (RK5_Bs6 * RK_data->RK6[SYS_DIM * indx + 0]) + dt * (RK5_Bs7 * RK_data->RK7[SYS_DIM * indx + 0]));
					dp_ho_step_y = run_data->u_hat[SYS_DIM * indx + 1] + (dt * (RK5_Bs1 * RK_data->RK1[SYS_DIM * indx + 1]) + dt * (RK5_Bs3 * RK_data->RK3[SYS_DIM * indx + 1]) + dt * (RK5_Bs4 * RK_data->RK4[SYS_DIM * indx + 1]) + dt * (RK5_Bs5 * RK_data->RK5[SYS_DIM * indx + 1]) + dt * (RK5_Bs6 * RK_data->RK6[SYS_DIM * indx + 1]) + dt * (RK5_Bs7 * RK_data->RK7[SYS_DIM * indx + 1]));
					dp_ho_step_z = run_data->u_hat[SYS_DIM * indx + 2] + (dt * (RK5_Bs1 * RK_data->RK1[SYS_DIM * indx + 2]) + dt * (RK5_Bs3 * RK_data->RK3[SYS_DIM * indx + 2]) + dt * (RK5_Bs4 * RK_data->RK4[SYS_DIM * indx + 2]) + dt * (RK5_Bs5 * RK_data->RK5[SYS_DIM * indx + 2]) + dt * (RK5_Bs6 * RK_data->RK6[SYS_DIM * indx + 2]) + dt * (RK5_Bs7 * RK_data->RK7[SYS_DIM * indx + 2]));
					

					// Denominator in the error
					err_denom_x = DP_ABS_TOL + DPMax(cabs(RK_data->u_hat_last[SYS_DIM * indx + 0]), cabs(run_data->u_hat[SYS_DIM * indx + 0])) * DP_REL_TOL;
					err_denom_y = DP_ABS_TOL + DPMax(cabs(RK_data->u_hat_last[SYS_DIM * indx + 1]), cabs(run_data->u_hat[SYS_DIM * indx + 1])) * DP_REL_TOL;
					err_denom_z = DP_ABS_TOL + DPMax(cabs(RK_data->u_hat_last[SYS_DIM * indx + 2]), cabs(run_data->u_hat[SYS_DIM * indx + 2])) * DP_REL_TOL;

					// Compute the sum for the error
					err_sum_x += pow((run_data->u_hat[SYS_DIM * indx + 0] - dp_ho_step_x) /  err_denom_x, 2.0);
					err_sum_y += pow((run_data->u_hat[SYS_DIM * indx + 1] - dp_ho_step_y) /  err_denom_y, 2.0);
					err_sum_z += pow((run_data->u_hat[SYS_DIM * indx + 2] - dp_ho_step_z) /  err_denom_z, 2.0);
				}
				#endif
			}
		}
	}


	#if defined(__DPRK5)
	if (iters > 1) {
		// Reduce and sync the error sum across the processes
		MPI_Allreduce(MPI_IN_PLACE, &err_sum_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &err_sum_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &err_sum_z, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// Compute the error
		err_x = sqrt(1.0 / (Nx * Ny * Nz_Fourier) * err_sum_x);
		err_y = sqrt(1.0 / (Nx * Ny * Nz_Fourier) * err_sum_y);
		err_z = sqrt(1.0 / (Nx * Ny * Nz_Fourier) * err_sum_z);
		RK_data->DP_err = DPErrMax(err_x, err_y, err_z);

		// Record the Fourier velocity for the next step
		for (int i = 0; i < local_Nx; ++i) {
			tmp1 = i * Ny;
			for (int j = 0; j < Ny; ++j) {
				tmp2 = Nz_Fourier * (tmp1 + j);
				for (int k = 0; k < Nz_Fourier; ++k) {
					indx = tmp2 + k;

					// Record the vorticity
					RK_data->u_hat_last[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0];
					RK_data->u_hat_last[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1];
					RK_data->u_hat_last[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2];
				}
			}
		}
	}
	#endif
}
#endif
#if defined(__DPRK5)
/**
 * Function used to find the max between three numbers -> used in the Dormand Prince scheme
 * @param  a Double that will be used to find the max
 * @param  b Double that will be used to find the max
 * @param  c Double that will be used to find the max
 * @return   The max between the two inputs
 */
double DPErrMax(double a, double b, double c) {

	// Initailize max
	double max = c;

	// Check Max
	if (a > max) {
		max = a;
	}
	if (b > max) {
		max = b;
	}

	// Return max
	return max;
}
/**
 * Function used to find the max between two numbers -> used in the Dormand Prince scheme
 * @param  a Double that will be used to find the max
 * @param  b Double that will be used to find the max
 * @return   The max between the two inputs
 */
double DPMax(double a, double b) {

	// Initailize max
	double max;

	// Check Max
	if (a > b) {
		max = a;
	}
	else {
		max = b;
	}

	// Return max
	return max;
}
/**
 * Function used to find the min between two numbers
 * @param  a Double that will be used to find the min
 * @param  b Double that will be used to find the min
 * @return   The minimum of the two inputs
 */
double DPMin(double a, double b) {

	// Initialize min
	double min;

	if (a < b) {
		min = a;
	}
	else {
		min = b;
	}

	// return the result
	return min;
}
#endif
/**
 * Function to perform one step using the 4th order Runge-Kutta method
 * @param dt       The current timestep of the system
 * @param N        Array containing the dimensions of the system
 * @param local_Nx Int indicating the local size of the first dimension of the arrays	
 * @param RK_data  Struct pointing the Integration variables: stages, tmp arrays, rhs and arrays needed for NonlinearRHS function
 */
#if defined(__RK4)
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Nx, RK_data_struct* RK_data) {

	// Initialize vairables
	int tmp1, tmp2;
	int indx;
	const long int Ny 		  = N[1];
	const long int Nz_Fourier = N[2] / 2 + 1;
	#if defined(__NAVIER)
	double k_sqr;
	double D_fac;
	#endif


	/////////////////////
	/// RK STAGES
	/////////////////////
	// ----------------------- Stage 1
	NonlinearRHSBatch(run_data->u_hat, RK_data->RK1, RK_data->curl, RK_data->vel, RK_data->vort);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Update temporary input for nonlinear term
				RK_data->RK_tmp[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + dt * RK4_A21 * RK_data->RK1[SYS_DIM * indx + 0];
				RK_data->RK_tmp[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + dt * RK4_A21 * RK_data->RK1[SYS_DIM * indx + 1];
				RK_data->RK_tmp[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + dt * RK4_A21 * RK_data->RK1[SYS_DIM * indx + 2];
			}
		}
	}
	// ----------------------- Stage 2
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK2, RK_data->curl, RK_data->vel, RK_data->vort);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Update temporary input for nonlinear term
				RK_data->RK_tmp[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + dt * RK4_A32 * RK_data->RK2[SYS_DIM * indx + 0];
				RK_data->RK_tmp[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + dt * RK4_A32 * RK_data->RK2[SYS_DIM * indx + 1];
				RK_data->RK_tmp[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + dt * RK4_A32 * RK_data->RK2[SYS_DIM * indx + 2];
			}
		}
	}
	// ----------------------- Stage 3
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK3, RK_data->curl, RK_data->vel, RK_data->vort);
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Update temporary input for nonlinear term
				RK_data->RK_tmp[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + dt * RK4_A43 * RK_data->RK3[SYS_DIM * indx + 0];
				RK_data->RK_tmp[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + dt * RK4_A43 * RK_data->RK3[SYS_DIM * indx + 1];
				RK_data->RK_tmp[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + dt * RK4_A43 * RK_data->RK3[SYS_DIM * indx + 2];
			}
		}
	}
	// ----------------------- Stage 4
	NonlinearRHSBatch(RK_data->RK_tmp, RK_data->RK4, RK_data->curl, RK_data->vel, RK_data->vort);
	
	
	/////////////////////
	/// UPDATE STEP
	/////////////////////
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				#if defined(__EULER)
				// Update Fourier velocity with the RHS
				run_data->u_hat[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] + (dt * (RK4_B1 * RK_data->RK1[SYS_DIM * indx + 0]) + dt * (RK4_B2 * RK_data->RK2[SYS_DIM * indx + 0]) + dt * (RK4_B3 * RK_data->RK3[SYS_DIM * indx + 0]) + dt * (RK4_B4 * RK_data->RK4[SYS_DIM * indx + 0]));
				run_data->u_hat[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] + (dt * (RK4_B1 * RK_data->RK1[SYS_DIM * indx + 1]) + dt * (RK4_B2 * RK_data->RK2[SYS_DIM * indx + 1]) + dt * (RK4_B3 * RK_data->RK3[SYS_DIM * indx + 1]) + dt * (RK4_B4 * RK_data->RK4[SYS_DIM * indx + 1]));
				run_data->u_hat[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] + (dt * (RK4_B1 * RK_data->RK1[SYS_DIM * indx + 2]) + dt * (RK4_B2 * RK_data->RK2[SYS_DIM * indx + 2]) + dt * (RK4_B3 * RK_data->RK3[SYS_DIM * indx + 2]) + dt * (RK4_B4 * RK_data->RK4[SYS_DIM * indx + 2]));
				#elif defined(__NAVIER)
				// Compute the pre factors for the RK4CN update step
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + run_data->k[2][k] * run_data->k[2][k]);
				
				#if defined(HYPER_VISC) 
				// Hyperviscosity 
				D_fac = dt * (sys_vars->NU * pow(k_sqr, VIS_POW)); 
				#else 
				// No hyper viscosity -> just normal viscosity
				D_fac = dt * (sys_vars->NU * k_sqr); 
				#endif

				// Update Fourier vorticity
				run_data->u_hat[SYS_DIM * indx + 0] = run_data->u_hat[SYS_DIM * indx + 0] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK4_B1 * RK_data->RK1[SYS_DIM * indx + 0] + RK4_B2 * RK_data->RK2[SYS_DIM * indx + 0] + RK4_B3 * RK_data->RK3[SYS_DIM * indx + 0] + RK4_B4 * RK_data->RK4[SYS_DIM * indx + 0]);
				run_data->u_hat[SYS_DIM * indx + 1] = run_data->u_hat[SYS_DIM * indx + 1] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK4_B1 * RK_data->RK1[SYS_DIM * indx + 1] + RK4_B2 * RK_data->RK2[SYS_DIM * indx + 1] + RK4_B3 * RK_data->RK3[SYS_DIM * indx + 1] + RK4_B4 * RK_data->RK4[SYS_DIM * indx + 1]);
				run_data->u_hat[SYS_DIM * indx + 2] = run_data->u_hat[SYS_DIM * indx + 2] * ((2.0 - D_fac) / (2.0 + D_fac)) + (2.0 * dt / (2.0 + D_fac)) * (RK4_B1 * RK_data->RK1[SYS_DIM * indx + 2] + RK4_B2 * RK_data->RK2[SYS_DIM * indx + 2] + RK4_B3 * RK_data->RK3[SYS_DIM * indx + 2] + RK4_B4 * RK_data->RK4[SYS_DIM * indx + 2]);
				#endif
			}
		}
	}
}
#endif
/**
 * Function that performs the evluation of the nonlinear term by transforming the velocity and vorticity back to real space where 
 * curl is performed before transforming back to Fourier space. Dealiasing is applied to the result
 * 
 * @param u_hat     Input array: contains the current Fourier velocity of the system
 * @param dw_hat_dt Output array: Contains the result of the dealiased nonlinear term. Also used as an intermediate array to save memory
 * @param curl      Array to hold the curl of velocity and vorticity in Real Space
 * @param u         Array to hold the real space velocities
 * @param vort      Array to hold the real space vorticity derivatives
 */
void NonlinearRHSBatch(fftw_complex* u_hat, fftw_complex* dw_hat_dt, double* curl, double* u, double* vort) {

	// Initialize variables
	int tmp1, tmp2, indx;
	const ptrdiff_t local_Nx  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	fftw_complex k_dot_nl;
	double k_sqr_inv;
	double fftw_norm_fac = 1.0 / pow(Nx * Ny * Nz, 2.0);

	// -----------------------------------
	// Compute Fourier Space Vorticity
	// -----------------------------------
	// Compute w_k from the Fourier space velocity u_k
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Compute the curl of u_k
				dw_hat_dt[SYS_DIM * indx + 0] = I * (run_data->k[1][j] * u_hat[SYS_DIM * indx + 2] - run_data->k[2][k] * u_hat[SYS_DIM * indx + 1]);
				dw_hat_dt[SYS_DIM * indx + 1] = I * (run_data->k[2][k] * u_hat[SYS_DIM * indx + 0] - run_data->k[0][i] * u_hat[SYS_DIM * indx + 2]);
				dw_hat_dt[SYS_DIM * indx + 2] = I * (run_data->k[0][i] * u_hat[SYS_DIM * indx + 1] - run_data->k[1][j] * u_hat[SYS_DIM * indx + 0]);
			}
		}
	}

	// ----------------------------------
	// Transform to Real Space
	// ----------------------------------
	// Batch transform Fourier velocites to real space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_3d_dft_trans_batch_c2r), dw_hat_dt, vort);
	// Batch transform Fourier vorticity to real space
	fftw_mpi_execute_dft_c2r((sys_vars->fftw_3d_dft_trans_batch_c2r), u_hat, u);

	// ---------------------------------------------
	// Perform Cross Product in Real Space
	// ---------------------------------------------
	// Compute u x w in real space
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = (Nz + 2) * (tmp1 + j);
			for (int k = 0; k < Nz; ++k) {
				indx = tmp2 + k;

				// Compute the u curl w
				curl[SYS_DIM * indx + 0] = u[SYS_DIM * indx + 1] * vort[SYS_DIM * indx + 2] - u[SYS_DIM * indx + 2] * vort[SYS_DIM * indx + 1];
				curl[SYS_DIM * indx + 1] = u[SYS_DIM * indx + 2] * vort[SYS_DIM * indx + 0] - u[SYS_DIM * indx + 0] * vort[SYS_DIM * indx + 2];
				curl[SYS_DIM * indx + 2] = u[SYS_DIM * indx + 0] * vort[SYS_DIM * indx + 1] - u[SYS_DIM * indx + 1] * vort[SYS_DIM * indx + 0];
			}			
		}
	}

	// ----------------------------------
	// Transform to Fourier Space
	// ----------------------------------
	// Batch transform both fourier vorticity derivatives to real space -> normalize in next loop
	fftw_mpi_execute_dft_r2c((sys_vars->fftw_3d_dft_trans_batch_r2c), curl, dw_hat_dt);

  	// -------------------------------------
 	// Subtract the Pressure Term
 	// -------------------------------------
 	// Compute the pressure term and subtract from the nonlinear term -> perform the post dft normalization here
 	for (int i = 0; i < local_Nx; ++i) {
 		tmp1 = i * Ny;
 		for (int j = 0; j < Ny; ++j) {
 			tmp2 = Nz_Fourier * (tmp1 + j);
 			for (int k = 0; k < Nz_Fourier; ++k) {
 				indx = tmp2 + k;

 				// Normalize the nonlinear term after DFT
 				dw_hat_dt[SYS_DIM * indx + 0] *= fftw_norm_fac;
 				dw_hat_dt[SYS_DIM * indx + 1] *= fftw_norm_fac;
 				dw_hat_dt[SYS_DIM * indx + 2] *= fftw_norm_fac;

 				// Compute 1/k^2
 				if ((run_data->k[0][i] != 0.0) || (run_data->k[1][j] != 0.0) || (run_data->k[2][k] != 0.0)) {
 					k_sqr_inv = 1.0 / (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + run_data->k[2][k] * run_data->k[2][k]);

 					// Compute the dot product of k and the nonlinear term
 					k_dot_nl = run_data->k[0][i] * dw_hat_dt[SYS_DIM * indx + 0] + run_data->k[1][j] * dw_hat_dt[SYS_DIM * indx + 1] + run_data->k[2][k] * dw_hat_dt[SYS_DIM * indx + 2]; 

 					// Subtract the pressure term from the nonlinear term
 					dw_hat_dt[SYS_DIM * indx + 0] -= run_data->k[0][i] * k_sqr_inv * k_dot_nl;
 					dw_hat_dt[SYS_DIM * indx + 1] -= run_data->k[1][j] * k_sqr_inv * k_dot_nl; 
 					dw_hat_dt[SYS_DIM * indx + 2] -= run_data->k[2][k] * k_sqr_inv * k_dot_nl; 
 				}
 				else {
 					// The zero mode
 					dw_hat_dt[SYS_DIM * indx + 0] = 0.0 + 0.0 * I;
 					dw_hat_dt[SYS_DIM * indx + 1] = 0.0 + 0.0 * I;
 					dw_hat_dt[SYS_DIM * indx + 2] = 0.0 + 0.0 * I;
	 			}
 			}
 		}
 	}

 	// -------------------------------------
 	// Apply Dealiasing & Forcing
 	// -------------------------------------
 	// Apply dealiasing 
 	ApplyDealiasing(dw_hat_dt, SYS_DIM, sys_vars->N);

 	// // Forcing 
 	// ApplyForcing(dw_hat_dt, sys_vars->N);
}
/**
 * Function to compute the total energy in the system at the current timestep
 * @return  The total energy in the system
 */
double TotalEnergy(void) {

	// Initialize variables
	int tmp1, tmp2;
	int indx;
	double tot_energy = 0.0;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny * Nz, 2.0);

	// ------------------------------------------
	// Compute Energy in Fourier Space
	// ------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Ny; ++k) {
				indx = tmp2 + k;

				if ((k == 0) || (k == Nz_Fourier - 1)) {
					// Update the sum for the total energy
					tot_energy += cabs(run_data->u_hat[SYS_DIM * indx + 0] * conj(run_data->u_hat[SYS_DIM * indx + 0]));
					tot_energy += cabs(run_data->u_hat[SYS_DIM * indx + 1] * conj(run_data->u_hat[SYS_DIM * indx + 1]));
					tot_energy += cabs(run_data->u_hat[SYS_DIM * indx + 2] * conj(run_data->u_hat[SYS_DIM * indx + 2]));
				}
				else {
					// Update the sum for the total energy
					tot_energy += 2.0 * cabs(run_data->u_hat[SYS_DIM * indx + 0] * conj(run_data->u_hat[SYS_DIM * indx + 0])); 
					tot_energy += 2.0 * cabs(run_data->u_hat[SYS_DIM * indx + 1] * conj(run_data->u_hat[SYS_DIM * indx + 1]));
					tot_energy += 2.0 * cabs(run_data->u_hat[SYS_DIM * indx + 2] * conj(run_data->u_hat[SYS_DIM * indx + 2]));
				}
			}
		}
	}
	
	// Return result
	return 8.0 * pow(M_PI, 3.0) * tot_energy * norm_fac;
}
/**
 * Function to compute the total enstrophy in the system at the current timestep
 * @return  The total enstrophy in the system
 */
double TotalEnstrophy(void) {

	// Initialize variables
	int tmp1, tmp2;
	int indx;
	double tot_enst = 0.0;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny * Nz, 2.0);
	fftw_complex w_hat_x, w_hat_y, w_hat_z;

	// --------------------------------------------------
	// Compute Vorticity and Enstrophy in Fourier Space
	// --------------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Ny; ++k) {
				indx = tmp2 + k;

				// Compute the Fourier space vorticity
				w_hat_x = I * (run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 2] - run_data->k[2][k] * run_data->u_hat[SYS_DIM * indx + 1]);
				w_hat_y = I * (run_data->k[2][k] * run_data->u_hat[SYS_DIM * indx + 0] - run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 2]);
				w_hat_z = I * (run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 1] - run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 0]);

				if ((k == 0) || (k == Nz_Fourier - 1)) {
					// Update the sum for the total enstrophy
					tot_enst += cabs(w_hat_x * conj(w_hat_x));
					tot_enst += cabs(w_hat_y * conj(w_hat_y));
					tot_enst += cabs(w_hat_z * conj(w_hat_z));
				}
				else {
					// Update the sum for the total enstrophy
					tot_enst += 2.0 * cabs(w_hat_x * conj(w_hat_x)); 
					tot_enst += 2.0 * cabs(w_hat_y * conj(w_hat_y));
					tot_enst += 2.0 * cabs(w_hat_z * conj(w_hat_z));
				}
			}
		}
	}
	
	// Return result
	return 8.0 * pow(M_PI, 3.0) * tot_enst * norm_fac;
}
/**
 * Function to compute the total helicity in the system at the current timestep
 * @return  The total helicity in the system
 */
double TotalHelicity(void) {

	// Initialize variables
	int tmp1, tmp2;
	int indx;
	double tot_heli = 0.0;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny * Nz, 2.0);
	fftw_complex w_hat_x, w_hat_y, w_hat_z;

	// --------------------------------------------------
	// Compute Vorticity and Enstrophy in Fourier Space
	// --------------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Ny; ++k) {
				indx = tmp2 + k;

				// Compute the Fourier space vorticity
				w_hat_x = I * (run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 2] - run_data->k[2][k] * run_data->u_hat[SYS_DIM * indx + 1]);
				w_hat_y = I * (run_data->k[2][k] * run_data->u_hat[SYS_DIM * indx + 0] - run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 2]);
				w_hat_z = I * (run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 1] - run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 0]);

				if ((k == 0) || (k == Nz_Fourier - 1)) {
					// Update the sum for the total helicity
					tot_heli += run_data->u_hat[SYS_DIM * indx + 0] * w_hat_x;
					tot_heli += run_data->u_hat[SYS_DIM * indx + 1] * w_hat_y;
					tot_heli += run_data->u_hat[SYS_DIM * indx + 2] * w_hat_z;
				}
				else {
					// Update the sum for the total heli
					tot_heli += 2.0 * run_data->u_hat[SYS_DIM * indx + 0] * w_hat_x; 
					tot_heli += 2.0 * run_data->u_hat[SYS_DIM * indx + 1] * w_hat_y;
					tot_heli += 2.0 * run_data->u_hat[SYS_DIM * indx + 2] * w_hat_z;
				}
			}
		}
	}
	
	// Return result
	return 8.0 * pow(M_PI, 3.0) * tot_heli * norm_fac;
}
/**
 * Function to compute the total palinstrophy in the system at the current timestep
 * @return  The total palinstrophy in the system
 */
double TotalPalinstrophy(void) {

	// Initialize variables
	int tmp1, tmp2;
	int indx;
	double tot_palin = 0.0;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny * Nz, 2.0);
	fftw_complex w_hat_x, w_hat_y, w_hat_z;
	fftw_complex curl_hat_x, curl_hat_y, curl_hat_z;

	// --------------------------------------------------
	// Compute Vorticity and Enstrophy in Fourier Space
	// --------------------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Ny; ++k) {
				indx = tmp2 + k;

				// Compute the Fourier space vorticity
				w_hat_x = I * (run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 2] - run_data->k[2][k] * run_data->u_hat[SYS_DIM * indx + 1]);
				w_hat_y = I * (run_data->k[2][k] * run_data->u_hat[SYS_DIM * indx + 0] - run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 2]);
				w_hat_z = I * (run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 1] - run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 0]);

				// Take the curl of the Fourier space vorticity
				curl_hat_x = I * (run_data->k[1][j] * w_hat_z - run_data->k[2][k] * w_hat_y);
				curl_hat_y = I * (run_data->k[2][k] * w_hat_x - run_data->k[0][i] * w_hat_z);
				curl_hat_z = I * (run_data->k[0][i] * w_hat_y - run_data->k[1][j] * w_hat_x);

				if ((k == 0) || (k == Nz_Fourier - 1)) {
					// Update the sum for the total palinstrophy
					tot_palin += cabs(curl_hat_x * conj(curl_hat_x));
					tot_palin += cabs(curl_hat_y * conj(curl_hat_y));
					tot_palin += cabs(curl_hat_z * conj(curl_hat_z));
				}
				else {
					// Update the sum for the total palinstrophy
					tot_palin += 2.0 * cabs(curl_hat_x * conj(curl_hat_x)); 
					tot_palin += 2.0 * cabs(curl_hat_y * conj(curl_hat_y));
					tot_palin += 2.0 * cabs(curl_hat_z * conj(curl_hat_z));
				}
			}
		}
	}
	
	// Return result
	return 8.0 * pow(M_PI, 3.0) * tot_palin * norm_fac;
}
/**
 * Function used to compute the Enstrophy spectrum of the current iteration. The energy spectrum is defined as all(sum) of the energy contained in concentric spherical shells in
 * wavenumber space. 	
 */
void EnstrophySpectrum(void) {

	// Initialize variables
	int tmp1, tmp2;
	int indx;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny * Nz, 2.0);
    double const_fac = 8.0 * pow(M_PI, 3.0);
    fftw_complex w_hat_x, w_hat_y, w_hat_z;

	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enst_spect[i] = 0.0;
	}

	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
				spec_indx = (int) round( sqrt( (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + run_data->k[2][k] * run_data->k[2][k])));

				// Compute the Fourier space vorticity
				w_hat_x = I * (run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 2] - run_data->k[2][k] * run_data->u_hat[SYS_DIM * indx + 1]);
				w_hat_y = I * (run_data->k[2][k] * run_data->u_hat[SYS_DIM * indx + 0] - run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 2]);
				w_hat_z = I * (run_data->k[0][i] * run_data->u_hat[SYS_DIM * indx + 1] - run_data->k[1][j] * run_data->u_hat[SYS_DIM * indx + 0]);

				// Update the current bin with this mode
				if ((k == 0) || (k == Nz_Fourier - 1)) { // These modes do not have conjugates so are counted once
					run_data->enst_spect[spec_indx] += const_fac * norm_fac * (cabs(w_hat_x * conj(w_hat_x)) + cabs(w_hat_y * conj(w_hat_y)) + cabs(w_hat_z * conj(w_hat_z)));;
				}
				else { // These modes have conjugates so are counted twice
					run_data->enst_spect[spec_indx] += 2.0 * const_fac * norm_fac * (cabs(w_hat_x * conj(w_hat_x)) + cabs(w_hat_y * conj(w_hat_y)) + cabs(w_hat_z * conj(w_hat_z)));;
				}
			}
		}
	}
}
/**
 * Function used to compute the Energy spectrum of the current iteration. The energy spectrum is defined as all(sum) of the energy contained in concentric spherical shells in
 * wavenumber space. 	
 */
void EnergySpectrum(void) {

	// Initialize variables
	int tmp1, tmp2;
	int indx;
	int spec_indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	double norm_fac = 0.5 / pow(Nx * Ny * Nz, 2.0);
    double const_fac = 8.0 * pow(M_PI, 3.0);
    fftw_complex u_hat_x, u_hat_y, u_hat_z;

	// ------------------------------------
	// Initialize Spectrum Array
	// ------------------------------------
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		run_data->enrg_spect[i] = 0.0;
	}

	// ------------------------------------
	// Compute Spectrum
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Nz_Fourier; ++k) {
				indx = tmp2 + k;

				// Get spectrum index -> spectrum is computed by summing over the energy contained in concentric annuli in wavenumber space
				spec_indx = (int) round( sqrt( (double)(run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + run_data->k[2][k] * run_data->k[2][k])));

				// Get the current velocity components
				u_hat_x = run_data->u_hat[SYS_DIM * indx + 0];
				u_hat_y = run_data->u_hat[SYS_DIM * indx + 1];
				u_hat_z = run_data->u_hat[SYS_DIM * indx + 2];
				
				// Update the current bin with this mode
				if ((k == 0) || (k == Nz_Fourier - 1)) { // These modes do not have conjugates so are counted once
					run_data->enrg_spect[spec_indx] += const_fac * norm_fac * (cabs(u_hat_x * conj(u_hat_x)) + cabs(u_hat_y * conj(u_hat_y)) + cabs(u_hat_z * conj(u_hat_z)));
				}
				else { // These modes have conjugates so are counted twice
					run_data->enrg_spect[spec_indx] += 2.0 * const_fac * norm_fac * (cabs(u_hat_x * conj(u_hat_x)) + cabs(u_hat_y * conj(u_hat_y)) + cabs(u_hat_z * conj(u_hat_z)));
				}
			}
		}
	}
}
/**
 * Function to compute the system measurables such as energy, enstrophy, palinstrophy, helicity, energy and enstrophy dissipation rates, and spectra at once on the local processes for the current timestep
 * @param iter The index in the system arrays for the current timestep
 */
void ComputeSystemMeasurables(int iter) {

	// Initialize variables
	int tmp1, tmp2;
	int indx;
	ptrdiff_t local_Nx 		  = sys_vars->local_Nx;
	const long int Nx         = sys_vars->N[0];
	const long int Ny         = sys_vars->N[1];
	const long int Nz         = sys_vars->N[2];
	const long int Nz_Fourier = sys_vars->N[2] / 2 + 1;
	double k_sqr, pre_fac;
	double norm_fac  = 0.5 / pow(Nx * Ny * Nz, 2.0);
    double const_fac = 8.0 * pow(M_PI, 3.0);
	fftw_complex w_hat_x, w_hat_y, w_hat_z;
	fftw_complex u_hat_x, u_hat_y, u_hat_z;
	fftw_complex curl_hat_x, curl_hat_y, curl_hat_z;

	// ------------------------------------
	// Initialize Measurables
	// ------------------------------------
	#if defined(__SYS_MEASURES)
	// Initialize totals
	run_data->tot_enstr[iter]  = 0.0;
	run_data->tot_palin[iter]  = 0.0;
	run_data->tot_heli[iter]   = 0.0;
	run_data->tot_energy[iter] = 0.0;
	run_data->enrg_diss[iter]  = 0.0;
	#endif 
	#if defined(__ENRG_SPECT) || defined(__ENST_SPECT)
	// Initialize spectra
	for (int i = 0; i < sys_vars->n_spect; ++i) {
		#if defined(__ENRG_SPECT)
		run_data->enrg_spect[i] = 0.0;
		#endif
		#if defined(__ENST_SPECT)
		run_data->enst_spect[i] = 0.0;
		#endif
	}
	#endif

	// ------------------------------------
	// Compute Measureables in Fourier Space
	// ------------------------------------
	for (int i = 0; i < local_Nx; ++i) {
		tmp1 = i * Ny;
		for (int j = 0; j < Ny; ++j) {
			tmp2 = Nz_Fourier * (tmp1 + j);
			for (int k = 0; k < Ny; ++k) {
				indx = tmp2 + k;

				// Get the current velocity components
				u_hat_x = run_data->u_hat[SYS_DIM * indx + 0];
				u_hat_y = run_data->u_hat[SYS_DIM * indx + 1];
				u_hat_z = run_data->u_hat[SYS_DIM * indx + 2];

				// Compute the Fourier space vorticity
				w_hat_x = I * (run_data->k[1][j] * u_hat_z - run_data->k[2][k] * u_hat_y);
				w_hat_y = I * (run_data->k[2][k] * u_hat_x - run_data->k[0][i] * u_hat_z);
				w_hat_z = I * (run_data->k[0][i] * u_hat_y - run_data->k[1][j] * u_hat_x);

				///------------------------------------------ System Totals
				#if defined(__SYS_MEASURES)
				// Take the curl of the Fourier space vorticity
				curl_hat_x = I * (run_data->k[1][j] * w_hat_z - run_data->k[2][k] * w_hat_y);
				curl_hat_y = I * (run_data->k[2][k] * w_hat_x - run_data->k[0][i] * w_hat_z);
				curl_hat_z = I * (run_data->k[0][i] * w_hat_y - run_data->k[1][j] * w_hat_x);

				// Compute |k|^2
				k_sqr = (double) (run_data->k[0][i] * run_data->k[0][i] + run_data->k[1][j] * run_data->k[1][j] + run_data->k[2][k] * run_data->k[2][k]);

				// Determine the prefactor for the dissipation rate
				#if defined(HYPER_VISC) 
				// Hyperviscosity 
				pre_fac = sys_vars->NU * pow(k_sqr, VIS_POW); 
				#else 
				// No hyper viscosity -> just normal viscosity
				pre_fac = sys_vars->NU * k_sqr; 
				#endif

				// Update the sum for the totals
				if ((k == 0) || (k == Nz_Fourier - 1)) { // these modes do not have conjugates so counted once
					run_data->enrg_diss[iter]  += pre_fac * (cabs(u_hat_x * conj(u_hat_x)) + cabs(u_hat_y * conj(u_hat_y)) + cabs(u_hat_z * conj(u_hat_z)));
					run_data->tot_enstr[iter]  += cabs(w_hat_x * conj(w_hat_x)) + cabs(w_hat_y * conj(w_hat_y)) + cabs(w_hat_z * conj(w_hat_z));
					run_data->tot_palin[iter]  += cabs(curl_hat_x * conj(curl_hat_x)) + cabs(curl_hat_y * conj(curl_hat_y)) + cabs(curl_hat_z * conj(curl_hat_z));
					run_data->tot_heli[iter]   += u_hat_x * w_hat_x + u_hat_y * w_hat_y + u_hat_z * w_hat_z;
					run_data->tot_energy[iter] += cabs(u_hat_x * conj(u_hat_x)) + cabs(u_hat_y * conj(u_hat_y)) + cabs(u_hat_z * conj(u_hat_z));
				}
				else { // these modes have conjugates, so counted twice
					run_data->enrg_diss[iter]  += 2.0 * pre_fac * (cabs(u_hat_x * conj(u_hat_x)) + cabs(u_hat_y * conj(u_hat_y)) + cabs(u_hat_z * conj(u_hat_z)));
					run_data->tot_enstr[iter]  += 2.0 * cabs(w_hat_x * conj(w_hat_x)) + cabs(w_hat_y * conj(w_hat_y)) + cabs(w_hat_z * conj(w_hat_z));
					run_data->tot_palin[iter]  += 2.0 * cabs(curl_hat_x * conj(curl_hat_x)) + cabs(curl_hat_y * conj(curl_hat_y)) + cabs(curl_hat_z * conj(curl_hat_z));
					run_data->tot_heli[iter]   += 2.0 * u_hat_x * w_hat_x + u_hat_y * w_hat_y + u_hat_z * w_hat_z;
					run_data->tot_energy[iter] += 2.0 * cabs(u_hat_x * conj(u_hat_x)) + cabs(u_hat_y * conj(u_hat_y)) + cabs(u_hat_z * conj(u_hat_z));				
				}
				#endif	

				///------------------------------------------ System Spectra
				#if defined(__ENRG_SPECT) || defined(__ENST_SPECT)
				if ((k == 0) || (k == Nz_Fourier - 1)) { // these modes do not have conjugates so counted once
					#if defined(__ENRG_SPECT)
					run_data->enrg_spect[i] += const_fac * norm_fac * (cabs(u_hat_x * conj(u_hat_x)) + cabs(u_hat_y * conj(u_hat_y)) + cabs(u_hat_z * conj(u_hat_z)));
					#endif
					#if defined(__ENST_SPECT)
					run_data->enst_spect[i] += const_fac * norm_fac * (cabs(w_hat_x * conj(w_hat_x)) + cabs(w_hat_y * conj(w_hat_y)) + cabs(w_hat_z * conj(w_hat_z)));;
					#endif
				}
				else { // these modes have conjugates, so counted twice
					#if defined(__ENRG_SPECT)
					run_data->enrg_spect[i] += 2.0 * const_fac * norm_fac * (cabs(u_hat_x * conj(u_hat_x)) + cabs(u_hat_y * conj(u_hat_y)) + cabs(u_hat_z * conj(u_hat_z)));
					#endif
					#if defined(__ENST_SPECT)
					run_data->enst_spect[i] += 2.0 * const_fac * norm_fac * (cabs(w_hat_x * conj(w_hat_x)) + cabs(w_hat_y * conj(w_hat_y)) + cabs(w_hat_z * conj(w_hat_z)));;
					#endif
				}
				#endif	
			}
		}
	}
	
	// ------------------------------------
	// Normalize Measureables 
	// ------------------------------------	
	#if defined(__SYS_MEASURES)
	// Normalize results and take into account computation in Fourier space
	run_data->enrg_diss[iter]  *= 2.0 * const_fac * norm_fac;
	run_data->tot_enstr[iter]  *= const_fac * norm_fac;
	run_data->tot_palin[iter]  *= const_fac * norm_fac;
	run_data->tot_heli[iter]   *= const_fac * norm_fac;
	run_data->tot_energy[iter] *= const_fac * norm_fac;
	#endif
}
/**
 * Function to record the system measures for the current timestep 
 * @param t          The current time in the simulation
 * @param print_indx The current index of the measurables arrays
 * @param RK_data 	 The Runge-Kutta struct containing the arrays to compute the nonlinear term for the fluxes
 */
void RecordSystemMeasures(double t, int print_indx, RK_data_struct* RK_data) {

	// -------------------------------
	// Record the System Measures 
	// -------------------------------	
	// The integration time
	#if defined(__TIME)
	if (!(sys_vars->rank)) {
		run_data->time[print_indx] = t;
	}
	#endif

	// Check if within memory limits
	if (print_indx < sys_vars->num_print_steps) {
		#if defined(__SYS_MEASURES) || defined(__ENRG_SPECT) || defined(__ENST_SPECT)
		// System totals: energy, enstrophy, palinstrophy, helicity, energy and enstrophy dissipation rates and spectra
		ComputeSystemMeasurables(print_indx);
		#endif
	}
	else {
		printf("\n["MAGENTA"WARNING"RESET"] --- Unable to write system measures at Indx: [%d] t: [%lf] ---- Number of intergration steps is now greater then memory allocated\n", print_indx, t);
	}
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

	// Total Helicity
	run_data->tot_heli = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->tot_heli == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Total Helicity");
		exit(1);
	}

	// Energy Dissipation Rate
	run_data->enrg_diss = (double* )fftw_malloc(sizeof(double) * print_steps);
	if (run_data->enrg_diss == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Dissipation Rate");
		exit(1);
	}	
	#endif
	#if defined(__ENST_SPECT)
	// Enstrophy Spectrum
	run_data->enst_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enst_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Spectrum");
		exit(1);
	}	
	#endif
	#if defined(__ENRG_SPECT)
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
	#if defined(__ENST_FLUX_SPECT)
	// Enstrophy Spectrum
	run_data->enst_flux_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enst_flux_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Enstrophy Flux Spectrum");
		exit(1);
	}	
	#endif
	#if defined(__ENRG_FLUX_SPECT)
	// Energy Spectrum
	run_data->enrg_flux_spect = (double* )fftw_malloc(sizeof(double) * n_spect);
	if (run_data->enrg_flux_spect == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for the ["CYAN"%s"RESET"]\n-->> Exiting!!!\n", "Energy Flux Spectrum");
		exit(1);
	}	
	#endif

	// ----------------------------
	// Get Measurables of the ICs
	// ----------------------------
	#if defined(__SYS_MEASURES) || defined(__ENST_SPECT) || defined(__ENRG_SPECT)
	// Compute system quantities
	ComputeSystemMeasurables(0);
	#endif
	#if defined(__TIME)
	if (!(sys_vars->rank)) {
		run_data->time[0] = sys_vars->t0;
	}
	#endif
	// #if defined(__ENST_FLUX)
	// // Enstrophy Flux and dissipation from/to Subset of modes
	// EnstrophyFlux(&(run_data->enst_flux_sbst[0]), &(run_data->enst_diss_sbst[0]), RK_data);
	// #endif
	// #if defined(__ENRG_FLUX)
	// // Energy Flux and dissipation from/to a subset of modes
	// EnergyFlux(&(run_data->enrg_flux_sbst[0]), &(run_data->enrg_diss_sbst[0]), RK_data);
	// #endif
	// Time

	// // ----------------------------
	// // Get Spectra of the ICs
	// // ----------------------------
	// #if defined(__ENRG_FLUX_SPECT)
	// EnergyFluxSpectrum(RK_data);
	// #endif
	// #if defined(__ENST_FLUX_SPECT)
	// EnstrophyFluxSpectrum(RK_data);
	// #endif
}
/**
 * Function to compute the initial condition for the integration
 * @param u_hat Fourier space velocity
 * @param u     Real space velocities in batch layout - both u and v
 * @param N     Array containing the dimensions of the system
 */
void InitialConditions(fftw_complex* u_hat, double* u, const long int* N) {

	// Initialize variables
	int tmp1, tmp2, indx;
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
	RK_data->curl = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->curl == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Curl");
		exit(1);
	}
	RK_data->vel = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->vel == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Vel");
		exit(1);
	}
	RK_data->vort = (double* )fftw_malloc(sizeof(double) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->vort == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "Vort");
		exit(1);
	}
	#if defined(__RK5) || defined(__DPRK5)
	RK_data->RK5 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->RK5 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK5");
		exit(1);
	}
	RK_data->RK6 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->RK6 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK6");
		exit(1);
	}
	#endif
	#if defined(__DPRK5)
	RK_data->RK7 = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->RK7 == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "RK7");
		exit(1);
	}
	RK_data->u_hat_last = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * 2 * sys_vars->alloc_local_batch);
	if (RK_data->u_hat_last == NULL) {
		fprintf(stderr, "\n["RED"ERROR"RESET"] --- Unable to allocate memory for Integration Array ["CYAN"%s"RESET"] \n-->> Exiting!!!\n", "w_hat_last");
		exit(1);
	}
	#endif

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
				for (int d = 0; d < SYS_DIM; ++d) {
					if (k < Nz_Fourier) {
						run_data->w_hat[SYS_DIM * indx_f + d] = 0.0 + 0.0 * I;
						run_data->u_hat[SYS_DIM * indx_f + d] = 0.0 + 0.0 * I;
						RK_data->RK1[SYS_DIM * indx_f + d]    = 0.0 + 0.0 * I;
						RK_data->RK2[SYS_DIM * indx_f + d]    = 0.0 + 0.0 * I;
						RK_data->RK3[SYS_DIM * indx_f + d]    = 0.0 + 0.0 * I;
						RK_data->RK4[SYS_DIM * indx_f + d]    = 0.0 + 0.0 * I;
						RK_data->RK_tmp[SYS_DIM * indx_f + d] = 0.0 + 0.0 * I;
						#if defined(__RK5) || defined(__DPRK5)
						RK_data->RK5[SYS_DIM * indx_f + d] = 0.0 + 0.0 * I;
						RK_data->RK6[SYS_DIM * indx_f + d] = 0.0 + 0.0 * I;						
						#endif
						#if defined(__DPRK5)
						RK_data->RK7[SYS_DIM * indx_f + d] 	  	  = 0.0 + 0.0 * I;
						RK_data->u_hat_last[SYS_DIM * indx_f + d] = 0.0 + 0.0 * I;						
						#endif
					}
					run_data->w[SYS_DIM * indx_r + d]   = 0.0;
					run_data->u[SYS_DIM * indx_r + d] 	= 0.0;
					RK_data->vel[SYS_DIM * indx_r + d]  = 0.0;
					RK_data->vort[SYS_DIM * indx_r + d] = 0.0;
					RK_data->curl[SYS_DIM * indx_r + d] = 0.0;
				}
			}
		}
	}
}
/**
 * Wrapper function that initializes the FFTW plans using MPI
 * @param N       Array containing the dimensions of the system
 * @param NTBatch Array containing the dimensions of the system for the transposed plans
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
	fftw_free(run_data->tot_heli);
	fftw_free(run_data->enrg_diss);
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
	fftw_free(RK_data->RK_tmp);
	fftw_free(RK_data->curl);
	fftw_free(RK_data->vel);
	fftw_free(RK_data->vort);
	#if defined(__RK5) || defined(__DPRK5)
	fftw_free(RK_data->RK5);
	fftw_free(RK_data->RK6);						
	#endif
	#if defined(__DPRK5)
	fftw_free(RK_data->RK7);
	fftw_free(RK_data->u_hat_last);						
	#endif

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