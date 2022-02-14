/**
* @file solver.h
* @author Enda Carroll
* @date Feb 2022
* @brief Header file containing the function prototypes for the solver.c file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "data_types.h"




// ---------------------------------------------------------------------
//  Function Prototpyes
// ---------------------------------------------------------------------
// Main function for the pseudospectral solver
void SpectralSolve(void);
#if defined(__RK5) || defined(__DPRK5)
void RK5DPStep(const double dt, const long int* N, const int iters, const ptrdiff_t local_Nx, RK_data_struct* RK_data);
#endif
#if defined(__DPRK5)
double DPErrMax(double a, double b, double c);
double DPMax(double a, double b);
double DPMin(double a, double b);
#endif
#if defined(__RK4)
void RK4Step(const double dt, const long int* N, const ptrdiff_t local_Nx, RK_data_struct* RK_data);
#endif
void NonlinearRHSBatch(fftw_complex* u_hat, fftw_complex* dw_hat_dt, double* curl, double* u, double* vort);
double TotalEnergy(void);
double TotalEnstrophy(void);
double TotalHelicity(void);
double TotalPalinstrophy(void);
void ComputeSystemTotals(int iter);
void InitializeSystemMeasurables(RK_data_struct* RK_data);
void RecordSystemMeasures(double t, int print_indx, RK_data_struct* RK_data);
void InitializeIntegrationVariables(double* t0, double* t, double* dt, double* T, long int* trans_steps);
void InitialConditions(fftw_complex* u_hat, double* u, const long int* N);
void ApplyDealiasing(fftw_complex* array, int array_dim, const long int* N);
void InitializeSpaceVariables(double** x, int** k, const long int* N);
void InitializeFFTWPlans(const long int* N, const long int* NTBatch);
void FreeMemory(RK_data_struct* RK_data);
void AllocateMemory(const long int* NBatch, RK_data_struct* RK_data);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------