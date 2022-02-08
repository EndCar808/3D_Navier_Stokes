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
void InitialConditions(fftw_complex* w_hat, double* u, fftw_complex* u_hat, const long int* N);
void InitializeSpaceVariables(double** x, int** k, const long int* N);
void InitializeFFTWPlans(const long int* N, const long int* NTBatch);
void FreeMemory(RK_data_struct* RK_data);
void AllocateMemory(const long int* NBatch, RK_data_struct* RK_data);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------