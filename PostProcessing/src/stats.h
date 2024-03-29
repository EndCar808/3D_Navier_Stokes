/**
* @file hdf5_funcs.h  
* @author Enda Carroll
* @date Feb 2023
* @brief File containing function prototpyes for stats file
*/
// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
//  Function Prototpyes
// ---------------------------------------------------------------------
void AllocateStatsObjects(void);
void FreeStatsObjects(void);
void Precompute(void);
void ComputeStats(int s);
void WriteStatsData(const int snap, int indx);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------