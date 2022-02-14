/**
* @file hdf5_funcs.h  
* @author Enda Carroll
* @date Feb 2022
* @brief File containing function prototpyes for hdf5_funcs file
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
void CreateOutputFilesWriteICs(const long int* N, double dt);
void GetOutputDirPath(void);
void WriteDataToFile(double t, double dt, long int iters);
void WriteGroupDataFourier(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, fftw_complex* data);
void WriteGroupDataReal(double t, int iters, hid_t group_id, char* dset_name, hid_t dtype, int dset_rank, hsize_t* dset_dims, hsize_t* slab_dims, hsize_t* mem_space_dims, int offset_Nx, double* data);
hid_t CreateGroup(hid_t file_handle, char* filename, char* group_name, double t, double dt, long int iters);
void FinalWriteAndCloseOutputFile(const long int* N, int iters, int save_data_indx);
hid_t CreateComplexDatatype(void);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------