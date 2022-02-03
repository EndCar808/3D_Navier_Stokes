/**
* @file utils.c  
* @author Enda Carroll
* @date Feb 2022
* @brief File containing the utilities functions for the pseudospectral solver
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

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"




// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function to read in arguements given at the command line upon execution of the solver
 * @param  argc The number of arguments given
 * @param  argv Array containg the arguments specified
 * @return      Returns 0 if the parsing of arguments has been successful
 */
int GetCMLArgs(int argc, char** argv) {

    // Initialize Variables
    int c;
    int dim_flag = 0;
    int force_flag = 0;
    int output_dir_flag = 0;

    return 0;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------