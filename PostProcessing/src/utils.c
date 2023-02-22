/**
* @file utils.c  
* @author Enda Carroll
* @date Feb 2023
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
#include <time.h>
#include <sys/time.h>
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
    int output_dir_flag = 0;
    int input_dir_flag  = 0;
    int threads_flag    = 0;

    // -------------------------------
    // Initialize Default Values
    // -------------------------------
    // Output & Input file directory
    strncpy(file_info->output_dir, "NONE", 1024);  // Set default output directory to the Tmp folder
    strncpy(file_info->input_dir, "NONE", 1024);  // Set default output directory to the Tmp folder
    strncpy(file_info->output_tag, "No-Tag", 64);
    sys_vars->num_threads      = 1;
    sys_vars->num_fftw_threads = 1;
    
    // -------------------------------
    // Parse CML Arguments
    // -------------------------------
    while ((c = getopt(argc, argv, "o:i:p:t:")) != -1) {
        switch(c) {
            case 'o':
                if (output_dir_flag == 0) {
                    // Read in location of output directory
                    strncpy(file_info->output_dir, optarg, 1024);
                    output_dir_flag++;
                }
                break;
            case 'i':
                if (input_dir_flag == 0) {
                    // Read in location of input directory
                    strncpy(file_info->input_dir, optarg, 1024);
                    input_dir_flag++;
                }
                break;       
            case 'p':
                // Get the number of threads to use
                if (threads_flag == 0) {
                    sys_vars->num_threads = atoi(optarg); 
                    if (sys_vars->num_threads <= 0) {
                        fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], number of OMP threads must be greater than or equal to 1, umber provided ["CYAN"%d"RESET"]\n--->> Now Exiting!\n", "sys_vars->num_threads", sys_vars->num_threads);
                        exit(1);
                    }
                    threads_flag = 1;
                    break;
                }
                else if (threads_flag == 1) {
                    sys_vars->num_fftw_threads = atoi(optarg); 
                    if (sys_vars->num_fftw_threads <= 0 || sys_vars->num_fftw_threads > 16) {
                        fprintf(stderr, "\n["RED"ERROR"RESET"]: Error in reading in command line agument ["CYAN"%s"RESET"], number of FFTW threads must be greater than or equal to 1 and less than 16, number provided ["CYAN"%d"RESET"]\n--->> Now Exiting!\n", "sys_vars->num_fftw_threads", sys_vars->num_fftw_threads);
                        exit(1);
                    }
                    threads_flag = 2;
                    break;  
                }
                break;
            case 't':
                // Get the tag for the output file
                strncpy(file_info->output_tag, optarg, 64);
                break;
            default:
                fprintf(stderr, "\n["RED"ERROR"RESET"] Incorrect command line flag encountered\n");     
                fprintf(stderr, "Use"YELLOW" -o"RESET" to specify the output directory\n");
                fprintf(stderr, "Use"YELLOW" -i"RESET" to specify the input directory\n");
                fprintf(stderr, "Use"YELLOW" -t"RESET" to specify the tag for the output file\n");
                fprintf(stderr, "\n-->> Now Exiting!\n\n");
                exit(1);
        }
    }

    return 0;
}
/**
 * Converts time in seconds into hours, minutes, seconds and prints to screen
 * @param start The wall time at the start of timing
 * @param end   The wall time at the end of timing
 */
void PrintTime(time_t start, time_t end) {

    // Get time spent in seconds
    double time_spent = (double)(end - start);

    // Get the hours, minutes and seconds
    int hh = (int) time_spent / 3600;
    int mm = ((int )time_spent - hh * 3600) / 60;
    int ss = time_spent - (hh * 3600) - (mm * 60);

    // Print hours minutes and second to screen
    printf("Time taken: ["CYAN"%5.10lf"RESET"] --> "CYAN"%d"RESET" hrs : "CYAN"%d"RESET" mins : "CYAN"%d"RESET" secs\n\n", time_spent, hh, mm, ss);

}
// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------