# The results directory 
res_dir="$1"

# Where the results directory is located
post_in_dir="/work/projects/TurbPhase/Phase_Dynamics_Navier_Stokes/Dublin_Results/RESULTS_3D/$res_dir"

# Where to store the stats data
post_out_dir="/home/ecarroll/PhD/3D_Navier_Stokes/Data/Stats/"

# Tag to go with the stats data file
post_file_tag="$2"

fftw_threads=1

# Run the post executable command to compute and save the stats data
post_cmd="time PostProcessing/bin/main -i $post_in_dir -o $post_in_dir -t $post_file_tag -p 1 -p $fftw_threads"
echo -e "\nCommand run:\n\t \033[1;36m $post_cmd \033[0m"
$post_cmd

# # Run the plotting command once this is done
# plot_cmd="python3 Plotting/plot_increment_pdfs.py -i $post_in_dir$res_dir -t $post_file_tag-402"
# echo -e "\nCommand run:\n\t \033[1;36m $plot_cmd \033[0m"
# $plot_cmd