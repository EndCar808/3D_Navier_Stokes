import numpy as np 
import h5py as h5
import sys
import getopt
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
if mpl.__version__ > '2':    
       mpl.rcParams['text.usetex'] = True
       mpl.rcParams['font.family'] = 'serif'
       mpl.rcParams['font.serif']  = 'Computer Modern Roman'
from matplotlib import pyplot as plt

class tc:
    H         = '\033[95m'
    B         = '\033[94m'
    C         = '\033[96m'
    G         = '\033[92m'
    Y         = '\033[93m'
    R         = '\033[91m'
    Rst       = '\033[0m'
    Bold      = '\033[1m'
    Underline = '\033[4m'

#################################
##          MISC               ##
#################################
def parse_cml(argv):

    """
    Parses command line arguments
    """

    ## Create arguments class
    class cmd_args:

        """
        Class for command line arguments
        """

        def __init__(self, in_dir = None, out_dir = None, info_dir = None, plotting = False):
            self.in_dir         = in_dir
            self.out_dir_info   = out_dir
            self.in_file        = out_dir
            self.plotting       = plotting
            self.tag = "None"


    ## Initialize class
    cargs = cmd_args()

    try:
        ## Gather command line arguments
        opts, args = getopt.getopt(argv, "i:o:f:t:", ["plot"])
    except Exception as e:
        print("[" + tc.R + "ERROR" + tc.Rst + "] ---> Incorrect Command Line Arguements.")
        print(e)
        sys.exit()

    ## Parse command line args
    for opt, arg in opts:

        if opt in ['-i']:
            ## Read input directory
            cargs.in_dir = str(arg)
            print("\nInput Folder: " + tc.C + "{}".format(cargs.in_dir) + tc.Rst)

            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        if opt in ['-o']:
            ## Read output directory
            cargs.out_dir = str(arg)
            print("Output Folder: " + tc.C + "{}".format(cargs.out_dir) + tc.Rst)

        elif opt in ['-f']:
            ## Read input directory
            cargs.in_file = str(arg)
            print("Input Post Processing File: " + tc.C + "{}".format(cargs.in_file) + tc.Rst)

        elif opt in ['--plot']:
            ## Read in plotting indicator
            cargs.plotting = True

        elif opt in ['-t']:
            cargs.tag = str(arg)

    return cargs


def compute_pdf(hist, bin_edges, normed = False, remove_zeros = False):

	# Convert edges to bin centres
	bin_centres = (bin_edges[1:] + bin_edges[:-1]) * 0.5
	bin_wdt     = bin_edges[1] - bin_edges[0]

	# Remove 0s if necessary
	if remove_zeros:
		## Get only non_zero counts
		non_zero_indx = np.argwhere(hist != 0)
		bin_centres = bin_centres[non_zero_indx]
		hist = hist[non_zero_indx]

    # Compute the pdf from the counts and bins data
	pdf = hist / (np.sum(hist) * bin_wdt)

	# Normalize the pdf and centres by the std deviation - normalized so that the PDF has std =1
	if normed:
		# Compute the std 
		std = np.sqrt(np.sum(pdf * bin_centres**2 * bin_wdt))
		bin_centres /= std
		bin_wdt /= std
		pdf *= std

	return pdf, bin_centres, bin_wdt



if __name__ == "__main__":

	# -------------------------------------
	# # --------- Parse Commnad Line
	# -------------------------------------
	cmdargs = parse_cml(sys.argv[1:])
	
	# Construct input file path
	input_file_path = cmdargs.in_dir + "/Stats_HDF_Data_TAG[{}].h5".format(cmdargs.tag)

	with h5.File(input_file_path, 'r') as f:
		print("Reading C Data")
		u_incr_hist = f["LongitudinalVelIncrements_BinCounts"][:, :]
		u_incr_ranges = f["LongitudinalVelIncrements_BinRanges"][:, :]

	# Plot the PDFs combined on one plot
	text_width=12.25
	plt.figure(figsize=(text_width, 1.5 * text_width))

	num_incrs = u_incr_hist.shape[0]
	colors    = plt.cm.magma(np.linspace(0, 0.75, num_incrs))

	for r in range(num_incrs):
		
		print("Plotting C Data Combined {}".format(r))

		pdf, bin_centres, _ = compute_pdf(u_incr_hist[r, :], u_incr_ranges[r, :], normed = True, remove_zeros = True)
		if r == (num_incrs - 1):
			plt.plot(bin_centres, pdf / 12**r, label=r"$r_{max} = \pi / N$", color = colors[r])
		else:
			plt.plot(bin_centres, pdf / 12**r, label=r"$r_{} = {}\Delta x$".format(r, 2**r), color = colors[r])
	plt.yscale('log')
	plt.ylabel(r"$\sigma$ PDF")
	plt.xlim(-27.5, 25)
	plt.grid()
	plt.legend()
	plt.xlabel(r"$\delta_{\parallel}\mathbf{u} / \sigma$")
	plt.savefig(cmdargs.out_dir + "/PDF_c_COMBINED.png".format(r))
	plt.close()

	# # Plot the individual PDFs
	# for r in range(u_incr_hist.shape[0]):
		
	# 	print("Plotting C Data {}".format(r))

	# 	plt.figure()
	# 	pdf, bin_centres, _ = compute_pdf(u_incr_hist[r, :], u_incr_ranges[r, :])
	# 	plt.plot(bin_centres, pdf, label=r"$r = {}$".format(r))
	# 	plt.yscale('log')
	# 	plt.ylabel(r"PDF")
	# 	plt.legend()
	# 	plt.xlabel(r"$\delta_{\parallel}\mathbf{u}$")
	# 	plt.savefig(cmdargs.out_dir + "/PDF_c_r{}.png".format(r))
	# 	plt.close()

