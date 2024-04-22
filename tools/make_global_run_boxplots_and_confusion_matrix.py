#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# This script computes the global errors boxplots for obtained from every execution of the model on n first folds of a run for each physiological parameter
# and each timestep. This script also create a global_run_boxplot_summaries.csv file.
# Moreover this script also computes the global confusion matrix of the run computed on n first folds.

def read_raw_distribution_file(fold_dirpath, output_size, nb_physio_parameters):
    """Reads absolute and relative raw distribution files of a fold and 
    return a dictionary with keys = physiological parameter names, values = dict with keys = timestep header, values = distribution values for current timestep
    Arguments:
    fold_dirpath: path to fold directory
    output_size: model's output size
    nb_physio_parameters: number of physiological parameters taken into account by the model"""

    # get raw distribution files filepath
    custom_dir = os.path.join(fold_dirpath, "custom_evaluation")
    absolute_dir = os.path.join(custom_dir, "absolute_percentage_error_boxplots")
    relative_dir = os.path.join(custom_dir, "relative_percentage_error_boxplots")
    absolute_distribution_file = os.path.join(absolute_dir, "absolute_percentage_error_evaluation_raw_distribution.csv")
    relative_distribution_file = os.path.join(relative_dir, "relative_percentage_error_evaluation_raw_distribution.csv")

    # create dict to store raw distributions
    absolute_distribution = {}
    relative_distribution = {}

    # store line indices where there is a physiological parameter name
    param_name_lines = [v for v in range(0, (output_size*nb_physio_parameters)+nb_physio_parameters, output_size+1)]

    # read absolute raw distrib file
    with open(absolute_distribution_file, 'r') as f:
        lines = f.readlines()
        
        current_param = ""
        for i, line in enumerate(lines):
            if i in param_name_lines:
                param = line.rsplit('\n')[0]
                absolute_distribution[param] = {}
                continue

            line = line.rsplit('\n')[0].split(',')
            absolute_distribution[param][line[0]] = [] # add timestep number to current param dict entry (line[0] correspo nds to timestep x)

            for v in line[1:]: # for values for corresponding timestep line
                absolute_distribution[param][line[0]].append(np.float32(v))
    
    # read relative raw distrib file
    with open(relative_distribution_file, 'r') as f:
        lines = f.readlines()
        
        current_param = ""
        for i, line in enumerate(lines):
            if i in param_name_lines:
                param = line.rsplit('\n')[0]
                relative_distribution[param] = {}
                continue

            line = line.rsplit('\n')[0].split(',')
            relative_distribution[param][line[0]] = [] # add timestep number to current param dict entry (line[0] correspo nds to timestep x)

            for v in line[1:]: # for values for corresponding timestep line
                relative_distribution[param][line[0]].append(np.float32(v))

    return absolute_distribution, relative_distribution

########################################################################################################

def convert_dict_list_to_numpy_array(relative_dict_list, absolute_dict_list, nb_physio_parameters, output_size):
    """Converts list of dictionaries (each dictionary comes from read_raw_distribution_file function) into 3D shapes arrays (nb_physiological_parameters, output_size, nb_values)
    relative_dict_list: list of relative errors distribution obtained by read_raw_distribution_file
    absolute_dict_list: list of absolute errors distribution obtained by read_raw_distribution_file
    nb_physio_parameters: number of physiological parameters considered
    output_size: model's output size"""

    relative_array = [ [list() for j in range(0, output_size)] for i in range(0, nb_physio_parameters)]
    absolute_array = [ [list() for j in range(0, output_size)] for i in range(0, nb_physio_parameters)]

    for nb_dict in range(0, len(relative_dict_list)):
        for i, param in enumerate(relative_dict_list[nb_dict].keys()):
            for j, timestep in enumerate(relative_dict_list[nb_dict][param].keys()):
                relative_array[i][j].extend(relative_dict_list[nb_dict][param][timestep])
                absolute_array[i][j].extend(absolute_dict_list[nb_dict][param][timestep])
                
    return np.array(absolute_array), np.array(relative_array)

########################################################################################################

def create_raw_error_distribution_csv_file(errors_data, output_file, headers):
        """Creates file and writes raw errors distribution (on time serie values) for each parameter and each timestep.
        Arguments:
        errors_data: 3D numpy array representing the computed errors on evaluation data: shape = (nb_physiological_parameter, output_size, nb_values)
        output_file: file to create and write
        headers: physiological parameter names in the same order as they are presented in raw distribution csv files"""
        try:
            with open(output_file, 'a') as f:
                for param in range(0, errors_data.shape[0]):
                    f.write(f"{headers[param]}\n")
                    for timestep in range(0, errors_data.shape[1]):
                        f.write(f"timestep {timestep},")
                        for val in range(0, errors_data.shape[2]):
                            if val == errors_data.shape[2] - 1:
                                f.write(f"{errors_data[param, timestep, val]}\n")
                                continue
                            #else
                            f.write(f"{errors_data[param, timestep, val]},")
        except:
            sys.stderr.write(f"\n[CreateFileError] Unable to create file '{output_file}'.\n")
        else:
            sys.stdout.write(f"\n[CreateFileSuccess] File '{output_file}' created !\n")

########################################################################################################

def write_boxplots_descriptions(csv_path, output_size, param, means, stds, meds, q1, q3, iqr, nb_outliers, eval_dataset_size, percent_outliers, whisker_bounds):
        """Writes description of one boxplot file (one physiological parameter) for each timestep
        Arguments:
        csv_path: csv_filepath (string)
        output_size: model's output_size
        param: current physiological parameter to study
        means: list of means for each timestep
        stds: list of standard error for each timestep
        q1: list of first quartile values for each timestep
        q3: list of third quartile values for each timestep
        iqr: list of inter-quatiles range for each timestep
        nb_outliers: number of outliers for each timestep
        eval_dataset_size: dataset size for all timesteps
        percent_outliers: list of percentage of outliers for each timestep
        whisker_bounds: nested list. 2D list. The list in position 0 refers to whisker low bounds list and the list in position 1 refers to whisker high bounds list.
        """

        # set file opening mode ("append" mode if file already exists)
        mode = 'w'
        if os.path.exists(csv_path):
            mode = 'a'
        
        # write in files
        with open(csv_path, mode) as csv:
            csv.write(f"{param}\n")
            csv.write("means,stds,meds,q1,q3,iqr,whisker_low,whisker_high,nb_outliers,total_timestep_eval_size,percent_outliers\n")
            for i in range(0, output_size):
                csv.write(f"{means[i]},{stds[i]},{meds[i]},{q1[i]},{q3[i]},{iqr[i]},{whisker_bounds[0][i]},{whisker_bounds[1][i]},{nb_outliers[i]},{eval_dataset_size},{percent_outliers[i]}\n")
            csv.write('\n') # skip line between each parameter
        
        return

########################################################################################################

def make_custom_evaluation_boxplots(errors_data, global_dir, boxplot_dir, nb_physio_parameters, headers, output_size, absolute=False):
    """
    Create global run boxplots from error distribution data. Also creates the global raw distribution file which gathers the raw error distribution of
    all considered folds.
    Arguments:
    errors_data: 3D shape numpy array representing errors data: shape = (nb_physiological_parameters, output_size, number of values)
    global_dir: directory in which to store the global boxplot subdirectories (boxplot dir)
    boxplot_dir: directory in which to store global boxplot and global raw distribution file
    nb_physio_parameters: number of physiological parameters to consider
    headers: physiological parameter names list in the same order as presented in raw distribution files
    output_size: model's output_size
    absolute: boolean to know if we're ploting asolute or relative errors boxplot
    """
    
    # example: transform shape = (40, 700) to (40/output_size, output_size, 700) and transpose to obtain all the data 2D arrays for each feature
    
    # creates output directory for boxplots
    boxplot_dir = os.path.join(global_dir, boxplot_dir)
    os.makedirs(boxplot_dir, exist_ok=True)

    # set boxplot csv description filepath
    boxplot_csv = os.path.join(boxplot_dir, "boxplot_summaries.csv")

    # set raw distribution file
    errors_distribution_file = os.path.join(boxplot_dir, "global_relative_percentage_error_evaluation_raw_distribution.csv")
    if absolute:
        errors_distribution_file = os.path.join(boxplot_dir, "global_absolute_percentage_error_evaluation_raw_distribution.csv")

    create_raw_error_distribution_csv_file(errors_data, errors_distribution_file, headers)

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    figsize=(1690*px, 900*px)

    for p in range(0, nb_physio_parameters):
        fig = plt.figure(figsize = figsize)

        # Creating axes instance
        ax = plt.gca()
        
        y_label = "Relative percentage error"
        if absolute:
            y_label = "Absolute percentage error"

        ax.set_ylabel(y_label, fontweight="bold", fontsize=25)

        # Creating plot
        boxplot = ax.boxplot(list(errors_data[p]), patch_artist=True, showmeans=True, meanline=True, showcaps=True, showbox=True, showfliers=True)
        plt.title(f"{headers[p]}")
        
        colors = [ 
            "orangered", "darkred", "yellow" ,"red", "cyan", "gold", "lightgreen", "purple", "blue", "pink",
            "darkgreen", "darkorange", "mediumpurple", "aqua", "khaki", "coral", "grey", "azure",
            "seagreen", "navy", "greenyellow", "chocolate", "salmon", "peru", "linen", "stateblue", "antiquewhite",
            "blanchedalmond", "burlywood", "firebrick", "brown", "darkviolet", "mediumaquamarine", "ivory", "goldenrod",
            "bisque", "cornflowerblue", "midnightblue", "lightseagreen", "darkorchid", "thisle", "fuchsia"  
        ] # colors of the different boxes (one per feature )

        # Computing boxplot statistics in order to display values on boxplot

        # statistic parameters for each timestep of the current physiological parameter
        means = []
        stds = []
        meds = []
        q1 = [] # lower quartile
        q3 = [] # upper quartile
        iqr = [] # interquarile region
        whisker_bounds = [ list(), list() ] # whiskers
        nb_outliers = [] # nb of outliers for each timestep
        eval_dataset_size = len(errors_data[p][0]) # all timesteps evaluation dataset size
        percent_outliers = [] # proportion of outliers

        for t, timestep in enumerate(errors_data[p]):
            # compute means
            means.append(np.mean(timestep))
            # compute std error
            stds.append(np.std(timestep))
            # compute median
            meds.append(np.median(timestep))
            # find the 1st quartile
            q1.append(np.quantile(timestep, 0.25))
            # find the 3rd quartile
            q3.append(np.quantile(timestep, 0.75))
            # finding the iqr region
            iqr.append(q3[t]-q1[t])
            # finding upper and lower whiskers (extrémités moustache)
            whisker_bounds[0].append(q1[t]-(1.5*iqr[t])) # lower bound
            whisker_bounds[1].append(q3[t]+(1.5*iqr[t])) # upper bound
            # finding outliers
            outliers = timestep[(timestep <= whisker_bounds[0][t]) | (timestep >= whisker_bounds[1][t])]
            # finding number of outliers
            nb_outliers.append(len(outliers))
            # finding outliers percentage
            percent_outliers.append((nb_outliers[t]/eval_dataset_size)*100)

        # Customize boxplot
        ax.set_yticks(np.arange(0, np.max(errors_data[p])+5, 5), labels=None, minor=True)

        # Change box colors
        for patch, color in zip(boxplot['boxes'], colors[0:len(boxplot['boxes'])]):
            patch.set_facecolor(color)

        # Modify whiskers
        for i, whisker in enumerate(boxplot['whiskers']):
            whisker.set(color = '#443266', linewidth = 1.5) # modify color

        # Change caps colors
        for cap in boxplot['caps']:
            cap.set(color = '#443266', linewidth = 2)

        # Chage flyers style
        for flier in boxplot['fliers']:
            flier.set(marker = 'D', color = '#e7298a', alpha = 0.5)
        
        image_name = f"{headers[p]}_relative_percentage_error_distrib_per_timestep"
        if absolute:
            image_name = f"{headers[p]}_absolute_percentage_error_distrib_per_timestep"

        output_path = os.path.join(boxplot_dir, image_name) #chemin d'accès final du fichiercsv

        # try to save the figure
        try:
            plt.savefig(f"{output_path}", format='png') #sauvegarde du boxplot
        except:
            sys.stderr.write(f"\n[CreateFileError] Unable to create file '{output_path}'.\n")
        else:
            sys.stdout.write(f"\n[CreateFileSuccess] File '{output_path}' created !\n")
        
        # write boxplots info 
        write_boxplots_descriptions(boxplot_csv, output_size, headers[p], means, stds, meds, q1, q3, iqr, nb_outliers, eval_dataset_size, percent_outliers, whisker_bounds)

        # close create figure
        plt.close()

########################################################################################################

def get_confusion_matrix(fold_dir):
    """Reads a confusion matrix file and return a numpy matrix.
    Arguments:
    fold_dir: fold directory path"""

    confusion_matrix_file = os.path.join(fold_dir, "custom_evaluation/confusion_matrix.csv")

    confusion_matrix = []
    headers = []
    with open(confusion_matrix_file, 'r') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.rsplit('\n')[0].split(',')
            if i == 0:
                headers = line[0:-1]
                confusion_matrix = np.empty(shape=(len(headers)-1, len(headers)-1), dtype=int)
                continue
            
            confusion_matrix[i-1] = [int(v) for v in line[1:-1]] # line[1:] because we don't want to keep line headers of the confusion matrix

    return headers, confusion_matrix
    
########################################################################################################

def write_confusion_matrix(global_dir, confusion_matrix, headers):
    """Writes a confusion matrix to a csv file
    Arguments:
    global_dir: directoryu in which the confusion matrix will be stores
    confusion_matrix: numpy 2D array (matrix) representing the confusion matrix
    headers: confusion matrix headers list"""

    try:
        with open(os.path.join(global_dir, "global_confusion_matrix.csv"), 'w') as f:
            # write headers
            for n, head in enumerate(headers):
                if n == len(headers) - 1:
                    f.write(f"{head}\n")
                    break
                f.write(f"{head},")

            # write values 
            for i in range(0, confusion_matrix.shape[0]):
                f.write(f"{headers[i+1]},")
                for j in range(0, confusion_matrix.shape[1]):
                    if j == confusion_matrix.shape[1] - 1:
                        f.write(f"{confusion_matrix[i, j]}\n")
                        break
                    f.write(f"{confusion_matrix[i, j]},")
    except:
        sys.stderr.write(f"\n[CreateFileError] Unable to confusion_matrix file'.\n")
    else:
        sys.stdout.write(f"\n[CreateFileSuccess] Confusion matrix file created !\n")
    
########################################################################################################

def main():
    """Main function"""
    try:
        n_folds = int(sys.argv[1]) # number of folds on which to compute statistics and create boxplot
        run_path = sys.argv[2] # path to run folder to consider
        output_size = int(sys.argv[3])
        nb_physio_parameters = int(sys.argv[4])
    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: python3 make_global_run_boxplots_and_confusion_matrix.py <n_folds> <run_path> <model_output_size> <nb_physio_parameters>\n")
        exit(1)
    except TypeError:
        sys.stderr.write("[TypeError] One or several arguments are of wrong type: Usage: \n")
        exit(2)

    # set list of fold folders
    run_dir_elm = os.listdir(run_path)

    # remove non fold directories from run_dir_elm
    to_delete = []
    for elm in run_dir_elm:
        splitted = elm.split('_')
        if "fold" not in splitted:
            to_delete.append(elm)
    for d in to_delete:
        del run_dir_elm[run_dir_elm.index(d)]

    # limit fold list to only the first n folds
    run_dir_elm = run_dir_elm[0:n_folds]

    # read raw absolute and relative distribution fold's files / read confusion matrix files for each folds
    absolute_dict_list = []
    relative_dict_list = []
    confusion_headers = []
    confusion_matrix = []

    for i, fold_dir in enumerate(run_dir_elm):
        fold_dir = os.path.join(run_path, fold_dir)

        # get raw distributions
        abs_dict, relative_dict = read_raw_distribution_file(fold_dir, output_size, nb_physio_parameters)
        absolute_dict_list.append(abs_dict)
        relative_dict_list.append(relative_dict)

        # get confusion matrix
        if i == 0: # first iteration, we keep headers for confusion matrix and initialize confusion matrix
            confusion_headers, current_confusion_matrix = get_confusion_matrix(fold_dir)
            confusion_matrix = current_confusion_matrix
        else: # we don't keep headers, we add old confusion matrix with current confusion matrix
            _, current_confusion_matrix = get_confusion_matrix(fold_dir)
            confusion_matrix = current_confusion_matrix + confusion_matrix

    # get physio param names
    headers = [k for k in absolute_dict_list[0].keys()]
    timesteps = [t for t in absolute_dict_list[0][headers[0]].keys()]
    
    # gather every dict entries into a 3D shape numpy array
    absolute_errors_array, relative_errors_array = convert_dict_list_to_numpy_array(relative_dict_list, absolute_dict_list, nb_physio_parameters, output_size)

    # create boxpots and gathered raw distribution file
    global_dir = os.path.join(run_path, "global_run_boxplots_confusion_matrix")
    make_custom_evaluation_boxplots(absolute_errors_array, global_dir, "absolute_percentage_error_boxplots", nb_physio_parameters, headers, output_size, absolute=True)
    make_custom_evaluation_boxplots(relative_errors_array, global_dir, "relative_percentage_error_boxplots", nb_physio_parameters, headers, output_size, absolute=False)

    # create global confusion matrix file
    write_confusion_matrix(global_dir, confusion_matrix, confusion_headers)

    print(f"\n[END] Global run's boxplots and run's confusion matrix created with success for the {n_folds} first folds of run {run_path}.\n")

    return 0

if __name__ == "__main__":
    main()