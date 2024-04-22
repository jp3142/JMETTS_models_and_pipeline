#!/usr/bin/env python3

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def fetch_boxplot_summaries_for_all_folds(run_folder, n_folds):
    """Reads boxplot summaries of the first n_folds folders. 
    Return a multidimensional dictionary with:
    keys = physiological_parameter, values = dictionary with 
    keys = boxplot metric name, values = dictionary with
    keys = timestep number, values = list of all fold values for the corresponding physiological parameter / metric_name / timestep
    Arguments:
    run_folder: path to run_folder to consider
    n_folds: number of folds on which to compute general run's results"""

    # get directory element lists
    run_dir_list = os.listdir(run_folder)

    # find in run_dir_list every element that aren't fold folders
    to_delete = []
    for i in range(0, len(run_dir_list)):
        if run_dir_list[i].find('fold') == -1: # if fold isn't find in run_dir_list elm name
            to_delete.append(run_dir_list[i])

    # delete all elements that aren't fold folders
    for elm in to_delete:
        indice = run_dir_list.index(elm)
        del run_dir_list[indice]

    # keep only n_folds first fold
    run_dir_list = run_dir_list[0:n_folds]

    # init dictionary to store every values
    run_dict = {}

    # go through the different folds
    for fold_folder in run_dir_list:
        fold_folder = os.path.join(run_folder, fold_folder)

        # read boxplot_summary file of current fold
        boxplot_summary_file = os.path.join(fold_folder, "custom_evaluation/relative_percentage_error_boxplots/boxplot_summaries.csv")
        run_dict = read_one_boxplot_summary_csv_file(run_dict, boxplot_summary_file)

    return run_dict


def read_one_boxplot_summary_csv_file(run_dict, boxplot_summary_file):
    """Update run_dictionary (run_dict) by reading a boxplot_summaries.csv file.
    If a key doesn't exists, it is automatically created.
    Arguments:
    run_dict: multidimensional dictionary with:
    keys = physiological_parameter, values = dictionary with 
    keys = boxplot metric name, values = dictionary with
    keys = timestep number, values = list of all fold values for the corresponding physiological parameter / metric_name / timestep
    boxplot_summary_file: boxplot_summaries.csv file to read."""

    current_param = ""
    headers = []
    count_timesteps = 0
    run_dict = run_dict

    with open(boxplot_summary_file, 'r') as f:
        for line in f:
            
            line = line.rsplit('\n')[0].split(',') # get rid of \n at end of line and split csv line with delimiter ','

            if line[0] == '': # to not consider blank lines
                # re init variables for each physiological parameter
                count_timesteps = 0
                headers = []
                current_param = ""
                continue
            
            # if line with physio param name
            if len(line) == 1:
                current_param = line[0]
                if not current_param in run_dict.keys():
                    run_dict[current_param] = {}
                continue

            # if line with headers (metric names)
            if "means" in line:
                headers = line
                for metric in headers:
                    if not metric in run_dict[current_param].keys():
                        run_dict[current_param][metric] = {}
                continue
            
            # else
            for head in headers:
                if not count_timesteps in run_dict[current_param][head].keys():
                    run_dict[current_param][head][count_timesteps] = []
                run_dict[current_param][head][count_timesteps].append(line[headers.index(head)])
            
            count_timesteps += 1
    
    return run_dict

#####################################################################

def compute_global_stat_by_param_physio_by_metric_by_timesteps(run_dict):
    """Computes the mean of values for all fold for each physiological parameter, each metric and each timestep within run_dictionary.
    Arguments:
    run_dict: multidimensional dictionary with:
    keys = physiological_parameter, values = dictionary with 
    keys = boxplot metric name, values = dictionary with
    keys = timestep number, values = list of all fold values for the corresponding physiological parameter / metric_name / timestep"""

    for param_physio in run_dict.keys():
        for metric in ["means", "nb_outliers", "total_timestep_eval_size", "percent_outliers", "meds", "q1", "q3", "iqr", "stds"]: # raw list of stats to determine global stat values in the good order
            for timestep in run_dict[param_physio][metric]:
                if metric in ["means", "nb_outliers", "total_timestep_eval_size", "percent_outliers"]:
                    run_dict[param_physio][metric][timestep] = np.mean( [ np.float32(v) for v in run_dict[param_physio][metric][timestep] ] ) 
                elif metric == "meds":
                    run_dict[param_physio][metric][timestep] = np.median( [ np.float32(v) for v in run_dict[param_physio][metric][timestep] ] )
                elif metric == "q1":
                    run_dict[param_physio][metric][timestep] = np.percentile( [ np.float32(v) for v in run_dict[param_physio][metric][timestep] ], 25, interpolation='midpoint')
                elif metric == "q3":
                    run_dict[param_physio][metric][timestep] = np.percentile( [ np.float32(v) for v in run_dict[param_physio][metric][timestep] ], 75, interpolation='midpoint')
                elif metric == "iqr":
                    run_dict[param_physio][metric][timestep] = run_dict[param_physio]['q3'][timestep] - run_dict[param_physio]['q1'][timestep]
                elif metric == "stds": # estimated global standard deviation from global iqr
                    run_dict[param_physio][metric][timestep] = (3/4)*run_dict[param_physio]["iqr"][timestep]

    return run_dict

#####################################################################

def compute_whisker_bounds(run_dict):
    """Add whisker bounds to each timestep of each physiological_parameter.
    The whiskers are computes after the computation of the mean among all considered folds of each metric for each physiological parameters 
    Arguments:
    run_dict: multidimensional dictionary with:
    keys = physiological_parameter, values = dictionary with 
    keys = boxplot metric name, values = dictionary with
    keys = timestep number, values = list of all fold values for the corresponding physiological parameter / metric_name / timestep"""

    for param_physio in run_dict.keys():
        run_dict[param_physio]["whisker_up"] = {}
        run_dict[param_physio]["whisker_low"] = {}
        for timestep in run_dict[param_physio]["means"].keys():
            run_dict[param_physio]["whisker_up"][timestep] = run_dict[param_physio]["q3"][timestep]+(run_dict[param_physio]["iqr"][timestep]*1.5)
            run_dict[param_physio]["whisker_low"][timestep] = run_dict[param_physio]["q1"][timestep]-(run_dict[param_physio]["iqr"][timestep]*1.5)

    return run_dict

#####################################################################

def write_boxplots_descriptions(boxplot_csv_filepath, physio_param, run_dict):
    """Writes global run boxplot statistic descriptions in a csv file
    Arguments:
    boxplot_csv_filepath: filepath to csv file in which to write statistics
    physio_param: considered physiological parameter (str)
    run_dict:multidimensional dictionary with:
    keys = physiological_parameter, values = dictionary with 
    keys = boxplot metric name, values = dictionary with
    keys = timestep number, values = list of all fold values for the corresponding physiological parameter / metric_name / timestep"""

    with open(boxplot_csv_filepath, 'a') as f:
            f.write(f"{physio_param}\n") # write headers physio param
            for i, metric in enumerate(run_dict[physio_param].keys()): # write headers metric stats boxplot
                if i == len(run_dict[physio_param].keys()) - 1: # if last metric we skip line
                    f.write(f"{metric}\n")
                else:
                    f.write(f"{metric},")
            
            # writes values for each timestep and for each metric
            for timestep in run_dict[physio_param]["means"].keys():
                for j, metric in enumerate(run_dict[physio_param].keys()):
                    if j == len(run_dict[physio_param].keys()) - 1: # if last metric we skip line
                        f.write(f"{run_dict[physio_param][metric][timestep]}\n")
                    else:
                        f.write(f"{run_dict[physio_param][metric][timestep]},")
            f.write('\n')

#####################################################################

def make_custom_evaluation_boxplots(run_dict, run_folder):
    """Creates global run boxplots and csv boxplot description files
    Arguments:
    run_dict:multidimensional dictionary with:
    keys = physiological_parameter, values = dictionary with 
    keys = boxplot metric name, values = dictionary with
    keys = timestep number, values = list of all fold values for the corresponding physiological parameter / metric_name / timestep
    run_folder: run folder path"""

    # creates global run directoryu in which to store global boxplot
    global_boxplot_dir = os.path.join(run_folder, "global_run_boxplot/") # directory for run's global boxplot
    os.makedirs(global_boxplot_dir, exist_ok=True)

    # set boxplot csv description filepath
    boxplot_csv = os.path.join(global_boxplot_dir, "boxplot_summaries.csv")

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    figsize=(1690*px, 900*px)

    for param in run_dict.keys():
        fig = plt.figure(figsize = figsize)

        # Creating axes instance
        ax = plt.gca()
        ax.set_ylabel("Run global relative percentage error", fontweight="bold", fontsize=25)

        # Creating plot for current param
        stats = []
        whishi = [] # to store all whisker high values and find maximum later
        for timestep in run_dict[param]["means"].keys(): # for each timestep
            stats.append({
                'med':run_dict[param]['meds'][timestep],
                'mean':run_dict[param]['means'][timestep],
                'q1':run_dict[param]['q1'][timestep],
                'q3':run_dict[param]['q3'][timestep],
                'whislo':run_dict[param]['whisker_low'][timestep],
                'whishi':run_dict[param]['whisker_up'][timestep]
            })
            whishi.append(run_dict[param]['whisker_up'][timestep])
        
        boxplot = ax.bxp(stats, showfliers=False, patch_artist=True, showmeans=True, meanline=True, showcaps=True, showbox=True)

        #boxplot = ax.bxp(, patch_artist=True, showmeans=True, meanline=True, showcaps=True, showbox=True, showfliers=True)
        plt.title(f"{param}")
        
        colors = [ 
            "orangered", "darkred", "yellow" ,"red", "cyan", "gold", "lightgreen", "purple", "blue", "pink",
            "darkgreen", "darkorange", "mediumpurple", "aqua", "khaki", "coral", "grey", "azure",
            "seagreen", "navy", "greenyellow", "chocolate", "salmon", "peru", "linen", "stateblue", "antiquewhite",
            "blanchedalmond", "burlywood", "firebrick", "brown", "darkviolet", "mediumaquamarine", "ivory", "goldenrod",
            "bisque", "cornflowerblue", "midnightblue", "lightseagreen", "darkorchid", "thisle", "fuchsia"  
        ] # colors of the different boxes (one per feature )

        # Computing boxplot statistics in order to display values on boxplot

        # Customize boxplot
        ax.set_yticks(np.arange(0, np.max(whishi)+5, 5), labels=None, minor=True)

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
        
        # final filepath to boxplot file
        boxplot_output_file = os.path.join(global_boxplot_dir, f"{param}_run_relative_percentage_error_distrib_per_timestep")

        # try to save the figure
        try:
            plt.savefig(f"{boxplot_output_file}", format='png') #sauvegarde du boxplot
        except:
            sys.stderr.write(f"\n[CreateFileError] Unable to create file '{boxplot_output_file}'.\n")
        else:
            sys.stdout.write(f"\n[CreateFileSuccess] File '{boxplot_output_file}' created !\n")
        
        # write boxplots info in a csv file
        write_boxplots_descriptions(boxplot_csv, param, run_dict)

        # close create figure
        plt.close()

#####################################################################

def main():
    """"""
    try:
        run_folder = sys.argv[1] # run folder to consider
        n_folds = int(sys.argv[2]) # number of folds on which to compute global run performances. Make sure that each fold folder to consider contains the custom_evaluation folder create

    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: make_global_run_boxplots.py <run_folder> <n_folds>")
    except TypeError:
        sys.stderr.write("[ArgumentError] run_folder must be str() type and n_folds must be int() type\n")
        exit(1)

    run_dict = fetch_boxplot_summaries_for_all_folds(run_folder, n_folds)
    run_dict = compute_global_stat_by_param_physio_by_metric_by_timesteps(run_dict)
    run_dict = compute_whisker_bounds(run_dict)
    make_custom_evaluation_boxplots(run_dict, run_folder)

    print("\n[END] Global run boxplots created succesfully\n")

    return 0

if __name__ == "__main__":
    main()