#!/usr/bin/env python3

from GridSearch import *

#### /!\ TO EXECUTE AT THE END OF A CCIPL JOB ###
# This script computes mean and stds of all metrics from all runs of all experiments in "my_logs/" folder (you can give another name to this folder containing all experiment folders).
# First of all, this script computes the common maximum number of folds available in all experiment dirs. The mean and std of 
# each metric will then be computed according the this number of folds (to have the same number of folds in each experiment
# to enable model comparisons)
#################################################

def find_max_fold(experiments_dir):
    """Returns the maximum common number of folds among all runs of all experiments.
    Allows to compute best score mean and stds on common folds.
    Arguments:
    experiments_dir: directory containing all experiment folders
    """

    # initialization
    min_fold = 999999
    associated_run = ""
    associated_experiment = ""

    for experiment_dir in os.listdir(experiments_dir): # for each experiment (combination of layers in encoder and decoder)
        experiment_dir = os.path.join(experiments_dir, experiment_dir)
        for run_dir in os.listdir(experiment_dir): # for each run
            run_dir = os.path.join(experiment_dir, run_dir)
            with open(os.path.join(run_dir, "run_results_table.csv"), "r") as f:
                nb_fold = sum(1 for line in f)-1
                if nb_fold < min_fold:
                    min_fold = nb_fold
                    associated_run = run_dir
                    associated_experiment = experiment_dir
    
    # write a txt file indicating the minimum number of folds for a run in all the results of all experiments and
    # gives name of experiment and run.
    with open("n_fold_stats.txt", 'w') as minf:
        minf.write(f"experiment: {associated_experiment}\nrun: {associated_run}\nnumber of folds for each run stats: {min_fold}")

    return min_fold # nombre de fold sur lequel on calcul les moyennes et écarts-types pour chaque métrique.

#####################################################################################################################

def read_csv_by_run(run_dir, exp_name, nb_fold):
    """ For each metric in a csv run results table, computes mean and standard error among all folds of the run.
    Return a dictionary with adapted entries representing mean and std of each metric.
    Arguments:
    run_dir: run directory filepath
    nb_fold: number of folds to compute mean and std on. Uses the first nb_fold folds available in each run."""

    metric_dict = {} # pour avoir un dictionnaire en key: nom_metric, value: list of values for all folds
    with open(os.path.join(run_dir, "run_results_table.csv"), 'r') as f:
        lines = f.readlines()
        headers = lines[0].split(',')
        del lines[0]
        headers[-1] = headers[-1].rsplit('\n')[0] # remove \n on last header and keep only metric name not empty list

        for col_name in headers:
            metric_dict[col_name] = [] # instancie list val dans chaque metric key

        for i in range(0, nb_fold):
            lines[i] = lines[i].split(',')
            lines[i][-1] = lines[i][-1].rsplit('\n')[0]
            
            j = 0
            for col, val in zip(headers, lines[i]):
                if j==0:
                    metric_dict[col].append(val)
                else:
                    metric_dict[col].append(np.float32(val))
                j+=1

    # create mean and std for each metric except for first one (run_name) for the run_dir
    run_dict = {"name":f"{exp_name}-{os.path.basename(run_dir)}"} # instanciate dict for mean and std of the run with run_name key == run_dir name
    
    for k in metric_dict.keys():
        if k != "name":
            run_dict[f"mean_{k}"] = np.mean(metric_dict[k])
            run_dict[f"std_{k}"] = np.std(metric_dict[k])
    
    return run_dict

#####################################################################################################################

def write_experiment_results_csv(experiments_dir, run_dict_list, name="experiment_results_table.csv"):
    """Write all experiments run means and std of each metric in a common csv file
    Arguments:
    experiments_dir: directory of experiments path (containing all the experiment dirs)
    run_dict_list: list of dictionaries representing mean and std of each metric for each run.
    name: name of the output csv file
    """

    res_file = os.path.join(experiments_dir, name)
    with open(res_file, 'w') as f:
        for cols in run_dict_list[0].keys():
            f.write(f"{cols},")
        
        f.write('\n')
        
        for run_dict in run_dict_list:
            for v in run_dict.values():
                f.write(f"{v},")
            f.write('\n')
    
    return res_file

#####################################################################################################################

def main():
    """MAIN function"""
    try:
        experiments_dir = sys.argv[1] # directory containing all experiment directories
    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: python3 after_ccipl.py <experiments_dir>\n")
        exit(0)
    
    # find number of folds on which to compute mean and stds
    n_fold = find_max_fold(experiments_dir)

    # get mean and std for each run of each experiment (go through the file tree)
    run_dict_list = []
    experiments = os.listdir(experiments_dir)
    for exp in experiments:
        runs_dir = os.path.join(experiments_dir, exp)
        runs = os.listdir(runs_dir)
        for run in runs:
            run = os.path.join(runs_dir, run)
            run_dict_list.append(read_csv_by_run(run, exp, n_fold))

    # write all means and stds of metrics of all runs in one results_table file (experiments_results_table.csv)
    res_file = write_experiment_results_csv(experiments_dir, run_dict_list, name="experiment_results_table.csv")

    # find best model among all runs from all experiments
    best_models(res_file, experiments_dir, len(run_dict_list), "best_experiment_models.txt", experiment_best=True)

if __name__ == "__main__":
    main()