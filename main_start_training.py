#!/usr/bin/env python3

import sys
import os
from fct.data_preprocessingLSTM import *
from fct.GridSearch import *
from classes.custom_metrics import *
from classes.model_classes import *
from classes.layer_classes import *
from fct.main_funcs import *
from main_start_training_2 import *

################################################

# CONSTANTS
SEED=42 # seed for all np.random and skutils.shuffle calls

###############################################

def main():
    """ MAIN function"""

    # Call arguments check
    try:
        parameters_file = sys.argv[1] # hyperparameters file
        csv_folder = sys.argv[2] # serie values dataset folder
        events_folder = sys.argv[3] # events dataset folder
        tensorboard_dir_name = str(sys.argv[4]) # corresponds to the experiments dir (containing all experiments)
        experiment_name = str(sys.argv[5]) # corresponds to experiment
    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: python3 main_start_training.py <parameters_file> <csv_folder> <events_folder> <tensorboard_dir_name>\n")
        exit(1)

    np.random.seed(SEED) # use same random state to obtain same split on all subdatasets for all runs

    # get params
    multivalues_param_dict = get_parameters(parameters_file) # get one hyperparameters dictionary with all parameter's names as keys and a list of all possible options values as value

    # make list of param dictionaries (param grid)
    param_dict_list = make_GridParams(multivalues_param_dict) # make a list of hyperparameter dictionaries representing the grid
    param_dict_list = check_encoder_decoder_params(param_dict_list, multivalues_param_dict) # check bad parameter combinations for encoder-decoder models
    
    # Get tensorboard folder
    root_logdir = os.path.join(os.getcwd(), os.path.join(tensorboard_dir_name, experiment_name))
    os.makedirs(root_logdir, exist_ok=True) # creates directories if not exist

    print('\n') # skip one line for formating
        
    # for each parameter dictionaries
    for run_num, run_param in enumerate(param_dict_list):
        # start run
        _, _ = start_run(run_param, run_num, root_logdir, experiment_name, csv_folder, events_folder, SEED) # start run

    os.system(f"python3 fct/best_experiment.py {tensorboard_dir_name}") # start after ccipl script to compute best experiment models
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    print(f"\nMAIN 1 - Returned exit code {exit_code}\n")
