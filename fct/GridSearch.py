#!/usr/bin/env python3

import itertools as it
import os
import sys
import numpy as np

def make_GridParams(multivalues_param_dict):
    """Create the parameters grid represented by a list of parameter dictionaries with only
    one value for each key.
    Arguments:
    multivalues_param_dict: parameter dictionary with a list of values associated to at least one of the keys
    """

    allNames = sorted(multivalues_param_dict) # keep name order in dictionaries to avoid mixing up
    
    list_params_dict = {} # dictionary of multivalue parameters
    non_list_param_names = [] # # list of of single value parameters' name

    # check for non list / range parameters
    for k in allNames:
        if isinstance(multivalues_param_dict[k], list):
            list_params_dict[k] = multivalues_param_dict[k]
        else:
            non_list_param_names.append(k)

    # create all combinations between list parameters 
    combinations = it.product(*[multivalues_param_dict[Name] for Name in list_params_dict])
        
    # create list parameters dictionaries
    list_params_name = [name for name in allNames if name in list_params_dict.keys()]
    dict_list = []
    for combi in combinations:
        current_param_dict = dict()
        for param_name, param_value in zip(list_params_name, combi):
            current_param_dict[param_name] = param_value
        dict_list.append(current_param_dict)

    # add constant parameters to list parameters dictionaries to create final dictionaries
    for _ in non_list_param_names: # _ = name -> used as range
        for param_dict in dict_list:
            for k in non_list_param_names:
                param_dict[k] = multivalues_param_dict[k]

    # force keep same batch size between training, validation and evaluation steps if value=None for eval and val batch size
    for param_dict in dict_list:
        if param_dict.get('eval_batch_size') is None:
            param_dict['eval_batch_size'] = param_dict['batch_size']

        if param_dict.get('validation_batch_size') is None:
            param_dict['validation_batch_size'] = param_dict['batch_size']
    
    # if n_best_models set to None, class all models from the experiment from best to worst
    # change n_best_models for all param_dicts
    if dict_list[0].get('n_best_models') is None:
        for dictp in dict_list:
            dictp['n_best_models'] = len(dict_list)
  
    return dict_list

##################################################################################

def check_encoder_decoder_params(dict_list, multivalues_param_dict):
    """Delete impossible encoder decoder number of neurons combinations. The first layer of a decoder should always have
    the same number of neurons as the last layer of the encoder.
    Return the updated list of parameter ditionaries.
    Arguments:
    dict_list: list of parameters dictionaries
    multivalues_param_dict: parameter dictionary with a list of values for the value associated to at least one key
    """

    # delete impossible encoder decoder params combinations
    dict_index_to_delete = []
    for d in range(0, len(dict_list)):
        if dict_list[d].get('encoder_neurons')[-1] != dict_list[d].get('decoder_neurons')[0]:
            dict_index_to_delete.append(d)
    
    # delete dictionaries with incorrect combinations
    dict_index_to_delete.sort(reverse=True) # reverse order to not whange indexes while deleting and have IndexError message
    for i in dict_index_to_delete:
        del dict_list[i]

    # look for unused sublist in encoder_neurons or decoder_neurons parameters
    unused_sublist = []
    for sublist in multivalues_param_dict.get('encoder_neurons'):
        found = False
        for param_dict in dict_list:
            if sublist in param_dict.get('encoder_neurons'):
                found = True
                break
        if not found:
            unused_sublist.append(sublist)
    
    # error if any unused sublist
    if not unused_sublist: # if list not empty
        sys.stdout.write("[GridSearchError] Some parameters sublists have not been used in parameters file. Unused parameter sublists: {unused_sublist}.\nPlease check parameters file.\n")
        exit(7)

    return dict_list

##################################################################################   

def write_results_eval_headers_in_csv(eval_dict, prefix, file_handler, last_headers=False):
    """Write the different types of evaluation metrics headers (add suheaders for metrics associated to each class of medical action in the
    dataset)
    Arguments:
    eval_dict: any evaluation_dictionary (keys: metric, values: dictionary of single values in each key 
    (key = class name, value = metric value computed for this class) or single value
    prefix: String indicating in which part of the pipeline we are. [mean_eval, std_eval, eval]
    file_handler: python file object
    last_headers: boolean to know if we're writing the headers of the last evaluation dictionary"""

    for j, param_name in enumerate(eval_dict.keys()):
        skip_line = ","
        # to create new header for metrics returning values in dictionary (f1_score by class (average = None)...)
        if isinstance(eval_dict[param_name], dict):
            param_name = [f"{str(param_name)}_{str(event)}" for event in eval_dict[param_name].keys()]
            for evnt in range(0, len(param_name)):
                if j == len(eval_dict.keys())-1 and evnt == len(param_name)-1 and last_headers:
                    skip_line = "\n"
                file_handler.write(f"{prefix}_{str(param_name[evnt])}{skip_line}")
            continue
        
        if j == len(eval_dict.keys())-1 and last_headers:
            skip_line = "\n"
        
        file_handler.write(f"{prefix}_{str(param_name)}{skip_line}")

##################################################################################

def create_results_csv(results_directory, history_dicts, eval_dicts, filename="tmp", num=0):
    """Create csv results file and add column names. Return the full csv results filepath.
    Arguments:
    results_directory: directory in which to store file named by filename argument.
    experiment_dirname: experiment directory path
    history_dicts: tuple of dictionaries of final training and/or validation losses and metrics 
    eval_dicts: tuple of dictionary of final evaluation metrics and/or losses
    filename: name of the csv file to be created
    num= model number representing the loop iteration current indice (used for paralelization). Avoid recreating
    the file for each fold of a run.
    """
    
    results_csv_filepath = os.path.join(results_directory, filename) # create filepath for results file
    
    if num == 0:
        if isinstance(history_dicts, tuple) and isinstance(eval_dicts, tuple):
            mean_history_dict, std_history_dict = history_dicts
            mean_eval_dict, std_eval_dict = eval_dicts

            # add headers (column names) to the file
            with open(results_csv_filepath, 'w') as f:
                f.write('name,')
                
                # write history headers (metrics and losses' names)
                for param_name in mean_history_dict.keys():
                    f.write("mean_"+str(param_name)+',')
                
                for param_name in std_history_dict.keys():
                    f.write("std_"+str(param_name)+',')

                # write mean_evaluation metric headers
                write_results_eval_headers_in_csv(mean_eval_dict, "mean_eval", f, last_headers=False)

                # write standard deviation_evaluation metric headers
                write_results_eval_headers_in_csv(std_eval_dict, "std_eval", f, last_headers=True)

        else: # if not std and means dictionary tuples  
            with open(results_csv_filepath, 'w') as f:
                f.write('name,')

                # write historyu metric names
                for param_name in history_dicts.keys():
                    f.write(f"{str(param_name)},")
                
                # write fold results in run csv file
                write_results_eval_headers_in_csv(eval_dicts, "eval", f, last_headers=True)

        os.system(f"chmod 777 {results_csv_filepath}") # give permission to other precessors to write in this file

    return results_csv_filepath

##################################################################################

def write_eval_metric_results_dict_format_in_csv(eval_dict, file_handler, last_values=False):
    """Write evaluation metric dictionary format results in csv file.
    Arguments:
    eval_dicts: tuple of dictionary of final evaluation metrics and/or losses
    file_handler: python file object
    last_values: boolean to indicates if it's the last dictionary (True), else False"""
    for j, param_name in enumerate(eval_dict.keys()):
        skip_line = ","
        # to create new header for metrics returning values in dictionary (f1_score by class (average = None)...)
        if isinstance(eval_dict[param_name], dict):
            events = [event for event in eval_dict[param_name].keys()]
            for i, evnt in enumerate(events):
                if j == len(eval_dict.keys())-1 and i == len(param_name)-1 and last_values:
                    skip_line = "\n"
                file_handler.write(f"{str((eval_dict[param_name])[evnt])}{skip_line}")
            continue
        
        if j == len(eval_dict.keys())-1 and last_values:
            skip_line = "\n"
        
        file_handler.write(f"{str(eval_dict[param_name])}{skip_line}")

##################################################################################

def write_in_csv(results_csvTable_filepath, run_logdir, history_dicts, eval_dicts):
    """Add losses and metrics values from one fold to the corresping columns in the results csv file
    Arguments: 
    results_csvTable_filepath: file path to csv file containing all results of a run or experiment
    run_logdir: run directory
    history_dicts: tuple of dictionary or simple dictionary containing headers as keys and values for current run or fold
    evals_dict: tuple of dictionary or simple dictionary containing headers as keys and values for current run or fold
    """

    # if tuple of dictionaries, unpack mean and stds dictionaries for training and evaluation
    if isinstance(history_dicts, tuple) and isinstance(eval_dicts, tuple):
        mean_history_dict, std_history_dict = history_dicts
        mean_eval_dict, std_eval_dict = eval_dicts

        with open(results_csvTable_filepath, 'a') as f:
            f.write(os.path.basename(run_logdir)+',')

            for col in mean_history_dict.keys():
                f.write(str(mean_history_dict.get(col))+',')
            
            for col in std_history_dict.keys():
                f.write(str(std_history_dict.get(col))+',')
            
            write_eval_metric_results_dict_format_in_csv(mean_eval_dict, f, last_values=False)

            write_eval_metric_results_dict_format_in_csv(std_eval_dict, f, last_values=True)
    else: # if not tuple of dictionaries

        with open(results_csvTable_filepath, 'a') as f:
            f.write(os.path.basename(run_logdir)+',')
            for col in history_dicts.keys():
                f.write(str(history_dicts.get(col))+',')
            
            write_eval_metric_results_dict_format_in_csv(eval_dicts, f, last_values=True)
    return

##################################################################################

def get_by_metric_by_fold_or_run_dict(result_dict_by_fold, headers):
    """Get all run's values for a specific metric and return a dictionary with keys = run_name and values = metric value for the current run
    Arguments:
    result_dict_by_fold: dictionary with keys = fold names and values = list of all losses and metrics values (the metric values should correspond to the metrics' names order in headers)
    headers: list of metric names (str) without fold_name header.
    """

    metric_dict = {}

    for metric_name in headers: # for each metric name in headers (without fold_name)
        metric_dict[metric_name] = {fold:result_dict_by_fold[fold][headers.index(metric_name)] for fold in result_dict_by_fold.keys()} # get metric value

    return metric_dict

##################################################################################

def convert_dict_of_dict_values_to_float(lookup):
    """Convert values of each dictionary in lookup (which is also a dictionary) to numpy float
    Arguments:
    lookup: dictionary of dictionaries: keys = metric_name, values = dict with keys= fold_name, values = metric value 
    for the corresponding fold. (str type)"""

    for k1 in lookup.keys():
        for k2 in lookup[k1].keys():
            try:
                lookup[k1][k2] = np.float32(lookup[k1][k2]) # convert value to numpy float if possible
            except ValueError: # if we try to convert string (like "None" if ZeroDivisionError-> nn_acc, ne_pe, acc_with_lag, string_acc) -> keep str type
                lookup[k1][k2] = str(lookup[k1][k2])
    
    return lookup

##################################################################################

def get_best_values_dict_by_metric(lookup, n_models, reverse=True):
    """Return a dictionary containing the dictionary entries with best values among the n first models (folds / runs).
    Arguments:
    lookup: dictionary with keys = metric names, values = dictionaries with key = fold / run name and values = metric value for this fold/run
    n_models: number of best models to select.
    reverse: bool used for the sorted function reverse argument. If false, sort the values in 
    ascending order. If true, sort the values in descending order."""
    
    best_dict = {}

    for metric_name in lookup.keys(): # for each metric in lookup
        one_metric_best_dict = {} # dictionary with keys = run/fold names, values = current metric values for each fold or run

        sorted_values = sorted(list(lookup[metric_name].values()), reverse=reverse)[0:n_models] # get all fold or run values for one metric and sort them from best to worse
        
        # find back the corresponding fold/run to sorted values
        for v in range(0, len(sorted_values)):
            for key, val in lookup[metric_name].items():
                if sorted_values[v] == val:
                    one_metric_best_dict[key] = sorted_values[v]

        # add one_metric_best_dict to the corresponding metric name key in best_dict
        best_dict[metric_name] = one_metric_best_dict

    return best_dict

############################################################### 

def write_best_models_file(best_result_filepath, best_metric_dict_of_dicts):
    """Write best metrics / losses dicts values and names in a csv best results file
    Arguments:
    best_results_filepath: filepath to the file to create
    best_metric_dicts_of_dicts: dictionary with key = metric name associated to the dictionary, value = dictionaries with keys = fold names and values = corresponding metric value for the run
    """

    with open(best_result_filepath, 'w') as f:

        for metric_name in best_metric_dict_of_dicts.keys():
            # write headers
            f.write(f"{metric_name}\n")
            f.write(f"ranking,name,value\n")

            rank=1 # ranking counter (best_metric_dict already ordered from best to worse)
            for model_name, val in best_metric_dict_of_dicts.get(metric_name).items():
                f.write("{},{},{:.8f}\n".format(rank, model_name, np.float32(val)))
                rank+=1

            f.write('\n') # skip a line between each metric ranking

    if os.path.exists(best_result_filepath):    
        print("[CreateFileSuccess] Best models' results file created.\n")
    else:
        sys.stderr.write("[CreateFileError] A problem occured while creating the best models' results file.\n")

    return 

############################################################### 

def best_models(results_filepath, dirpath, n_models, best_filename, experiment_best=False):
    """Orders models within a run or within a fold (all folds of a run or all run of an experiment) from best models to worse for each metric.
    Create a csv file to store results.
    Arguments:
    results_filepath: results_csv_table file (results from experiment or from a specific run)
    dirpath: full path to run or fold directory
    n_models: number of best models to keep
    best_filename: best results filename
    experiment_best: boolean to know if we're creating best model dictionaries for a run (experiment_best=False) or for a whole experiment (experiment_best=True)
    """
    
    best_models_file = os.path.join(dirpath, best_filename)

    with open(results_filepath, 'r') as f:
        lines = f.readlines()

        # get csv headers
        headers = lines[0].split(',')

        if headers[-1] == '\n':
            headers = headers[0:-1] # get rid of '\n' entry
        else:
            headers[-1] = headers[-1].rsplit('\n')[0] # get rid of "\n" at the end of last metric name

        # remove headers from file content
        lines = lines[1:]

        result_dict_by_name = {} # init dict with key = run name and values = list of metric losses (same order as headers)

        # Get all model values for each metric / loss
        # for each run
        for line in lines:
            line = line.split(',')
            line[-1] = line[-1].rsplit('\n')[0]
            # append results to result_by_run dictionary
            result_dict_by_name[line[0]] = [line[i] for i in range(1, len(line))]

        # Best model(s) determination #
        by_metric_by_name_dict = {} # init dictionary with key = metric_name : value = dict with key = fold_name, value = metric value for this run 

        headers = headers[1:] # remove fold_name header column

        # get a dictionary with keys = metric name, values = dictionary with keys = fold_name: value = corresponding value for each fold and metric
        by_metric_by_name_dict = get_by_metric_by_fold_or_run_dict(result_dict_by_name, headers)

        # accuracy, precision and recall metric names selection, to be sorted in descending order (reverse=True) -> tmp dict created
        reverse_true_metrics_dict = {}
        for metric_name, v in by_metric_by_name_dict.items():
            splitted = metric_name.split('_')
            keywords_for_reverse_true_metrics = ["acc", "accuracy", "asa", "precision", "recall", "F1"] # add keywords to this list when using other metrics that have to be sorted in descending order
            for elm in keywords_for_reverse_true_metrics:
                try:
                    splitted.index(elm)
                except ValueError: # valueError raised if index doesn't find element elm in splitted list.
                    continue # don't take ValueError into account, try another element from elm list
                else:
                    reverse_true_metrics_dict[metric_name] = v
                    break # break 2nd loop once we found one correspondance 

         # all the other metrics are sorted with reverse=False (ascending order) -> tmp dict created

        reverse_false_metrics_dict = {k:v for k, v in by_metric_by_name_dict.items() if k not in reverse_true_metrics_dict.keys()}

        # pass std metric from reverse_true dict to reverse_false dict
        if experiment_best:
            to_delete = []
            for keys in reverse_true_metrics_dict.keys():
                splitt = keys.split('_')
                if "std" in splitt:
                    reverse_false_metrics_dict[keys] = reverse_true_metrics_dict[keys]
                    to_delete.append(keys)
            for key in to_delete: # delete keys from reverse_true_metrics_dict that have been added to reverse_false_metrics_dict
                del reverse_true_metrics_dict[key]
        
        # convert values of dict of dicts to numpy float32 (to allow sorting according to float values and not according to str)
        reverse_false_metrics_dict = convert_dict_of_dict_values_to_float(reverse_false_metrics_dict)
        reverse_true_metrics_dict = convert_dict_of_dict_values_to_float(reverse_true_metrics_dict)

        del by_metric_by_name_dict

        # get best values dictionary (n_models best values (folds) for each metric)
        best_reverse_true_metrics = get_best_values_dict_by_metric(reverse_true_metrics_dict, n_models, reverse=True)
        best_reverse_false_metrics = get_best_values_dict_by_metric(reverse_false_metrics_dict, n_models, reverse=False)

        best_dict = best_reverse_true_metrics 
        
        del best_reverse_true_metrics

        # gather reverse_false and reverse_true dictionaries in one single dictionary
        for k, v in best_reverse_false_metrics.items():
            best_dict[k] = v
        
        del best_reverse_false_metrics

    write_best_models_file(best_models_file, best_dict)

###############################################################

def write_model_parameters(run_logdir, param_dict):
    """Writes parameters dictionary keys and values of a specific run in a txt file to avoid long file names
    Arguments:
    run_logdir: run directory path
    param_dict: dictionary of parameters
    """

    filepath = os.path.join(run_logdir, "parameters_resume.txt")

    with open(filepath, 'w') as f:
        for k, v in param_dict.items():
            f.write(f"{k}:{v}\n")

    if not os.path.exists(filepath):
        sys.stderr.write(f"[CreateFileError] Unable to create file \"{filepath}\"\n")
        return
    #else
    sys.stderr.write(f"[CreateFileSuccess] File \"{filepath}\" created.\n\n")
    
    return

###############################################################

# UNUSED FUNCTION --> fitting performances can be seen visually via tensorboard (more practical)
# def compute_fitting_perf(train_dict, val_dict):
#     """Compute the generalization (fitting) capacity of the model corresponding the absolute difference between
#     training losses and metrics and validation losses and metrics. |train_metric - val_metric|.
#     The goal is to detect how much a model is overfitting or underfitting which can lead to generalization problem.
#     Arguments:
#     train_dict: dictionary of loss / metric values during training for all runs. Key: run_name, value: corresponding dictionary metric value.
#     val_dict: dictionary of loss / metric values during validation for all runs. Key: run_name, value: corresponding dictionary metric value.
#     Example: loss_dict = {run1: 12, run2: 15 ...}"""

#     fit_dict = {key:abs(np.float32(train_dict.get(key)) - np.float32(val_dict.get(key))) for key in train_dict.keys()}

#     sorted_values = sorted(list(fit_dict.values()), reverse=False)

#     ordered_dict = {}
    
#     for v in range(0, len(sorted_values)):
#         for key, val in fit_dict.items():
#             if sorted_values[v] == val:
#                 ordered_dict[key] = sorted_values[v]
    
#     return ordered_dict

############################################################### 