#!/usr/bin/env python3

import time
import os
import numpy as np
import sys
import pickle
import io

# Tensorboard run log dir

def get_run_logdir(root_logdir, run_num, n_layer_1, n_layer_2):
    """Get subdirectory name in which to put logs of a specific run
    n_layer_1 and n_layer_2 refers to number of layers in encoder and decoder
    Arguments:
    root_logdir: experiment directory
    run_num: run number corresponding to its id
    n_layer_1: number of layers in encoder
    n_layer_2: number of layers in decoder"""

    run_id = str(time.strftime(f"run_%Y_%m_%d-%H_%M_%S_E{n_layer_1}D{n_layer_2}_id={run_num}")) # create run name with date and a random integer called id
    
    return os.path.join(root_logdir, run_id), run_id # return path to run directory and run identifier (run folder name)

#######################################################################

def update_dict_values(dict_to_update, dict_to_add):
    """Append values of dict_to_add to corresponding `dict_to_update` keys. If the key doesn't exist
    in dict_to_update, the entry will be automatically created for you.
    Arguments:
    dict_to_update: dict() - keys: whatever, values: list() of values
    dict_to_add: dict() - keys: whatever, values: single value for each key
    """
    for key in dict_to_add.keys():
        if key not in dict_to_update.keys():
            dict_to_update[key] = list()

        dict_to_update[key].append(dict_to_add[key])
                    
    return dict_to_update

#######################################################################

def get_mean_std_dict_from_dict_of_lists(dict_of_lists):
    """ Compute mean and standard deviation at each key of a dictionary float32 values lists.
    Return 2 dictionaries with the same keys as `dict_of_lists` argument, one for the mean values and another one for the standard deviation values.
    Arguments:
    dict_of_lists: dict - keys: whatever, values: list of float32 values"""
    mean_dict = {}
    std_dict = {}

    for key in dict_of_lists.keys():
        mean_dict[key] = np.mean(dict_of_lists[key], dtype=np.float32)
        std_dict[key] = np.std(dict_of_lists[key], dtype=np.float32)
        
    return mean_dict, std_dict

#######################################################################

def replace_last_decoder_layer_neurons(param_dict, encoding_width, nb_parameters, split_outputs=False):
    """ If split_outputs is set to True, will replace last decoder layer number of neurons to make split_function work whatever is the user input (the script won't work otherwise).
    Set number of neurons of last decoder layer to output_size*(encoding_width+nb_parameters) so split_function can split event end regular outputs
    Arguments:
    param_dict: parameters dictionary
    encoding_width: number of events in dataset
    nb_parameters: number of serie values variables
    split_outputs: boolean. Set to True only if split_output function is used in call method of model
    """

    if split_outputs: # if split_outputs is True, number of neurons in last decoder layer will be set to output_size*(encoding_width+nb_parameters)
        if param_dict.get('decoder_neurons')[-1] != param_dict.get('output_size')*(encoding_width+nb_parameters):
            param_dict['decoder_neurons'][-1] = param_dict.get('output_size')*(encoding_width+nb_parameters)
            print(f"[ParametersWarning] Setting last decoder layer number of neurons to {param_dict.get('decoder_neurons')[-1]} for split outputs function (output_size = {param_dict.get('output_size')}).\n")

        # check if this change keeps encoder last number of neurons and decoder first number of neurons equal (in case of only one layer)
        try:
            assert param_dict['decoder_neurons'][0] == param_dict['encoder_neurons'][-1] 
        except AssertionError:
            print(f"\n[ParametersWarning] First decoder layer's number of neurons \"{param_dict.get('decoder_neurons')[0]}\" is incompatible with last encoder layer number of neurons \"{param_dict.get('encoder_neurons')[-1]}\".\n")
            exit(7)

    return param_dict

#######################################################################

def serialize(param_dict, tmp_file):
    """Serialize a parameters dictionary
    Arguments:
    param_dict: parameters dictionary
    tmp_file: name for the binary temporary file
    """
    with open(tmp_file, 'wb') as handle:
        pickle.dump(param_dict, handle)

    return tmp_file

#######################################################################

def deserialize(tmp_file, del_file=False):
    """Deserialize parameters dictionary binary file and delete it after reading
    Arguments:
    tmp_file: parameters dictionaryu tmp file
    del_file: Set to true if you want to delete the binary file after reading (once you won't need it anymore)
    """
    try:
        with open(tmp_file, 'rb') as f:
            depickler = pickle.Unpickler(f)
            data = depickler.load()
    except FileNotFoundError:
        sys.stderr.write(f"[FileNotFoundError] Unable to open rb file \"{tmp_file}\" (serialized object)")
        exit(9)
    else: # delete file after loading
        if del_file:
            os.remove(tmp_file)

    return data

###############################################

def get_n_nodes(n_tasks_tot, n_tasks_per_node):
     """Compute the number of required nodes to parallelize all tasks, each task allocates one core in CCIPL
     Arguments:
     n_tasks_tot: total number of parameters combinations
     n_tasks_per_node: number of tasks on each node"""

     n_nodes = n_tasks_tot//n_tasks_per_node
     if n_tasks_tot % n_tasks_per_node > 0:
         n_nodes += 1
    
     return n_nodes

##############################################

def get_model(model_name, available_models):
    """Check if the model exists in a dictionary of available models before loading model.
    Return the model if the name corresponds to any entry in available_models. If the model name doesn't exist, throw an error
    and stops program.
    Arguments:
    model_name: parameters dictionary model name entry
    available_models: dictionary of available models in model_classes.py file. key = class name, value = class instance"""
    model = None # instanciate model to keep it out of try except block
    try:
        model = available_models[model_name]
    except KeyError:
        sys.stderr.write(f"[ModelNameError] No matching model for {model_name} in classes/model_classes.py file.\n")
        exit(10)
    
    return model

#############################################

def write_vocabulary_table(uniq_events, output_path, name="vocabulary_table.csv"):
    """Write the index_to_events dictionary to keep track of vocabulary indexing.
    Return final concatenated path to the vocab file.
    Arguments:
    uniq_events: list of uniq different events in dataset
    output_path: output_directoryu to save the file
    name: name of the output file"""

    full_path = os.path.join(output_path, name)

    with open(full_path, 'w') as f:
        f.write("index,medical action\n")
        for i in range(0, len(uniq_events)):
            f.write(f"{i},{uniq_events[i]}\n")

    return full_path 

#############################################

def write_model_summary_to_file(model, output_filename="model_summary.txt"):
    """Gets tensorflow model's summary as a string and writes it to a file
    Arguments:
    model: tensorflow model object
    output_filename: name of the output file containing the model's summary."""

    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()

    with open(output_filename, 'w') as f:
        f.write(summary_string)
    
    return summary_string

############################################

def add_global_multiplied_loss(eval_dict):
    """Adds a metric called multiplied_loss = eval_regression_loss * eval_event_loss.
    This metric can be used to select best models after evaluation step. It represents the global evaluation loss metric.
    This metric can't be used as a loss for gradient descent.
    Arguments:
    eval_dict: evaluation dictionary from Tensoflow evaluation method."""

    eval_dict['multiplied_loss'] = np.float32(eval_dict['regression_output_loss']*eval_dict['event_output_loss'])

    return eval_dict

############################################

def add_last_time_step_multiplied_loss(eval_dict):
    """Adds a metric calculated called last_time_step_multiplied_loss = eval_last_time_step_regression_loss * eval_last_time_step_event_loss.
    This metric can be used to select best models after evaluation step. It represents the evaluation loss metric obtained from last_time_step prediction.
    This metric can't be used for gradient descent.
    Arguments:
    eval_dict: evaluation dictionary from Tensoflow evaluation method."""

    eval_dict['last_time_step_multiplied_loss'] = np.float32(eval_dict['regression_output_last_time_step_mse']*eval_dict['event_output_last_time_step_CategoricalCrossentropy'])

    return eval_dict

############################################

def scale_loss_according_to_weights(history_dict, eval_dict, loss_weight, event_loss_weight):
    """Multiply losses to their corresponding weight in order to have dictionaries with correctly
    scaled losses. This function is specific to metric names used in EncoderDecoder classes tensorflow's compile method
    Arguments:
    history_dict: tensorflow dictionary of training and validation metrics and losses. 
    eval_dict: tensorflow dictionary of evaluation metric and losses
    loss_weight: weight associated to regression losses
    event_loss_weight: weight associated to event losses"""

    # scale history dict loss values
    for kh in history_dict.keys():
        hist_split = kh.split('_')
        if ("event" in hist_split and "loss" in hist_split) or "CategoricalCrossentropy" in hist_split:
            history_dict[kh] = event_loss_weight*history_dict[kh]
        elif ("regression" in hist_split and "loss" in hist_split) or "mse" in hist_split:
            history_dict[kh] = loss_weight*history_dict[kh]
    
    # scale eval_dict loss values
    for ke in eval_dict.keys():
        eval_split = ke.split('_')
        if ("event" in eval_split and "loss" in eval_split) or "CategoricalCrossentropy" in eval_split:
            eval_dict[ke] = event_loss_weight*eval_dict[ke]
        elif ("regression" in eval_split and "loss" in eval_split) or "mse" in eval_split:
            eval_dict[ke] = loss_weight*eval_dict[ke]
    
    return history_dict, eval_dict