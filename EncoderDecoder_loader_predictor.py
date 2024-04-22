#!/usr/bin/env python3

import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from classes.custom_metrics import *
from classes.model_classes import *
from fct.data_preprocessingLSTM import *

def get_prediction_logdir(root_logdir):
    """Get subdirectory name in which to put logs of this specific run of the programm"""
    run_id = time.strftime("inferences_%Y_%m_%d-%H_%M_%S") # create run name with date
    return os.path.join(root_logdir, run_id)

####################################################################

def format_params_prediction(param_dict, param_file):
    """Format parameters for talos autoML
    Arguments:
    param_dict: raw parameters dictionary
    param_file: parameters file name
    Return: formatted parameters dictionary"""

    try:
        for k, v in param_dict.items():
            # EDITED FOR ONLY ONE VALUE PER PARAMETERS

            # check end of value (undesired characters)           
            if v[-1] in [',', '.', ' ', '']:
                raise SyntaxError
            
            if v[-1] == '\n':
                param_dict[k] = v[0:-1]

            if v == "None" or v=="":
                param_dict[k] = [None]
                continue
            
            # convert integer parameters
            if k in ['input_size', 'output_size', 'dataset_size'] and v is not None:
                param_dict[k] = int(v)
                continue

    except ValueError:
        sys.stderr.write(f"[ValueError] One or several parameter values are invalid or lacking in file \"{param_file}\".\n")
        exit(2)
    except KeyError:
        sys.stderr.write(f"[KeyError] One or several parameter names are invalid. Unknown parameter(s) in \"{param_file}\".\n.")
        exit(3)
    except TypeError:
        sys.stderr.write(f"[TypeError] One or several parameters aren't of the correct type in file \"{param_file}\".\n")
        exit(4)
    except SyntaxError:
        sys.stderr.write(f"[SyntaxParameterError] Presence of undesired characters in file \"{param_file}\".\n")
        exit(5)

    return param_dict

####################################################################

def get_parameters_prediction(param_file):
    param_dict = {}

    # read parameters and add them to the dictionary
    with open(param_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith(' ') or line.startswith('\n'):
                continue

            line = line.rsplit('\n')
            line = line[0].split(':')

            param_dict[str(line[0])] = str(line[1]) # create dict entry

    param_dict = format_params_prediction(param_dict, param_file)
    
    return param_dict

####################################################################

def plot_exmpl_graphs(true_series, predict_series, x_events, true_events, predict_events, nb_parameters, param_dict, results_dir, headers):
    """
    Plots some example of predictions obtained via input test dataset
    Arguments:
    true_series: target series, numpy array of shape = (test_dataset_size, n_features (no slope feature), input_size+output_size)
    predict_series: predicted series, numpy array of shape (test_dataset_size, n_features (no slope feature), input_size+output_size)
    x_events: input test decoded (string) events - array of shape (test_dataset_size, input_size)
    true_events: target test decoded events - array of shape (test_dataset_size, output_size)
    predict_events: predicted test decoded events - array of shape (test_dataset_size, output_size)
    """

    # check max number of graph to create
    if param_dict.get('dataset_size') > true_series.shape[0]:
        param_dict['prediction_number'] = true_series.shape[0]
        sys.stderr.write(f"[PredictionWarning] The number of plots to create is greater than the actual amount of samples in the test dataset.\n nb_graphs set to {true_series.shape[0]}.\n")

    time_axes = np.arange(0, param_dict.get('input_size')+param_dict.get('output_size'), 1) # time axes (in timesteps)
    
    # plot each mts
    for n in range(0, param_dict.get('dataset_size')):
        fig = plt.figure(figsize=(30,20), dpi=100)
        plt.title(f"MTS prediction {n+1}", fontweight="bold", fontsize=20)
        axes = plt.gca()
        axes2 = axes.twiny()
        axes2.set_xlim(axes.get_xlim())
        axes2.xaxis.set_ticks_position('top')
        axes2.xaxis.set_label_position('top')
        axes2.xaxis.set_ticks(time_axes)

        # create displayable event series
        predicted = [f"{y}/{y_pred}" for y, y_pred in zip(true_events[n], predict_events[n])]

        axes2.xaxis.set_ticklabels(np.concatenate((x_events[n], predicted), axis=-1), rotation=90, color='blue', fontsize=7.5)
        axes2.xaxis.set_tick_params(direction='out', pad=15, length=15)

        # Lists used for y anx x max and min axes ticks
        maxes = []
        mins = []
        for p in range(0, nb_parameters): # for each params
            max_ = np.max(predict_series[n, p])
            min_ = np.min(predict_series[n, p])
            maxes.append(max_)
            mins.append(min_)
            plt.plot(time_axes, predict_series[n, p], label=f"{headers[p]} - prediction", linewidth=1)
            plt.plot(time_axes, true_series[n, p], label=f"{headers[p]} - target", linewidth=1)
            plt.legend(loc="upper left")
        
        axes.xaxis.set_ticks(time_axes)
        axes.yaxis.set_ticks(np.arange(int(min(mins)-5), int(max(maxes)+5), 5))
        axes.xaxis.set_ticks_position('bottom')
        axes.set_ylabel("Physiological parameters value", fontweight="bold", fontsize=15)
        axes.set_xlabel("Time (in timesteps)", fontweight="bold", fontsize=15)
        plt.text(-0.3, max(maxes)+15, "true/predict", rotation=90, fontweight="bold")
        plt.tight_layout()

        os.makedirs(os.path.join(results_dir, "prediction_example_plots/"), exist_ok=True)

        plt.savefig(os.path.join(results_dir, f"prediction_example_plots/prediction_{n+1}.pdf"), dpi=100)
        plt.clf()
        plt.close()

    if param_dict.get('dataset_size') > 0:
        if os.path.exists(os.path.join(results_dir, 'prediction_example_plots/')):
            print("[CreateFileSuccess] Prediction plots created.\n")
        else:
            sys.stderr.write(f"[CreateFileError] An error occured while creating the prediction plots.\n.")

    return 

#######################################################################################

def main():
    """"""
    try:
        model_filepath = sys.argv[1] # relative path to the model dir (saved model)
        series_dataset_filepath = sys.argv[2]
        events_dataset_filepath = sys.argv[3]
        results_dir = sys.argv[4]
        param_file = sys.argv[5]

    except IndexError:
        sys.stderr.write("[Arguments Error] Usage: python3 predictor.py <model_filepath> <series_dataset_filepath> <events_dataset_filepath> <results_dir> <param_file>\n")
        exit(1)

    # get parameters and dataset
    results_dir = get_prediction_logdir(results_dir)

    param_dict = get_parameters_prediction(param_file)
    
    whole_dataset, whole_events, _, headers, nb_parameters = get_multivariate_dataset(series_dataset_filepath, events_dataset_filepath, param_dict)
    inputs, targets, input_events, target_events = prepare_dataset(whole_dataset, whole_events, nb_parameters, param_dict, seed=None) 
    
    nb_parameters = len(headers)

    # Retrieve the config
    model = tf.keras.models.load_model(
        model_filepath,
        custom_objects= {
            "normalization":MinMaxNormalizationLayer,
            "standardization":StandardizationLayer,
            "onehot":OneHotEncodingLayer,
            "SoftMaxLayer":SoftMaxLayer,
            "last_time_step_errors_mean_errors":last_time_step_errors_mean_errors,
            "last_time_step_mse":last_time_step_mse,
            "last_time_step_categorical_accuracy":last_time_step_categorical_accuracy,
            "relative_percentage_error":relative_percentage_error,
            "absolute_percentage_error":absolute_percentage_error,
            "last_time_step_CategoricalCrossentropy":last_time_step_CategoricalCrossentropy,
            "last_time_step_F1_score":last_time_step_F1_score,
            "last_time_step_F1_score_macro_average_geometric_mean": last_time_step_F1_score_macro_average_geometric_mean,
            "last_time_step_Recall_by_label":last_time_step_Recall_by_label,
            "last_time_step_Precision_by_label":last_time_step_Precision_by_label,
            "NN_accuracy":NN_accuracy,
            "NE_percentage_error":NE_percentage_error,
            "action_specific_accuracy":action_specific_accuracy,
            "accuracy_with_lag":accuracy_with_lag,
            "create_multiclass_confusion_matrix":create_multiclass_confusion_matrix,    
        }
    )

    # convert inputs and targets to tensors (keep only last RNN step for targets)
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    input_events = tf.convert_to_tensor(input_events, dtype=tf.string)
    regular_y = tf.convert_to_tensor(targets[:, -1], dtype=tf.float32)
    event_y = tf.convert_to_tensor(target_events[:, -1], dtype=tf.string)
    
    # get the correct call signature function 'export_signature' for inference mode, 'serving_default' for training mode
    infer_call = model.signatures['export_signature'] 
    #print(list(model.signatures.values()))
    #print(list(model.signatures.keys()))

    # run model for predictions
    predictions = infer_call(inputs=inputs, inputs_1=input_events)
    
    # unstack output types
    regular_y_pred = predictions['regression_output']
    event_y_pred = predictions['event_output'] # one hot decoding already take last recurrent timestep so no need to do it manually

    # denormalize and destandardize regular outputs
    regular_y_pred = model.normalization.inverse_transform(regular_y_pred) # input=32 10 80, output: 32 80
    regular_y_pred = model.standardization.inverse_transform(regular_y_pred) # input = 32 80, output = 32 80
    
    # decode one hot events predictions
    event_y_pred = model.onehot.decode(event_y_pred)

    ##### Data formating to prepare ploting #####

    # get rid of slope values
    inputs = inputs[:, :, 0:nb_parameters]

    # reshape windows with regular inputs + regular targets and outputs
    true_series = tf.transpose(tf.concat([inputs, tf.reshape(regular_y, shape=[tf.shape(inputs)[0], param_dict.get('output_size'), nb_parameters*2])[:, :, 0:nb_parameters]], axis=1), perm=[0, 2, 1])
    predict_series = tf.transpose(tf.concat([inputs, tf.reshape(regular_y_pred, shape=[tf.shape(inputs)[0], param_dict.get('output_size'), nb_parameters*2])[:, :, 0:nb_parameters]], axis=1), perm=[0, 2, 1])

    # convert tensors back to numpy arrays
    true_series = np.array(true_series, dtype=np.float32)
    predict_series = np.array(predict_series, dtype=np.float32)
    input_events = np.array(input_events, dtype=object)
    event_y = np.array(event_y, dtype=object)
    event_y_pred = np.array(event_y_pred, dtype=object)
    
    plot_exmpl_graphs(true_series, predict_series, input_events, event_y, event_y_pred, nb_parameters, param_dict, results_dir, headers)
    print("\n[END] End predictions.\n")

    return 0

if __name__ == "__main__":
    main()