#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import numpy as np
import sklearn.metrics as skm
import os
import sys

# Deep learning metrics for evaluation and prediction

###################### TENSORFLOW FUNCTIONS (added to tensorflow graph) ##############################

@tf.function
def last_time_step_mse(targets, predicts):
    """Custom mse only on the prediction from the last input time step. This metric is used for evaluation of prediction."""
    return keras.metrics.mean_squared_error(targets[:, -1], predicts[:, -1])

@tf.function
def last_time_step_categorical_accuracy(target_events, predict_events):
    """Compute accuracy for one hot events"""
    return keras.metrics.categorical_accuracy(target_events[:, -1], predict_events[:, -1])

@tf.function
def last_time_step_CategoricalCrossentropy(target_events, predict_events):
    """Compute categorical cross entropy only on last time step (RNN temporal step) used for prediction and evaluation"""
    return keras.metrics.categorical_crossentropy(target_events[:, -1], predict_events[:, -1])

@tf.function
def relative_percentage_error(targets, predicts):
    """Computes relative percentage error: 100 * (y - ypred) / y """
    return 100 * (targets-predicts)/targets

@tf.function
def absolute_percentage_error(targets, predicts):
    """Computes absolute percentage error: 100 * |y - y_pred| / y"""
    return 100 * abs(targets - predicts)/targets

###################### NON TENSORFLOW FUNCTIONS ##############################

def last_time_step_F1_score(target_events, predict_events, ordered_events, not_predicted_events, average=False):
    """Compute F1 score according to settings. Return a list of values (one for each action).
    Arguments:
    target_events: 4D shaped medical action target one hot vector 
    predict_events: 4D shaped medical action predictions from softmax layer
    ordered_events: correct ordered list of events (order used to create vocabulary hash tables before trianing and in OneHotEncodingLayer.
    not_predicted_events: set of not_predicted_events (events only in first window of each patient)
    average: False (by class F1 score), True: macro average F1 score (arithmetic mean)"""
    
    target_events, predict_events = reshape_for_recall_precision_f1score(target_events, predict_events)
    
    metric = tfa.metrics.F1Score(num_classes=len(ordered_events), average=None)

    metric.update_state(target_events, predict_events)

    results = metric.result()
    results = results.numpy()

    if not average:
        result_dict = {ordered_events[i]:results[i] for i in range(0, len(results)) if not ordered_events[i] in not_predicted_events}
        return result_dict
    else: # computes macro average f1 score
        return np.mean([results[i] for i in range(0, len(results)) if not ordered_events[i] in not_predicted_events], dtype=np.float32)
    

########################################################################

def last_time_step_F1_score_macro_average_geometric_mean(target_events, predict_events, ordered_events, not_predicted_events):
    """Computes F1 score macro average using a geometric mean (because of use of imbalanced events dataset)
    Arguments:
    target_events: 4D shaped medical action target one hot vector 
    predict_events: 4D shaped medical action predictions from softmax layer
    ordered_events: correct ordered list of events (order used to create vocabulary hash tables before trianing and in OneHotEncodingLayer.
    not_predicted_events: set of not_predicted_events (events only in first window of each patient) """

    # compute f1_score for each class independently
    f1_score_by_label_dict = last_time_step_F1_score(target_events, predict_events, ordered_events, not_predicted_events, average=False)
    ordered_f1_score_values = np.array([v for v in f1_score_by_label_dict.values()], dtype=np.float32)

    # geometric mean : sqrt{a+b+...}[f-s(classe1)**a x f-s(classe1)**b x ...] with ab, b, c ... corresponding to coefficient of the F1score from each class

    # product
    mul = 1
    for i in range(0, len(ordered_f1_score_values)):
        if not ordered_events[i] in not_predicted_events:
            mul = mul * ordered_f1_score_values[i]

    # root to power of encoding_width = len(ordered_events)
    return np.power(mul, 1/(len(ordered_f1_score_values)))

########################################################################

def reshape_for_recall_precision_f1score(target_events, predict_events):
    """Reshape event targets and inputs for last_time_step_Recall_by_label, last_time_step_Precision_by_label, last_time_step_F1score
    For example from a matrix of shape = (32,10,10,36), the resulting dimension will be shape=(320, 36). Only the last time_step of 
    the 2nd dimension is kept. 
    Arguments:
    target_events: 4D shaped medical action target one hot vector 
    predict_events: 4D shaped medical action predictions from softmax layer
    """

    target_events = target_events[:, -1]
    predict_events = predict_events[:, -1]

    target_events = tf.reshape(target_events, shape=[target_events.shape[0]*target_events.shape[1], target_events.shape[2]])
    predict_events = tf.reshape(predict_events, shape=[predict_events.shape[0]*predict_events.shape[1], predict_events.shape[2]])

    return target_events, predict_events

#######################################################################

def last_time_step_Recall_by_label(target_events, predict_events, ordered_events, not_predicted_events):
    """Computes the Recall metric for each label/class in the dataset.
    Return a dictionary of recall values (one for each label).
    Arguments:
    target_events: 4D shaped medical action target one hot vector 
    predict_events: 4D shaped medical action predictions from softmax layer
    ordered_events: correct ordered list of events (order used to create vocabulary hash tables before training and in OneHotEncodingLayer.
    not_predicted_events: set of not_predicted_events (events only in first window of each patient)"""

    target_events, predict_events = reshape_for_recall_precision_f1score(target_events, predict_events)

    # get recall percentage for each class in dataset
    results_by_label = {}

    for i in range(0, len(ordered_events)):
        if not ordered_events[i] in not_predicted_events:
            metric = tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=i)
            metric.update_state(target_events, predict_events)
            result = metric.result()
            results_by_label[ordered_events[i]] = result.numpy()
        
    return results_by_label

#######################################################################

def last_time_step_Precision_by_label(target_events, predict_events, ordered_events, not_predicted_events):
    """Computes the Precision metric for each label/class in the dataset.
    Return a list of recall values (one for each label). The position of the value in the list refers to 
    the index associated to an action.
    target_events: 4D shaped medical action target one hot vector 
    predict_events: 4D shaped medical action predictions from softmax layer
    ordered_events: correct ordered list of events (order used to create vocabulary hash tables before training and in OneHotEncodingLayer.
    not_predicted_events: set of not_predicted_events (events only in first window of each patient)"""

    target_events, predict_events = reshape_for_recall_precision_f1score(target_events, predict_events)

    # get recall percentage for each class in dataset
    results_by_label = {}

    for i in range(0, len(ordered_events)):
        if not ordered_events[i] in not_predicted_events:
            metric = tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=i)
            metric.update_state(target_events, predict_events)
            result = metric.result()
            results_by_label[ordered_events[i]] = result.numpy()
    
    return results_by_label

########################################################################

def last_time_step_errors_mean_errors(targets, predicts, nb_parameters, absolute=False):
    """ Computes mean relative percentage errors and relative percentage error: mean(target - predict)/target * 100 for each timestep and parameter
    or mean absolute percentage errors and absolute percentage errors : mean(abs(target-predict))/target * 100
    For predict and targets with RNN, we have to pass only the last prediction for each window (3Dimensional to 2Dimensional). 
    Use for example: targets[:, -1].
    Arguments:
    targets: shape=(dataset_size, output_size*nb_parameters)
    predicts: shape=(dataset_size, output_size*nb_parameters)
    nb_parameters: number of physiological parameters in the dataset
    absolute: boolean to know if we compute absolute percentage error (True) or relative percentage error (False)
    """

    # group by feature (timestep per parameter)
    # shape = (nb_features=nb_timestep*nb_parameters, dataset_size)
    predicts_by_timestep = np.empty(shape=(targets.shape[1], targets.shape[0]))
    targets_by_timestep = np.empty(shape=(targets.shape[1], targets.shape[0]))
    for i in range(0, targets.shape[1]):
        predicts_by_timestep[i] = predicts[:, i]
        targets_by_timestep[i] = targets[:, i]

    # compute relative errors and mape (mean absolute percentage error)
    errors = []

    # different types of percentage errors
    if absolute:
        errors = absolute_percentage_error(targets_by_timestep, predicts_by_timestep)
    else:
        errors = relative_percentage_error(targets_by_timestep, predicts_by_timestep)

    errors = errors.numpy()
    mean_errors = np.mean(errors, axis=1)

    # reshape mean_errors: [ [Param1Val1, Param1Val2 ...],
    #                 [Param2Val1, Param2Val2, ...],
    #                 ...
    #               ]
    mean_errors = np.transpose(np.split(mean_errors, len(mean_errors)/nb_parameters))

    return errors, mean_errors

########################################################################

def NN_accuracy(targets, predicts):
    """Compute the percentage of None prediction among all None targets.
    Return "None" if ZeroDivisionError.
    Arguments:
    targets: 2D list/array targets as strings
    predicts: 2D list/array predictions as strings"""

    total = 0
    n_good_pred = 0
    nn_acc = 0.00

    for target, predict in zip(targets, predicts):
        for event_t, event_p in zip(target, predict):
            if event_t == bytes("None", 'utf8'):
                if event_p == event_t:
                    n_good_pred += 1
                total += 1
    try:
        nn_acc = (n_good_pred/total) * 100 # None_pred/None_targets accuracy
    except ZeroDivisionError():
        nn_acc = "None"
    
    return nn_acc

########################################################################

def NE_percentage_error(targets, predicts):
    """Compute the percentage of None prediction among all non None targets.
    Return None if ZeroDivisionError.
    Arguments:
    targets: 2D list/array targets as strings
    predicts: 2D list/array predictions as strings"""

    total = 0
    n_none_pred = 0
    ne_errp = 0.00

    for target, predict in zip(targets, predicts):
        for event_t, event_p in zip(target, predict):
            if event_t != bytes("None", 'utf8'):
                if event_p == bytes("None", 'utf8'):
                    n_none_pred += 1
                total += 1

    try:
        ne_errp = (n_none_pred/total) * 100 # None_pred/None_targets accuracy
    except ZeroDivisionError():
        ne_errp = "None"

    return ne_errp
    
########################################################################

def action_specific_accuracy(targets, predicts):
    """Action Specific Accuracy
    Compute the percentage of good event prediction among all non None targets
    Arguments:
    targets: 2D list/array targets as strings
    predicts: 2D list/array predictions as strings"""

    total = 0
    n_good_pred = 0
    e_acc = 0.00

    for target, predict in zip(targets, predicts):
        for event_t, event_p in zip(target, predict):
            if event_t != bytes("None", 'utf8'):
                if event_t == event_p:
                    n_good_pred += 1
                total += 1

    try:
        e_acc = (n_good_pred/total) * 100
    except ZeroDivisionError():
        e_acc = "None"

    return e_acc

########################################################################

def string_accuracy(targets, predicts):
    """Categorical accuracy with strings instead of one hot vectors
    Custom function to use with list of list of strings
    Arguments: 
    targets: 2D list/array targets as strings
    predicts: 2D list/array predictions as strings
    """

    total = 0
    n_good_pred = 0

    for target, predict in zip(targets, predicts):
        for event_t, event_p in zip(target, predict):
            if event_p == event_t:
                n_good_pred += 1
            total += 1
    
    return (n_good_pred/total) * 100
    
########################################################################    

def accuracy_with_lag(targets, predicts, delay=1):
    """Categorical accuracy allowing a time delay in the prediction of an action.
    For each target, we look in current and neighbour timestep predictions according to delay.
    Arguments:
    targets: 2D list/array targets as strings
    predicts: 2D list/array predictions as strings
    delay: authorized delay (before and after) to count a prediction as a good prediction - int
    Exemple: delay = 1 will allow a shift of one timestep for a good prediction
    """
    total = 0
    n_good_pred = 0

    # for each couple of windows (target, prediction) we itarate through targets and for each target, find if a good prediction
    # is in the neighborhood and make total
    for target, predict in zip(targets, predicts):
        for i, event_tp in enumerate(zip(target, predict)): # event tp = (event_target, event_predict)
            event_t = event_tp[0]
            # compute range for iteration through direct neighbours of current event
            min_range = int( bool( (i - delay) >= 0 ) ) * (i - delay) # min_range = i - delay if i > 0, else min_range = 0
            max_range = i + delay

            # get list of neighbour predictions according to delay
            neighbour_predictions = [predict[j] for j in range(min_range, max_range, 1)]
            
            if event_t in neighbour_predictions:
                n_good_pred += 1
            total += 1
    
    return (n_good_pred/total) * 100

#########################################################################

def create_multiclass_confusion_matrix(target_events, predict_events, ordered_events, not_predicted_events, output_file):
    """Creates the multiclass confusion matrix. Lines correspond to true indices and columns to predicted indices.
    The function returns the confusion matrix.
    Arguments:
    target_events: 4D shaped target events matrix
    predict_events: 4D shaped predict events matrix
    ordered_events: list of uniq events in the dataset. The order corresponds to the indices used to create lookup table during training.
    not_predicted_events: set of not_predicted_events (events only in first window of each patient)
    output_file: confusion_matrix_csv output filepath"""

    # get 2D matrix from original 3D matrix
    target_events, predict_events = reshape_for_recall_precision_f1score(target_events, predict_events)

    # convert tensor to numpy
    target_events = target_events.numpy()
    predict_events = predict_events.numpy()

    # get indices of max value in each event vector
    predict_events = np.argmax(predict_events, axis=-1)
    target_events = np.argmax(target_events, axis=-1)
    
    # get confusion matrix (multiclass)
    # lines (toward bottom): true indices (in order) -> actual
    # columns (toward right): prediction indices (in order) -> predictions
    confusion_matrix = skm.confusion_matrix(target_events, predict_events, labels=np.arange(0, len(ordered_events), 1))

    write_confusion_matrix_to_csv(confusion_matrix, ordered_events, not_predicted_events, output_file)
    
    return confusion_matrix
    
#########################################################################

def write_confusion_matrix_to_csv(confusion_matrix, ordered_events, not_predicted_events, output_file):
    """Writes confusion matrix from sklearn.metrics.confusion_matrix in a csv file.
    Arguments:
    confusion_matrix: confusion matrix from sklearn.metrics.confusion_matrix
    ordered_events: list of uniq events in the dataset. The order corresponds to the indices used to create lookup table during training.
    not_predicted_events: set of not_predicted_events (events only in first window of each patient)
    output_file: confusion_matrix_csv output filepath"""

    with open(output_file, 'w') as f:
        f.write("actual/prediction,") # writes upper left corner block
        for event in ordered_events: # write headers (prediction event names-> cols)
            if not event in not_predicted_events:
                f.write(f"{event},")
        f.write('\n')
        
        for i, actual in enumerate(confusion_matrix):
            if not ordered_events[i] in not_predicted_events:
                f.write(f"{ordered_events[i]},") # write actual event name
                for j, val in enumerate(actual):
                    if not ordered_events[j] in not_predicted_events:
                        f.write(f"{val},") # write value between current actual and all predict events (columns) 
                f.write('\n')
    
    # check file creation
    if os.path.exists(output_file):
        print(f"\n[CreateFileSuccess] Multiclass confusion matrix file create as {output_file}.")
    else:
        sys.stderr.write(f"[CreateFileError] Unable to create multiclass confusion matrix file {output_file}.\n")
    
    return

#########################################################################