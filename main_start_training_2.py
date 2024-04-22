#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras as keras
from fct.data_preprocessingLSTM import *
from fct.GridSearch import *
from classes.custom_metrics import *
from classes.layer_classes import *
from classes.model_classes import *
from fct.main_funcs import *
import keras.backend as K

def start_run(param_dict, run_num, root_logdir, experiment_name, csv_folder, events_folder, SEED):
    """Performs one run loop (all folds or "max_fold" folds (dictionary parameter entry))
    Return 2 dictionaries: one to store training results for all folds of a run and one to store evaluation results for all folds of the considered run.
    Arguments:
    param_dict: dictionary of parameters
    run_num: run id
    root_logdir: experiments directory path (containing all experiments)
    experiment_name: current experiment name
    csv folder: serie values multivariate training dataset folder path
    events_folder: events training dataset folder path
    SEED: common seed to use for all trainings to have comparable results - constant
    """

    print("###################################")
    print(f"############  RUN N°{run_num}  ############")
    print("###################################\n")
    
    # dictionaries to store list of values for each metric and for each fold
    historys_dict = {}
    evals_dict = {}

    # define parameters lookup tables  
          
    available_reg_losses = {
        # add options if needed
        'mse': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.losses.MeanAbsoluteError(),
        'mape': tf.keras.losses.MeanAbsolutePercentageError(),
        'msle': tf.keras.losses.MeanSquaredLogarithmicError(),
        'huber': tf.keras.losses.Huber()
    }

    available_categorical_losses = {
        # add options if needed
        'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy()
    }

    # define customizable lookup parameters tables
    available_optimizers = {
        # add options for each optimizer
        'adam': tf.keras.optimizers.Adam(learning_rate=param_dict.get('lr')),
        'nadam': tf.keras.optimizers.Nadam(learning_rate=param_dict.get('lr')),
        "adamax": tf.keras.optimizers.Adamax(learning_rate=param_dict.get('lr')),
        'sgd': tf.keras.optimizers.SGD(learning_rate=param_dict.get('lr')),
        'adadelta': tf.keras.optimizers.Adadelta(learning_rate=param_dict.get('lr')),
        'adagrad': tf.keras.optimizers.Adagrad(learning_rate=param_dict.get('lr'),),
        'ftrl': tf.keras.optimizers.Ftrl(learning_rate=param_dict.get('lr')),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=param_dict.get('lr'))
    }
    
    # get run directory
    run_logdir, run_id = get_run_logdir(root_logdir, run_num, param_dict.get('n_encoder_layers'), param_dict.get('n_decoder_layers'))
    os.makedirs(run_logdir, exist_ok=True) # create run directoryu

    # declare run results_filepath
    run_results_file = ""

    print(f"### RUN NAME: \"{run_id}\" ###\n")

    # get raw dataset, uniq events in dataset, physiological parameters headers, number of parameters
    whole_dataset, whole_events, uniq_events, headers, nb_parameters = get_multivariate_dataset(csv_folder, events_folder, param_dict)
    _ = write_vocabulary_table(uniq_events, run_logdir) # write vocabulary indexing into file

    # get one hot encoding matching dictionaries
    events_to_index, _, encoding_width = get_onehot_dict(uniq_events)

    # Get dataset folds indices (along patients axis (first axis)) for cross validation
    split_indices = create_cross_validation_datasets_indices(whole_dataset, param_dict, shuffle=False, random_state=None)
    param_dict['max_fold'] =  min([len(split_indices), param_dict.get('max_fold')]) # set number of folds to train. Use the minimum between max fold set by user and max_fold available from cross validation

    # if split_outputs is True replace last decoder layer number of neurons to make split_outputs compatible, no effect otherwise
    param_dict = replace_last_decoder_layer_neurons(param_dict, encoding_width, nb_parameters, split_outputs=param_dict.get('split_outputs'))

    # write params in txt file before training (the file will be present even if all folds of a run didn't finish)
    print("\nWriting parameters file ...\n")
    write_model_parameters(run_logdir, param_dict)

    # for each dataset fold
    for n_fold in range (0, param_dict.get('max_fold')):

        fold_logdir = os.path.join(run_logdir, f"fold_{n_fold+1}") # sub run log dir for model with this specific fold 

        print(f"\n############ FOLD n°{n_fold+1} ############")

        ### Get inputs and targets train, val, and test datasets for current fold + windowing + data preprocessing according to current fold ###

        # create current dataset fold of patients
        x_train, x_val, x_test = dataset_Kfolding(whole_dataset, split_indices[n_fold])
        x_train_events, x_val_events, x_test_events = dataset_Kfolding(whole_events, split_indices[n_fold])

        # create dataset of windows + get targets for each sub datasets
        x_train, y_train, x_train_events, y_train_events = prepare_dataset(x_train, x_train_events, nb_parameters, param_dict, seed=SEED) # split/flatten/normalize/unsplit_unflatten
        x_val, y_val, x_val_events, y_val_events = prepare_dataset(x_val, x_val_events, nb_parameters, param_dict, seed=SEED)
        x_test, y_test, x_test_events, y_test_events = prepare_dataset(x_test, x_test_events, nb_parameters, param_dict, seed=SEED)

        # detect actions that aren't in all target events datasets
        not_predicted_events = {event for l in [detect_not_predicted_target_events(uniq_events, target) for target in [y_train_events, y_val_events, y_test_events]] for event in l}
        write_not_predicted_events_file(not_predicted_events, run_logdir)

        # Preprocessing only training data and targets (only for regular inputs)
        x_train, y_train, y_val, y_test, scalers = standardization_normalization(x_train, y_train, y_val, y_test, param_dict.get('min_norm_threshold'), param_dict.get('max_norm_threshold'), param_dict.get('output_size'))

        # Preprocessing events only on training data and targets shape=(Batch_size, output_size, output_size, encoding_width)
        x_train_events, y_train_events, y_val_events, y_test_events = one_hot_encoding(x_train_events, y_train_events, y_val_events, y_test_events, events_to_index, encoding_width)
        
        # instanciate model for each fold
        available_models = {
            # add entries to this dictionary if you create new model classes
            "CustomEncoderDecoder":CustomEncoderDecoder(param_dict, headers, run_logdir, scalers, uniq_events, encoding_width, not_predicted_events),
            "EncoderDecoder_no_split_outputs":EncoderDecoder_no_split_outputs(param_dict, headers, run_logdir, scalers, uniq_events, encoding_width, not_predicted_events),
            "EncoderDecoder_no_dense_event_layer":EncoderDecoder_no_dense_event_layer(param_dict, headers, run_logdir, scalers, uniq_events, encoding_width, not_predicted_events),
            "EncoderDecoder_with_attention":EncoderDecoder_with_attention(param_dict, headers, run_logdir, scalers, uniq_events, encoding_width, not_predicted_events) # add other model classes if needed
        }

        model = get_model(param_dict.get('model_name'), available_models) # get model according to model name parameter
        
        # compile model (tensorflow method) with losses, scalers for losses, optimizer, and metrics used during training
        model.compile(
            optimizer=available_optimizers[param_dict.get('optimizer')],
            loss = {
                "regression_output":available_reg_losses[param_dict.get('loss')],
                "event_output":available_categorical_losses[param_dict.get('event_loss')]
            },
            metrics = {
                "regression_output": [last_time_step_mse],
                "event_output": [last_time_step_categorical_accuracy, tf.keras.metrics.categorical_accuracy, last_time_step_CategoricalCrossentropy]
            },
            loss_weights= {
                "regression_output":param_dict.get('loss_weight'),
                "event_output":param_dict.get('event_loss_weight')
            }
        )

        # Model callbacks
        callbacks_list = [] # init list of tensorflow callbacks

        # Tensorboard callback
        tensorboard_callbk_fit = keras.callbacks.TensorBoard(fold_logdir, histogram_freq=1) # se charge de la création de tous les répertoires (même parents si besoin) pour stocker les logs,
        callbacks_list.append(tensorboard_callbk_fit)

        if param_dict.get('checkpoint_save'): # add model checkpoint callback if set to True (saves model at every epoch if performances are better from the previous epoch)
            checkpoint_filepath = os.path.join(fold_logdir, f"tmp/checkpoint")
            os.makedirs(checkpoint_filepath, exist_ok=True)

            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            callbacks_list.append(model_checkpoint_callback)

        # Early Stopping callback
        if param_dict.get('early_stopper'): # add early stopper callback if set to True (regularisation technique to avoid overfitting)
            early_stopper_callbk = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=param_dict.get('min_delta'),
                patience=param_dict.get('patience'),
                verbose=1,
                mode='min',
                baseline=None,
                restore_best_weights=True # restore best weights from best performance epoch among amont set of patience epoch
            )
            callbacks_list.append(early_stopper_callbk)

        ### TRAINING PROCESS ###

        # TRAINING (+ VALIDATION at the same time) 
        print("\nStart training ...\n")
        history = model.fit( 
            x=[x_train, x_train_events],
            y={"regression_output":y_train, "event_output":y_train_events}, 
            epochs=param_dict.get('epochs'), 
            validation_data = [[x_val, x_val_events], {"regression_output":y_val, "event_output":y_val_events}],
            batch_size=param_dict.get('batch_size'),
            validation_batch_size=param_dict.get('validation_batch_size'),
            use_multiprocessing=True,
            callbacks=callbacks_list,    #model_checkpoint_callbacks
            verbose=2
        )

        # get model history as a dictionary
        history_dict = history.history

        # get only final metrics and losses values
        for k in history_dict.keys():
            history_dict[k] = history_dict[k][-1]

        # EVALUATION 
        print("\nModel evaluation ...\n")
        eval_dict = model.evaluate( 
            x=[x_test, x_test_events],
            y={"regression_output": y_test, "event_output":y_test_events},
            return_dict=True, 
            batch_size=param_dict.get('eval_batch_size'), 
            use_multiprocessing=True,
            callbacks=[tensorboard_callbk_fit],
        )

        # scale losses accord to losses weights
        history_dict, eval_dict = scale_loss_according_to_weights(history_dict, eval_dict, param_dict.get('loss_weight'), param_dict.get('event_loss_weight'))

        # CUSTOM METRICS EVALUATION (2nd evaluation for custom metrics)
        print("\nModel custom evaluation ...\n")
        custom_eval_dict = model.custom_evaluate(x=[x_test, x_test_events], y={"regression_output": y_test, "event_output":y_test_events}, scalers=scalers, fold_logdir=fold_logdir) # pazss numpy array in custom evaluation for simplicity

        # remove mrpe (mean relative percentage errors) and mape (mean absolute percentage errors) values from custom_eval_dict because they're not used here. You can keep these values if needed, but they are already displayed
        # in boxplot summaries
        del custom_eval_dict['mean_relative_percentage_error_per_feature']
        del custom_eval_dict['mean_absolute_percentage_error_per_feature']

        # saving model only if save model parameter is set to true
        print(f"\nSaving model ... [{param_dict.get('save_model')}]\n")
        model.save_model(fold_logdir) # model will be saved only if "save_model" parameter is set to True

        # METRICS DICTIONARIES UPDATES

        # add history_dict values to historys_dict which is a list of values for each metric and for each fold
        historys_dict = update_dict_values(historys_dict, history_dict)

        # add custom eval metric entries to eval_dict
        for k in custom_eval_dict.keys():
            eval_dict[k] = custom_eval_dict[k]

        # add multiplied_loss and last_time_step_multiplied_loss metrics to evaluation dictionary
        eval_dict = add_global_multiplied_loss(eval_dict)
        eval_dict = add_last_time_step_multiplied_loss(eval_dict)

        # add eval_dict values to evals_dict which is a list of values for each metric and for each fold
        evals_dict = update_dict_values(evals_dict, eval_dict)

        # WRITING IN RUN RESULTS FILE

        # write all fold results in a file corresponding to current run results
        print("\nWriting run's results file ...\n")
        run_results_file = create_results_csv(run_logdir, history_dict, eval_dict, filename="run_results_table.csv", num=n_fold) # we create the file with headers only in the first iteration (first fold)
        write_in_csv(run_results_file, fold_logdir, history_dict, eval_dict) # append current fold values to corresponding metric and losses columns

        # write model's in txt file
        _ = write_model_summary_to_file(model, output_filename=os.path.join(run_logdir, "model_summary.txt"))

        # delete model from memory to delete weights
        K.clear_session()
        del model
    
    # find best models among the current run
    print("\nWriting best_run_models file ...\n")
    best_models(run_results_file, os.path.join(experiment_name, run_logdir), param_dict.get('max_fold'), "best_run_models.txt", experiment_best=False) # n_best_models = len(split_indices) to get a ranking of all fold results 
    
    print(f"\n[END] End of cross validation training for run \"{run_id}\".\n")
    
    return historys_dict, evals_dict
