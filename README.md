# General Description

**Author**: Julien Paris, Intern in LS2N (Laboratoire des Sciences du Numérique), Nantes University, 2022.

**Authors' note**: I designed this software during my last year Bioinformatics master degree 6 months internship in "Laboratoire des Sciences du Numérique (LS2N)" of Nantes University within the DUKe team (Data User Knowledge). I was supervized by Christine Sinoquet, Associate Professor.

This software performs hyper parameters gridsearch and crossvalidation to optimize neural network model trainings. The neural networks proposed in this software are designed to perform a joint modeling of continuous timeserie values and associated events (literal) records. This software was specifically created for DBLBS (Dama-Boisaubert-Lejus-Bourdeau-Sinoquet) dataset. DBLBS is a realistic chirurgical dataset generator designed by the DUKe team of LS2N laboratory. However this software can be enhanced to any kind of joint data (continuous timeserie values and associated discontinuous events) with the only condition of creating a dataset following the same pattern as DBLBS dataset. The DBLBS dataset is for now specialized in the generation of 4 physiological parameters for one specific chirurgical intervention: 
- FC: heart rate (Fréquence Cardiaque)
- PAS: systolic arterial pressure (Pression Artérielle Systolique)
- PAD: diastolic arterial pressure (Pression Artérielle Diastolique)
- PAM: average arterial pressure (Pression Artérielle Moyenne) ).

Two different versions of this software are proposed.
The sequential version perform a gridsearch sequentially and the parallelized version allows to parallelize the execution of multiple hyper parameters combinations on several core of the CCIPL (Centre de Calculs Intensifs des Pays-de-la-Loire) computation center.

The software was created using Ubuntu, python 3.9.12 and the librairies scikit-learn, tensorflow, and keras.

# Quick vocabulary definition

A specific vocabulary is used in this software. To avoid any misunderstanding, the few needed word definitions are described below:

- **Experiment**: An experiment corresponds to a set of run all gathered in a same directory called the "experiment folder". For instance, an experiment folder can gather all hyper-parameter combinations for a specific combination of number of layers in the used neural network. You can find more details by reading the hyper-parameters and parameters description section.

- **Run**: A run corresponds to a specific hyper-parameters combination. Inside a run folder, we can find different execution of the model on different folds of the dataset. The global run statistics are computed at the end of the last fold for each metric and loss.

- **Fold**: A fold refers to a specific combination of training, validation and evaluation dataset blocks for the cross-validation process.

# Description of working directory

- **main_start_training.py**: Main file. Execute this file to start the training with hyperparameters grid search and cross validation. This script will train a model for each fold of the dataset (cross validation) and for each hyperparameter combinations. The gridsearch process will compute result statistics for each experiment, run and fold in order to select best hyperparameters combination.
- **main_start_training_2.py**: Python script designed to be called by *main_start_training.py* script. This script starts one run at a time.
- **training_parameters.txt**: Txt file to configure the training parameters. The parameter and hyperparameter descriptions can be found in the parameters section.
- **tools**: Directory in which you can find different useful tools. These tools will be described further.
- **classes**: Directory in which you can find diferent python modules:
1. *model_classes.py*: Python module to store all available custom model architecture classes.
2. *layer_classes.py*: Python module to store all available custom layer classes.
3. *custom_metrics.py*: Python module to store model classes associated custom metric functions.
- **fct**: Directory in which you can find different python modules and scripts:
1. *best_experiment.py*: Python script to create best experiment results table and best experiment best file model. This script has to be executed manually at the end of a gridsearch pipeline if using parallelized version. THis script is automatically started at the end of an experiment if using sequential version.
2. *data_preprocessingLSTM.py*: Python datapreprocessing module designed especially for neural networks using LSTM or any recurrent layers.
3. *GridSearch.py*: Python module storing functions for gridearch pipeline process.
4. *main_funcs.py*: Python module storing functions used in main scripts.
- **multivariate_datasets**: Directory which contains the DBLBS medical action dataset (**dataset_train1000_events**) and timeserie values dataset (**dataset_train_1000**). The dataset is composed of 1000 timeseries and corresponding medical actions files.

# Installing required libraries

To install the required librairies, a conda environment is highly recommended [conda website](https://conda.io).
A yaml file is provided with the software to automatically download the correct librairies versions.

**Steps**:
1. Open the yaml file and set the prefix path to where you want to install the **env.yml** environment. By default, the environment is installed in */home/miniconda3/envs/env* but errors can be triggered if some directories don't exist.
2. Install the conda environment using the command: `conda env create -f env.yml`

# Parameters file

The **training_parameters.txt** file contains all the different parameters available. This file is read by the gridsearch pipeline and every possible combination of parameters are computed.

## General parameters syntax
This file follows a specific formatting for parsing purposes.
Any blank lines and lines starting with "#" are ignored. The lines starting by "#" are used for comments.
Within each line, don't insert any space between any characters or after any character or errors may occur.
Different type of parameter syntaxes are available. You are allowed to mix these syntaxes within a single parameters file.

- **Single value parameter syntax**: `parameter_name:value`. Use this syntax when a parameter stays constant for each run. Only one value is used for the parameter in the gridsearch process. Every parameter can use this syntax. Some parameters can only be used with this syntax, they are marked with the sign **(*)** in the parameters file.
- **Multiple values parameter syntax**: Use this syntax when you want a parameter to be set with a range or a list of values. Each run will use one of these values and every combination with every other parameters will be performed. *Range of values syntax*: `parameter_name:(start,end,step)`, with this syntax, the range of values will be converted by the software in an extended list. *List of values syntax*: `parameter_name:[val1,val2,...,valn]`. These 2 syntaxes can be used with almost every parameter unless it is specified by signs (*) or ([]) in the parameters file.
- **Double list values parameter syntax**: Some parameters need nested lists as value. This syntax allows to test during the gridsearch process different set of values for each layer of 
the neural network model, example to set a parameter for each of the 3 layers of a lambda model: `parameter_name:[[val1,val2,val3],[val4,val5,val6],[val7,val8,val9]]`.
The parameters specified by the symbol **([])** can only be used with this nested list syntax. For these marked ([]) parameters, make sure to always use this syntax even if only one single value is set, for example `name:[[value]]`.

## Parameter descriptions

- **model_name**: model to use among available models - *values: CustomEncoderDecoder, EncoderDecoder_no_split_outputs, EncoderDecoder_no_dense_event_layer, EncoderDecoder_with_attention*.
- **dataset_size**: Number of patients to take in the database to create the dataset to work with (at least 60 if split outputs parameter is True to get rid of errors due to an insuffisant amount of windows). To avoid train, val and test dataset proportions errors, make sure that: **dataset_size x test_size** results in an integer and that **dataset_size - (dataset_size x test_size) x val_split** also results in an integer - *integer*.
- **input_size**: input window size - *integer*.
- **output_size**: output window size (corresponds to target window size) - *integer*.
- **split_outputs (*)**:  boolean to know if you are using or not a model using split outputs function. Set this parameter to True if you are using *CustomEncoderDecoder* or *EncoderDecoder_no_dense_event_layer* models. If you are using a list of models in the *model name* parameter, please make sure every model uses the *split_output* function because the use of split output function can lead to changes in the number of neurons of some layers. If you want to mix models using and not using split outputs function, use separate parameter files - *boolean values: True | False*.
- **save_model (*)**: boolean to save or not models from each fold as a tensorflow binary file. Be careful, saved models uses a lot of disk memory - *boolean values: True | False*.
- **checkpoint_save (*)**: boolean to save or not model at each epoch (only if performances are better than on previous fold) - *boolean values: True | False*.
- **early_stopper (*)**: boolean to stop or not training when performances don't get better. If set to true, make sure to set *patience* and *min_delta* parameters - *boolean values: True | False*.
Using an early stopper is a form of neural network regularization.
- **nb_graphs (*)**: number of example prediction plots to create. If set to 0, no plots will be created - *integer*.
- **n_best_models (*)**: number of selected best models within the *best_experiments_results.csv* file. If set to None, classify all models from best to worst for each metric and loss - *values None | integer*.
- **min_norm_threshold (*)**: minimum parameter ("a") for MinMaxNormalization layer - *values: integer or float*
- **max_norm_threshold (*)**: maximum parameter ("b") for MinMaxNormalization layer - *values: integer or float*.
- **last_activation**: activation function used in the last layer of regression output (output for timeserie values). Please make sure to use the correct activation function depending on min and max norm_thresholds parameters giving the range of output values - *values: tanh, sigmoid, relu, gelu, selu, elu, linear*.
- **rnn_output_dense_activation**: "rnn output dense layer" activation function to use. This parameter is taken into account only for *EncoderDecoder_with_attention* model - *values: tanh, sigmoid, relu, gelu, selu, elu, linear*.
- **rnn_output_dense_neurons**: "rnn output dense layer" number of neurons. This parameter is taken into account only for *EncoderDecoder_with_attention* model - *integer*.
- **kernel_initializer**: neural network weights initialization - *values: glorotUniform, heUniform, glorotNormal, heNormal, zeros, ones, constant (=0.5)*.
- **optimizer**: gradient descent algorithm to use - *values: adam, nadam, adamax, sgd, adadelta, adagrad, ftrl, rmsprop*.
- **test_size (*)**: percentage of dataset used for evaluation (depending on dataset_size, k folding of the dataset can result in slightly different percentages for each partition) - *float*.
- **val_split (*)**: percentage of training dataset used for validation (depending on validation+training dataset_size, k folding of the dataset can result in slightly different percentages of each partition)
- **epochs (*)**: number of epochs (iteration through dataset) to train on - *integer*.
- **max_fold (*)**: maximum number of fold to train on by run - *integer >= 1 <= total number of folds available, depending on dataset_size, test_size, val_split*.
- **batch_size**: training batch size - *integer*
- **validation_batch_size**: validation step batch size. Set to None to set *validation_batch_size* equal to *batch_size* parameter. - *integer | None*
- **eval_batch_size**: evaluation step batch size. Set to None to set *eval_batch_size* equal to *batch_size* parameter. - *integer | None*
- **lr**: learning rate - *float*.
- **loss**: loss associated to time serie values to minimize - *values: mse, huber, rmse, mape (mean absolute percentage error), mae (mean absolute error)*. Custom metrics (last_time_step loss) are only available for mse loss, if you want to use another loss, please create the corresponding custom metrics for the chosen loss. 
- **loss_weight (*)**: weight associated to *loss* parameter. This parameter enable to scale the loss associated to timeserie values to make it fit with the range of values of any other loss associated to actions.
- **event_loss**: loss to minimize associated to actions - *only available values: categorical_crossentropy*.
- **event_loss_weight**: weight associated to *event_loss* parameter. This parameter enable to scale the loss associated to actions to make it fit with the range of values of any other loss associated to timeserie values - *integer*.
- **delay (*)**: delay to consider a well predicted event, for *accuracy_with_lag* metric - *integer < output_size/2*
- **min_delta (*)**: tensorflow early stopper minimum delta parameter. This parameter is considered only if *early_stopper* parameter is set to True. The *min_delta* parameter indicates the minimum evolution of a validation loss function between 2 epochs under which the training will be stopped *float*.
- **patience (*)**: Number of epochs to wait for once the validation loss between 2 epochs didn't improve of at least *min_delta*. This parameter isn't considered if *early_stopper* parameter is set to True. The best weights of the model will be restaured among the set of *patience* epochs - *integer >= 1*.
- **n_encoder_layers (*)**: number of layers in the encoder module - *integer*.
- **encoder_neurons ([])**: number of neurons in each encoder layer - *integer nested lists*, for example: [[10,10,10],[5,6,7]]. We have here 2 combinations of 3 number of neurons, in this example we consider *n_encoder_layers*=3, the first number is the number of neurons for the first layer, the second is the number of neurons for the second layer etc ... The 2 sublists symbolize that we want to try 2 different sets of encoder number of neurons (with the position of the value in the sublist indicating which layer we consider in the encoder). Make sure give only *n_encoder_layers* values in each sublist to keep lenght of number of neurons sublist and encoder module number of layers compatible. Moreover, last encoder layer number of neurons must be equal to first decoder layer number of neurons in order to enable decoder states initialization.
- **n_decoder_layers (*)**: number of layers in the decoder module - *integer*.
- **decoder_neurons ([])**: number of neurons in each decoder layer. The last number of neurons in each sublist is ignored and automatically computed if split outputs is set to True. With split outputs set to True, only certain shapes of outputs are possible  depending on input_size/output_size and encoding_width. Last decoder layer output_shape = (batch_size, input_size, output_size*nb_parameters). In any cases you must asigned a number of neurons to last decoder layer for parsing purposes. Moreover, last encoder layer number of neurons must be equal to first decoder layer number of neurons in order to enable decoder states initialization - *integer nested lists* (same principle as for *encoder_neurons* parameter).
- **encoder_dropout ([])**: dropout (after each recurrent layer) associated to each layer of the encoder. Set this parameter to None if you don't want to use any dropout - *integer nested lists* (use the same principle as in *encoder_neurons* parameter) | None.
- **decoder_dropout ([])**: dropout (after each recurrent layer) associated to each layer of the decoder. Set this parameter to None if you don't want to use any dropout - *integer nested lists* (use the same principle as in *encoder_neurons* parameter) | None.
- **encoder_recurrent_dropout ([])**: dropout (within each recurrent layer - applied to each hidden state of each time step) associated to each layer of the encoder. Set this parameter to None if you don't want to use any recurrent dropout - *integer nested lists* (use the same principle as in *encoder_neurons* parameter) | None.
- **decoder_recurrent_dropout ([])**: dropout (within each recurrent layer - applied to each hidden state of each time step) associated to each layer of the decoder. Set this parameter to None if you don't want to use any recurrent dropout - *integer nested lists* (use the same principle as in *encoder_neurons* parameter) | None.

**!! Warning !!** 
- You have to create different parameter files (which will create different experiment) for different combinations of encoder and decoder number of layers. That's why we often consider an experiment as a specific number of layers combination*.
- The parameters associated only to model *EncoderDecoder_with_attention* aren't considered when using other models. For parsing purposes, even if the value isn't used, please set a value for these parameters to avoid any parsing errors.

## SEED

The seed of random number generator for numpy.random calls and skutils.shuflle call is set inside *main_start_training.py* file as a constant line 16.

## Configuring parallization on CCIPL using SLURM (Centre de Calculs Intensifs des Pays de la Loire)

Slurm variables are declared and initialized as constants in *main_start_training.py* script of parallelized version of the sofware. You can find them from line 18 to 23 of the script to modify them.

# Dataset descriptions

The **multivariate_datasets** folder is splitted into 2 subfolders:
- **dataset_train1000** folder contains the 1000 timeseries values txt files. Each of the files represent a patient and a chirurgical operation. Each file present the same format with the first line corresponding to headers (Time,FC,PAS,PAM,PAD) and the rest of the lines for the values. The first timestep called "-1" corresponds to the initial state of the patient when entering the operating room. each file has a uniq identifier from 1 to 1000.
- **dataset_train1000_events** folder contains the 1000 timeseries associated event sequences txt files. Each file represent the sequence of events that occured for a patient during the chirurgical intervention. Each file present the same format with the first column for the time associated to an event, and the second column for the event as a word. Each file has an uniq identifier from 1 to 1000.

Each time serie values file id in *dataset_train1000* has its associated event sequence in *dataset_train1000_events*.

# Start training

To start the training you need to:
1. Use correctly formatted dataset folders and files.
2. Edit *training_parameters.txt* file. If any parameter has a wrong format the script won't work and an error message will be displayed.
3. Activate conda environment: `conda activate env`
4. Start traing (from working directory): `python3 main_start_training.py <parameters_file> <dataset_series_directory_path> <dataset_events_directory_path> <experiments_directory> <experiment_name>`. 
Example: `python3 main_start_training.py training_parameters.txt multivariate_datasets/dataset_train1000/ multivariate_datasets/dataset_train1000_events/ my_logs/ experiment_test`.
We recommend to call `<experiments_directory>` "my_logs" to avoid any future problems during data visualization section.
5. If you are using the parallelized version, please execute at the end of the experiment(s) the script *best_experiment.py* on the newly generated *experiments_directory* (from working directory): `python3 fct/best_experiment.py <experiments_directory>`. This script will generate 2 files in <experiments_directory>: **experiment_results_table.csv** and **best_experiment_models.txt** gathering results all runs of all experiments available in `<experiments directory>`. If you are using the sequential version, these files will be automatically generated at the end of the experiment.
This script will generate a file in the working directory called *n_fold_stats.txt* which indicates from how many common folds to all runs of all experiments in *my_logs* directory the different averages and standard deviations statistics were computed. For example, if an experiment contains 2 runs, one trained on 3 folds (folds 1 to 3) and 1 trained on 4 folds (folds 1 to 4), the *best_experiment_models.txt* and the *experiment_results_table.csv* files will compute averages and standard deviations of each metric on the first 3 common folds of the 2 runs. This script is particularly designed for partial results causing unbalanced number of folds in a set of runs (within several experiments or not). 
6. After results analysis thanks to *best_experiment_models.txt* file classifying models from best to worse for each metric and loss, we can have an idea of the best run in all experiments. Once a run is spotted as the best run with all the experiments, we can execute the script **make_global_run_boxplots_and_confusion_matrix.py** to create a global confusion matrix (obtained by doing the sumation of all folds' confusion matrices) and the global percentage error boxplots (relative percentage error and absolute percentage error) for each timestep and each physiological parameters (from working directory):`python3 tools/make_global_run_boxplots_and_confusion_matrix.py <n_folds> <run_directory_path> <model's_output_size> <number_of_physiologial_parameters>`. *n_folds* argument refers to the number of folds to consider within the run to create the confusion matrix and the boxplots. The description of the output directory is mention in next section *Description of output directory*.

# Description of output directory

my_logs **directory containing all experiments subdirectories**

+-- experiment directories **contains all run directories of an experiment**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- best_experiment_models.txt **file classifying best runs in all experiments for each metric and loss**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- experiment_results_table.csv **csv file gathering all runs' means and standard deviation (std) results for each metric / loss**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- run directories **directory containing all run' models (one for each dataset fold) for a specific set of hyperparameters combination combination**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- best_run_models.txt **file classfying the best models among one run (one hyperparameters combination) for each metric / loss**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- run_results_table.csv **csv file gathering all results (one for each dataset fold) of a run (a specific hyperparameter combination)**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- model_summary.txt **tensorflow model summary description**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- parameters_resume.txt **parameters combination used for this run.**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- vocabulary_table.csv **vocabulary dictionary used for events encoding**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- not_predicted_events.txt **txt file containing the list of not predicted events. Indeed some medical actions might not be present in the target dataets because we can only find them in the first input window of each patient events sequence. The model won't take these events into account to compute model's performances on medical actions**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- global_run_boxplots_confusion_matrix **directory available only after executing "make_global_run_boxplots_and_confusion_matrix.py". This folder contains global run boxplots, global raw error distribution files and global confusion matrix file**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- global_confusion_matrix.csv **csv file of global confusion matrix computed as the sum of n considered folds confusion matrix**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- absolute_percentage_error_boxplots **directory which contains absolute percentage error boxplots, boxplot summaries csv file and a file containing the global raw absolute percentage error distribution by physiological parameters and by timestep (the global raw distribution here corresponds to the concatenation of n considered folds raw absolute distribution)**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- relative_percentage_error_boxplots **directory which contains relative percentage error boxplots, boxplot summaries csv file and a file containing the global raw relative percentage error distribution by physiological parameters and by timestep (the global raw distribution here corresponds to the concatenation of n considered folds raw relative distribution)****

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- fold directories **directory containing the model tensorboard and saves for a specific fold of the dataset and for a specific hyperparameters combination (if save model or checkpoint options enabled)**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- custom evaluation directory **custom evaluation directory results**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- confusion_matrix.csv **confusion matrix associated to events for current fold**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- relative_percentage_error_boxplots directory **directory which contains relative percentage error boxplots, boxplots stats summaries csv file and the raw relative error distribution file**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- absolute_percentage_error_boxplots directory **directory which contains absolute percentage error boxplots, boxplots stats summaries csv file and the raw absolute error distribution file**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- prediction_example_plots directory **example predictions plots for the corresponding model**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- saved_model directory **contains all tensorflow model saves and data if saved model option = True**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- assets directory

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- keras_metadata.pb

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- saved_model.pb **binary saved model file**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- variables directory **model variables binary files directory

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- variable tensorflow files

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- tmp directory **tensorflow checkpoint directory - only if checkpoint sazve option = True**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- checkpoint **directory which contains best epoch model's save** 

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- assets directory

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- keras_metadata.pb

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- saved_model.pb

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- variables directory

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- variable tensorflow files

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- train directory **tensorboard binary training results directory**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- events.out tensorflow files

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- validation directory **tensorboard binary validation results directory**

|-&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+-- events.out tensorflow files


# Data visualization

Tensorflow uses a specific software to visualize data called **tensorboard**. 
You can plot every training results (loss / metrics / plots) of a whole `<experiments_dir`> directory (containing all experiments). To use this script please make sure that this directory is called **"my_logs"** otherwise the script won't be able to open tensorboard API. This script will start tensorboard in a new Firefox tab. Make sure firefox is installed on your computer (or edit the sh script to use another web browser).
This script starts tensorboard on port 6006. The script kills the process when pressing a key in the command prompt. To see the current processes id present on this port you can use the command `lsof -i:6006`. To kill any concurrent process (for example when trying to start tensorboard if the previous tensorboard process is still running) you can use `kill <process_id>`.

To execute tensorboard and visualize all data: `./run_tensorboard.sh my_logs/`. Make sure *env.yml* conda environment is activated before using this script or an error message will be displayed.

# Metric names and descriptions

In every "results" files or "best" files in the outprut directories, you will find different metric names. The metric names work on key words. You can find different keywords and metrics.

- *val* indicates that the metric or loss is a validation metric or loss.
- *eval* indicates that the metric or loss is an evaluation metric or loss.
- *last_time_step* indicates that the metric is calculated from only the last time step prediction produced by the model.
- *regression_output* indicates that the metric or loss is associated to time serie values output.
- *event_output* indicates that the metric or loss is associated to action output.
- *no val or no eval in the metric name* indicates the metric or loss is a training metric or loss.
- *loss* refers to the loss that we try to minimize (or maximize).
- *recall* refers to evaluation metrics computing recall
- *precision* refers to evaluation metrics computing precision
- *F1_score* refers to evaluation metrics computing F1 score
- *arithmetic_mean* refers to arithmetic mean of a metric
- *geometric_mean* refers to geometric mean of a metric
- *asa* refers to action specific accuracy metric computing accuracy for all "non none" actions
- *ne_PE* refers to "None-Event" percentage error metric computing percentage error of None prediction when we should have predicted an event.
- *nn_acc* refers to "None-None" accuracy computing the accuracy of None prediction.
- *acc_with_lag* refers to categorical accuracy computed with an authorized delay in the prediction. For each prediction, we look if a target corresponds to the prediction in a range of timestep determined by *[current-timestep - delay, current-timestep + delay]*
- *eval_multiplied_loss* and *eval_last_time_step_multiplied_loss* refers to evaluation metrics used to determine the best run among a set of experiments/runs or the best fold within a run. *eval_multiplied_loss* is computed as **eval_regression_output_loss x eval_event_output_loss** and *eval_last_time_step_multiplied_loss* is computed as **eval_regression_output_last_time_step_mse x eval_event_output_last_time_step_CategoricalCrossentropy**.
- *mean*: average of the considered metric / loss computed for all available folds (within one run or within an experiment)
- *std*: standard deviation of the considered metric / loss for all available folds (within one run or within an experiment)

# Perform inferences with a loaded serialized saved model

To perform inferences from a previously trained model you can use the script **EncoderDecoder_loader_predictor.py**. 
This script allows to load a trained model to perform inferences.
This script is also an example on how to load the saved EncoderDecoder model.
Before executing the script you can edit the *prediction_parameters.txt* file in the *tools* directory.
3 parameters are available:
- **dataset_size**: number of predictions to perform - *integer*.
- **input_size**: input window size (number of time steps) the loadable saved model was trained with - *integer*.
- **output_size**: output window size (number of time steps) the loadable saved model was trained with - *integer*.

The saved models are available if you set the parameter *save_model* to True in the training parameters file before starting the training.
The binary saved model can be loaded by giving the path to the *saved_model* directory within any fold directories.

The syntax to execute the script is the following (from working directory): 
Example command (from working diretory):`python3 EncoderDecoder_loader_predictor.py <path_to_saved_model_directory> <path_to_series_dataset_folder> <path_to_events_dataset_folder>/ <output_inference_directory> <path_to_prediction_parameters_file>`

The *output_inference_directory* (choose the name you want) will be created. This directory can be used to store all your inferences test on different models. For each excecution of the *EncoderDecoder_loader_predictor.py* script, a subfolder within the *output_inference_directory* will be created, to store the specific *example_prediction_plots* subfolder containing the prediction window plots. 

**!!!Warning!!!** Model's loading works well on the architectures *CustomEncoderDecoder* (EncoderDecoder using split outputs function) and *EncoderDecoder_no_split_outputs* (EncoderDecoder without split output function). However, the script might not load other model architectures properly.
