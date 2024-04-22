# Toy example

This readme file will explain you how to how to perform model's training with gridsearch and cross validation using this software. The parameters file *training_parameters_toy_example.txt* is configured to execute the grid search pipeline on 2 different combination of hyper parameter tuning (called "run"). By using this parameters file, each run will train the model on 2 cross-validation split of the dataset. The loadable model binary files will also be created by using this parameters file.

Please try to execute the toy example before starting any proper experiments in order to check good python environment installation and execution of the software. 

Please make sure that the conda python environment **"julien_deepL"** is correctly installed and activated `conda activate julien_deepL`.

# How to run the script

## 1. Start the training and grid search hyperparameters tuning

Execute the command from working directory: `python3 main_start_training.py training_parameters_toy_example.txt multivariate_datasets/dataset_train1000/ multivariate_datasets/dataset_train1000_events/ my_logs/ toy_example_exp`.

## 2. Display results in Tensorboard

You can visualize results in tensorboard during and after the training process of the models.
Note that if the grid search process isn't finished, some results may not be available.
To visualize the results execute the command from working directory: `./tools/run_tensorboard.sh my_logs/`
The results for each execution of each run will be displayed in a firefox tab. 

If you want to see the results only for a specific run or a specific fold, give the full path to the run of fold directory (*my_logs/<experiment name>/<run name>/<fold name>*).

## 3. Create best_experiment_models.txt and experiment_results_table.csv files

### Sequential version procedure
These files are automatically generated at the end of the experiment if you're using the sequential version of the software. 
However, if you use the sequential version, these 2 files will be generated for each of the experiments in the *"my_logs"* folder. 
To create the files storing the models for all experiments at the same time, you will need to execute manually the file *fct/best_experiment.py* (refers to parallelized version procedure). 
If you are using the sequential version with only one experiment (use only one parameters file) you can skip this section

### Parallelized version procedure

If you are using the SLURM CCIPL parallelized version of the software, the file *best_experiment_models.txt* and *experiment_results_table.csv* won't be automatically generated because of parallelization implementation. Once every experiment is finished, execute the command from working directory: `python3 fct/best_experiment.py my_logs/`. This will create the 2 files storing the global experiment results.

## 4. Compute global run performances

Once the gridsearch process is finished, you can compute the global performances (global performances computed on all folds of a run) of any run.
The script **tools/make_global_run_boxplots_and_confusion_matrix.py** will produce the global relative and absolute errors boxplots for each physiological parameter and output timestep and the global confusion matrix, computed as the sum of each fold confusion matrix.

Execute the command from working directory: `python3 tools/make_global_run_boxplots_and_confusion_matrix.py 2 my_logs/toy_example_exp/<insert_run_dir_name>/ 10 4`.
A directory called *global_run_boxplots_confusion_matrix* will be created in the run directory. This folder contains the different global boxplots and boxplot summaries files and the global confusion matrix. There are also files storing the global raw distribution of absolute and relative percentage errors by physiological parameter and timestep.

## 5. Load a serialized model to make inferences on a testing dataset

To perform inferences from a previously trained model you can use the script **EncoderDecoder_loader_predictor.py**. This script allows to load a trained model to perform inferences.
This script is also an example on how to load the saved EncoderDecoder model.
Before executing the script you can edit the *prediction_parameters.txt* file in the *tools* directory. The description of these parameters can be found in the *README.md* file.


Example command (from working diretory):`python3 EncoderDecoder_loader_predictor.py my_logs/<insert_experiment_dir_name>/<insert_run_name_folder>/<insert_fold_name_folder>saved_model/ multivariate_datasets/<insert_test_series_dataset_folder>/ multivariate_datasets/<insert_test_events_dataset_folder>/ inferences_test/ tools/prediction_parameters.txt`

The resulting directory in this case will be **inferences_test** but you are free to give any other names. This folder can store all your predictions for all your models. For each run of the prediction script, a dated and timed **"inference"** subfolder will be created within the parent's *inferences_test* folder (in this case) and will contain a **"prediction_example_plots"** subfolder containing all resulting prediction plots.

**!!!Warning!!!** Model's loading works well on the architectures *CustomEncoderDecoder* (EncoderDecoder using split outputs function) and *EncoderDecoder_no_split_outputs* (EncoderDecoder without split output function). However, the script might not load other model architectures properly.
