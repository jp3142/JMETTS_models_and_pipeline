#!/usr/bin/env py

import sys
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
import sklearn.utils as skutils

def multivariate_dataset_max_size(dataset, param_dict):
    """Computes the maximum number of windows (samples) for a given dataset, a window and an output size
    Arguments:
    dataset: dataset of patients serie values - list (number of patients, number of timesteps, number of features)
    (*) Note that number of timesteps and number of features can be variable across the different patients
    param_dict: dictionary of paramters
    """
    max_size = 0
    for i in range(0, len(dataset)):
        max_size += len(dataset[i])-(param_dict.get('input_size')+param_dict.get('output_size')) # computes the len of the first univariate serie of the mts (all series in a mts are of the same size)
    
    return max_size

####################################################################################

def prepare_dataset(dataset, events_dataset, nb_parameters, param_dict, seed=None):
    """From a non windowed dataset, return a shuffled windowed dataset
    dataset: non windowed dataset of patients serie values - 3D list with shape = (number of patients, number of timesteps, number of features)
    (*) Note that number of timesteps and number of features can be variable across the different patients
    events_dataset: non windowed dataset of actions 
    nb_parameters: number of physiological parameters. 2 features for each parameter (value + slope)"""

    # create random seed if None value has been passed as argument for seed
    if seed is None:
        seed = int(np.random.randint(0, 100000, 1, dtype=np.int))

    # check dataset size capacity
    max_size = multivariate_dataset_max_size(dataset, param_dict)

    # initialization
    cpt_window = 0

    # create dataset from raw files
    prepared_dataset = np.empty(shape=(max_size, param_dict.get('input_size')+param_dict.get('output_size'), nb_parameters), dtype=np.float32)
    prepared_events = np.empty(shape=(max_size, param_dict.get('input_size')+param_dict.get('output_size')), dtype=object)

    # create 3D dataset of windows of mt timesteps [ [val1,val2,val3,val4,val1......max = (input+output)*nb_parameters] [...] [...] ]
    for mts in range(0, len(dataset)):
        for mt in range(0, len(dataset[mts])-(param_dict.get('input_size')+param_dict.get('output_size'))): # mt = multivariate timestep
            if cpt_window >= max_size: # >= car cpt commence à 0
                break
            prepared_dataset[cpt_window] = np.array(dataset[mts][mt:mt+param_dict.get('input_size')+param_dict.get('output_size')])#.flatten(order='C')
            prepared_events[cpt_window] = np.array(events_dataset[mts][mt:mt+param_dict.get('input_size')+param_dict.get('output_size')]).reshape(-1)
            cpt_window+=1
        if cpt_window >= max_size: # >= because cpt >= 0
                break
    
    del dataset # delete old dataset from memory, we don't need it anymore

    # Shuffle windows synchroneously
    prepared_dataset, prepared_events = skutils.shuffle(prepared_dataset, prepared_events, random_state=seed) 
    
    # create input datasets
    inputs_series = prepared_dataset[:,0:param_dict.get('input_size')]
    inputs_events = prepared_events[:,0:param_dict.get('input_size')]
    
    # create target datasets (create intermediate timesteps predictions also for sequence to sequence model)
    targets_series = np.empty((max_size, param_dict.get('input_size'), param_dict.get('output_size')*nb_parameters), dtype=np.float32)
    targets_events = np.empty((max_size, param_dict.get('input_size'), param_dict.get('output_size')), dtype=object)
    
    for n_inputs in range(0, max_size): # for each window in the dataset
        for t_inputs in range(0, param_dict.get('input_size')): # for each timestep in the input
            targets_series[n_inputs, t_inputs] = np.reshape(prepared_dataset[n_inputs, t_inputs+1:t_inputs+1+param_dict.get('output_size')], -1)
            targets_events[n_inputs, t_inputs] = np.reshape(prepared_events[n_inputs, t_inputs+1:t_inputs+1+param_dict.get('output_size')], -1)
    
    return inputs_series.astype(np.float32), targets_series.astype(np.float32), inputs_events, targets_events

####################################################################################

def computes_slope(prev, current, interval):
    """Computes slope between 2 timesteps of a multivariate timeserie
    Arguments:
    prev: previous timestep values
    current: current timestep values
    interval: number of sub timesteps between each timestep"""
    return [ (current[i]-prev[i]) / interval for i in range(0, len(current)) ]

####################################################################################

def get_multivariate_dataset(csv_folder, events_folder, param_dict):
    """Read csv dataset files and create non windowed dataset. Add slope feature
    for each feature in the dataset. Doesn't keep -1 timestep only used for slopes
    computation. Explicitly remove patient exit action to event sequences.
    Arguments:
    csv_folder: serie values dataset folder
    events_folder: events dataset folder
    param_dict: dictionary of parameters"""

    nb_parameters = 0 # initialize
    whole_dataset = [] # to store the 3D list of raw serie dataset
    whole_events = [] # to store the 2D list of raw event dataset
    headers = [] # to store physiological parameters names
    h = 1 # 1 to not get time column if available, 0 to get time column if available

    # get whole dataset in memory
    list_dir = os.listdir(csv_folder) # list files in dataset directory
    np.random.shuffle(list_dir) # to shuffle dataset to not always have the multivariate timeseries in the same order

    # Get dataset_size number of series 
    for x in range(0, param_dict.get('dataset_size')):
        # find corresponding events file
        events_file = f"{list_dir[x][0:list_dir[x].find('_')]}_events.txt"
        with open(os.path.join(events_folder, events_file), 'r') as ef:
            # serie file opening
            events = ef.readlines()[1:] # don't keep headers
            events = [ tuple(t.split(',')) for t in [event.rsplit('\n')[0] for event in events] ]
        
        # file opening
        with open(os.path.join(csv_folder, list_dir[x]), 'r') as f:
            lines = f.readlines() # get all lines of file in a list
            
            # get headers only once and check if time column is present
            if x==0:
                headers = lines[0].rsplit('\n') 
                headers = headers[0].split(',')

                # delete 'Times' value from parameters_names
                try:
                    del headers[headers.index('Time')] #delete value 'Time' only if it exists -> check only once so all files have to be of the same format
                except ValueError:
                    h = 0 # Possible ValueError if Time isn't present
                    continue
                except KeyError:
                    h = 0 # Possible KeyError if Time isn't present
                    continue
            
                nb_parameters = len(headers) # because of slopes features

            initial_state = lines[1].rsplit('\n')
            initial_state = initial_state[0].split(',')
            initial_state = [ np.float32(initial_state[n]) for n in range(h, len(initial_state)) ] # get initial patient's state for this serie

            lines = lines[2:] # delete the first line of the file (headers, and -1 timestep = initial state)
            
            # create matrix for current MTS
            mts = np.empty(shape=(len(lines), nb_parameters*2), dtype=np.float32)
            mts_events = np.empty(shape=(len(lines), 1), dtype=object)

            # for each timestep - get values and slopes (evolution data) (nb_parameters = nb/2)
            e = 0 # Non None events counter
            a = 0 # all events counter
            for i in range(0, len(lines)):
                
                # transform line into a list of str
                lines[i] = lines[i].rsplit('\n')
                lines[i] = lines[i][0].split(',')
                
                if np.float32(events[e][0]) == np.float32(lines[i][0]): # if same timestep between corresponding file -> means there's an event
                    if events[e][1].lower() == "patient exit": # avoid keeping patient exit parameter
                        mts_events[a] = "None"
                    else:
                        mts_events[a] = events[e][1].lower() # add the event, else add None
                    e += 1
                    a += 1
                else:
                    mts_events[a] = "None"
                    a += 1
                
                # add timeserie timestep values to the mts
                mts[i, 0:nb_parameters] = [ np.float32(lines[i][j]) for j in range(h, len(lines[i])) ]
                
                if i == 0: # if were on timestep 0 (not -1) we compute the slope with timestep -1 (initial patient's state) but we dont use timestep -1 as a window
                    mts[i, nb_parameters:] = computes_slope(initial_state, mts[i, 0:nb_parameters], 2)
                else:
                    # compute slopes between previous timestep and current one
                    mts[i, nb_parameters:] = computes_slope(mts[i-1, 0:nb_parameters], mts[i, 0:nb_parameters], 2)

            # add MTS to the dataset
            whole_dataset.append(mts) 
            whole_events.append(mts_events)

    nb_parameters *= 2 # *2 because of slopes data
    
    uniq_events = find_uniq_events(whole_events)

    return whole_dataset, whole_events, uniq_events, headers, nb_parameters  # whole dataset = list = dataset of array = mts-size=len file of arrays = timesteps - size = nb_parameters

####################################################################################################

def format_params(param_dict, param_file):
    """Parameters formating, return a formatted multivalues parameter dictionary
    Arguments:
    param_dict: raw parameters dictionary
    param_file: parameters file name
    """

    try:
        for k, v in param_dict.items():

            # check no values parameters
            if v == "None" or v=="":
                param_dict[k] = [None]
                continue   
            
            # check end of value (undesired characters)  
            if v[-1] in [',', '.']:
                raise SyntaxError
                
            if v[-1] == '\n':
                param_dict[k] = v[0:-1]
            
            # check if parameter of type list
            list_key = False
            if v.startswith('[') and v[len(v)-1] == (']'):
                param_dict[k] = param_dict[k][1:-1]
                list_key = True

            # check if parameter of type tuple
            tuple_key = False
            if v.startswith('(') and v[len(v)-1] == (')'):
                param_dict[k] = param_dict[k][1:-1]
                tuple_key = True

            # check if [] or () have been forgotten
            if not tuple_key and not list_key:
                if ',' in v:
                    raise SyntaxError

            # format list type parameters
            if list_key:
                param_dict[k] = param_dict[k].split(',')

                # format specific parameters (double list params)
                if k in ['encoder_dropout', 'encoder_recurrent_dropout', 'decoder_dropout', 'decoder_recurrent_dropout', 'encoder_neurons', 'decoder_neurons']:
                    if param_dict[k] == None:
                        continue
                    else:
                        n_list = list()
                        n_sublists = -1
                        for value in param_dict[k]:

                            if '[' in value:
                                value = value.split('[')[1:]
                                value = value[0]
                                n_list.append(list())
                                n_sublists += 1 
                            
                            if ']' in value:
                                value = value.split(']')[0]
                            
                            n_list[n_sublists].append(value)

                        if k in ['encoder_dropout', 'encoder_recurrent_dropout', 'decoder_dropout', 'decoder_recurrent_dropout']:
                            n_list = [[float(v) for v in l] for l in n_list]
                        elif k in ['encoder_neurons', 'decoder_neurons']:
                            n_list = [[int(v) for v in l] for l in n_list]

                        param_dict[k] = n_list

                        continue               
                
                #integer parameters
                if k in ['input_size', 'dataset_size', 'output_size', 'nb_graphs', 'epochs', 'batch_size', 'validation_batch_size', 'eval_batch_size', 'encoder_neurons', 'decoder_neurons', 'rnn_output_dense_neurons']:
                    param_dict[k] = [int(i) for i in param_dict[k]]
                    continue

                # float parameters
                if k in ['test_size', 'val_split', 'lr', 'max_norm_threshold', 'min_norm_threshold']:
                    param_dict[k] = [float(i) for i in param_dict[k]]
                    continue
                
                # else str parameters
                param_dict[k] = [str(i) for i in param_dict[k]] # convert to str list
                continue
                
            # format from list to tuple if tuple type parameter
            if tuple_key:         
                param_dict[k] = param_dict[k].split(',')

                #integer parameters
                if k in ['input_size', 'dataset_size', 'output_size', 'nb_graphs', 'epochs', 'batch_size', 'validation_batch_size', 'eval_batch_size']:
                    param_dict[k] = tuple([int(i) for i in param_dict[k]])
                    continue

                # float parameters
                if k in ['test_size', 'val_split', 'lr', 'max_norm_threshold', 'min_norm_threshold']:
                    param_dict[k] = tuple([float(i) for i in param_dict[k]])
                    continue
                
                if isinstance(param_dict[k][0], str): # no str range tuple
                    sys.stderr.write("\n[TypeError] Range tuples can't take string types as values.\n")
                    raise TypeError

                # verify range tuple shape
                if len(param_dict[k]) != 3: # to have a correct range tuple shape (3 elms)
                        raise SyntaxError
                    
            # else format as single value param

            # convert integer parameters
            if k in ['input_size', 'dataset_size', 'output_size', 'epochs', 'batch_size', 'validation_batch_size', 'eval_batch_size', 'nb_graphs', 'n_best_models', 'n_encoder_layers', 'n_decoder_layers', 'patience', 'max_fold', 'delay', 'rnn_output_dense_neurons']:
                param_dict[k] = int(v)
                continue
                
            # convert to float parameters
            if k in ['test_size', 'val_split', 'lr', 'min_delta', 'loss_weight', 'event_loss_weight', 'max_norm_threshold', 'min_norm_threshold']:
                param_dict[k] = float(v)
                continue
            
            if k in ["split_outputs", "save_model", "checkpoint_save", "early_stopper"]:
                if param_dict[k] == "True":
                    param_dict[k] = True
                else:
                    param_dict[k] = False
                continue
            
            #else convert to string parameter
            param_dict[k] = str(v)

        # convert range tuple into extended list
        for k, v in param_dict.items():
            if isinstance(v, tuple):
                # for int tuple
                if isinstance(v[0], int) and isinstance(v[1], int) and isinstance(v[2], int):
                    param_dict[k] = [i for i in range(v[0], v[1]+v[2], v[2])]
                else:
                    v = list(v) # convert back to list
                    # for float tuple
                    values_list = [] # initialize list
                    values_list.append(v[0])
                    n_it = int((v[1]-v[0])/v[2])+2 # computes the number of iterations to create list from tuple

                    # replace tuple by extended list
                    for j in range(1, n_it):
                        values_list.append(values_list[j-1]+v[2])

                    param_dict[k] = values_list

        # specific parameter verifications
        if isinstance(param_dict.get('val_split'), list):
            for values in param_dict.get("val_split"):
                assert values > 0 and values < 1
        else:
            assert param_dict.get("val_split") > 0 and param_dict.get("val_split") < 1

        if isinstance(param_dict.get('test_size'), list):
            for values in param_dict.get("test_size"):
                assert values > 0 and values < 1
        else:
            assert param_dict.get('test_size') > 0 and param_dict.get('test_size') < 1
        
        param_dict['delay'] = abs(param_dict.get('delay')) # to be sure to have a positive value for delay of accuracy with lag
        
        # check for none values in encoder / decoder params
        for key in ['encoder_dropout', 'encoder_recurrent_dropout', 'decoder_dropout', 'decoder_recurrent_dropout']:
            if param_dict[key] == [None]:
                param_dict[key] = [[0.0 for _ in range(0, param_dict.get(f"n_{key[0:key.find('_')]}_layers"))]] # if None, initialize list with zeros
        
        # assert that each sublist of double list parameters contain the sazme number of values and that this number is equal to the fixed number of layers
        for key in ['encoder_dropout', 'encoder_recurrent_dropout', 'decoder_dropout', 'decoder_recurrent_dropout']:
            if param_dict.get(key) == [[0.0 for _ in range(0, param_dict.get(f"n_{key[0:key.find('_')]}_layers"))]]:
                continue
            #else
            for sublist in param_dict[key]: 
                # check if each sublist contain the correct number of values (should be equal to number of layers for encder / decoder )
                if len(sublist) != param_dict.get(f"n_{key[0:key.find('_')]}_layers"):
                    raise ValueError # raise a value error if not the case
        
        for sublist_enc, sublist_dec in zip(param_dict['encoder_neurons'], param_dict['decoder_neurons']):
            # check if number of neurons in layers (n_encoder_neurons and n_decoder_neurons params) matches number of layers (n_encoder_layers and n_decoder_layers params)
            try:
                assert len(sublist_enc) == param_dict.get('n_encoder_layers')
                assert len(sublist_dec) == param_dict.get('n_decoder_layers')
            except AssertionError:
                sys.stderr.write(f"[IncompatibleParameters] Length of list of number of neurons in layers (n_encoder_neurons and n_decoder_neurons params) doesn't match number of layers (n_encoder_layers and n_decoder_layers params)\n")
                exit(11)

                    # check that each last number of encoder neurons param is at least equal to one decoder neurons param first sublist value (to avoid having unused sublist)
            if sublist_enc[-1] not in [sublist_dec[0] for sublist_dec in param_dict['decoder_neurons']]:
                sys.stderr.write(f"[EncoderDecoderModelError] Last number of neurons in encoder_neurons sublist \"{sublist_enc}\" doesn't match any first number of neurons in decoder_neurons sublists. (leads to ValueError)\n")
                raise ValueError

        # test if val_split, test_size and dataset_size values are compatible
        try:
            # compute test and val bloc sizes
            test_bloc_size = (param_dict.get('test_size')*param_dict.get('dataset_size'))
            val_bloc_size = (( param_dict.get('dataset_size') - ( param_dict.get('dataset_size')*param_dict.get('test_size') ) )*param_dict.get('val_split'))
            
            # assert that test_bloca and val_bloc sizes are integer
            assert test_bloc_size.is_integer()
            assert val_bloc_size.is_integer()

            # if integer, convert type to integer
            test_bloc_size = int(test_bloc_size)
            val_bloc_size = int(val_bloc_size)
            
            # assert that the dataset_size and train_val_dataset_size can be splited in equal test_blocs sizes and val_blocs sizes
            assert param_dict.get('dataset_size')%test_bloc_size == 0 and (param_dict.get('dataset_size')-test_bloc_size)%val_bloc_size == 0
            
        except AssertionError:
            sys.stderr.write("\n[ProportionsError] dataset_size, and/or test_size and/or val_split have incompatible values.\n")
            exit(8)

        assert param_dict['max_fold'] >= 1 # check to run at least one fold

        if param_dict.get('output_size') == 1:
            param_dict['delay'] = 0
            print("[ParametersWarning] Setting \"delay\" parameter to 0 because output_size == 1 (no accuracy_with_lag).\n")
        else:
            assert param_dict['delay'] <= ( param_dict.get('output_size') // 2 ) # delay has to be <= to half of output window size

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
    except AssertionError:
        sys.stderr.write(f"[AssertionError] One or several parameters have out of scale values. Please check the file \"{param_file}\".\n")
        exit(6)
    
    return param_dict

#######################################################################################

def get_parameters(param_file):
    """
    Read parameters file and return a raw unformatted (str only) dictionary.
    Arguments:
    param_file: parameters_filepath
    """
    param_dict = {}

    # read parameters and add them to the dictionary
    with open(param_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith(' ') or line.startswith('\n'):
                continue

            line = line.rsplit('\n')
            line = line[0].split(':')

            param_dict[str(line[0])] = str(line[1]) # create dict entry

    param_dict = format_params(param_dict, param_file)
    
    return param_dict

#######################################################################################

def val_split(training_dataset, training_targets, val_split, string=False):
    """Simulate val_split by taken x% of windows at the beginning of the dataset
    Do not use this function for cross_validation"""

    nb_windows_to_get = int(len(training_dataset)*val_split)
    training_val, targets_val = training_dataset[0:nb_windows_to_get], training_targets[0:nb_windows_to_get]
    
    # remove validation values from training values
    training_dataset, training_targets = training_dataset[nb_windows_to_get:], training_targets[nb_windows_to_get:]

    if string:
        return training_dataset.astype(np.object), training_targets.astype(np.object), training_val.astype(np.object), targets_val.astype(np.object)
        
    return training_dataset.astype(np.float32), training_targets.astype(np.float32), training_val.astype(np.float32), targets_val.astype(np.float32)

############################ tf function version ###########################################################
#@tf.function
#def tf_val_split(training_dataset, training_targets, val_split):
#    """Simulate val_split by taking val_split% at the beginning of the dataset"""
#
#    nb_windows_to_get = int(training_dataset.shape[0]*val_split)
#    training_val, targets_val = training_dataset[0:nb_windows_to_get], training_targets[0:nb_windows_to_get]
#    
#    # remove validation values from training values
#    training_dataset, training_targets = training_dataset[nb_windows_to_get:], training_targets[nb_windows_to_get:]
#
#    return training_dataset, training_targets, training_val, targets_val

#######################################################################################

def standardization_normalization(x_train, targets_train, targets_val, targets_test, a, b, output_size):
    """Preprocessing before model learning. Normalize data (3D shape) (StandardScaler Followed by MimaxScaler on train_set (x_train + targets_train) to have all the possible windows.
    (all values at each timestep). 
    The resulting statistical parameters are applied to  the rest of the sub datasets (Validation and test (not x_test or x_val because they are normalized on the go)
    The preprocessing is performed for each variable/feature in the dataset.
    Arguments:
    x_train: training input windowed dataset
    targets_train: training targets windowed dataset
    targets_val: validation targets windowed dataset
    targets_test: test targets windowed dataset
    a: low value of range used to normalize data (MinMaxNormalization)
    b: high value of range used to normalise data (MinMaxNormalization)
    output_size: window output_size"""

    # list of initial shapes to reshape after fit transform
    initial_shapes = [x_train.shape, targets_train.shape, targets_val.shape, targets_test.shape ]
    
    # reshape x_train and all targets for use of scalers
    x_train = np.reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2])).astype(np.float32)
    targets_train = np.reshape(targets_train, (targets_train.shape[0]*targets_train.shape[1]*output_size, x_train.shape[1])).astype(np.float32)
    targets_val = np.reshape(targets_val, (targets_val.shape[0]*targets_val.shape[1]*output_size, x_train.shape[1])).astype(np.float32)
    targets_test = np.reshape(targets_test, (targets_test.shape[0]*targets_test.shape[1]*output_size, x_train.shape[1])).astype(np.float32)
    
    # standard fit data
    std_scaler = StandardScaler()
    std_scaler.fit(np.concatenate((x_train, targets_train), axis=0))

    # standard transform data (mean = 0, std = 1)
    x_train = std_scaler.transform(x_train).astype(np.float32)
    targets_train = std_scaler.transform(targets_train).astype(np.float32)
    targets_val = std_scaler.transform(targets_val).astype(np.float32)
    targets_test = std_scaler.transform(targets_test).astype(np.float32)

    # normalize fit data
    minmax_scaler = MinMaxScaler(feature_range=(a, b))
    minmax_scaler.fit(np.concatenate((x_train, targets_train), axis=0))

    # normalize transform data (between 0 and 1)
    x_train = minmax_scaler.transform(x_train).astype(np.float32)
    targets_train = minmax_scaler.transform(targets_train).astype(np.float32)
    targets_val = minmax_scaler.transform(targets_val).astype(np.float32)
    targets_test = minmax_scaler.transform(targets_test).astype(np.float32)

    # reshaping to original shape
    x_train = np.reshape(x_train, initial_shapes[0])
    targets_train = np.reshape(targets_train, initial_shapes[1])
    targets_val= np.reshape(targets_val, initial_shapes[2])
    targets_test = np.reshape(targets_test, initial_shapes[3])

    return x_train.astype(np.float32), targets_train.astype(np.float32), targets_val.astype(np.float32), targets_test.astype(np.float32), (std_scaler, minmax_scaler)

#######################################################################################

def find_uniq_events(events_dataset):
    """Return a set of uniq elements in a dataset (string datatype -> used for events dataset)
    Arguments:
    events_dataset: dataset of events - shape=(dataset_size, input_size+output_size)""" 
    return list({events[0] for mts in events_dataset for events in mts}) # convert uniq_events set into list to allow futur conversion into a tensor

#######################################################################################

def get_onehot_dict(uniq_events):
    """
    From a set of events, return correspondance dictionaries for one hot encoding of events.
    events_to_index: dictionary with keys = events and values = indexes (position of one i one hot vector)
    index_to_events: dictionary with keys = indexes and value = events
    Arguments:
    uniq_events: set of events
    """
    events_to_index = dict((event, index) for index, event in enumerate(uniq_events))
    index_to_events = dict((index, event) for index, event in enumerate(uniq_events))
    encoding_width = len(events_to_index) # number of values in each one-hot vector

    return events_to_index, index_to_events, encoding_width

#######################################################################################

def one_hot_encoding(x_train, targets_train, targets_val, targets_test, events_to_index, encoding_width):
    """
    Perform one-hot encoding of events. Return x and targets train, validation and evaluation encoded datasets.
    Arguments: 
    x_train: input training windowed dataset
    targets_train: targets training windowed dataset
    targets_val: targets validation windowed dataset
    targets_test: targets evaluation windowed dataset
    events_to_index: dictionary making correspondance between events and indexes to generate one hot vectors
    encoding_width: number of values in each one hot vector = number of events (classes)
    """
    
    #encode inputs into one hot vectors (x_train)
    encoded_x_train = np.zeros(shape=(x_train.shape[0], x_train.shape[1], encoding_width))
    for i, window in enumerate(x_train):
        for j, event in enumerate(window):
            encoded_x_train[i, j, events_to_index[event]] = 1

    # encode targets into one hot vectors (all targets)
    concat_targets = np.concatenate((targets_train, targets_val, targets_test), axis=0) # concat all targets set to iterate only via one loop
    encoded_targets = np.zeros(shape=(concat_targets.shape[0], concat_targets.shape[1], concat_targets.shape[2], encoding_width))
    for k, window in enumerate(concat_targets):
        for l, timestep in enumerate(window):
            for m, target_event in enumerate(timestep):
                encoded_targets[k, l, m, events_to_index[target_event]] = 1

    # re split previously concatenated targets
    encoded_targets_train = encoded_targets[0:targets_train.shape[0]]
    encoded_targets_val = encoded_targets[targets_train.shape[0]:targets_train.shape[0]+targets_val.shape[0]]
    encoded_targets_test = encoded_targets[targets_train.shape[0]+targets_val.shape[0]:]

    return encoded_x_train, encoded_targets_train, encoded_targets_val, encoded_targets_test

#######################################################################################

def concatenate_features(regular, events):
    """Concatenate events and regular features for each timestep.
    Return a concatenated 3D matrix (dataset_size, n_timesteps, n_features)
    Arguments:
    regular: series values features to concatenate
    events: events features to concatenate
    """
    # reshape regular targets to have a shape similar to targets_events shape
    regular = regular.reshape((regular.shape[0], regular.shape[1], regular.shape[1], int(regular.shape[2]/regular.shape[1])))

    # create concatenated mixed inputs (events and regulat float values)
    concat = np.concatenate((regular, events), axis=-1)
    concat = np.reshape(concat, (concat.shape[0], concat.shape[1], concat.shape[2]*concat.shape[3]))

    return concat

#######################################################################################

def create_cross_validation_datasets_indices(inputs, param_dict, shuffle=False, random_state=None):
    """
    For a given dataset, create different splits of the dataset according to its first dimension. 
    This function will return a list of dataset fold represented by a list of tuples (train_set_indices, val_set_indices, eval_set_indices) 
    Arguments:
    inputs: whatever input dataset (only used by KFold function to get dataset size). Targets and train datasets should be of same length.)
    test_size: test dataset size (%) (float)
    val_size: validation dataset size (% of training dataset) (float)
    shuffle: bool - to shuffle dataset
    random_state: bool - seed for random shuffle - set this parameter to None if shuffle = False
    """
    kf_test= KFold(n_splits=int(1/param_dict.get('test_size')), random_state=random_state, shuffle=False)
    kf_val = KFold(n_splits=int(1/param_dict.get('val_split')), random_state=random_state, shuffle=False)

    # return list of tuples train indices, val indices, test indices) or tuple(train_index, val_index, test_index)
    return [(train_index, val_index, test_index) for train_val_index, test_index in kf_test.split(inputs) for train_index, val_index in kf_val.split(train_val_index)]

#######################################################################################

def dataset_Kfolding(dataset, fold_index):
    """Perform the split of a dataset using a list of fold indices returned by create_cross_validation_datasets_indices function.
    Arguments:
    dataset: dataset to split according to split indices
    fold_index: tuple(train_index, test_index) or tuple(train_index, val_index, test_index)
    """
    return [[dataset[j] for j in fold_index[i]] for i in range(0, len(fold_index))] # return train, test or train, val, test dataset for this split

#######################################################################################

def detect_not_predicted_target_events(uniq_events, target_dataset):
    """From a list of all uniq actions (classes / labels) present in a whole raw dataset, detects all actions that are that aren't in a target dataset target dataset.
    Arguments:
    uniq_events: set or list of all different (uniq in the list) events in the dataset
    target_dataset: target dataset (non encoded) - 3D numpy array (dataset_size, input_size, output_size) containing literal actions
    """

    target_dataset = np.array(target_dataset, dtype=object)
    not_predicted_events = []
    
    for event in uniq_events:
        found = False
        for window in target_dataset[:, -1, :]: #check last target's slice
            if event in window:
                found = True
                break
        if not found:
            not_predicted_events.append(event)

    return not_predicted_events 

#######################################################################################
    
def write_not_predicted_events_file(not_predicted_events, run_logdir):
    """Writes not_predicted_events returned from detect_not_predicted_target_events function to a file.
    Arguments:
    not_predicted_events: set of not predicted events (events present only in first window of each patient)"""

    output_file = os.path.join(run_logdir, "not_predicted_events.txt")

    if not os.path.exists(output_file):
        try:
            with open(output_file, 'w') as f:
                f.write(f"# This file contains the set of medical actions that are only present in first patients window.\n# Thus, these actions are not in the target datasets and no prediction performances are computed for these actions.\n")
                for event in not_predicted_events:
                    f.write(f"{event}\n")
        except:
            sys.stderr.write(f"\n[CreateFileError] Unable to create file {output_file}.\n")
        else:
            print(f"\n[CreateFileSuccess] File {output_file} created with success\n")


