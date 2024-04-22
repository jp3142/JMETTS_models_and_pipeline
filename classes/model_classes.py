#!/usr/bin/env python3

import os
import sys
import io
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from .layer_classes import *
from .custom_metrics import *

### MODULE FILE FOR MODEL CLASS TEMPLATES ###

class CustomEncoderDecoder(tf.keras.Model):
    """
    ########################################################
    ##  TEMPLATE CLASS. Encoder-Decoder ##
    ########################################################
    => (no timedistributed for serie values normalized between -1 and 1 because we directly use LSTM layer output as final output -> tanh)                        

    Encoder-Decoder class. Backpropagation is in a end-to-end fashion with encoder and decoder trained together.
    Classic LSTM layers are used to create Encoder and Decoder.
    The hyper-parameters of the model can be changed via the parameters dictionary gathering hyperparameters 
    file informations."""
    
    def __init__(self, param_dict, headers, logdir, scalers, ordered_events, encoding_width, not_predicted_events, **kwargs):
        """Class constructor 
        Encoder_decoder class for joint modeling.
        Arguments:
        param_dict: parameters dictionary
        headers: list of physiological variable names
        logdir: run directory
        scalers: tuple of scaler objects (StandardScaler object, MinMaxScaler object) obtained from scikit learn library.
        ordered_events: oredered array of uniq actions in the dataset, used for OneHotEncoding Layer
        encoding_width: number of classes (actions) in the dataset
        """
        super(CustomEncoderDecoder, self).__init__(**kwargs) # initialize parent class

        # def self variables
        self.full_path_model = ""
        self.nb_parameters = len(headers)*2
        self.headers = headers
        self.param_dict = param_dict
        self.run_logdir = logdir # mean that we have to create another instnace of model for each run
        std_scaler, minmax_scaler = scalers # no serialization, used only to instanciate Standardization and Normalization Layers
        self.ordered_events = ordered_events
        self.encoding_width = encoding_width
        self.not_predicted_events = not_predicted_events

        # correspondance dictionnary for weights initializers
        available_kernel_init = {
            "glorotUniform": keras.initializers.GlorotUniform(),
            "heUniform": keras.initializers.HeUniform(),
            "glorotNormal": keras.initializers.GlorotNormal(),
            "heNormal": keras.initializers.HeNormal(),
            "zeros": keras.initializers.Zeros(),
            "ones": keras.initializers.Ones(),
            "constant": keras.initializers.Constant(value=0.5),
            # add others if needed
        }

        # correspondance dictionary for activation functions
        available_activations = {
            "tanh": keras.activations.tanh,
            "sigmoid": keras.activations.sigmoid,
            "relu": keras.activations.relu,
            "elu": keras.activations.elu,
            "linear": keras.activations.linear,
            "selu": keras.activations.selu,
            "gelu":keras.activations.gelu
            # add others if needed
        }
        
        # Preprocessing layers
        self.standardization = StandardizationLayer(means=std_scaler.mean_, stds=std_scaler.scale_, nb_parameters=self.nb_parameters, output_size=self.param_dict.get('output_size'), name="TestStdLayer")
        self.normalization = MinMaxNormalizationLayer(minmax_scaler.data_min_, minmax_scaler.data_max_, param_dict.get('min_norm_threshold'), param_dict.get('max_norm_threshold'), nb_parameters=self.nb_parameters, output_size=self.param_dict.get('output_size'), name="TestNormLayer")
        self.onehot = OneHotEncodingLayer(ordered_events, encoding_width, name="TestOneHotEncodingLayer")
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Encoder module 
        self.encoder = []
        for i in range(0, self.param_dict.get('n_encoder_layers')):
            # default return_sequences and return state parameter for more than one layer models
            return_seq = True

            # boolean set last layer return seq parameter: 
            # i != (n_encoder_layers -1) -> return_seq = True
            # i == (n_encoder_layers - 1) -> return_seq = False
            return_seq = bool(i != (self.param_dict.get('n_encoder_layers') - 1) )
            
            if i == 0:
                self.encoder.append(
                    tf.keras.layers.LSTM(
                        self.param_dict.get('encoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh,
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'encoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("encoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("encoder_dropout")[i],
                        input_shape=[None, None], # output_size, 8+36
                        return_sequences=return_seq,
                        return_state=True # return state at each layer (avoid conditions in call) but will use only the last layer state as initial state for first decoder layer
                    )
                )
            else:
                self.encoder.append(
                    tf.keras.layers.LSTM(
                        self.param_dict.get('encoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh,
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'encoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("encoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("encoder_dropout")[i],
                        return_sequences=return_seq,
                        return_state=True # return state at each layer (avoid conditions in call) but will use only the last layer state as initial state for first decoder layer
                    )
                )
            
        # Repeat last time step prediction from last encoder layer, in order to use repaetedVector as input for decoder's first layer
        self.repeat = tf.keras.layers.RepeatVector(n=self.param_dict.get('input_size'))
        
        # Decoder module
        self.decoder = []
        for i in range(0, self.param_dict.get('n_decoder_layers')):
            # default return_sequences and return state parameter for more than one layer models
            if i == 0:
                self.decoder.append(
                    tf.keras.layers.LSTM( # pour créer des cocuhe perso il faut utiliser la classe RNN de base en leur pazssaznt une couche (cell class) perso
                        self.param_dict.get('decoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh, # tanh: fonction saturante pour éviter l'instabilité des gradients.(évite que les valeurs ou gradients explosent, variation faible si grosses valeurs, donc les non saturantes comme relu aident pas)
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'decoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("decoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("decoder_dropout")[i],
                        input_shape=[None, None], # output_size, len(output vect from last encoder layer)
                        return_sequences=True,
                        return_state=False
                    )
                )
            else:
                self.decoder.append(
                    tf.keras.layers.LSTM( # pour créer des cocuhe perso il faut utiliser la classe RNN de base en leur pazssaznt une couche (cell class) perso
                        self.param_dict.get('decoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh, # tanh: fonction saturante pour éviter l'instabilité des gradients.(évite que les valeurs ou gradients explosent, variation faible si grosses valeurs, donc les non saturantes comme relu aident pas)
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'decoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("decoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("decoder_dropout")[i],
                        return_sequences=True,
                        return_state=False
                    )
                )

        self.eventDense = keras.layers.TimeDistributed(
            keras.layers.Dense(
                self.param_dict.get('output_size')*self.encoding_width,
                kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                use_bias = True,
                bias_initializer='zeros',
                name='events_output_layer',
                activation = keras.activations.linear
            )
        )

        # TimeDistributed layers
        self.regularDense = lambda x: x # layer doing no operation (represented as a simple lambda function)

        # if not tanh range for normalization, we add a new output layer to the model
        if self.param_dict.get('min_norm_threshold') == -1 and self.param_dict.get('max_norm_threshold') == 1:
            pass
        else:
            self.regularDense = keras.layers.TimeDistributed( # we could use an RNN layer but a dense layer allows to chosse the activation and has the same capacities (our values between 0 and 1 so we cant use tanh)
                keras.layers.Dense(
                    self.param_dict.get('output_size')*self.nb_parameters, # output -> taille = nb_pas de temps qu'on veut prédire (chaque pas de temps =  nb_parametre)
                    kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                    use_bias = True,
                    bias_initializer='zeros',
                    name='regular_output_layer',
                    activation = available_activations[self.param_dict.get('last_activation')]
                )
            )
        
        # softmax layer for one hot encoder prediction events
        self.softmax_layer = SoftMaxLayer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="split_outputs")])
    def split_outputs(self, outputs):
        """Split output from a Recurrent Layer to keep actions output values and series values on 2 different sides.
        Return 2 tensors:
        regular_outputs: tensor for serie values of shape = (batch_size, input_size, n_features*output_size)
        event_outputs: tensor for one hot action vectors of shape (batch_size, input_size, encoding_width*output_size)
        Arguments:
        outputs: tensor representing sequence output from RNN layer. Shape = (batch_size, input_size, n_features*output_size)"""

        # split by timestep to have shape = (batch_size, input_size, output_size, n_features + encoding_width)
        outputs = tf.reshape(tf.cast(outputs, dtype=tf.float32), shape=[tf.shape(outputs)[0], self.param_dict.get('input_size'), self.param_dict.get('output_size'), self.nb_parameters+self.encoding_width])

        # get regular outputs and event_outputs
        regular_outputs = outputs[:, :, :, 0:self.nb_parameters]
        event_outputs = outputs[:, :, :, self.nb_parameters:]

        # reshape to original shape
        regular_outputs = tf.reshape(regular_outputs, shape=[tf.shape(regular_outputs)[0], self.param_dict.get('input_size'), self.param_dict.get('output_size')*self.nb_parameters])
        event_outputs = tf.reshape(event_outputs, shape=[tf.shape(event_outputs)[0], self.param_dict.get('input_size'), self.param_dict.get('output_size')*self.encoding_width])
        
        return regular_outputs, event_outputs

    @tf.function
    def call(self, inputs, training=None, **kwargs): # we have to tell the forward pass with () at end of line, beucause non sequential model
        """Main method. This function is executed when calling an instance of this model class.
        Defines the forward pass used when using model.fit
        inputs: list of input types: [serie_values_input, action_onehot_input]
        for training:
            - serie_values_input: (batch_size, input_size, n_features)
            - action_one_hot_input: (batch_size, input_size, encoding_width)
        for inference:
            - serie_values_input: (batch_size, input_size, n_features)
            - action_one_hot_input: (batch_size, input_size)
        training: boolean indicating behavior of function during training. Used for preprocessing layers and dropout layers training argument
        **kwargs: any Tensorflow Model Class argument"""

        # get regular and events inputs
        regular_inputs, events = inputs

        #### INFERENCE ONLY (for x_val and x_test) ####
        regular_inputs = self.standardization(regular_inputs, training=training)
        regular_inputs = self.normalization(regular_inputs, training=training)
        events = self.onehot(events, training=training)
        ##########################################

        concat_inputs = self.concatenate([regular_inputs, events])

        # ENCODER DECODER

        # Encoder
        encoder_output = [concat_inputs] # initialize encoder_output with concatenated preprocessed inputs
        states_c = [] # initialize list of long term memory vector outputs

        # for each layer, compute output from previous output
        for i in range(0, len(self.encoder)):
            output, _, state_c = self.encoder[i](encoder_output[i])
            encoder_output.append(output)
            states_c.append(state_c)
            
        encoder_output = encoder_output[-1] # keep only output from last time step of recurrent layer
        # last encoder output = (batch_size, last_layer_number_of_neurons) because no return seq = state_h-> one weights matrix for each instance get encoded version of inputs (dense vector representation)
        states_c = states_c[-1] # get only last layer state shape = (batch_size, last_layer_number_of_neurons)

        # Decoder

        # initialize decoder_output list with repeated encoded output
        decoder_output = [self.repeat(encoder_output)] # repeat output (last time step) = state h repeated for each timestep to form input x for first decoder layer

        # first iteration
        # use h and c from last encoder layer to initalize h_state and c_state of first decoder layer for the first iteration.
        # use repeated state h as input
        decoder_output.append(self.decoder[0](decoder_output[0], initial_state=[encoder_output, states_c])) # init_state != None only for first iteration
        
        # compute other layer states and outputs 
        for i in range(1, len(self.decoder)):
            decoder_output.append(self.decoder[i](decoder_output[i], initial_state=None))

        self.regular_outputs, self.event_outputs = self.split_outputs(decoder_output[-1]) # split sequence output from last encoder layer

        # Timedistributed layers
        # regular_outputs: (batch_size, input_size, output_size*n_features)
        # event_outputs: (batch_size, input_size, encoding_width)
        self.regular_outputs = self.regularDense(self.regular_outputs)
        self.event_outputs = self.eventDense(self.event_outputs)# (batch_size, input_size, output_size*nb_features) after

        # Softmax performed on each one hot vector
        # reshape to (batch_size, input_size, outputsize, encoding_width)
        self.event_outputs = tf.reshape(self.event_outputs, shape=[tf.shape(self.event_outputs)[0], self.param_dict.get('input_size'), self.param_dict.get('output_size'), self.encoding_width])
        
        # apply softmax to each encoding_width values vectors
        self.event_outputs = self.softmax_layer(self.event_outputs)
        
        return {"regression_output": self.regular_outputs, "event_output": self.event_outputs}

    ############# CUSTOM CLASS FUNCTIONS #############

    def plot_exmpl_graphs(self, true_series, predict_series, x_events, true_events, predict_events):
        """
        Plots some example of predictions obtained via input test dataset
        Arguments:
        true_series: target series, numpy array of shape = (test_dataset_size, n_features (no slope feature), input_size+output_size)
        predict_series: predicted series, numpy array of shape (test_dataset_size, n_features (no slope feature), input_size+output_size)
        x_events: input test decoded (string) events - array of shape (test_dataset_size, input_size)
        true_events: target test decoded events - array of shape (test_dataset_size, output_size)
        predict_events: predicted test decoded events - array of shape (test_dataset_size, output_size)
        """

        n_params = int(self.nb_parameters/2) # ignore slope parameters

        # check max number of graph to create
        if self.param_dict.get('nb_graphs') > true_series.shape[0]:
            self.param_dict['nb_graphs'] = true_series.shape[0]
            sys.stderr.write(f"[PredictionWarning] The number of graphs to create is greater than the actual amount of samples in the test dataset.\n nb_graphs set to {true_series.shape[0]}.\n")

        time_axes = np.arange(0, self.param_dict.get('input_size')+self.param_dict.get('output_size'), 1) # time axes (in timesteps)
        
        # plot each mts
        for n in range(0, self.param_dict.get('nb_graphs')):
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
            for p in range(0, n_params): # for each params
                max_ = np.max(predict_series[n, p])
                min_ = np.min(predict_series[n, p])
                maxes.append(max_)
                mins.append(min_)
                plt.plot(time_axes, predict_series[n, p], label=f"{self.headers[p]} - prediction", linewidth=1)
                plt.plot(time_axes, true_series[n, p], label=f"{self.headers[p]} - target", linewidth=1)
                plt.legend(loc="upper left")
            
            axes.xaxis.set_ticks(time_axes)
            axes.yaxis.set_ticks(np.arange(int(min(mins)-5), int(max(maxes)+5), 5))
            axes.xaxis.set_ticks_position('bottom')
            axes.set_ylabel("Physiological parameters value", fontweight="bold", fontsize=15)
            axes.set_xlabel("Time (in timesteps)", fontweight="bold", fontsize=15)
            plt.text(-0.3, max(maxes)+15, "true/predict", rotation=90, fontweight="bold")
            plt.tight_layout()

            os.makedirs(os.path.join(self.test_logdir, "prediction_example_plots/"), exist_ok=True)

            plt.savefig(os.path.join(self.test_logdir, f"prediction_example_plots/prediction_{n+1}.pdf"), dpi=100)
            plt.clf()
            plt.close()

        if self.param_dict.get('nb_graphs') > 0:
            if os.path.exists(os.path.join(self.test_logdir, 'prediction_example_plots/')):
                print("[CreateFileSuccess] Prediction plots created.\n")
            else:
                sys.stderr.write(f"[CreateFileError] An error occured while creating the prediction plots.\n.")

        return 
    
    def write_boxplots_descriptions(self, csv_path, param, means, stds, meds, q1, q3, iqr, nb_outliers, eval_dataset_size, percent_outliers, whisker_bounds):
        """Writes description of one boxplot file (one physiological parameter) for each timestep
        Arguments:
        csv_path: csv_filepath (string)
        param: current physiological parameter to study
        means: list of means for each timestep
        stds: list of standard error for each timestep
        q1: list of first quartile values for each timestep
        q3: list of third quartile values for each timestep
        iqr: list of inter-quatiles range for each timestep
        nb_outliers: number of outliers for each timestep
        eval_dataset_size: dataset size for all timesteps
        percent_outliers: list of percentage of outliers for each timestep
        whisker_bounds: nested list. 2D list. The list in position 0 refers to whisker low bounds list and the list in position 1 refers to whisker high bounds list.
        """

        # set file opening mode ("append" mode if file already exists)
        mode = 'w'
        if os.path.exists(csv_path):
            mode = 'a'
        
        # write in files
        with open(csv_path, mode) as csv:
            csv.write(f"{param}\n")
            csv.write("means,stds,meds,q1,q3,iqr,whisker_low,whisker_high,nb_outliers,total_timestep_eval_size,percent_outliers\n")
            for i in range(0, self.param_dict.get('output_size')):
                csv.write(f"{means[i]},{stds[i]},{meds[i]},{q1[i]},{q3[i]},{iqr[i]},{whisker_bounds[0][i]},{whisker_bounds[1][i]},{nb_outliers[i]},{eval_dataset_size},{percent_outliers[i]}\n")
            csv.write('\n') # skip line between each parameter
        
        return
    
    def plot_to_image(self, figure):
        """Official documentation: Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        Arguments: 
        figure: matplotlib figure object"""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from behing shown in notebook if used 
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        return image
    
    def create_raw_error_distribution_csv_file(self, errors_data, output_file):
        """Creates file and writes raw errors distribution (on time serie values) for each parameter and each timestep.
        Arguments:
        errors_data: 3D numpy array representing the computed errors on evaluation data: shape = (nb_physiological_parameter, output_size, nb_values)
        output_file: file to create and write"""

        try:
            with open(output_file, 'a') as f:
                for param in range(0, errors_data.shape[0]):
                    f.write(f"{self.headers[param]}\n")
                    for timestep in range(0, errors_data.shape[1]):
                        f.write(f"timestep {timestep},")
                        for val in range(0, errors_data.shape[2]):
                            if val == errors_data.shape[2] - 1:
                                f.write(f"{errors_data[param, timestep, val]}\n")
                                continue
                            #else
                            f.write(f"{errors_data[param, timestep, val]},")
        except:
            sys.stderr.write(f"[CreateFileError] Unable to create file '{output_file}'.\n")
        else:
            sys.stdout.write(f"[CreateFileSuccess] File '{output_file}' created !\n")

    def make_custom_evaluation_boxplots(self, errors_data, boxplot_dir, absolute=False):
        """Creates distribution boxplots of relative percentage error for each feature of a dataset (and for each timestep) and add images of boxplots to tensorboard
        Arguments:
        errors_data: numpy 2D arrays. Ex: [ [0, 1], [2, 3] ]
        boxplot_dir: output directory
        absolute: boolean to know if we are plotting absolute (True) or relative percentage errors (False)"""

        n_params = int(self.nb_parameters/2) # get number of params without slope features

        # example: transform shape = (40, 700) to (40/output_size, output_size, 700) and transpose to obtain all the data 2D arrays for each feature
        errors_data = np.array(np.split(errors_data, self.param_dict.get('output_size'))).transpose(1,0,2)
        
        # set output directory for boxplots
        boxplot_dir = os.path.join(self.test_logdir, boxplot_dir)
        os.makedirs(boxplot_dir, exist_ok=True)
        boxplot_writer = tf.summary.create_file_writer(boxplot_dir)

        # set boxplot csv description filepath
        boxplot_csv = os.path.join(boxplot_dir, "boxplot_summaries.csv")

        # set raw distribution file
        errors_distribution_file = os.path.join(boxplot_dir, "relative_percentage_error_evaluation_raw_distribution.csv")
        if absolute:
            errors_distribution_file = os.path.join(boxplot_dir, "absolute_percentage_error_evaluation_raw_distribution.csv")

        self.create_raw_error_distribution_csv_file(errors_data, errors_distribution_file)

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        figsize=(1690*px, 900*px)

        for p in range(0, n_params):
            fig = plt.figure(figsize = figsize)
    
            # Creating axes instance
            ax = plt.gca()
            
            y_label = "Relative percentage error"
            if absolute:
                y_label = "Absolute percentage error"

            ax.set_ylabel(y_label, fontweight="bold", fontsize=25)
    
            # Creating plot
            boxplot = ax.boxplot(list(errors_data[p]), patch_artist=True, showmeans=True, meanline=True, showcaps=True, showbox=True, showfliers=True)
            plt.title(f"{self.headers[p]}")
            
            colors = [ 
                "orangered", "darkred", "yellow" ,"red", "cyan", "gold", "lightgreen", "purple", "blue", "pink",
                "darkgreen", "darkorange", "mediumpurple", "aqua", "khaki", "coral", "grey", "azure",
                "seagreen", "navy", "greenyellow", "chocolate", "salmon", "peru", "linen", "stateblue", "antiquewhite",
                "blanchedalmond", "burlywood", "firebrick", "brown", "darkviolet", "mediumaquamarine", "ivory", "goldenrod",
                "bisque", "cornflowerblue", "midnightblue", "lightseagreen", "darkorchid", "thisle", "fuchsia"  
            ] # colors of the different boxes (one per feature )

            # Computing boxplot statistics in order to display values on boxplot

            # statistic parameters for each timestep of the current physiological parameter
            means = []
            stds = []
            meds = []
            q1 = [] # lower quartile
            q3 = [] # upper quartile
            iqr = [] # interquarile region
            whisker_bounds = [ list(), list() ] # whiskers
            nb_outliers = [] # nb of outliers for each timestep
            eval_dataset_size = len(errors_data[p][0]) # all timesteps evaluation dataset size
            percent_outliers = [] # proportion of outliers

            for t, timestep in enumerate(errors_data[p]):
                # compute means
                means.append(np.mean(timestep))
                # compute std error
                stds.append(np.std(timestep))
                # compute median
                meds.append(np.median(timestep))
                # find the 1st quartile
                q1.append(np.quantile(timestep, 0.25))
                # find the 3rd quartile
                q3.append(np.quantile(timestep, 0.75))
                # finding the iqr region
                iqr.append(q3[t]-q1[t])
                # finding upper and lower whiskers (extrémités moustache)
                whisker_bounds[0].append(q1[t]-(1.5*iqr[t])) # lower bound
                whisker_bounds[1].append(q3[t]+(1.5*iqr[t])) # upper bound
                # finding outliers
                outliers = timestep[(timestep <= whisker_bounds[0][t]) | (timestep >= whisker_bounds[1][t])]
                # finding number of outliers
                nb_outliers.append(len(outliers))
                # finding outliers percentage
                percent_outliers.append((nb_outliers[t]/eval_dataset_size)*100)

            # Customize boxplot
            ax.set_yticks(np.arange(0, np.max(errors_data[p])+5, 5), labels=None, minor=True)

            # Change box colors
            for patch, color in zip(boxplot['boxes'], colors[0:len(boxplot['boxes'])]):
                patch.set_facecolor(color)

            # Modify whiskers
            for i, whisker in enumerate(boxplot['whiskers']):
                whisker.set(color = '#443266', linewidth = 1.5) # modify color

            # Change caps colors
            for cap in boxplot['caps']:
                cap.set(color = '#443266', linewidth = 2)

            # Chage flyers style
            for flier in boxplot['fliers']:
                flier.set(marker = 'D', color = '#e7298a', alpha = 0.5)
            
            image_name = f"{self.headers[p]}_relative_percentage_error_distrib_per_timestep"
            if absolute:
                image_name = f"{self.headers[p]}_absolute_percentage_error_distrib_per_timestep"

            output_path = os.path.join(boxplot_dir, image_name) #chemin d'accès final du fichiercsv

            # try to save the figure
            try:
                plt.savefig(f"{output_path}", format='png') #sauvegarde du boxplot
            except:
                sys.stderr.write(f"[CreateFileError] Unable to create file '{output_path}'.\n")
            else:
                sys.stdout.write(f"[CreateFileSuccess] File '{output_path}' created !\n")

                # write figures in tensorboard
                with boxplot_writer.as_default():
                    tf.summary.image(image_name, self.plot_to_image(fig), step=0) # plot to image closes also the matplotlib figure
            
            # write boxplots info 
            self.write_boxplots_descriptions(boxplot_csv, self.headers[p], means, stds, meds, q1, q3, iqr, nb_outliers, eval_dataset_size, percent_outliers, whisker_bounds)

            # close create figure
            plt.close()

    def custom_evaluate(self, x, y, scalers, fold_logdir):
        """ Computes relative percentage error for each feature at each timestep in the dataset.
        Creates distribution boxplots of relative percentage error for each physiological parameter and for each timestep.
        Plots some example of contextualized multivariate predictions.
        Also compute different evaluation metrics for action predictions (F1score by class, F1 score macro average with arithmetic and geometric mean, Precision/Recall by class), creates confusion matrix.
        Arguments:
        x: tuple or list [serie_values_input_test, action_string_input_test]
        - serie_values_input_test: array of shape = (test_dataset_size, input_size, n_features)
        - action_string_input_test: array of shape = (test_dataset_size, input_size, encoding_width)
        y: tuple or list [serie_values_target_test, action_string_target_test]
        - serie_value_target_test: array of shape = (test_dataset_size, input_size, output_size*n_features)
        - action_string_target_test: array of shape = (test_dataset_size, input_size, output_size, encoding_width)
        fold_logdir: fold directory path
        """

        # create custom_evaluation_dictionary
        custom_eval = {}

       # create log directory for custom evaluation, and create log writer object
        self.test_logdir = os.path.join(fold_logdir, "custom_evaluation")
        os.makedirs(self.test_logdir, exist_ok=True)

        test_summary_writer = tf.summary.create_file_writer(self.test_logdir)
        
        # unpack scaler objects
        std_scaler, min_max_scaler = scalers
        
        # predict with previously trained neural network (training == False to avoid any weights update while inferring)
        y_pred = self(x, training=False)
        
        # get only last time step (full prediction/cible of the next window)
        regular_x, event_x = x
        regular_y = y['regression_output'][:, -1]
        regular_y_pred = y_pred['regression_output'][:, -1]
        event_y = y['event_output']
        event_y_pred = y_pred['event_output']

        # F1 scores, recall and precision
        F1_score_by_class = last_time_step_F1_score(event_y, event_y_pred, self.ordered_events, self.not_predicted_events, average=False)
        F1_score_macro_arithmetic_mean = last_time_step_F1_score(event_y, event_y_pred, self.ordered_events, self.not_predicted_events, average=True)
        F1_score_macro_geometric_mean = last_time_step_F1_score_macro_average_geometric_mean(event_y, event_y_pred, self.ordered_events, self.not_predicted_events)
        recall_by_label = last_time_step_Recall_by_label(event_y, event_y_pred, self.ordered_events, self.not_predicted_events)
        precision_by_label = last_time_step_Precision_by_label(event_y, event_y_pred, self.ordered_events, self.not_predicted_events)

        # confusion matrix - get + create csv file
        _ = create_multiclass_confusion_matrix(event_y, event_y_pred, self.ordered_events, self.not_predicted_events, os.path.join(self.test_logdir, "confusion_matrix.csv")) 

        # get decoded tensor events (decode only keeps last time step prediction of RNN)
        event_y = self.onehot.decode(event_y)
        event_y_pred = self.onehot.decode(event_y_pred)

        # get numpy array versions of tensor
        event_y = np.array(event_y, dtype=object)
        event_y_pred = np.array(event_y_pred, dtype=object)
        
        # reshape inputs for denormalization with scalers
        initial_shapes = [ regular_x.shape, regular_y.shape ]

        regular_x = np.reshape(regular_x, (regular_x.shape[0]*regular_x.shape[1], regular_x.shape[2]))
        regular_y = np.reshape(regular_y, (regular_y.shape[0]*self.param_dict.get('output_size'), self.nb_parameters))
        regular_y_pred = np.reshape(regular_y_pred, regular_y.shape)

        # denormalization with MinMaxScaler
        regular_y = min_max_scaler.inverse_transform(regular_y)
        regular_y_pred = min_max_scaler.inverse_transform(regular_y_pred)

        # destandardization with StandarScaler
        regular_y = std_scaler.inverse_transform(regular_y)
        regular_y_pred = std_scaler.inverse_transform(regular_y_pred)

        # get rid of gradient feature values
        regular_x = regular_x[:, 0:int(self.nb_parameters/2)] # nb_parameters/2 because of computation of gradients (not used in series error computation for evaluation)
        regular_y = regular_y[:, 0:int(self.nb_parameters/2)]
        regular_y_pred = regular_y_pred[:, 0:int(self.nb_parameters/2)]
        
        # reconstitute full input+(outputs|targets) series and reshape to initial shapes of inputs
        # transpose each series for plot_exmpl_graphs from shape:
        # [ 
        #   fc1, pas1, pad1 ... // each timestep
        #   fc2, pas2, pad2 ... 
        #    ... *(input_size+output_size)*nb_param], // each timestep
        # ]
        # 
        # to shape:
        # [
        #   [fc1, fc2, fc3 ... * input_size+output_size]
        #   [pas1 ...] # each line = 1 parameter
        #       ... * nb_parameters
        # ]

        true_series = np.concatenate( (np.reshape(regular_x, (initial_shapes[0][0], self.param_dict.get('input_size'), int(initial_shapes[0][2]/2))), np.reshape(regular_y, (initial_shapes[0][0], self.param_dict.get('output_size'), int(initial_shapes[0][2]/2))) ), axis=1 ).transpose(0, 2, 1)
        predict_series = np.concatenate( (np.reshape(regular_x, (initial_shapes[0][0], self.param_dict.get('input_size'), int(initial_shapes[0][2]/2))), np.reshape(regular_y_pred, (initial_shapes[0][0], self.param_dict.get('output_size'), int(initial_shapes[0][2]/2))) ), axis=1 ).transpose(0, 2, 1)
        # initial_shape[0][2]/2 -> because we previously got rid of slope features

        # flatten windows for last_time_step_RE_MAPE_per_feature
        regular_y = np.reshape(regular_y, (initial_shapes[1][0], int(initial_shapes[1][1]/2)))
        regular_y_pred = np.reshape(regular_y_pred, (initial_shapes[1][0], int(initial_shapes[1][1]/2)))      
        
        # compute mean_relative errors and relative percentage error
        relative_errors, mrpe = last_time_step_errors_mean_errors(regular_y, regular_y_pred, int(self.nb_parameters/2), absolute=False)
        
        # compute mean absolute errors and absolute percentage error
        abs_errors, mape = last_time_step_errors_mean_errors(regular_y, regular_y_pred, int(self.nb_parameters/2), absolute=True)

        # add custom mrpe per feature to custom_eval_dict
        custom_eval['mean_relative_percentage_error_per_feature'] = {self.headers[i]:mrpe[i] for i in range(0, int(self.nb_parameters/2))} # this new key is also a dictionary, dict of 'nb_parameters' (4) list - one for each physiological parameter
        custom_eval['mean_absolute_percentage_error_per_feature'] = {self.headers[i]:mape[i] for i in range(0, int(self.nb_parameters/2))} # this new key is also a dictionary, dict of 'nb_parameters' (4) list - one for each physiological parameter

        # plot MRPE and MAPE in tensorboard (scalar mode)
        with test_summary_writer.as_default():
            for hist in range(0, int(self.nb_parameters/2)):
                for step_ in range(0, self.param_dict.get('output_size')):
                    tf.summary.scalar(f'mean_relative_percentage_error_{self.headers[hist]}', mrpe[hist, step_], step=step_) # take as input a scalar -> not a vector, step = pas de temps courant         
                    tf.summary.scalar(f'mean_absolute_percentage_error_{self.headers[hist]}', mape[hist, step_], step=step_)

        # plot relative_errors with matplotlib and in tensorboard (boxplot - image mode) and creates the boxplots in the corresponding log_subdiretory
        self.make_custom_evaluation_boxplots(relative_errors, "relative_percentage_error_boxplots", absolute=False)

        # plot absolute errors with matplotlib and in tensorboard (boxplot - image mode) and creates the boxplots in the corresponding log_subdiretory
        self.make_custom_evaluation_boxplots(abs_errors, "absolute_percentage_error_boxplots", absolute=True)

        # evaluate with custom events metrics (last time step)
        nn_acc = NN_accuracy(event_y, event_y_pred)
        ne_errp = NE_percentage_error(event_y, event_y_pred)
        events_acc = action_specific_accuracy(event_y, event_y_pred)
        asa_with_lag = accuracy_with_lag(event_y, event_y_pred, delay=self.param_dict.get('delay'))

        # add events metrics to custom evaluation dict
        custom_eval['last_time_step_nn_acc'] = nn_acc
        custom_eval['last_time_step_ne_PE'] = ne_errp
        custom_eval['last_time_step_asa'] = events_acc
        custom_eval['last_time_step_acc_with_lag'] = asa_with_lag
        custom_eval['last_time_step_precision_by_label'] = precision_by_label
        custom_eval['last_time_step_recall_by_label'] = recall_by_label
        custom_eval['last_time_step_F1_score_by_label'] = F1_score_by_class
        custom_eval['last_time_step_F1_score_macro_average_arithmetic_mean'] = F1_score_macro_arithmetic_mean
        custom_eval['last_time_step_F1_score_macro_average_geometric_mean'] = F1_score_macro_geometric_mean

        # plot some predictions graphs
        self.plot_exmpl_graphs(true_series, predict_series, event_x, event_y, event_y_pred)

        return custom_eval

    def save_model(self, fold_logdir):
        """Saves the model in specific directory in a tensorflow format.
        Arguments:
        fold_logdir: fold directory
        """
        # create directories to save model
        if self.param_dict.get('save_model'): # if save_model is True
            self.full_path_model = os.path.join(fold_logdir, "saved_model/") # get output_dir+model dir name
            os.makedirs(self.full_path_model, exist_ok = True) # créate the 2 directories (1 directory and a sub directory.)
            
            # save model
            signatures = { # get tensorflow converted functions with specific signatures for the exported model's call function (different behavior while training / infering)
                "serving_default": self.call.get_concrete_function(inputs=[tf.TensorSpec(shape=[None, self.param_dict.get('input_size'), self.nb_parameters], dtype=tf.float32), tf.TensorSpec(shape=[None, self.param_dict.get('input_size'), self.encoding_width], dtype=tf.float32)], training=True),
                "export_signature": self.call.get_concrete_function(inputs=[tf.TensorSpec(shape=[None, self.param_dict.get('input_size'), self.nb_parameters], dtype=tf.float32), tf.TensorSpec(shape=[None, self.param_dict.get('input_size')], dtype=tf.string)], training=False)
            }
            tf.keras.models.save_model(self, self.full_path_model, save_format="tf", signatures=signatures)

            # check if model is saved properly
            if os.path.exists(os.path.join(self.full_path_model, "saved_model.pb")) and os.path.exists(os.path.join(self.full_path_model, "variables/")) and os.path.exists(os.path.join(self.full_path_model, "assets/")):
                    print(f"\n[CreateModelSuccess] \"{self.full_path_model}\" and model files/directories successfully created.\n")
            else:
                sys.stderr.write(f"[SaveError] Unable to save model in \"{self.full_path_model}\".\n")
        
        return

    def get_config(self):
        """Defines layer configuration for tensorflow serialization.
        Warning: every parameter in init should be in config
        """
        # return config dictionary
        config = super(CustomEncoderDecoder, self).get_config()
        config['param_dict'] = self.param_dict
        config['headers'] = self.headers
        config['nb_parameters'] = self.nb_parameters
        config['normalization'] = self.normalization
        config['standardization'] = self.standardization
        config['onehot'] = self.onehot
        config['encoding_width'] = self.encoding_width
        config['ordered_events'] = self.ordered_events
        config['repeat'] = self.repeat
        config['concatenate'] = self.concatenate
        config['encoder'] = self.encoder
        config['decoder'] = self.decoder
        config['regularDense'] = self.regularDense
        config['eventDense'] = self.eventDense
        config['softmax_layer'] = self.softmax_layer
        config['last_time_step_mse'] = last_time_step_mse
        config['last_time_step_errors_mean_errors'] = last_time_step_errors_mean_errors
        config['relative_percentage_error'] = relative_percentage_error
        config['absolute_percentage_error'] = absolute_percentage_error
        config['last_time_step_categorical_accuracy'] = last_time_step_categorical_accuracy
        config['last_time_step_CategoricalCrossentropy'] = last_time_step_CategoricalCrossentropy
        config['last_time_step_F1_score'] = last_time_step_F1_score
        config['last_time_step_F1_score_macro_average_geometric_mean'] = last_time_step_F1_score_macro_average_geometric_mean
        config['last_time_step_Recall_by_label'] = last_time_step_Recall_by_label
        config['last_time_step_Precision_by_label'] = last_time_step_Precision_by_label
        config['NN_accuracy'] = NN_accuracy
        config['NE_percentage_error'] = NE_percentage_error
        config['action_specific_accuracy'] = action_specific_accuracy
        config['accuracy_with_lag'] = accuracy_with_lag
        config['split_outputs'] = self.split_outputs
        config['plot_to_image'] = self.plot_to_image
        config['create_multiclass_confusion_matrix'] = create_multiclass_confusion_matrix
        config['write_boxplot_descriptions'] = self.write_boxplots_descriptions
        config['make_custom_evaluation_boxplots'] = self.make_custom_evaluation_boxplots
        config['save_model'] = self.save_model
        config['plot_exampl_graphs'] = self.plot_exmpl_graphs
        config['custom_evaluate'] = self.custom_evaluate
        
    @classmethod
    def from_config(cls, config):
        """For tensorflow serialization. Customizable function.
        Here is default function"""
        return cls(**config)
        
##################################################################################################################

class EncoderDecoder_no_split_outputs(CustomEncoderDecoder):
    """EncoderDecoder without split_outputs function. We use the full output of the decoder as input for the TimeDistributed Layers."""
    def __init__(self, param_dict, headers, logdir, scalers, ordered_events, encoding_width, not_predicted_events, **kwargs):
        """Class constructor 
        Encoder_decoder class for joint modeling.
        Arguments:
        param_dict: parameters dictionary
        headers: list of physiological variable names
        logdir: run directory
        scalers: tuple of scaler objects (StandardScaler object, MinMaxScaler object) obtained from scikit learn library.
        ordered_events: oredered array of uniq actions in the dataset, used for OneHotEncoding Layer
        encoding_width: number of classes (actions) in the dataset
        """
        super(CustomEncoderDecoder, self).__init__(**kwargs) # initialize parent class

        # def self variables
        self.full_path_model = ""
        self.nb_parameters = len(headers)*2
        self.headers = headers
        self.param_dict = param_dict
        self.run_logdir = logdir # mean that we have to create another instnace of model for each run
        std_scaler, minmax_scaler = scalers # no serialization, used only to instanciate Standardization and Normalization Layers
        self.ordered_events = ordered_events
        self.encoding_width = encoding_width
        self.not_predicted_events = not_predicted_events

        # correspondance dictionnary for weights initializers
        available_kernel_init = {
            "glorotUniform": keras.initializers.GlorotUniform(),
            "heUniform": keras.initializers.HeUniform(),
            "glorotNormal": keras.initializers.GlorotNormal(),
            "heNormal": keras.initializers.HeNormal(),
            "zeros": keras.initializers.Zeros(),
            "ones": keras.initializers.Ones(),
            "constant": keras.initializers.Constant(value=0.5),
            # add others if needed
        }

        # correspondance dictionary for activation functions
        available_activations = {
            "tanh": keras.activations.tanh,
            "sigmoid": keras.activations.sigmoid,
            "relu": keras.activations.relu,
            "elu": keras.activations.elu,
            "linear": keras.activations.linear,
            "selu": keras.activations.selu,
            "gelu":keras.activations.gelu
            # add others if needed
        }
        
        # Preprocessing layers
        self.standardization = StandardizationLayer(means=std_scaler.mean_, stds=std_scaler.scale_, nb_parameters=self.nb_parameters, output_size=self.param_dict.get('output_size'), name="TestStdLayer")
        self.normalization = MinMaxNormalizationLayer(minmax_scaler.data_min_, minmax_scaler.data_max_, param_dict.get('min_norm_threshold'), param_dict.get('max_norm_threshold'), nb_parameters=self.nb_parameters, output_size=self.param_dict.get('output_size'), name="TestNormLayer")
        self.onehot = OneHotEncodingLayer(ordered_events, encoding_width)
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Encoder module 
        self.encoder = []
        for i in range(0, self.param_dict.get('n_encoder_layers')):
            # default return_sequences and return state parameter for more than one layer models
            return_seq = True

            # boolean set last layer return seq parameter: 
            # i != (n_encoder_layers -1) -> return_seq = True
            # i == (n_encoder_layers - 1) -> return_seq = False
            return_seq = bool(i != (self.param_dict.get('n_encoder_layers') - 1) )
            
            if i == 0:
                self.encoder.append(
                    tf.keras.layers.LSTM(
                        self.param_dict.get('encoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh,
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'encoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("encoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("encoder_dropout")[i],
                        input_shape=[None, None], # output_size, 8+36
                        return_sequences=return_seq,
                        return_state=True # return state at each layer (avoid conditions in call) but will use only the last layer state as initial state for first decoder layer
                    )
                )
            else:
                self.encoder.append(
                    tf.keras.layers.LSTM(
                        self.param_dict.get('encoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh,
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'encoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("encoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("encoder_dropout")[i],
                        return_sequences=return_seq,
                        return_state=True # return state at each layer (avoid conditions in call) but will use only the last layer state as initial state for first decoder layer
                    )
                )
            
        # Repeat last time step prediction from last encoder layer, in order to use repaetedVector as input for decoder's first layer
        self.repeat = tf.keras.layers.RepeatVector(n=self.param_dict.get('input_size'))
        
        # Decoder module
        self.decoder = []
        for i in range(0, self.param_dict.get('n_decoder_layers')):
            # default return_sequences and return state parameter for more than one layer models
            if i == 0:
                self.decoder.append(
                    tf.keras.layers.LSTM( # pour créer des cocuhe perso il faut utiliser la classe RNN de base en leur pazssaznt une couche (cell class) perso
                        self.param_dict.get('decoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh, # tanh: fonction saturante pour éviter l'instabilité des gradients.(évite que les valeurs ou gradients explosent, variation faible si grosses valeurs, donc les non saturantes comme relu aident pas)
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'decoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("decoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("decoder_dropout")[i],
                        input_shape=[None, None], # output_size, len(output vect from last encoder layer)
                        return_sequences=True,
                        return_state=False
                    )
                )
            else:
                self.decoder.append(
                    tf.keras.layers.LSTM( # pour créer des cocuhe perso il faut utiliser la classe RNN de base en leur pazssaznt une couche (cell class) perso
                        self.param_dict.get('decoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh, # tanh: fonction saturante pour éviter l'instabilité des gradients.(évite que les valeurs ou gradients explosent, variation faible si grosses valeurs, donc les non saturantes comme relu aident pas)
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'decoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("decoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("decoder_dropout")[i],
                        return_sequences=True,
                        return_state=False
                    )
                )
        
        # TimeDistributed layers
        self.eventDense = keras.layers.TimeDistributed(
            keras.layers.Dense(
                self.param_dict.get('output_size')*self.encoding_width,
                kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                use_bias = True,
                bias_initializer='zeros',
                name='events_output_layer',
                activation = keras.activations.linear
            )
        )

        # always a regular dense timedistributed layer here because no split outputs
        self.regularDense = keras.layers.TimeDistributed( # we could use an RNN layer but a dense layer allows to chosse the activation and has the same capacities (our values between 0 and 1 so we cant use tanh)
            keras.layers.Dense(
                self.param_dict.get('output_size')*self.nb_parameters, # output -> taille = nb_pas de temps qu'on veut prédire (chaque pas de temps =  nb_parametre)
                kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                use_bias = True,
                bias_initializer='zeros',
                name='regular_output_layer',
                activation = available_activations[self.param_dict.get('last_activation')]
            )
        )
        
         # softmax layer for one hot encoded action predictions
        self.softmax_layer = SoftMaxLayer()
        
    @tf.function
    def call(self, inputs, training=None, **kwargs): # we have to tell the forward pass with () at end of line, beucause non sequential model
        """Main method. This function is executed when calling an instance of this model class.
        Defines the forward pass used when using model.fit
        inputs: list of input types: [serie_values_input, action_onehot_input]
        for training:
            - serie_values_input: (batch_size, input_size, n_features)
            - action_one_hot_input: (batch_size, input_size, encoding_width)
        for inference:
            - serie_values_input: (batch_size, input_size, n_features)
            - action_one_hot_input: (batch_size, input_size)
        training: boolean indicating behavior of function during training. Used for preprocessing layers and dropout layers training argument
        **kwargs: any Tensorflow Model Class argument"""

        # get regular and events inputs
        regular_inputs, events = inputs

        #### INFERENCE ONLY (for x_val and x_test) ####
        regular_inputs = self.standardization(regular_inputs, training=training)
        regular_inputs = self.normalization(regular_inputs, training=training)
        events = self.onehot(events, training=training)
        ##########################################

        concat_inputs = self.concatenate([regular_inputs, events])

        # ENCODER DECODER

        # Encoder
        encoder_output = [concat_inputs] # initialize encoder_output with concatenated preprocessed inputs
        states_c = [] # initialize list of long term memory vector outputs

        # for each layer, compute output from previous output
        for i in range(0, len(self.encoder)):
            output, _, state_c = self.encoder[i](encoder_output[i])
            encoder_output.append(output)
            states_c.append(state_c)
            
        encoder_output = encoder_output[-1] # keep only output from last time step of recurrent layer
        # last encoder output = (batch_size, last_layer_number_of_neurons) because no return seq = state_h-> one weights matrix for each instance get encoded version of inputs (dense vector representation)
        states_c = states_c[-1] # get only last layer state shape = (batch_size, last_layer_number_of_neurons)

        # Decoder

        # initialize decoder_output list with repeated encoded output
        decoder_output = [self.repeat(encoder_output)] # repeat output (last time step) = state h repeated for each timestep to form input x for first decoder layer

        # first iteration
        # use h and c from last encoder layer to initalize h_state and c_state of first decoder layer for the first iteration.
        # use repeated state h as input
        decoder_output.append(self.decoder[0](decoder_output[0], initial_state=[encoder_output, states_c])) # init_state != None only for first iteration
        
        # compute other layer states and outputs 
        for i in range(1, len(self.decoder)):
            decoder_output.append(self.decoder[i](decoder_output[i], initial_state=None))

        # Timedistributed layers
        self.regular_outputs = self.regularDense(decoder_output[-1])
        self.event_outputs = self.eventDense(decoder_output[-1])

        # Softmax performed on each one hot vector
        # reshape to (batch_size, input_size, outputsize, encoding_width)
        self.event_outputs = tf.reshape(self.event_outputs, shape=[tf.shape(self.event_outputs)[0], self.param_dict.get('input_size'), self.param_dict.get('output_size'), self.encoding_width])
        
        # apply softmax to each encoding_width values vectors
        self.event_outputs = self.softmax_layer(self.event_outputs)
        
        return {"regression_output": self.regular_outputs, "event_output": self.event_outputs}

    def get_config(self):
        """Defines layer configuration for tensorflow serialization.
        Warning: every parameter in init should be in config
        """
        # return config dictionary
        config = super(CustomEncoderDecoder, self).get_config()
        config['param_dict'] = self.param_dict
        config['headers'] = self.headers
        config['nb_parameters'] = self.nb_parameters
        config['normalization'] = self.normalization
        config['standardization'] = self.standardization
        config['onehot'] = self.onehot
        config['encoding_width'] = self.encoding_width
        config['ordered_events'] = self.ordered_events
        config['repeat'] = self.repeat
        config['concatenate'] = self.concatenate
        config['encoder'] = self.encoder
        config['decoder'] = self.decoder
        config['regularDense'] = self.regularDense
        config['eventDense'] = self.eventDense
        config['softmax_layer'] = self.softmax_layer
        config['last_time_step_mse'] = last_time_step_mse
        config['last_time_step_errors_mean_errors'] = last_time_step_errors_mean_errors
        config['relative_percentage_error'] = relative_percentage_error
        config['absolute_percentage_error'] = absolute_percentage_error
        config['last_time_step_categorical_accuracy'] = last_time_step_categorical_accuracy
        config['last_time_step_CategoricalCrossentropy'] = last_time_step_CategoricalCrossentropy
        config['last_time_step_F1_score'] = last_time_step_F1_score
        config['last_time_step_F1_score_macro_average_geometric_mean'] = last_time_step_F1_score_macro_average_geometric_mean
        config['last_time_step_Recall_by_label'] = last_time_step_Recall_by_label
        config['last_time_step_Precision_by_label'] = last_time_step_Precision_by_label
        config['NN_accuracy'] = NN_accuracy
        config['NE_percentage_error'] = NE_percentage_error
        config['action_specific_accuracy'] = action_specific_accuracy
        config['accuracy_with_lag'] = accuracy_with_lag
        config['plot_to_image'] = self.plot_to_image
        config['create_multiclass_confusion_matrix'] = create_multiclass_confusion_matrix
        config['write_boxplot_descriptions'] = self.write_boxplots_descriptions
        config['make_custom_evaluation_boxplots'] = self.make_custom_evaluation_boxplots
        config['save_model'] = self.save_model
        config['plot_exampl_graphs'] = self.plot_exmpl_graphs
        config['custom_evaluate'] = self.custom_evaluate    

##################################################################################################################

class EncoderDecoder_no_dense_event_layer(CustomEncoderDecoder):
    """Encoder Decoder class using split outputs function but not using timedistributed and dense layer for events in the output layers. Use a timedistributed dense layer
    for serie values outputs if normalization not in range(-1,1)"""
    def __init__(self, param_dict, headers, logdir, scalers, ordered_events, encoding_width, not_predicted_events, **kwargs):
        """Class constructor 
        Encoder_decoder class for joint modeling.
        Arguments:
        param_dict: parameters dictionary
        headers: list of physiological variable names
        logdir: run directory
        scalers: tuple of scaler objects (StandardScaler object, MinMaxScaler object) obtained from scikit learn library.
        ordered_events: oredered array of uniq actions in the dataset, used for OneHotEncoding Layer
        encoding_width: number of classes (actions) in the dataset
        """
        super(CustomEncoderDecoder, self).__init__(**kwargs) # initialize parent class

        # def self variables
        self.full_path_model = ""
        self.nb_parameters = len(headers)*2
        self.headers = headers
        self.param_dict = param_dict
        self.run_logdir = logdir # mean that we have to create another instnace of model for each run
        std_scaler, minmax_scaler = scalers # no serialization, used only to instanciate Standardization and Normalization Layers
        self.ordered_events = ordered_events
        self.encoding_width = encoding_width
        self.not_predicted_events = not_predicted_events

        # correspondance dictionnary for weights initializers
        available_kernel_init = {
            "glorotUniform": keras.initializers.GlorotUniform(),
            "heUniform": keras.initializers.HeUniform(),
            "glorotNormal": keras.initializers.GlorotNormal(),
            "heNormal": keras.initializers.HeNormal(),
            "zeros": keras.initializers.Zeros(),
            "ones": keras.initializers.Ones(),
            "constant": keras.initializers.Constant(value=0.5),
            # add others if needed
        }

        # correspondance dictionary for activation functions
        available_activations = {
            "tanh": keras.activations.tanh,
            "sigmoid": keras.activations.sigmoid,
            "relu": keras.activations.relu,
            "elu": keras.activations.elu,
            "linear": keras.activations.linear,
            "selu": keras.activations.selu,
            "gelu":keras.activations.gelu
            # add others if needed
        }
        
        # Preprocessing layers
        self.standardization = StandardizationLayer(means=std_scaler.mean_, stds=std_scaler.scale_, nb_parameters=self.nb_parameters, output_size=self.param_dict.get('output_size'), name="TestStdLayer")
        self.normalization = MinMaxNormalizationLayer(minmax_scaler.data_min_, minmax_scaler.data_max_, param_dict.get('min_norm_threshold'), param_dict.get('max_norm_threshold'), nb_parameters=self.nb_parameters, output_size=self.param_dict.get('output_size'), name="TestNormLayer")
        self.onehot = OneHotEncodingLayer(ordered_events, encoding_width)
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Encoder module 
        self.encoder = []
        for i in range(0, self.param_dict.get('n_encoder_layers')):
            # default return_sequences and return state parameter for more than one layer models
            return_seq = True

            # boolean set last layer return seq parameter: 
            # i != (n_encoder_layers -1) -> return_seq = True
            # i == (n_encoder_layers - 1) -> return_seq = False
            return_seq = bool(i != (self.param_dict.get('n_encoder_layers') - 1) )
            
            if i == 0:
                self.encoder.append(
                    tf.keras.layers.LSTM(
                        self.param_dict.get('encoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh,
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'encoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("encoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("encoder_dropout")[i],
                        input_shape=[None, None], # output_size, 8+36
                        return_sequences=return_seq,
                        return_state=True # return state at each layer (avoid conditions in call) but will use only the last layer state as initial state for first decoder layer
                    )
                )
            else:
                self.encoder.append(
                    tf.keras.layers.LSTM(
                        self.param_dict.get('encoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh,
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'encoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("encoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("encoder_dropout")[i],
                        return_sequences=return_seq,
                        return_state=True # return state at each layer (avoid conditions in call) but will use only the last layer state as initial state for first decoder layer
                    )
                )
            
        # Repeat last time step prediction from last encoder layer, in order to use repaetedVector as input for decoder's first layer
        self.repeat = tf.keras.layers.RepeatVector(n=self.param_dict.get('input_size'))
        
        # Decoder module
        self.decoder = []
        for i in range(0, self.param_dict.get('n_decoder_layers')):
            # default return_sequences and return state parameter for more than one layer models
            if i == 0:
                self.decoder.append(
                    tf.keras.layers.LSTM( # pour créer des cocuhe perso il faut utiliser la classe RNN de base en leur pazssaznt une couche (cell class) perso
                        self.param_dict.get('decoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh, # tanh: fonction saturante pour éviter l'instabilité des gradients.(évite que les valeurs ou gradients explosent, variation faible si grosses valeurs, donc les non saturantes comme relu aident pas)
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'decoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("decoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("decoder_dropout")[i],
                        input_shape=[None, None], # output_size, len(output vect from last encoder layer)
                        return_sequences=True,
                        return_state=False
                    )
                )
            else:
                self.decoder.append(
                    tf.keras.layers.LSTM( # pour créer des cocuhe perso il faut utiliser la classe RNN de base en leur pazssaznt une couche (cell class) perso
                        self.param_dict.get('decoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh, # tanh: fonction saturante pour éviter l'instabilité des gradients.(évite que les valeurs ou gradients explosent, variation faible si grosses valeurs, donc les non saturantes comme relu aident pas)
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'decoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("decoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("decoder_dropout")[i],
                        return_sequences=True,
                        return_state=False
                    )
                )

         # TimeDistributed layers (for different normalization than -1/1)
        self.regularDense = lambda x: x # layer doing no operation

        if self.param_dict.get('min_norm_threshold') == -1 and self.param_dict.get('max_norm_threshold') == 1: # if not tanh range for normalization, we add a new output layer to the model
            pass
        else:
            self.regularDense = keras.layers.TimeDistributed( # we could use an RNN layer but a dense layer allows to chosse the activation and has the same capacities (our values between 0 and 1 so we cant use tanh)
                keras.layers.Dense(
                    self.param_dict.get('output_size')*self.nb_parameters, # output -> taille = nb_pas de temps qu'on veut prédire (chaque pas de temps =  nb_parametre)
                    kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                    use_bias = True,
                    bias_initializer='zeros',
                    name='regular_output_layer',
                    activation = available_activations[self.param_dict.get('last_activation')]
                )
            )
        
        # softmax layer for one hot encoded action predictions
        self.softmax_layer = SoftMaxLayer()

    @tf.function
    def call(self, inputs, training=None, **kwargs): # we have to tell the forward pass with () at end of line, beucause non sequential model
        """Main method. This function is executed when calling an instance of this model class.
        Defines the forward pass used when using model.fit
        inputs: list of input types: [serie_values_input, action_onehot_input]
        for training:
            - serie_values_input: (batch_size, input_size, n_features)
            - action_one_hot_input: (batch_size, input_size, encoding_width)
        for inference:
            - serie_values_input: (batch_size, input_size, n_features)
            - action_one_hot_input: (batch_size, input_size)
        training: boolean indicating behavior of function during training. Used for preprocessing layers and dropout layers training argument
        **kwargs: any Tensorflow Model Class argument"""

        # get regular and events inputs
        regular_inputs, events = inputs

        #### INFERENCE ONLY (for x_val and x_test) ####
        regular_inputs = self.standardization(regular_inputs, training=training)
        regular_inputs = self.normalization(regular_inputs, training=training)
        events = self.onehot(events, training=training)
        ##########################################

        concat_inputs = self.concatenate([regular_inputs, events])

        # ENCODER DECODER

        # Encoder
        encoder_output = [concat_inputs] # initialize encoder_output with concatenated preprocessed inputs
        states_c = [] # initialize list of long term memory vector outputs

        # for each layer, compute output from previous output
        for i in range(0, len(self.encoder)):
            output, _, state_c = self.encoder[i](encoder_output[i])
            encoder_output.append(output)
            states_c.append(state_c)
            
        encoder_output = encoder_output[-1] # keep only output from last time step of recurrent layer
        # last encoder output = (batch_size, last_layer_number_of_neurons) because no return seq = state_h-> one weights matrix for each instance get encoded version of inputs (dense vector representation)
        states_c = states_c[-1] # get only last layer state shape = (batch_size, last_layer_number_of_neurons)

        # Decoder

        # initialize decoder_output list with repeated encoded output
        decoder_output = [self.repeat(encoder_output)] # repeat output (last time step) = state h repeated for each timestep to form input x for first decoder layer

        # first iteration
        # use h and c from last encoder layer to initalize h_state and c_state of first decoder layer for the first iteration.
        # use repeated state h as input
        decoder_output.append(self.decoder[0](decoder_output[0], initial_state=[encoder_output, states_c])) # init_state != None only for first iteration
        
        # compute other layer states and outputs 
        for i in range(1, len(self.decoder)):
            decoder_output.append(self.decoder[i](decoder_output[i], initial_state=None))

        self.regular_outputs, self.event_outputs = self.split_outputs(decoder_output[-1]) # split sequence output from last encoder layer

        self.regular_outputs = self.regularDense(self.regular_outputs)

        # Softmax performed on each one hot vector
        # reshape to (batch_size, input_size, outputsize, encoding_width)
        self.event_outputs = tf.reshape(self.event_outputs, shape=[tf.shape(self.event_outputs)[0], self.param_dict.get('input_size'), self.param_dict.get('output_size'), self.encoding_width])
        
        # apply softmax to each encoding_width values vectors
        self.event_outputs = self.softmax_layer(self.event_outputs)
        
        return {"regression_output": self.regular_outputs, "event_output": self.event_outputs}
        
##################################################################################################################

class EncoderDecoder_with_attention(CustomEncoderDecoder):
    """EncoderDecoder implementation using Attention mechanism to add relevant informations to hidden vector.
    *** No error, runs well but not sure about correct use of attention ***. 
    This class is an adaptation of Fadoua's LSTM encoder-decoder attention model."""

    def __init__(self, param_dict, headers, logdir, scalers, ordered_events, encoding_width, not_predicted_events, **kwargs):
        """Class constructor 
        Encoder_decoder class for joint modeling.
        Arguments:
        param_dict: parameters dictionary
        headers: list of physiological variable names
        logdir: run directory
        scalers: tuple of scaler objects (StandardScaler object, MinMaxScaler object) obtained from scikit learn library.
        ordered_events: oredered array of uniq actions in the dataset, used for OneHotEncoding Layer
        encoding_width: number of classes (actions) in the dataset
        """
        super(CustomEncoderDecoder, self).__init__(**kwargs) # initialize parent class

        # def self variables
        self.full_path_model = ""
        self.nb_parameters = len(headers)*2
        self.headers = headers
        self.param_dict = param_dict
        self.run_logdir = logdir # mean that we have to create another instnace of model for each run
        std_scaler, minmax_scaler = scalers # no serialization, used only to instanciate Standardization and Normalization Layers
        self.ordered_events = ordered_events
        self.encoding_width = encoding_width
        self.not_predicted_events = not_predicted_events

        # correspondance dictionnary for weights initializers
        available_kernel_init = {
            "glorotUniform": keras.initializers.GlorotUniform(),
            "heUniform": keras.initializers.HeUniform(),
            "glorotNormal": keras.initializers.GlorotNormal(),
            "heNormal": keras.initializers.HeNormal(),
            "zeros": keras.initializers.Zeros(),
            "ones": keras.initializers.Ones(),
            "constant": keras.initializers.Constant(value=0.5),
            # add others if needed
        }

        # correspondance dictionary for activation functions
        available_activations = {
            "tanh": keras.activations.tanh,
            "sigmoid": keras.activations.sigmoid,
            "relu": keras.activations.relu,
            "elu": keras.activations.elu,
            "linear": keras.activations.linear,
            "selu": keras.activations.selu,
            "gelu":keras.activations.gelu
            # add others if needed
        }
        
        # Preprocessing layers
        self.standardization = StandardizationLayer(means=std_scaler.mean_, stds=std_scaler.scale_, nb_parameters=self.nb_parameters, output_size=self.param_dict.get('output_size'), name="TestStdLayer")
        self.normalization = MinMaxNormalizationLayer(minmax_scaler.data_min_, minmax_scaler.data_max_, param_dict.get('min_norm_threshold'), param_dict.get('max_norm_threshold'), nb_parameters=self.nb_parameters, output_size=self.param_dict.get('output_size'), name="TestNormLayer")
        self.onehot = OneHotEncodingLayer(ordered_events, encoding_width)
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Encoder module 
        self.encoder = []
        for i in range(0, self.param_dict.get('n_encoder_layers')):
            # default return_sequences and return state parameter for more than one layer models
            return_seq = True

            # boolean set last layer return seq parameter: 
            # i != (n_encoder_layers -1) -> return_seq = True
            # i == (n_encoder_layers - 1) -> return_seq = False
            return_seq = bool(i != (self.param_dict.get('n_encoder_layers') - 1) )
            
            if i == 0:
                self.encoder.append(
                    tf.keras.layers.LSTM(
                        self.param_dict.get('encoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh,
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'encoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("encoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("encoder_dropout")[i],
                        input_shape=[None, None], # output_size, 8+36
                        return_sequences=return_seq,
                        return_state=True # return state at each layer (avoid conditions in call) but will use only the last layer state as initial state for first decoder layer
                    )
                )
            else:
                self.encoder.append(
                    tf.keras.layers.LSTM(
                        self.param_dict.get('encoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh,
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'encoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("encoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("encoder_dropout")[i],
                        return_sequences=return_seq,
                        return_state=True # return state at each layer (avoid conditions in call) but will use only the last layer state as initial state for first decoder layer
                    )
                )
            
        # Repeat last time step prediction from last encoder layer, in order to use repaetedVector as input for decoder's first layer
        self.repeat = tf.keras.layers.RepeatVector(n=self.param_dict.get('input_size'))
        
        # Decoder module
        self.decoder = []
        for i in range(0, self.param_dict.get('n_decoder_layers')):
            # default return_sequences and return state parameter for more than one layer models
            if i == 0:
                self.decoder.append(
                    tf.keras.layers.LSTM( # pour créer des cocuhe perso il faut utiliser la classe RNN de base en leur pazssaznt une couche (cell class) perso
                        self.param_dict.get('decoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh, # tanh: fonction saturante pour éviter l'instabilité des gradients.(évite que les valeurs ou gradients explosent, variation faible si grosses valeurs, donc les non saturantes comme relu aident pas)
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'decoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("decoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("decoder_dropout")[i],
                        input_shape=[None, None], # output_size, len(output vect from last encoder layer)
                        return_sequences=True,
                        return_state=False
                    )
                )
            else:
                self.decoder.append(
                    tf.keras.layers.LSTM( # pour créer des cocuhe perso il faut utiliser la classe RNN de base en leur pazssaznt une couche (cell class) perso
                        self.param_dict.get('decoder_neurons')[i], # output per timestep for the layer
                        activation=keras.activations.tanh, # tanh: fonction saturante pour éviter l'instabilité des gradients.(évite que les valeurs ou gradients explosent, variation faible si grosses valeurs, donc les non saturantes comme relu aident pas)
                        recurrent_activation=keras.activations.sigmoid, 
                        use_bias=True, 
                        kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                        recurrent_initializer='orthogonal', 
                        bias_initializer='zeros',
                        name=f'decoder_layer_{i}',
                        recurrent_dropout=self.param_dict.get("decoder_recurrent_dropout")[i],
                        dropout=self.param_dict.get("decoder_dropout")[i],
                        return_sequences=True,
                        return_state=False
                    )
                )
        
        # Classical decoder output (decoder_output1)
        self.decoder_classic_output_dense = keras.layers.TimeDistributed(
            keras.layers.Dense(
                self.param_dict.get('decoder_neurons')[-1],
                kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                use_bias = True,
                bias_initializer='zeros',
                name='decoder_classic_output_dense',
                activation = keras.activations.linear
            )
        )

        self.attention_layer = keras.layers.Attention(name="attention_layer")

        self.rnn_output_dense = keras.layers.Dense(
                self.param_dict.get('rnn_output_dense_neurons'),
                kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                use_bias = True,
                bias_initializer='zeros',
                name='rnn_output_dense',
                activation = available_activations[self.param_dict.get('rnn_output_dense_activation')]
        )

        self.concatenate_classic_decoder_output_attention_output = tf.keras.layers.Concatenate(name="concatenate_decoder_output_attention_output")

        self.eventDense = keras.layers.TimeDistributed(
            keras.layers.Dense(
                self.param_dict.get('output_size')*self.encoding_width,
                kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                use_bias = True,
                bias_initializer='zeros',
                name='events_output_layer',
                activation = keras.activations.linear
            )
        )

        self.regularDense = keras.layers.TimeDistributed( # we could use an RNN layer but a dense layer allows to chosse the activation and has the same capacities (our values between 0 and 1 so we cant use tanh)
            keras.layers.Dense(
                self.param_dict.get('output_size')*self.nb_parameters, # output -> taille = nb_pas de temps qu'on veut prédire (chaque pas de temps =  nb_parametre)
                kernel_initializer=available_kernel_init[self.param_dict.get('kernel_initializer')],
                use_bias = True,
                bias_initializer='zeros',
                name='regular_output_layer',
                activation = available_activations[self.param_dict.get('last_activation')]
            )
        )
        
        # softmax layer for onehot decoded action predictions
        self.softmax_layer = SoftMaxLayer()

    @tf.function
    def call(self, inputs, training=None, **kwargs): # we have to tell the forward pass with () at end of line, beucause non sequential model
        """Main method. This function is executed when calling an instance of this model class.
        Defines the forward pass used when using model.fit
        inputs: list of input types: [serie_values_input, action_onehot_input]
        for training:
            - serie_values_input: (batch_size, input_size, n_features)
            - action_one_hot_input: (batch_size, input_size, encoding_width)
        for inference:
            - serie_values_input: (batch_size, input_size, n_features)
            - action_one_hot_input: (batch_size, input_size)
        training: boolean indicating behavior of function during training. Used for preprocessing layers and dropout layers training argument
        **kwargs: any Tensorflow Model Class argument"""

        # get regular and events inputs
        regular_inputs, events = inputs

        #### INFERENCE ONLY (for x_val and x_test) ####
        regular_inputs = self.standardization(regular_inputs, training=training)
        regular_inputs = self.normalization(regular_inputs, training=training)
        events = self.onehot(events, training=training)
        ##########################################

        concat_inputs = self.concatenate([regular_inputs, events])

        # ENCODER DECODER

        # Encoder
        encoder_output = [concat_inputs] # initialize encoder_output with concatenated preprocessed inputs
        states_c = [] # initialize list of long term memory vector outputs

        # for each layer, compute output from previous output
        for i in range(0, len(self.encoder)):
            output, _, state_c = self.encoder[i](encoder_output[i])
            encoder_output.append(output)
            states_c.append(state_c)
            
        encoder_output = encoder_output[-1] # keep only output from last time step of recurrent layer
        # last encoder output = (batch_size, last_layer_number_of_neurons) because no return seq = state_h-> one weights matrix for each instance get encoded version of inputs (dense vector representation)
        states_c = states_c[-1] # get only last layer state shape = (batch_size, last_layer_number_of_neurons)

        # Decoder

        # initialize decoder_output list with repeated encoded output
        decoder_output = [self.repeat(encoder_output)] # repeat output (last time step) = state h repeated for each timestep to form input x for first decoder layer

        # first iteration
        # use h and c from last encoder layer to initalize h_state and c_state of first decoder layer for the first iteration.
        # use repeated state h as input
        decoder_output.append(self.decoder[0](decoder_output[0], initial_state=[encoder_output, states_c])) # init_state != None only for first iteration
        
        # compute other layer states and outputs 
        for i in range(1, len(self.decoder)):
            decoder_output.append(self.decoder[i](decoder_output[i], initial_state=None))
        
        # computing classic decoder output
        classic_decoder_output = self.decoder_classic_output_dense(decoder_output[-1]) 
        # shape after = 32 10 440

        attention_output = self.attention_layer( [ tf.reshape( encoder_output, shape=[tf.shape(encoder_output)[0], self.param_dict.get('output_size'), tf.shape(concat_inputs)[2]] ), concat_inputs ] ) # query, value
        # shape after = 32 10 44

        concat_classic_attention_output = self.concatenate_classic_decoder_output_attention_output([classic_decoder_output, attention_output])
        # shape after = [32 10 484] (440+44)

        rnn_output_dense = self.rnn_output_dense(concat_classic_attention_output)
        # shape after = 32 10 rnn_output_dense_neurons

        # Timedistributed layers
        # regular_outputs: (batch_size, input_size, output_size*n_features)
        # event_outputs: (batch_size, input_size, encoding_width)
        self.regular_outputs = self.regularDense(rnn_output_dense)
        self.event_outputs = self.eventDense(rnn_output_dense)# (batch_size, input_size, output_size*nb_features) after

        # Softmax performed on each one hot vector
        # reshape to (batch_size, input_size, outputsize, encoding_width)
        self.event_outputs = tf.reshape(self.event_outputs, shape=[tf.shape(self.event_outputs)[0], self.param_dict.get('input_size'), self.param_dict.get('output_size'), self.encoding_width])
        
        # apply softmax to each encoding_width values vectors
        self.event_outputs = self.softmax_layer(self.event_outputs)
        
        return {"regression_output": self.regular_outputs, "event_output": self.event_outputs}
    
    def get_config(self):
        """Defines layer configuration for tensorflow serialization.
        Warning: every parameter in init should be in config
        """
        # return config dictionary
        config = super(CustomEncoderDecoder, self).get_config()
        config['param_dict'] = self.param_dict
        config['headers'] = self.headers
        config['nb_parameters'] = self.nb_parameters
        config['normalization'] = self.normalization
        config['standardization'] = self.standardization
        config['onehot'] = self.onehot
        config['encoding_width'] = self.encoding_width
        config['ordered_events'] = self.ordered_events
        config['softmax_layer'] = self.softmax_layer
        config['repeat'] = self.repeat
        config['concatenate'] = self.concatenate
        config['encoder'] = self.encoder
        config['decoder'] = self.decoder
        config['decoder_classic_output_dense'] = self.decoder_classic_output_dense,
        config['attention_layer'] = self.attention_layer,
        config['rnn_output_dense'] = self.rnn_output_dense,
        config['concatenate_classic_decoder_output_attention_output'] = self.concatenate_classic_decoder_output_attention_output,
        config['regularDense'] = self.regularDense
        config['eventDense'] = self.eventDense
        config['last_time_step_mse'] = last_time_step_mse
        config['last_time_step_errors_mean_errors'] = last_time_step_errors_mean_errors
        config['relative_percentage_error'] = relative_percentage_error
        config['absolute_percentage_error'] = absolute_percentage_error
        config['last_time_step_categorical_accuracy'] = last_time_step_categorical_accuracy
        config['last_time_step_CategoricalCrossentropy'] = last_time_step_CategoricalCrossentropy
        config['last_time_step_F1_score'] = last_time_step_F1_score
        config['last_time_step_F1_score_macro_average_geometric_mean'] = last_time_step_F1_score_macro_average_geometric_mean
        config['last_time_step_Recall_by_label'] = last_time_step_Recall_by_label
        config['last_time_step_Precision_by_label'] = last_time_step_Precision_by_label
        config['NN_accuracy'] = NN_accuracy
        config['NE_percentage_error'] = NE_percentage_error
        config['action_specific_accuracy'] = action_specific_accuracy
        config['accuracy_with_lag'] = accuracy_with_lag
        config['plot_to_image'] = self.plot_to_image
        config['create_multiclass_confusion_matrix'] = create_multiclass_confusion_matrix
        config['write_boxplot_descriptions'] = self.write_boxplots_descriptions
        config['make_custom_evaluation_boxplots'] = self.make_custom_evaluation_boxplots
        config['save_model'] = self.save_model
        config['plot_exampl_graphs'] = self.plot_exmpl_graphs
        config['custom_evaluate'] = self.custom_evaluate    