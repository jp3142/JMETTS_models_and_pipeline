#!/usr/bin/env python3

import tensorflow as tf

### MODULE FILE FOR LAYER CLASS TEMPLATES ###

class MinMaxNormalizationLayer(tf.keras.layers.Layer):
    """Normalization Layer. Use to transform data into range (a, b)
    This layer normalize data during inference processes (validation, evaluation) but has no effect during the training. Training data
    have to be preprocessed before using this layer."""
    
    def __init__(self, mins, maxs, a, b, nb_parameters, output_size, name="MinMaxNormalizationLayer", **kwargs): # scale = tuple (mins, maxsà
        """
        Class constructor. 
        Arguments:
        mins: list of n_features minimums (one for each variable at each timestep) -> (n_features*input_size,). This list
        can be obtained with data_min_ object of MinMaxScaler class in sci-kit-learn library.
        maxs: list of n_features maximums (one for each variable at each timestep). This list
        can be obtained with data_max_ object of MinMaxScaler class in sci-kit-learn library.
        a: lowest value in normalization range (a, b)
        b: highest value in normalization range (a, b)
        nb_parameters: number of features per timestep in input of call method - int
        output_size: output window size - int
        name: class name for tensorflow serialization backends
        **kwargs: any Tensorflow Layer Class argument
        """

        # define layer parameters
        self.mins_ = mins
        self.maxs_ = maxs
        self.nb_parameters_ = nb_parameters
        self.output_size_ = output_size
        self.a_ = a
        self.b_ = b
        super().__init__(name=name, **kwargs) # for inheritance from superclass Layer.

    @tf.function
    def call(self, inputs, training=None):
        """ Main method automatically called when calling an instance of this class.
        Transform data in range (a, b) for each feature in the dataset.
        Arguments:
        inputs: 3D tensor of shape 
        training: boolean indicating behavior of function during training."""

        if training: # if True, no preprocessing because training data are preprocessed before the network
            return inputs
        #else  
        mins = tf.cast(self.mins_, dtype=tf.float32)
        a = tf.cast(self.a_, dtype=tf.float32)

        return (tf.cast(self.b_, dtype=tf.float32) - a) * ( (tf.cast(inputs, dtype=tf.float32) - mins) / (tf.cast(self.maxs_, dtype=tf.float32) - mins)) + a
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)]) # give inputs number of dims for tensorflow graph
    def inverse_transform(self, outputs):
        """ Invert normalization from the given range (a, b) to the original range of values.
        This method is useful when using a trained model for predictions to get back original data. 
        Return only the inverted transformation of last time step prediction for each instance from an output RNN sequence.
        Return a 2D tensor of shape (batch_size, n_neurons_output_layer==n_features*output_size).
        Arguments:
        outputs: neural network outputs to denormalize - 3D tensor of shape (batch_size, n_timesteps, n_neurons_output_layer)"""

        # cast outputs and layer parameters
        mins = tf.cast(self.mins_, dtype=tf.float32)
        a = tf.cast(self.a_, dtype=tf.float32)
        unorm_outputs = tf.cast(outputs[:, -1], dtype=tf.float32) # keep only last time step prediction (the actual final prediction)

        # reshape to inverse transform
        unorm_outputs = tf.reshape(unorm_outputs, shape=[tf.shape(outputs)[0], self.output_size_, self.nb_parameters_])
        
        # inverse transform
        unorm_outputs = ( (unorm_outputs - a) / (tf.cast(self.b_, dtype=tf.float32) - a) ) * ( tf.cast(self.maxs_, dtype=tf.float32) - mins ) + mins
        
        # reconstitute initial dimensions of each window
        unorm_outputs = tf.reshape(unorm_outputs, shape=[tf.shape(unorm_outputs)[0], self.nb_parameters_*self.output_size_])

        return unorm_outputs

    def get_config(self):
        """Get layer configuration for tensorflow serialization.
        Warning: every parameter in init should be in config
        """
        config = super(MinMaxNormalizationLayer, self).get_config()
        config["a_"] = self.a_
        config['b_'] = self.b_
        config["mins_"] = self.mins_
        config["maxs_"] = self.maxs_
        config["nb_parameters_"] = self.nb_parameters_
        config["output_size_"] = self.output_size_
        config["inverse_transform"] = self.inverse_transform
        return config

    @classmethod
    def from_config(cls, config):
        """For tensorflow serialization. Customizable function.
        Here is default function"""
        return cls(**config)

########################################################################

class StandardizationLayer(tf.keras.layers.Layer):
    """Standardization Layer. Use to transform data according to their std (standard deviation) and mean.
    This layer normalize data during inference processes (validation, evaluation) but has no effect during the training. Training data
    have to be preprocessed before using this layer."""
    def __init__(self, means, stds, nb_parameters, output_size, name="StandardizationLayer", **kwargs): # scales tuple = means, stds
        """Class constructor. 
        Arguments:
        means: mean_ object (list of mean_ for each feature in the dataset) obtained from sci-kit-learn StandardScaler class
        stds: scale_ object (list of std deviation for each feature in the dataset) obtained from sci-kit-learn StandardScaler class 
        nb_parameters: number of features per timestep in input of call method - int
        output_size: output window size - int
        name: Class name for tensorflow serialization backends
        **kwargs: any Tensorflow Layer Class argument"""
        
        # define layers parameters
        self.means_ = means
        self.stds_ = stds
        self.nb_parameters_ = nb_parameters
        self.output_size_ = output_size
        super().__init__(name=name, **kwargs) 

    @tf.function
    def call(self, inputs, training=None):
        """Main method automatically called when calling an instance of this class.
        Transform data in according to: x'= (x-mean)/std for each feature in dataset.
        Arguments:
        inputs: 3D tensor of shape (batch_size, output_size, n_features) 
        training: boolean indicating behavior of function during training."""
        if training: # doesn't process training data. Process only inference data like validation and evaluation dataset.
            return inputs
        # else
        means = tf.cast(self.means_, dtype=tf.float32)
        stds = tf.cast(self.stds_, dtype=tf.float32)

        # return normalized data according to means and stds
        return (tf.cast(inputs, dtype=tf.float32)-means)/stds
        
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def inverse_transform(self, outputs):
        """Invert standardization. x = x' * std + mean
        This method is useful when using a trained model for predictions to get back original data. 
        Return only the inverted transformation of last time step prediction for each instance from an output RNN sequence.
        Return a 2D tensor of shape (batch_size, n_neurons_output_layer==n_features*output_size).
        Arguments:
        outputs: neural network outputs to denormalize - 3D tensor of shape (batch_size, n_timesteps, n_neurons_output_layer==n_features*output_size)""""""
        """
        # cast parameters and output
        means = tf.cast(self.means_, dtype=tf.float32)
        stds = tf.cast(self.stds_, dtype=tf.float32)
        unorm_outputs = tf.cast(outputs, dtype=tf.float32) # keep only last time step prediction

        # reshape to inverse transform
        unorm_outputs = tf.reshape(unorm_outputs, shape=[tf.shape(unorm_outputs)[0], self.output_size_, self.nb_parameters_])

        # inverse transform
        unorm_outputs = (unorm_outputs * stds) + means

        # reshape to original shape (with only last time step prediction)
        unorm_outputs = tf.reshape(unorm_outputs, shape=[tf.shape(outputs)[0], self.nb_parameters_*self.output_size_])

        return unorm_outputs

    def get_config(self):
        """Get layer configuration for tensorflow serialization.
        Warning: every parameter in init should be in config
        """
        config = super(StandardizationLayer, self).get_config()
        config["means_"] = self.means_
        config["stds_"] = self.stds_
        config['nb_parameters_'] = self.nb_parameters_
        config["output_size_"] = self.output_size_
        config["inverse_transform"] = self.inverse_transform
        return config
    
    @classmethod
    def from_config(cls, config):
        """For tensorflow serialization. Customizable function.
        Here is default function"""
        return cls(**config)
        
########################################################################

class OneHotEncodingLayer(tf.keras.layers.Layer):
    """One-hot Encoding Layer. Use to transform events in vector representation like [0, 0, 0, ...., 0, 1, 0] with only 0 and one 1.
    The position of the 1 in the vector indicates the associated action.
    This layer preprocess data during inference processes (validation, evaluation) but has no effect during the training. Training data
    have to be preprocessed before using this layer."""

    def __init__(self, ordered_events, encoding_width, name="OneHotEncodingLayer", **kwargs): # scales tuple = means, stds
        """Class constructor. Define 2 correspondances dictionaries for one-hot encoding.
        Arguments:
        ordered_events: ordered array of unique events corresponding to the array used to preprocess training actions. Allow to create the
        same hash tables as in pre processing of training data.
        encoding_width: number of classes in the dataset = length of one-hot vectors
        **kwargs: any Tensorflow Layer Class argument"""

        self.encoding_width_ = encoding_width # len of one hot vectors
        self.ordered_events_ = ordered_events # ordered vector of events
        
        # create lookup tables
        indices = tf.range(encoding_width, dtype=tf.int64) # get indices
        vocab = tf.constant(ordered_events, dtype=tf.string) # get vocabulary

        table_init_events_to_index = tf.lookup.KeyValueTensorInitializer(vocab, indices, key_dtype=tf.string, value_dtype=tf.int64) # initalize lookup table events -> index
        table_init_index_to_events = tf.lookup.KeyValueTensorInitializer(indices, vocab, key_dtype=tf.int64, value_dtype=tf.string) # initialize lookup table index -> events
        
        # create lookup table
        self.events_to_index_ = tf.lookup.StaticVocabularyTable(table_init_events_to_index, num_oov_buckets=1)
        self.index_to_events_ = tf.lookup.StaticHashTable(table_init_index_to_events, default_value="UNKNOWN")

        super().__init__(name=name, **kwargs) 

    @tf.function
    def call(self, inputs, training=None):
        """One-hot encoding main method. This function is excecuted when calling an instance of this layer class.
        Arguments:
        inputs: string tensor representing actions of shape (batch_size, input_size)
        training: boolean indicating behavior of function during training
        """
        # inputs = x events of the window (one event per timestep)
        if training or len(tf.shape(inputs)) == 3: # allow to not convert to one hot if training (training dataset already in one hot format, or if shape[3] exists corresponding to the presence of one hot vector)
            return inputs

        # get indices of events for the current window
        indices = self.events_to_index_.lookup(inputs)

        # get one hot vector for window of events
        encoded_inputs = tf.one_hot(indices, depth=self.encoding_width_)

        return encoded_inputs

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, 10, 10, 36], dtype=tf.float32, name="decode")]) 
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32, name="decode")]) 
    def decode(self, encoded_outputs):
        """Decoding from one-hot representation to strings (action names)
        To use during inference with a trained model.
        Return only decoded last time step prediction for each instance of the dataset.
        Arguments:
        encoded_outputs: network output vectors of probability distribution obtained from softmax. Shape = (batch_size, input_size, output_size, encoding_width)"""
        
        encoded_outputs = encoded_outputs[:, -1] # keep only last timestep prediction (final prediction)
        decoded_outputs = tf.argmax(encoded_outputs, axis=-1) # (argmax finds index of each value for each timestep event) decode one hot outputs to corresponding integers, keep shape except for last dimension
        decoded_outputs = self.index_to_events_.lookup(decoded_outputs) # decode each integer to corresponding event name
        
        return decoded_outputs

    def get_config(self):
        """Get layer configuration for tensorflow serialization.
        Warning: every parameter in init should be in config
        """
        config = super(OneHotEncodingLayer, self).get_config()
        config["ordered_events_"] = self.ordered_events_
        config["encoding_width_"] = self.encoding_width_
        config["decode"] = self.decode
        return config
    
    @classmethod
    def from_config(cls, config):
        """For tensorflow serialization. Customizable function.
        Here is default function"""
        return cls(**config)

########################################################################

class SoftMaxLayer(tf.keras.layers.Layer):
    """Softmax Layer. Perform softmax on last dimension of a multidimensional matrix"""
    
    def __init__(self, name="SoftMaxLayer", **kwargs): # scales tuple = means, stds
        """Class constructor"""    
        # define layers parameters
        super().__init__(name=name, **kwargs) 

    @tf.function
    def call(self, inputs):
        """Main method automatically called when calling an instance of this class."""
        return tf.keras.activations.softmax(inputs)
    
    def get_config(self):
        """Get layer configuration for tensorflow serialization.
        Warning: every parameter in init should be in config
        """
        config = super(SoftMaxLayer, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        """For tensorflow serialization. Customizable function.
        Here is default function"""
        return cls(**config)