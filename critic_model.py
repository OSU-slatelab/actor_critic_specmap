#!/u/drspeech/opt/anaconda3/bin/python3

"""
Critic model for actor-critic noisy speech recognition.

Author: Deblin Bagchi and Peter Plantinga
Date:   Fall 2017
"""

import tensorflow as tf

def lrelu(x, a):
    """ Leaky ReLU activation function """
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batch_norm(x, shape, training, decay = 0.999, epsilon = 1e-3):
    """ Batch Norm for controlling batch statistics """
    #Assume 2d [batch, values] tensor
    beta = tf.get_variable(name='beta', shape=shape[-1], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
    gamma = tf.get_variable(name='gamma', shape=shape[-1], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
    pop_mean = tf.get_variable('pop_mean',
                               shape[-1],
                               initializer=tf.constant_initializer(0.0),
                               trainable=False)
    pop_var = tf.get_variable('pop_var',
                              shape[-1],
                              initializer=tf.constant_initializer(1.0),
                              trainable=False)
    batch_mean, batch_var = tf.nn.moments(x, [0])

    train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean_op, train_var_op]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)

    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, gamma, epsilon)

    return tf.cond(training, batch_statistics, population_statistics)

def feedforward_layer(inputs, shape):
    """ Simple feedforward layer """

    weight = tf.get_variable("weight",
        shape,
        dtype=tf.float32,
        initializer = tf.random_normal_initializer(0,0.02))

    bias = tf.get_variable("bias",
        shape[-1],
        initializer=tf.zeros_initializer())

    return tf.matmul(inputs, weight) + bias

class Critic:
    """
    This critic model takes clean speech as input, and outputs senone labels.

    As part of the actor-critic model, this is trained jointly with
    the actor, by freezing the weights after it has been
    trained on clean speech.
    """

    def __init__(self,
            inputs,
            input_size,
            layer_size  = 1024,
            layers      = 6,
            output_size = 1999,
            dropout     = 0.5):
        """
        Create critic model.

        Params:
         * inputs : Tensor
            The input placeholder or tensor from actor
         * layer_size : int
            Number of neurons per layer
         * output_frames : int
            Number of output frames (should be the same as input frames for critic)
        """

        self.inputs = inputs
        self.input_size = input_size
        
        # Layer params
        self.dropout = dropout
        self.layer_size = layer_size
        self.layers = layers
        self.output_size = output_size
        
        # Whether or not we're training
        self.training = tf.placeholder(dtype = tf.bool, name = "training")

        self._create_model()
        

    def _create_model(self):
        """Put together all the parts of the critic model."""

        with tf.variable_scope("hidden1"):
          hidden = feedforward_layer(self.inputs, (self.input_size, self.layer_size))

        for i in range(2, self.layers + 1):
            with tf.variable_scope("hidden%d" % i):
                hidden = feedforward_layer(hidden, (self.layer_size, self.layer_size))

                # Batch norm, but not on first layer
                if i != 1:
                  hidden = batch_norm(hidden, (self.layer_size, self.layer_size), self.training)

                # Activate!
                hidden = lrelu(hidden, 0.3)

        with tf.variable_scope('output'):
            self.outputs = feedforward_layer(hidden, (self.layer_size, self.output_size))

