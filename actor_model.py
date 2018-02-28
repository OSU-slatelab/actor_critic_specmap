#!/u/drspeech/opt/anaconda3/bin/python3

"""
Actor model for actor-critic noisy speech recognition.

Author: Peter Plantinga
Date:   Fall 2017
"""

import tensorflow as tf

class Actor:
    """
    This actor model takes noisy speech as input, and outputs clean speech.

    As part of the actor-critic model, this is trained jointly with
    the critic, by freezing the weights of the critic after it has been
    trained on clean speech.
    """

    def __init__(self,
            input_shape,
            output_shape,
            layer_size = 2048,
            layers     = 2,
            dropout    = 0.5,
            batch_norm = True):
        """
        Create actor model.

        Params:
         * input_shape : tuple
            The shape of the input, 3D tensor [batch size x frame count x input frame size]
         * output_shape : tuple
            The shape of the output, 3D tensor [batch size x frame count x output frame size]
         * layer_size : int
            Number of neurons per layer
         * layers : int
            Number of layers
         * block_size : int
            Number of layers per block (adds residual connection)
         * dropout : float
            Percent of nodes to drop when training
        """

        # Compute shape of input and output
        self.inputs = tf.placeholder(dtype = tf.float32, shape = input_shape, name = "inputs")
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Every frame past the size of the output is a context frame
        self.context_frames = input_shape[1] - output_shape[1]

        # Layer params
        self.dropout = dropout
        self.layer_size = layer_size
        self.layers = layers
        self.block_size = block_size
        self.batch_norm = batch_norm

        # Whether or not we're training
        self.training = tf.placeholder(dtype = tf.bool, name = "training")

        self._create_model()


    def _create_model(self):
        """Put together all the parts of the actor model."""

        # Initialize graph
        a = self._dnn_frame_output(self.inputs[:, 0 : self.context_frames + 1], reuse = False)

        # Generate all the output frames
        output = [self._dnn_frame_output(self.inputs[:, i : i + self.context_frames + 1])
                for i in range(self.output_shape[1])]

        # Stack the output frames into a single tensor
        self.outputs = tf.stack(output, axis = 1)

    def _dnn_frame_output(self, inputs, reuse = True):
        """Generate the graph for a single frame of output"""

        layer = tf.reshape(inputs,
                shape = (-1, (self.context_frames + 1) * self.input_shape[2]))

        for i in range(self.layers):
            with tf.variable_scope('actor_layer' + str(i), reuse = reuse):
                layer = self._dense(layer)

        with tf.variable_scope('output_layer', reuse = reuse):
            output = tf.layers.dense(layer, self.output_shape[2])

        return output

    def _dense(self, inputs):
        """Fully connected layer, with activation, dropout, and batch norm."""

        layer = tf.layers.dense(
            inputs     = inputs,
            units      = self.layer_size,
            activation = tf.nn.relu,
        )

        if self.batch_norm:
            layer = tf.layers.batch_normalization(layer)

        layer = tf.layers.dropout(
            inputs   = layer,
            rate     = self.dropout,
            training = self.training,
        )

        return layer

