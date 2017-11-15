#!/u/drspeech/opt/anaconda3/bin/python3

"""
Actor model for actor-critic noisy speech recognition.

Author: Peter Plantinga
Date:   Fall 2017
"""

import tensorflow as tf
from critic_model import batch_norm

def prelu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

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
            layers = 2,
            block_size = 0,
            output_frames = 11,
            dropout = 0.1,
            batch_norm = False):
        """
        Create actor model.

        Params:
         * input_shape : tuple
            The shape of the input, 3D tensor [batch size x frame count x input frame size]
         * output_shape : tuple
            The shape of the output, 3D tensor [batch size x frame count x output frame size]
         * layer_size : int
            Number of neurons per layer
         * output_frames : int
            Number of output frames (should be the same as input frames for critic)
        """

        # Compute shape of input and output
        self.inputs = tf.placeholder(dtype = tf.float32, shape = input_shape, name = "inputs")
        self.input_shape = input_shape

        # Batch size x frame count x frame size
        self.output_shape = output_shape
        self.output_frames = output_frames
        self.targets = tf.placeholder(dtype = tf.float32, shape = self.output_shape, name = "targets")

        # Every frame past the size of the output is a context frame
        self.context_frames = self.input_shape[1] - output_frames

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
                for i in range(self.output_frames)]

        # Stack the output frames into a single tensor
        self.outputs = tf.stack(output, axis = 1)

    def _dnn_frame_output(self, inputs, reuse = True):
        """Generate the graph for a single frame of output"""

        inputs = tf.reshape(inputs,
                shape = (-1, (self.context_frames + 1) * self.input_shape[2]))

        inputs = tf.layers.dropout(inputs, self.dropout, self.training)

        with tf.variable_scope('actor_layer0', reuse = reuse):
            layer = self._dense(inputs)

        # Store residual for bypass
        residual = layer

        for i in range(1, self.layers):
            with tf.variable_scope('actor_layer' + str(i), reuse = reuse):
                layer = self._dense(layer)

                if self.block_size != 0 and i % self.block_size == 0:
                    layer += residual
                    residual = layer

        with tf.variable_scope('output_layer', reuse = reuse):
            output = tf.layers.dense(layer, self.output_shape[2])

        return output

    def _dense(self, inputs):
        """Fully connected layer, with activation, dropout, and batch norm."""

        layer = tf.layers.dense(
                inputs             = inputs,
                units              = self.layer_size)

        if self.batch_norm:
            layer = batch_norm(
                    x        = layer,
                    shape    = (self.layer_size, self.layer_size),
                    training = self.training)

        layer = prelu(layer)

        layer = tf.layers.dropout(
                inputs   = layer,
                rate     = self.dropout,
                training = self.training)

        return layer

