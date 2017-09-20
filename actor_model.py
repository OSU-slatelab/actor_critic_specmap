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

	def __init__(self, input_shape, layer_size = 1024, output_frames = 11):
		"""
		Create actor model.

		Params:
		 * input_shape : tuple
		    The shape of the input, 3D tensor [batch size x frame count x frame size]
		 * layer_size : int
		    Number of neurons per layer
		 * output_frames : int
		    Number of output frames (should be the same as input frames for critic)
		"""
		
		# Compute shape of input and output
		self.inputs = tf.placeholder(dtype = tf.float32, shape = input_shape, name = "inputs")
		self.input_shape = input_shape

		# Batch size x frame count x frame size
		self.output_shape = (input_shape[0], output_frames, input_shape[2])
		self.output_frames = output_frames
		self.targets = tf.placeholder(dtype = tf.float32, shape = self.output_shape, name = "targets")

		# Every frame past the size of the output is a context frame
		self.context_frames = self.input_shape[1] - output_frames

		# Layer params
		self.dropout = tf.placeholder(dtype = tf.float32, name = "dropout")
		self.layer_size = layer_size

		self._create_model()
		

	def _create_model(self):
		"""Put together all the parts of the actor model."""

		# Initialize graph
		a = self._frame_output(self.inputs[:, 0 : self.context_frames + 1], reuse = False)

		# Generate all the output frames
		output = [self._frame_output(self.inputs[:, i : i + self.context_frames + 1])
				for i in range(self.output_frames)]

		# Stack the output frames into a single tensor
		self.outputs = tf.stack(output, axis = 1)


	def _frame_output(self, inputs, reuse = True):
		"""Generate the graph for a single frame of output"""

		inputs = tf.reshape(inputs,
				shape = (self.input_shape[0], (self.context_frames + 1) * self.input_shape[2]))

		with tf.variable_scope('actor_layer1', reuse = reuse):
			layer1 = self._dense(inputs)

		with tf.variable_scope('actor_layer2', reuse = reuse):
			layer2 = self._dense(layer1)

		with tf.variable_scope('actor_layer3', reuse = reuse):
			layer3 = self._dense(layer2)

		with tf.variable_scope('output_layer', reuse = reuse):
			output = tf.layers.dense(layer3, self.output_shape[2])

		return output


	def _dense(self, inputs):
		"""Fully connected layer, with activation, dropout, and batch norm."""

		layer = tf.layers.dense(
				inputs             = inputs,
				units              = self.layer_size,
				activation         = tf.nn.relu,
				kernel_initializer = tf.random_normal_initializer(0, 0.02))

		layer = tf.layers.dropout(
				inputs = layer,
				rate   = self.dropout)

		layer = tf.layers.batch_normalization(
				inputs = layer)
		
		return layer
