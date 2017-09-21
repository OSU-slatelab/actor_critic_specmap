#!/u/drspeech/opt/anaconda3/bin/python3

"""
Train/test a denoising autoencoder using the actor model.

Author: Peter Plantinga
Date:   Fall 2017
"""

import tensorflow as tf
import numpy as np
import data_io

from actor_model import Actor

def input_dropout(inputs, proportion):
	mask = np.random.binomial(1, proportion, inputs.shape)
	return np.multiply(mask, inputs) #* (1/(1-proportion))

def make_fd(inputs, actor, context_frames, training):
	return {
			actor.inputs: input_dropout(inputs, 0.5),
			actor.targets: inputs[:, context_frames // 2 : -context_frames // 2],
			actor.training: training,
	}

def make_train_op(predicted_out, actual_out):
	loss = tf.losses.mean_squared_error(actual_out, predicted_out)
	train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
	return loss, train_op

def make_inputs(data, batch_size, context_frames, output_frames):
	inputs = []
	keys = list(data.keys())
	for i in range(batch_size):
		index = np.random.randint(len(data[keys[i]]) - context_frames - output_frames)
		inputs.append(data[keys[i]][index : index + context_frames + output_frames])
	return np.array(inputs)

def read_data(offset, batch_size):
	
	data, lines = data_io.read_kaldi_ark_from_scp(uid=0, offset=0,
			batch_size=batch_size, buffer_size=batch_size*10,
			scp_fn = "/data/data2/scratch/bagchid/specGAN-tf_old/data-fbank/dev_dt_05_clean/feats.scp",
			ark_base_dir = "/data/data2/scratch/bagchid/specGAN-tf_old/")

	return data


def main():
	context_frames = 10
	output_frames = 11
	batch_size = 1024
	frame_size = 40

	data = read_data(0, batch_size)

	actor = Actor(
			input_shape   = (batch_size, output_frames + context_frames, frame_size),
			output_frames = output_frames
	)

	loss, train_op = make_train_op(actor.outputs, actor.targets)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(10):
			for batch in range(100):
				inputs = make_inputs(data, batch_size, context_frames, output_frames)
				sess.run(train_op, make_fd(inputs, actor, context_frames, training=True))
				data = read_data((batch+1) * batch_size, batch_size)
			
			inputs = make_inputs(data, batch_size, context_frames, output_frames)
			print("Epoch", epoch, "loss:", sess.run(loss, make_fd(inputs, actor, context_frames, training=False)))

if __name__ == "__main__":
	main()
