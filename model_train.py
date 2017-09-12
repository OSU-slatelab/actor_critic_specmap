from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
from os import path
import json
import glob
import random
import collections
import math
import time
from data_io import read_kaldi_ark_from_scp
from six.moves import xrange 

Model = collections.namedtuple("Model", "outputs, discrim_loss, gen_loss_GAN, gen_loss_objective, gen_loss, discrim_train, gd_train,misc")

data_base_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("--noisy_train_file", default="data-spectrogram/train_si84_noisy/feats.scp", help="The input feature file for training")
parser.add_argument("--noisy_dev_file", default="data-spectrogram/dev_dt_05_noisy/feats.scp", help="The input feature file for cross-validation")
parser.add_argument("--clean_train_file", default="data-spectrogram/train_si84_clean/feats.scp", help="The feature file for clean training labels")
parser.add_argument("--clean_dev_file", default="data-spectrogram/dev_dt_05_clean/feats.scp", help="The feature file for clean cross-validation labels")
parser.add_argument("--buffer_size", default=10, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--exp_name", default=None, help="directory with checkpoint to resume training from or use for testing")
#Training
parser.add_argument("--lr", type=float, default = 0.0002, help = "initial learning rate")
#Model
parser.add_argument("--normalize_input", type=str, default="no", choices=["no", "sigmoid", "tanh"], help = "if no, do not normalize inputs; if sigmoid, normalize between [0,1], if tanh, normalize between [-1,1]")
parser.add_argument("--normalize_target", type=str, default="no", choices=["no", "sigmoid", "tanh"], help = "if no, do not normalize inputs; if sigmoid, normalize between [0,1], if tanh, normalize between [-1,1]")
parser.add_argument("--discrim_cond", type=str, default="full", choices=["full", "central"], help = "determines the form of the conditioning input to the discriminator; \"full\" uses full context window, and \"central\" uses only the central frame")
parser.add_argument("--alayers", type=int, default=2)
parser.add_argument("--aunits", type=int, default=2048)
parser.add_arguments("--clayers", type=int, default=6)
parser.add_argument("--cunits", type=int, default=1024)

parser.add_argument("--input_featdim", type=int, default=257)
parser.add_argument("--output_featdim", type=int, default=257)
parser.add_argument("--context", type=int, default=3)
parser.add_argument("--epsilon", type=float, default=1e-3, help="parameter for batch normalization")
parser.add_argument("--decay", type=float, default=0.999, help="parameter for batch normalization")
parser.add_argument("--max_global_norm", type=float, default=5.0, help="global max norm for clipping")
parser.add_argument("--keep_prob", type=float, default=0.5, help="keep percentage of neurons")
#Generator Noise options 
#parser.add_argument("--patience", type=int, default=5312*10, help="patience interval to keep track of improvements")
#parser.add_argument("--patience_increase", type=int, default=2, help="increase patience interval on improvement")
#parser.add_argument("--improvement_threshold", type=float, default=0.995, help="keep track of validation error improvement")
a = parser.parse_args()
in_min = 0
in_max = 0
tgt_min = 0
tgt_max = 0

def read_mats(uid, offset, file_name):
    #Read a buffer containing buffer_size*batch_size+offset 
    #Returns a line number of the scp file
    scp_fn = path.join(data_base_dir, file_name)
    ark_dict,uid = read_kaldi_ark_from_scp(uid, offset, a.batch_size, a.buffer_size, scp_fn, data_base_dir)
    return ark_dict,uid

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batch_norm(x, shape, training):
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

    train_mean_op = tf.assign(pop_mean, pop_mean * a.decay + batch_mean * (1 - a.decay))
    train_var_op = tf.assign(pop_var, pop_var * a.decay + batch_var * (1 - a.decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean_op, train_var_op]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, a.epsilon)

    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, gamma, a.epsilon)

    return tf.cond(training, batch_statistics, population_statistics)


def create_actor(inputs, is_training, noise=None):
    shape = [(a.input_featdim*(2*a.context+1)), a.aunits]
    last_layer = inputs
    for i in range(1,a.ayers+1):
        with tf.variable_scope("hidden%d" % i):
            weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,0.02))
            bias = tf.get_variable("bias", shape[-1], initializer=tf.zeros_initializer())
            linear = tf.matmul(last_layer, weight) + bias
            bn = batch_norm(linear, shape, is_training)
            hidden = tf.nn.relu(bn)
        shape = [a.aunits, a.aunits]
        last_layer = hidden

    # Linear
    shape = [a.aunits, a.output_featdim]
    with tf.variable_scope('output'):
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,0.02))
        bias = tf.get_variable("bias", shape[-1], initializer=tf.zeros_initializer())
        out = tf.matmul(last_layer, weight) + bias
        if a.normalize_target == "sigmoid":
            out = tf.sigmoid(out)
        elif a.normalize_target == "tanh":
            out = tf.tanh(out)
    return out

def create_critic(inputs, targets, is_training):

    last_layer = inputs
    shape = (a.input_featdim*a.context,a.cunits)

    for i in range(1, a.clayers+1):
        with tf.variable_scope("hidden%d"%i):
            weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,0.02))
            bias = tf.get_variable("bias", shape[-1], initializer=tf.zeros_initializer())
            linear = tf.matmul(last_layer, weight) + bias
            if i==1:
                bn = linear
            else:
                bn = batch_norm(linear, shape, is_training)
            hidden = lrelu(bn,0.3)
        shape = [a.cunits, a.cunits]
        last_layer = hidden
   
    shape = [a.cunits,1]
    with tf.variable_scope('output'):
        weight = tf.get_variable("weight",
                              shape,
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0,0.02))
        bias = tf.get_variable("bias", shape[-1], initializer=tf.zeros_initializer())
        linear = tf.matmul(last_layer, weight) + bias
        out = linear
    return out

def create_joint_model(inputs, targets, is_training):
    disc_cond_inputs = None
    if a.discrim_cond == "full":
        disc_cond_inputs = inputs
    elif a.discrim_cond == "central":
        disc_cond_inputs = tf.slice(inputs, [0, a.context*a.input_featdim], [-1, a.input_featdim])
     
    with tf.variable_scope('generator'):
        outputs = create_generator(inputs, is_training)

    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator'):
            predict_real = create_discriminator(disc_cond_inputs, targets, is_training)

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse = True):
            predict_fake = create_discriminator(disc_cond_inputs, outputs, is_training)

    with tf.name_scope('discriminator_loss'):
        discrim_loss = tf.reduce_mean(predict_fake) - tf.reduce_mean(predict_real)
    
    with tf.name_scope('generator_loss'):
        gen_loss = -tf.reduce_mean(predict_fake) 

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        discrim_grads = tf.gradients(discrim_loss, discrim_tvars)
        discrim_grads, _ = tf.clip_by_global_norm(discrim_grads, clip_norm=a.max_global_norm)
        discrim_grad_var_pairs = zip(discrim_grads, discrim_tvars)
        discrim_optim = tf.train.RMSPropOptimizer(a.lr)
        discrim_train = discrim_optim.apply_gradients(discrim_grad_var_pairs)
        clip_values = [-0.01, 0.01]
        clip_disc_weights = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for var in discrim_tvars]

    with tf.name_scope("generator_train"):
        gd_tvars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        gd_grads = tf.gradients(gen_loss, gd_tvars)
        gd_grads, _ = tf.clip_by_global_norm(gd_grads, clip_norm=a.max_global_norm)
        gd_grad_var_pairs = zip(gd_grads, gd_tvars)
        gd_optim = tf.train.RMSPropOptimizer(a.lr)
        gd_train = gd_optim.apply_gradients(gd_grad_var_pairs)
        
    return discrim_loss, gen_loss, discrim_train, gd_train, clip_disc_weights

def init_config():
    offset_frames_noisy = np.array([], dtype=np.float32).reshape(0,a.input_featdim)
    offset_frames_clean = np.array([], dtype=np.float32).reshape(0,a.output_featdim)
    frame_buffer_clean = np.array([], dtype=np.float32)
    frame_buffer_noisy = np.array([], dtype=np.float32)
    A = np.array([], dtype=np.int32)

    config = {'batch_index':0, 
                'uid':0, 
                'offset':0, 
                'offset_frames_noisy':offset_frames_noisy, 
                'offset_frames_clean':offset_frames_clean, 
                'frame_buffer_clean':frame_buffer_clean, 
                'frame_buffer_noisy':frame_buffer_noisy, 
                'perm':A}
    return config

def fill_feed_dict(noisy_pl, clean_pl, config, noisy_file, clean_file, shuffle):

    batch_index = config['batch_index']
    offset_frames_noisy = config['offset_frames_noisy']
    offset_frames_clean = config['offset_frames_clean']
    A = config['perm']

    def create_buffer(uid, offset):
        ark_dict_noisy,uid_new= read_mats(uid,offset,noisy_file)
        ark_dict_clean,uid_new = read_mats(uid,offset,clean_file)

        ids_noisy = sorted(ark_dict_noisy.keys())
        mats_noisy = [ark_dict_noisy[i] for i in ids_noisy]
        mats2_noisy = np.vstack(mats_noisy)
        nonlocal offset_frames_noisy
        mats2_noisy = np.concatenate((offset_frames_noisy,mats2_noisy),axis=0)

        ids_clean = sorted(ark_dict_clean.keys())
        mats_clean = [ark_dict_clean[i] for i in ids_clean]
        mats2_clean = np.vstack(mats_clean)
        nonlocal offset_frames_clean
        mats2_clean = np.concatenate((offset_frames_clean,mats2_clean),axis=0)
            
        if mats2_noisy.shape[0]>=(a.batch_size*a.buffer_size):
            offset_frames_noisy = mats2_noisy[a.batch_size*a.buffer_size:]
            mats2_noisy = mats2_noisy[:a.batch_size*a.buffer_size]
            offset_frames_clean = mats2_clean[a.batch_size*a.buffer_size:]
            mats2_clean = mats2_clean[:a.batch_size*a.buffer_size]
            offset = offset_frames_noisy.shape[0]
        return mats2_noisy, mats2_clean, uid_new, offset

    if batch_index==0:
        frame_buffer_noisy, frame_buffer_clean, uid_new, offset = create_buffer(config['uid'],
                                                                                config['offset'])
        if a.normalize_input == "sigmoid":
            frame_buffer_noisy = np.interp(frame_buffer_noisy, [in_min, in_max], [0.0, 1.0])
        elif a.normalize_input == "tanh":
            frame_buffer_noisy = np.interp(frame_buffer_noisy, [in_min, in_max], [-1.0, 1.0])

        if a.normalize_target == "sigmoid":
            frame_buffer_clean = np.interp(frame_buffer_clean, [tgt_min, tgt_max], [0.0, 1.0])
        elif a.normalize_target == "tanh":
            frame_buffer_clean = np.interp(frame_buffer_clean, [tgt_min, tgt_max], [-1.0, 1.0])

        frame_buffer_noisy = np.pad(frame_buffer_noisy,
                                    ((a.context,),(0,)),
                                    'constant',
                                    constant_values=0)
        if shuffle==True:
            A = np.random.permutation(frame_buffer_clean.shape[0])
        else:
            A = np.arange(frame_buffer_clean.shape[0])
        # we don't permute the noisy frames because we need to preserve context;
        # we take matching windows in the assignment to noisy_batch below;
        # this means we pass the permutation in config
        frame_buffer_clean = frame_buffer_clean[A]
 
    else:
        frame_buffer_noisy = config['frame_buffer_noisy']
        frame_buffer_clean = config['frame_buffer_clean']
        uid_new = config['uid']
        offset = config['offset']


    start = batch_index*a.batch_size
    end = min((batch_index+1)*a.batch_size,frame_buffer_clean.shape[0])
    config = {'batch_index':(batch_index+1)%a.buffer_size, 
                'uid':uid_new,
                'offset':offset, 
                'offset_frames_noisy':offset_frames_noisy,
                'offset_frames_clean':offset_frames_clean, 
                'frame_buffer_noisy':frame_buffer_noisy,
                'frame_buffer_clean':frame_buffer_clean, 
                'perm':A}
    noisy_batch = np.stack((frame_buffer_noisy[A[i]:A[i]+1+2*a.context,].flatten()
                            for i in range(start, end)), axis = 0)
    feed_dict = {noisy_pl:noisy_batch, clean_pl:frame_buffer_clean[start:end]}
    return (feed_dict, config)
 
        
def placeholder_inputs():
    shape = a.input_featdim*(2*a.context+1)
    noisy_placeholder = tf.placeholder(tf.float32, shape=(None,shape), name="noisy_placeholder")
    clean_placeholder = tf.placeholder(tf.float32, shape=(None,a.output_featdim), name="clean_placeholder")
    is_training = tf.placeholder(tf.bool, name="is_training")
    return noisy_placeholder, clean_placeholder, is_training


