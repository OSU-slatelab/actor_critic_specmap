from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
from os import path
import random
import collections
import math
import time
import sys
from data_io import read_kaldi_ark_from_scp, read_senones_from_text
from six.moves import xrange 


data_base_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("--frame_train_file", default="data-fbank/train_si84_clean/feats.scp", help="The input feature file for training")
parser.add_argument("--frame_dev_file", default="data-fbank/dev_dt_05_clean/feats.scp", help="The input feature file for cross-validation")
parser.add_argument("--senone_train_file", default="clean_labels_train.txt", help="The senone file for clean training labels")
parser.add_argument("--senone_dev_file", default="clean_labels_dev.txt", help="The senone file for clean cross-validation labels")

parser.add_argument("--buffer_size", default=10, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--exp_name", default="new_exp", help="directory with checkpoint to resume training from or use for testing")
#Training
parser.add_argument("--lr", type=float, default = 0.0002, help = "initial learning rate")
#Model
parser.add_argument("--alayers", type=int, default=2)
parser.add_argument("--aunits", type=int, default=2048)
parser.add_argument("--clayers", type=int, default=6)
parser.add_argument("--cunits", type=int, default=1024)

parser.add_argument("--input_featdim", type=int, default=40)
parser.add_argument("--senones", type=int, default=1999)
parser.add_argument("--context", type=int, default=5)
parser.add_argument("--epsilon", type=float, default=1e-3, help="parameter for batch normalization")
parser.add_argument("--decay", type=float, default=0.999, help="parameter for batch normalization")
parser.add_argument("--max_global_norm", type=float, default=5.0, help="global max norm for clipping")
parser.add_argument("--keep_prob", type=float, default=0.5, help="keep percentage of neurons")
a = parser.parse_args()

def read_mats(uid, offset, file_name):
    #Read a buffer containing buffer_size*batch_size+offset 
    #Returns a line number of the scp file
    scp_fn = path.join(data_base_dir, file_name)
    ark_dict,uid = read_kaldi_ark_from_scp(uid, offset, a.batch_size, a.buffer_size, scp_fn, data_base_dir)
    return ark_dict,uid

def read_senones(uid, offset, file_name):
    scp_fn = path.join(data_base_dir, file_name)
    senone_dict,uid = read_senones_from_text(uid, offset, a.batch_size, a.buffer_size, scp_fn, data_base_dir)
    return senone_dict,uid

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


def create_critic(inputs, is_training):

    last_layer = inputs
    shape = (a.input_featdim*(2*a.context+1),a.cunits)
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
   
    shape = [a.cunits,a.senones]
    with tf.variable_scope('output'):
        weight = tf.get_variable("weight",
                              shape,
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0,0.02))
        bias = tf.get_variable("bias", shape[-1], initializer=tf.zeros_initializer())
        linear = tf.matmul(last_layer, weight) + bias
        out = linear
    return out

def train_critic(inputs, targets, is_training, keep_prob):
    with tf.variable_scope('critic'):
        outputs = create_critic(inputs, is_training)

        with tf.name_scope('critic_loss_single'):
             critic_loss_single = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=targets))
        with tf.name_scope('critic_train'):
            critic_tvars = [var for var in tf.trainable_variables() if 'critic' in var.name]
            critic_grads = tf.gradients(critic_loss_single, critic_tvars)
            critic_grads, _ = tf.clip_by_global_norm(critic_grads, clip_norm=a.max_global_norm)
            critic_grad_var_pairs = zip(critic_grads, critic_tvars)
            critic_optim = tf.train.RMSPropOptimizer(a.lr)
            critic_train = critic_optim.apply_gradients(critic_grad_var_pairs)

    return critic_loss_single, critic_train

def init_config():
    offset_frames = np.array([], dtype=np.float32).reshape(0,a.input_featdim)
    frame_buffer = np.array([], dtype=np.float32)
    offset_senones = np.array([], dtype=np.float32).reshape(0,a.senones)
    senone_buffer = np.array([], dtype=np.float32)
    A = np.array([], dtype=np.int32)

    config = {'batch_index':0, 
                'uid':0, 
                'offset':0, 
                'offset_frames':offset_frames, 
                'frame_buffer':frame_buffer, 
                'offset_senones':offset_senones,
                'senone_buffer':senone_buffer,
                'perm':A}
    return config

def fill_feed_dict(frame_pl, senone_pl, config, frame_file, senone_file, shuffle):

    batch_index = config['batch_index']
    offset_frames = config['offset_frames']
    offset_senones = config['offset_senones']
    A = config['perm']

    def create_buffer(uid, offset):
        ark_dict,uid_new= read_mats(uid,offset,frame_file)
        senone_dict,uid_new = read_senones(uid,offset,senone_file)

        ids = sorted(ark_dict.keys())
        mats = [ark_dict[i] for i in ids]
        mats2 = np.vstack(mats)
        nonlocal offset_frames
        mats2 = np.concatenate((offset_frames,mats2),axis=0)

        mats_senone = [senone_dict[i] for i in ids]
        mats2_senone = np.vstack(mats_senone)
        nonlocal offset_senones
        mats2_senone = np.concatenate((offset_senones,mats2_senone),axis=0)
            
        if mats2.shape[0]>=(a.batch_size*a.buffer_size):
            offset_frames = mats2[a.batch_size*a.buffer_size:]
            mats2 = mats2[:a.batch_size*a.buffer_size]
            offset = offset_frames.shape[0]
            offset_senones = mats2_senone[a.batch_size*a.buffer_size:]
            mats2_senone = mats2_senone[:a.batch_size*a.buffer_size]
            
        return mats2, mats2_senone, uid_new, offset

    if batch_index==0:
        frame_buffer, senone_buffer, uid_new, offset = create_buffer(config['uid'], config['offset'])
        frame_buffer = np.pad(frame_buffer,
                                    ((a.context,),(0,)),
                                    'constant',
                                    constant_values=0)
        if shuffle==True:
            A = np.random.permutation(senone_buffer.shape[0])
        else:
            A = np.arange(senone_buffer.shape[0])
        senone_buffer = senone_buffer[A]
 
    else:
        frame_buffer = config['frame_buffer']
        senone_buffer = config['senone_buffer']
        uid_new = config['uid']
        offset = config['offset']


    start = batch_index*a.batch_size
    end = min((batch_index+1)*a.batch_size,senone_buffer.shape[0])
    config = {'batch_index':(batch_index+1)%a.buffer_size, 
                'uid':uid_new,
                'offset':offset, 
                'offset_frames':offset_frames,
                'offset_senones':offset_senones, 
                'frame_buffer':frame_buffer,
                'senone_buffer':senone_buffer, 
                'perm':A}
    frame_batch = np.stack((frame_buffer[A[i]:A[i]+1+2*a.context,].flatten()
                            for i in range(start, end)), axis = 0)
    senone_batch = senone_buffer[start:end] 
    feed_dict = {frame_pl:frame_batch, senone_pl:senone_batch}
    return (feed_dict, config)
 
        
def placeholder_inputs():
    shape = a.input_featdim*(2*a.context+1)
    frame_placeholder = tf.placeholder(tf.float32, shape=(None,shape), name="frame_placeholder")
    senone_placeholder = tf.placeholder(tf.float32, shape=(None,a.senones), name="senone_placeholder")
    is_training = tf.placeholder(tf.bool, name="is_training")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return frame_placeholder, senone_placeholder, is_training, keep_prob

def do_eval(sess, critic_loss_single, frame_pl, senone_pl, is_training, keep_prob):
    config = init_config()
    tot_loss_epoch = 0
    totframes = 0

    start_time = time.time()
    while(True):
        feed_dict, config = fill_feed_dict(frame_pl, senone_pl, config, a.frame_dev_file, a.senone_dev_file, shuffle=False)
        feed_dict[is_training] = False 
        feed_dict[keep_prob] = 1.0
        result = sess.run(critic_loss_single, feed_dict=feed_dict)
        tot_loss_epoch += feed_dict[frame_pl].shape[0]*result
        totframes += feed_dict[frame_pl].shape[0]
        
        if feed_dict[frame_pl].shape[0]<a.batch_size:
            break
    
    eval_loss = float(tot_loss_epoch)/totframes 
    duration = time.time() - start_time
    print ('Eval loss = %.6f (%.3f sec)' % (eval_loss, duration))
    return eval_loss


def run_training():
    if not os.path.isdir(a.exp_name):
        os.makedirs(a.exp_name)
    tot_loss_epoch = 0
    avg_loss_epoch = 0
    with tf.Graph().as_default():
        frame_pl, senone_pl, is_training, keep_prob = placeholder_inputs()
        critic_loss_single, critic_train  = train_critic(frame_pl, senone_pl, is_training, keep_prob)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        all_vars = tf.global_variables()
        saver = tf.train.Saver([var for var in all_vars])
        sess = tf.Session()

        sess.run(init)  
        #if a.checkpoint is not None:
        #    saver.restore(sess,a.checkpoint)
        start_time = time.time()
        main_step = 0

        totframes_train = 5436393
        #print("finding number of batches needed...")
        #print("for training.....")
        #config = init_config()
        #totframes_train = 0
        #totbatches_train  = 0
        #while(True):
        #    feed_dict, config = fill_feed_dict(frame_pl, senone_pl, config, a.frame_train_file, a.senone_train_file, shuffle=False)
        #    feed_dict[is_training] = False
        #    totframes_train += feed_dict[frame_pl].shape[0]
        #    totbatches_train += 1
        #    print(totbatches_train)
        #    if feed_dict[frame_pl].shape[0]<a.batch_size:
        #        break
        totbatches_train = totframes_train/a.batch_size
        totbatches_train = int(totbatches_train) + 1
        print ("Total number of frames:"+str(totframes_train))
        print("Total training batches:"+str(totbatches_train))
        while(True):
            config = init_config()
            for step in range(totbatches_train):
                feed_dict, config = fill_feed_dict(frame_pl, senone_pl, config, a.frame_train_file, a.senone_train_file, shuffle=True)
                feed_dict[is_training] = True
                feed_dict[keep_prob] = a.keep_prob

                critic_loss, _ = sess.run([critic_loss_single, critic_train], feed_dict=feed_dict)
                tot_loss_epoch += feed_dict[frame_pl].shape[0]*critic_loss
            avg_loss_epoch = float(tot_loss_epoch)/totframes_train
            tot_loss_epoch = 0
            duration = time.time() - start_time
            start_time = time.time()
            print ('Iteration: %d Gen:%.6f (%.3f sec)'
                   % ((main_step+1), avg_loss_epoch,duration))
            
            print ('Eval step:')
            eval_loss = do_eval(sess, critic_loss_single, frame_pl,
                                          senone_pl, is_training, keep_prob)
                 
        save_path = saver.save(sess, os.path.join(a.exp_name,"model.ckpt"+str(eval_loss)), global_step=main_step)
        main_step += 1

def find_min_max(scp_file):
    minimum = float("inf")
    maximum = -float("inf")
    uid = 0
    offset = 0
    ark_dict, uid = read_mats(uid, offset, scp_file)
    while ark_dict:
        for key in ark_dict.keys():
            mat_max = np.amax(ark_dict[key])
            mat_min = np.amin(ark_dict[key])
            if mat_max > maximum:
                maximum = mat_max
            if mat_min < minimum:
                minimum = mat_min
        ark_dict, uid = read_mats(uid, offset, scp_file)
    return minimum, maximum

def main():
    run_training()
    
if __name__=='__main__':
    main()    
    

