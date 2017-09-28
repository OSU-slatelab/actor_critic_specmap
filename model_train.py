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

from critic_model import Critic

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
parser.add_argument("--max_global_norm", type=float, default=5.0, help="global max norm for clipping")
parser.add_argument("--dropout", type=float, default=0.5, help="percentage of neurons to drop")
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

def train_critic(critic, targets):
    with tf.name_scope('critic_loss_single'):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=critic.outputs, labels=targets)
        critic_loss_single = tf.reduce_mean(loss)

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
    return frame_placeholder, senone_placeholder

def do_eval(sess, critic_loss_single, frame_pl, senone_pl, critic):
    config = init_config()
    tot_loss_epoch = 0
    totframes = 0

    start_time = time.time()
    while(True):
        feed_dict, config = fill_feed_dict(frame_pl, senone_pl, config, a.frame_dev_file, a.senone_dev_file, shuffle=False)
        feed_dict[critic.training] = False 
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
        frame_pl, senone_pl = placeholder_inputs()

        # Define our critic model
        with tf.variable_scope('critic'):
            critic = Critic(frame_pl,
                input_size  = a.input_featdim*(2*a.context+1),
                layer_size  = a.cunits,
                layers      = a.clayers,
                output_size = a.senones,
                dropout     = a.dropout)

        critic_loss_single, critic_train  = train_critic(critic, senone_pl)
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

                feed_dict[critic.training] = True
                critic_loss, _ = sess.run([critic_loss_single, critic_train], feed_dict=feed_dict)
                tot_loss_epoch += feed_dict[frame_pl].shape[0]*critic_loss
            avg_loss_epoch = float(tot_loss_epoch)/totframes_train
            tot_loss_epoch = 0
            duration = time.time() - start_time
            start_time = time.time()
            print ('Iteration: %d Gen:%.6f (%.3f sec)'
                   % ((main_step+1), avg_loss_epoch,duration))
            
            print ('Eval step:')
            eval_loss = do_eval(sess, critic_loss_single, frame_pl, senone_pl, critic)
                 
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
    

