from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import time
from data_io import DataLoader

from critic_model import Critic

parser = argparse.ArgumentParser()
parser.add_argument("--base_directory", default=os.getcwd(), help="The directory the data is in")
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

def do_eval(sess, critic_loss_single, critic):
    """ Compute loss over validation set """

    # Create loader
    loader = DataLoader(
        base_dir    = a.base_directory,
        frame_file  = a.frame_dev_file,
        senone_file = a.senone_dev_file,
        batch_size  = a.batch_size,
        buffer_size = a.buffer_size,
        context     = a.context,
        out_frames  = 1,
        shuffle     = False)

    # Initialize loop vars
    tot_loss_epoch = 0
    totframes = 0
    start_time = time.time()

    # Stop when we've reached a batch smaller than the batch size
    for frame_batch, senone_batch in loader.batchify():

        feed_dict = {
            critic.inputs: frame_batch,
            critic.labels: senone_batch,
            critic.training: False
        }

        result = sess.run(critic_loss_single, feed_dict=feed_dict)

        tot_loss_epoch += frame_batch.shape[0] * result
        totframes += frame_batch.shape[0]
    
    # Compute loss
    eval_loss = float(tot_loss_epoch)/totframes 
    duration = time.time() - start_time

    return eval_loss, duration

def do_train(sess, train_ops, critic):
    """ Perform one epoch of training """

    # Create loader for data
    loader = DataLoader(
            base_dir    = a.base_directory,
            frame_file  = a.frame_train_file,
            senone_file = a.senone_train_file,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1,
            shuffle     = True)

    print("Total train frames:", loader.frame_count)

    tot_loss_epoch = 0
    frames = 0
    start_time = time.time()

    # Iterate dataset
    for frame_batch, senone_batch in loader.batchify():

        frames += frame_batch.shape[0]
        update_progressbar(frames / loader.frame_count)

        feed_dict = {
            critic.inputs: frame_batch,
            critic.labels: senone_batch,
            critic.training: True,
        }

        critic_loss, _ = sess.run(train_ops, feed_dict=feed_dict)
        tot_loss_epoch += frame_batch.shape[0]*critic_loss

    # Compute loss
    avg_loss_epoch = float(tot_loss_epoch) / frames
    
    duration = time.time() - start_time

    return avg_loss_epoch, duration

def update_progressbar(progress):
    """ Make a very basic progress bar """

    length = 30
    intprog = int(round(progress * length))
    sys.stdout.write("\r[{0}] {1:2.1f}%".format("#"*intprog + "-"*(length-intprog), progress*100))
    sys.stdout.flush()

def run_training():
    """ Define our model and train it """

    # Create directory for saving models
    if not os.path.isdir(a.exp_name):
        os.makedirs(a.exp_name)

    with tf.Graph().as_default():
        shape = (None, 2*a.context + 1, a.input_featdim)
        frame_placeholder = tf.placeholder(tf.float32, shape=shape, name="frame_placeholder")

        # Define our critic model
        with tf.variable_scope('critic'):
            critic = Critic(
                inputs      = frame_placeholder,
                layer_size  = a.cunits,
                layers      = a.clayers,
                output_size = a.senones,
                dropout     = a.dropout)

        # Define ops
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        train_ops = critic.create_train_ops(a.max_global_norm, a.lr, var_list)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()

        # Begin session
        sess = tf.Session()
        sess.run(init)  
        
        #if a.checkpoint is not None:
        #    saver.restore(sess,a.checkpoint)

        # Perform training
        for epoch in range(1, 200):
            print('Epoch %d' % epoch)

            train_loss, duration = do_train(sess, train_ops, critic)
            print ('\nTrain loss: %.6f (%.3f sec)' % (train_loss, duration))

            eval_loss, duration = do_eval(sess, train_ops[0], critic)
            print('Eval loss: %.6f (%.3f sec)' % (eval_loss, duration))

            save_path = saver.save(sess, os.path.join(a.exp_name,"model.ckpt"+str(eval_loss)), global_step=epoch)

def main():
    run_training()
    
if __name__=='__main__':
    main()    
    

