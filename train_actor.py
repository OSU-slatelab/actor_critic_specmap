from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import time

from critic_model import Critic
from actor_model import Actor
from data_io import DataLoader

parser = argparse.ArgumentParser()

# Files
parser.add_argument("--base_directory", default=os.getcwd(), help="The directory the data is in")
parser.add_argument("--frame_train_file", default="data-fbank/train_si84_noisy/feats.scp", help="The input feature file for training")
parser.add_argument("--frame_dev_file", default="data-fbank/dev_dt_05_noisy/feats.scp", help="The input feature file for cross-validation")
parser.add_argument("--senone_train_file", default="clean_labels_train.txt", help="The senone file for clean training labels")
parser.add_argument("--senone_dev_file", default="clean_labels_dev.txt", help="The senone file for clean cross-validation labels")
parser.add_argument("--exp_name", default="new_exp", help="directory with critic weights")
parser.add_argument("--actor_checkpoints", default="actor_checkpoints", help="directory with actor weights")

# Training
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate")
parser.add_argument("--max_global_norm", type=float, default=5.0, help="global max norm for clipping")

# Model
parser.add_argument("--alayers", type=int, default=2)
parser.add_argument("--aunits", type=int, default=2048)
parser.add_argument("--clayers", type=int, default=6)
parser.add_argument("--cunits", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.5, help="percentage of neurons to drop")

# Data
parser.add_argument("--input_featdim", type=int, default=40)
parser.add_argument("--senones", type=int, default=1999)
parser.add_argument("--context", type=int, default=5)
parser.add_argument("--buffer_size", default=10, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
a = parser.parse_args()

def do_eval(sess, loss_single, critic, actor):
    """ Compute loss over validation set """

    # Create loader
    loader = DataLoader(
        base_dir    = a.base_directory,
        frame_file  = a.frame_dev_file,
        senone_file = a.senone_dev_file,
        batch_size  = a.batch_size,
        buffer_size = a.buffer_size,
        context     = a.context,
        out_frames  = 1 + 2 * a.context,
        shuffle     = False)

    # Initialize loop vars
    tot_loss_epoch = 0
    totframes = 0
    batch_size = a.batch_size
    start_time = time.time()

    # Stop when we've reached a batch smaller than the batch size
    while batch_size == a.batch_size:
        frame_batch, senone_batch = loader.make_inputs()

        feed_dict = {
            actor.inputs: frame_batch,
            critic.labels: senone_batch,
            actor.training: False,
            critic.training: False,
        }

        result = sess.run(loss_single, feed_dict=feed_dict)

        batch_size = frame_batch.shape[0]
        tot_loss_epoch += batch_size * result
        totframes += batch_size
    
    # Compute loss
    eval_loss = float(tot_loss_epoch)/totframes 
    duration = time.time() - start_time

    return eval_loss, duration

def do_train(sess, train_ops, critic, actor, totframes_train):
    """ Perform one epoch of training """

    # Create loader for data
    loader = DataLoader(
            base_dir    = a.base_directory,
            frame_file  = a.frame_train_file,
            senone_file = a.senone_train_file,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1 + 2 * a.context,
            shuffle     = True)

    tot_loss_epoch = 0
    avg_loss_epoch = 0
    start_time = time.time()

    # Iterate dataset
    totbatches_train = int(totframes_train/a.batch_size) + 1
    for step in range(totbatches_train):

        update_progressbar(step / totbatches_train)

        frame_batch, senone_batch = loader.make_inputs()

        feed_dict = {
            actor.inputs: frame_batch,
            critic.labels: senone_batch,
            actor.training: True,
            critic.training: False,
        }

        critic_loss, _ = sess.run(train_ops, feed_dict=feed_dict)
        tot_loss_epoch += frame_batch.shape[0]*critic_loss

    # Compute loss
    avg_loss_epoch = float(tot_loss_epoch) / totframes_train
    tot_loss_epoch = 0
    
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
    if not os.path.isdir(a.actor_checkpoints):
        os.makedirs(a.actor_checkpoints)

    with tf.Graph().as_default():

        # Define our actor model
        with tf.variable_scope('actor'):

            # Output of actor is input of critic, so output context plus frame
            output_frames = 2*a.context + 1
            shape = (None, output_frames + 2*a.context, a.input_featdim)
            actor = Actor(
                input_shape   = shape,
                layer_size    = a.aunits,
                layers        = a.alayers,
                output_frames = output_frames,
                dropout       = a.dropout,
            )

        # Define our critic model
        with tf.variable_scope('critic'):
            critic = Critic(
                inputs      = actor.outputs,
                layer_size  = a.cunits,
                layers      = a.clayers,
                output_size = a.senones,
                dropout     = a.dropout)

        # Define ops
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        train_ops = critic.create_train_ops(a.max_global_norm, a.lr, actor_vars)
        
        # Re-define actor vars so it includes train ops
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        
        # Saver is also loader
        actor_saver = tf.train.Saver(actor_vars)
        critic_saver = tf.train.Saver(critic_vars)

        # Begin session
        sess = tf.Session()

        # Load critic weights, initialize actor weights
        critic_saver.restore(sess, tf.train.latest_checkpoint(a.exp_name))
        sess.run(tf.variables_initializer(actor_vars))
        
        # Perform training
        totframes_train = 5436393
        print("Total number of frames:" + str(totframes_train))

        for epoch in range(1, 200):
            print('Epoch %d' % epoch)

            train_loss, duration = do_train(sess, train_ops, critic, actor, totframes_train)
            print ('\nTrain loss: %.6f (%.3f sec)' % (train_loss, duration))

            eval_loss, duration = do_eval(sess, train_ops[0], critic, actor)
            print('Eval loss: %.6f (%.3f sec)' % (eval_loss, duration))

            save_file = os.path.join(a.actor_checkpoints, "model.ckpt")
            save_path = actor_saver.save(sess, save_file, global_step=epoch)

def main():
    run_training()
    
if __name__=='__main__':
    main()    
    

