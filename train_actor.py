from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os

from critic_model import Critic
from actor_model import Actor
from data_io import DataLoader
from trainer import Trainer

parser = argparse.ArgumentParser()

# Files
parser.add_argument("--base_directory", default=os.getcwd(), help="The directory the data is in")
parser.add_argument("--frame_train_file", default="data-spectrogram/train_si84_noisy/feats.scp", help="The input feature file for training")
parser.add_argument("--frame_dev_file", default="data-spectrogram/dev_dt_05_delta_noisy/feats.scp.mod", help="The input feature file for cross-validation")
parser.add_argument("--clean_train_file", default=None, help="Clean speech for mse loss")
parser.add_argument("--clean_dev_file", default=None, help="Clean speech for mse loss")
parser.add_argument("--senone_train_file", default="clean_labels_train.txt", help="The senone file for clean training labels")
parser.add_argument("--senone_dev_file", default="clean_labels_dev_mod.txt", help="The senone file for clean cross-validation labels")
parser.add_argument("--exp_name", default="new_exp", help="directory with critic weights")
parser.add_argument("--actor_checkpoints", default="actor_checkpoints", help="directory with actor weights")
parser.add_argument("--actor_pretrain", default=None, help="directory with actor pretrained weights")

# Training
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate")
parser.add_argument("--max_global_norm", type=float, default=5.0, help="global max norm for clipping")
parser.add_argument("--l2_weight", type=float, default=0)
parser.add_argument("--mse_decay", type=float, default=0)

# Model
parser.add_argument("--alayers", type=int, default=2)
parser.add_argument("--aunits", type=int, default=2048)
parser.add_argument("--clayers", type=int, default=6)
parser.add_argument("--cunits", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.5, help="percentage of neurons to drop")

# Data
parser.add_argument("--input_featdim", type=int, default=771)
parser.add_argument("--output_featdim", type=int, default=40)
parser.add_argument("--senones", type=int, default=1999)
parser.add_argument("--context", type=int, default=5)
parser.add_argument("--buffer_size", default=10, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
a = parser.parse_args()

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
            output_shape = (None, output_frames + 2*a.context, a.output_featdim)
            actor = Actor(
                input_shape   = shape,
                output_shape  = output_shape,
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
            
        # Create loader for train data
        train_loader = DataLoader(
            base_dir    = a.base_directory,
            frame_file  = a.frame_train_file,
            senone_file = a.senone_train_file,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1 + 2 * a.context,
            shuffle     = False,
            clean_file  = a.clean_train_file)

        #print("Total train frames:", train_loader.frame_count)
        # Create loader
        dev_loader = DataLoader(
            base_dir    = a.base_directory,
            frame_file  = a.frame_dev_file,
            senone_file = a.senone_dev_file,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1 + 2 * a.context,
            shuffle     = False,
            clean_file  = a.clean_dev_file)

        #print("Total dev frames:", dev_loader.frame_count)

        with tf.variable_scope('trainer'):
            trainer = Trainer(a.lr, a.max_global_norm, a.l2_weight, a.mse_decay, critic, actor)

        # Saver is also loader
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        trainer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainer')
        actor_saver = tf.train.Saver(actor_vars)
        critic_saver = tf.train.Saver(critic_vars)

        # Begin session
        sess = tf.Session()

        # Load critic weights, actor weights and initialize trainer weights
        critic_saver.restore(sess, tf.train.latest_checkpoint(a.exp_name))
        if a.actor_pretrain:
            actor_saver.restore(sess, tf.train.latest_checkpoint(a.actor_pretrain))
        else:
            sess.run(tf.variables_initializer(actor_vars))
        sess.run(tf.variables_initializer(trainer_vars))
        
        # Perform training
        min_loss = float('inf')
        for epoch in range(1, 200):
            print('Epoch %d' % epoch)

            train_loss, duration = trainer.run_ops(sess, train_loader, training = True)
            print ('\nTrain loss: %.6f (%.3f sec)' % (train_loss, duration))

            eval_loss, duration = trainer.run_ops(sess, dev_loader, training = False)
            print('\nEval loss: %.6f (%.3f sec)' % (eval_loss, duration))

            # Save if we've got the best loss so far
            if eval_loss < min_loss:
                min_loss = eval_loss
                save_file = os.path.join(a.actor_checkpoints, f"model-{eval_loss}.ckpt")
                save_path = actor_saver.save(sess, save_file, global_step=epoch)

def main():
    run_training()
    
if __name__=='__main__':
    main()    
    

