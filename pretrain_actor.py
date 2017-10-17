from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os

from actor_model import Actor
from data_io import DataLoader
from trainer import Trainer

parser = argparse.ArgumentParser()

# Files
parser.add_argument("--base_directory", default=os.getcwd(), help="The directory the data is in")
parser.add_argument("--frame_train_file", default="data-fbank/train_si84_delta_noisy_global_normalized/feats.scp", help="The input feature file for training")
parser.add_argument("--frame_dev_file", default="data-fbank/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod", help="The input feature file for cross-validation")
parser.add_argument("--clean_train_file", default="data-fbank/train_si84_clean_global_normalized/feats.scp", help="The input feature file for training")
parser.add_argument("--clean_dev_file", default="data-fbank/dev_dt_05_clean_global_normalized/feats.scp.mod", help="The input feature file for cross-validation")
parser.add_argument("--senone_train_file", default="clean_labels_train.txt", help="The senone file for clean training labels")
parser.add_argument("--senone_dev_file", default="clean_labels_dev_mod.txt", help="The senone file for clean cross-validation labels")
parser.add_argument("--actor_pretrain", default="actor_pretrain", help="Directory to store pre-trained weights")

# Training
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate")
parser.add_argument("--max_global_norm", type=float, default=5.0, help="global max norm for clipping")
parser.add_argument("--l2_weight", type=float, default=0)

# Model
parser.add_argument("--alayers", type=int, default=2)
parser.add_argument("--aunits", type=int, default=2048)
parser.add_argument("--dropout", type=float, default=0.5, help="percentage of neurons to drop")

# Data
parser.add_argument("--input_featdim", type=int, default=120)
parser.add_argument("--output_featdim", type=int, default=40)
parser.add_argument("--context", type=int, default=5)
parser.add_argument("--buffer_size", default=10, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
a = parser.parse_args()

def run_training():
    """ Define our model and train it """

    # Create directory for saving models
    if not os.path.isdir(a.actor_pretrain):
        os.makedirs(a.actor_pretrain)

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

        # Create loader for train data
        train_loader = DataLoader(
            base_dir    = a.base_directory,
            frame_file  = a.frame_train_file,
            senone_file = a.senone_train_file,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1 + 2 * a.context,
            shuffle     = True,
            clean_file  = a.clean_train_file)

        print("Total train frames:", train_loader.frame_count)

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
            clean_file  = a.clean_train_file)

        print("Total dev frames:", dev_loader.frame_count)

        with tf.variable_scope('trainer'):
            trainer = Trainer(a.lr, a.max_global_norm, a.l2_weight, actor = actor)

        # Saver is also loader
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        trainer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='trainer')
        actor_saver = tf.train.Saver(actor_vars)

        # Begin session
        sess = tf.Session()

        # Load critic weights, initialize actor weights and trainer weights
        sess.run(tf.variables_initializer(actor_vars))
        sess.run(tf.variables_initializer(trainer_vars))
        
        # Perform training
        min_loss = float('inf')
        for epoch in range(1, 200):
            print('Epoch %d' % epoch)

            train_loss, duration = trainer.run_ops(sess, train_loader, training = True, pretrain = True)
            print ('\nTrain loss: %.6f (%.3f sec)' % (train_loss, duration))

            eval_loss, duration = trainer.run_ops(sess, dev_loader, training = False, pretrain = True)
            print('\nEval loss: %.6f (%.3f sec)' % (eval_loss, duration))

            # Save if we've got the best loss so far
            if eval_loss < min_loss:
                min_loss = eval_loss
                save_file = os.path.join(a.actor_pretrain, f"model-{eval_loss}.ckpt")
                save_path = actor_saver.save(sess, save_file, global_step=epoch)

def main():
    run_training()
    
if __name__=='__main__':
    main()    
    

