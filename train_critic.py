from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os

from critic_model import Critic
from data_io import DataLoader
from trainer import Trainer

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

        # Create loader for train data
        train_loader = DataLoader(
            base_dir    = a.base_directory,
            frame_file  = a.frame_train_file,
            senone_file = a.senone_train_file,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1,
            shuffle     = True)

        print("Frames in train data:", train_loader.frame_count)

        # Create loader for test data
        dev_loader = DataLoader(
            base_dir    = a.base_directory,
            frame_file  = a.frame_dev_file,
            senone_file = a.senone_dev_file,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = 1,
            shuffle     = False)

        print("Frames in dev data:", dev_loader.frame_count)

        # Class for training
        with tf.variable_scope('trainer'):
            trainer = Trainer(a.lr, a.max_global_norm, critic)

        # Save all variables
        saver = tf.train.Saver()

        # Begin session
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)
        
        #if a.checkpoint is not None:
        #    saver.restore(sess,a.checkpoint)

        # Perform training
        for epoch in range(1, 200):
            print('Epoch %d' % epoch)

            train_loss, duration = trainer.run_ops(sess, train_loader, training = True)
            print ('\nTrain loss: %.6f (%.3f sec)' % (train_loss, duration))

            eval_loss, duration = trainer.run_ops(sess, dev_loader, training = False)
            print('\nEval loss: %.6f (%.3f sec)' % (eval_loss, duration))

            save_path = saver.save(sess, os.path.join(a.exp_name, "model.ckpt"), global_step=epoch)

def main():
    run_training()
    
if __name__=='__main__':
    main()    
    

