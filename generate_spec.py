from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os

from actor_model import Actor
from data_io import DataLoader

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def printSpec(array, filename):
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(111)
    ax.imshow(np.flipud(array.T))
    fig.savefig(filename)
    plt.close(fig)



def run_training(a):
    """ Define our model and train it """

    # Create directory for saving models
    if not os.path.isdir(a.actor_checkpoints):
        os.makedirs(a.actor_checkpoints)

    with tf.Graph().as_default():

        # Define our actor model
        with tf.variable_scope('actor'):

            # Output of actor is input of critic, so output context plus frame
            output_frames = 100
            in_shape = (None, output_frames + 2*a.context, a.input_featdim)
            out_shape = (None, output_frames, a.output_featdim)
            actor = Actor(
                input_shape   = in_shape,
                output_shape  = out_shape,
                layer_size    = a.aunits,
                layers        = a.alayers,
                output_frames = output_frames,
                dropout       = a.dropout,
            )

        # Create loader
        dev_loader = DataLoader(
            base_dir    = a.base_directory,
            frame_file  = a.frame_dev_file,
            batch_size  = a.batch_size,
            buffer_size = a.buffer_size,
            context     = a.context,
            out_frames  = output_frames,
            shuffle     = False,
            clean_file  = a.clean_dev_file)

        # Saver is also loader
        actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        actor_saver = tf.train.Saver(actor_vars)

        # Begin session
        sess = tf.Session()

        # Load actor weights
        actor_saver.restore(sess, tf.train.latest_checkpoint(a.actor_checkpoints))

        for batch in dev_loader.batchify():

            fd = {actor.inputs : batch['frame'], actor.training: False}

            outputs = sess.run(actor.outputs, fd)

            printSpec(batch['frame'][501], "actor_input.png")
            printSpec(batch['clean'][501], "actor_clean.png")
            printSpec(outputs[501], "actor_output.png")
            
            break
            

def main():
    parser = argparse.ArgumentParser()

    # Files
    parser.add_argument("--base_directory", default=os.getcwd(), help="The directory the data is in")
    parser.add_argument("--frame_dev_file", default="data-fbank/dev_dt_05_delta_noisy_global_normalized/feats.scp.mod") 
    parser.add_argument("--clean_dev_file", default="data-fbank/dev_dt_05_clean_global_normalized/feats.scp.mod") 
    parser.add_argument("--actor_checkpoints", default="actor_checkpoints/", help="directory with actor weights")

    # Model
    parser.add_argument("--alayers", type=int, default=4)
    parser.add_argument("--aunits", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5, help="percentage of neurons to drop")

    # Data
    parser.add_argument("--input_featdim", type=int, default=120)
    parser.add_argument("--output_featdim", type=int, default=40)
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--buffer_size", default=10, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    a = parser.parse_args()

    run_training(a)
    
if __name__=='__main__':
    main()    
    

