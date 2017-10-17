"""
Trainer creates train ops and goes through all data to train or test.

Author: Peter Plantinga
Date: Fall 2017
"""

import tensorflow as tf
import time
import sys

def update_progressbar(progress):
    """ Make a very basic progress bar """

    length = 30
    intprog = int(round(progress * length))
    sys.stdout.write("\r[{0}] {1:2.1f}%".format("#"*intprog + "-"*(length-intprog), progress*100))
    sys.stdout.flush()

class Trainer:
    """ Train a model """

    def __init__(self, learning_rate, max_global_norm, l2_weight, critic = None, actor = None):
        """ 
        Params:
         * learning_rate : float
            Rate of gradient descent
         * max_global_norm : float
            For clipping norm
         * l2_weight : float
            Amount of l2 loss to include
         * critic : Critic
            model to train. If None, pretrain actor
         * actor : Actor
            (optional) model to train. If passed, critic is frozen.
        """
        
        self.feed_dict = {}

        # Critic is none if we're pretraining actor
        pretrain = critic is None

        # Actor is none if we're training critic
        if actor is None:
            self.inputs = critic.inputs
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
            self.training = critic.training
        else:
            self.inputs = actor.inputs
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
            self.training = actor.training

            if not pretrain:
                self.feed_dict[critic.training] = False

        if pretrain:
            self.outputs = actor.outputs
            self.labels = tf.placeholder(tf.float32, shape=actor.outputs.shape)
        else:
            self.outputs = critic.outputs
            self.labels = critic.labels

        self.learning_rate = learning_rate
        self.max_global_norm = max_global_norm
        self.l2_weight = l2_weight

        self._create_ops(pretrain)

    def _create_ops(self, pretrain = False):
        """ Define the loss and training ops """

        l2_loss = self.l2_weight * tf.reduce_sum([tf.nn.l2_loss(var) for var in self.var_list])

        if pretrain:
            loss = tf.losses.mean_squared_error(labels=self.labels, predictions=self.outputs)
            loss = tf.reduce_mean(loss)
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.labels)
            loss = tf.reduce_mean(loss)

        self.loss = loss + l2_loss
        
        grads = tf.gradients(self.loss, self.var_list)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_global_norm)
        grad_var_pairs = zip(grads, self.var_list)
        optim = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train = optim.apply_gradients(grad_var_pairs)

    def run_ops(self, sess, clean_loader, noisy_loader, training = True, pretrain = False):

        tot_loss_epoch = 0
        frames = 0
        start_time = time.time()
        self.feed_dict[self.training] = training
        
        print(clean_loader.batchify(pretrain))
        #noisy_batch, senone_batch = noisy_loader.batchify(pretrain)

        print(a.shape)
        print(b.shape)
        # Iterate dataset
        for frame_batch, senone_batch in izip(loader.batchify(pretrain),clean_loader.batchify(pretrain)):

            self.feed_dict[self.inputs] = frame_batch
            self.feed_dict[self.labels] = senone_batch
            #print("after batchify")
            #print(frame_batch.shape)
            #print(senone_batch.shape)
            if training:
                batch_loss, _ = sess.run([self.loss, self.train], feed_dict = self.feed_dict)
            else:
                batch_loss = sess.run(self.loss, feed_dict = self.feed_dict)

            frames += frame_batch.shape[0]
            update_progressbar(frames / loader.frame_count)
            tot_loss_epoch += frame_batch.shape[0] * batch_loss

        # Compute loss
        avg_loss_epoch = float(tot_loss_epoch) / frames
        duration = time.time() - start_time

        loader.reset()

        return avg_loss_epoch, duration

