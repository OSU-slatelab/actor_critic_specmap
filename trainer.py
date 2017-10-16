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

    def __init__(self,
            learning_rate,
            max_global_norm,
            l2_weight = 0.,
            mse_decay = 0.,
            critic = None,
            actor = None):
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

        if mse_decay > 0:
            self.actor_out = actor.outputs
            self.clean = tf.placeholder(tf.float32, shape=actor.outputs.shape)
            self.mse_weight = tf.placeholder(tf.float32)
            self.current_mse_weight = 1.0

        self.learning_rate = learning_rate
        self.max_global_norm = max_global_norm
        self.l2_weight = l2_weight
        self.mse_decay = mse_decay

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

            if self.mse_decay > 0:
                loss2 = tf.losses.mean_squared_error(labels=self.clean, predictions=self.actor_out)
                loss = (1-self.mse_weight) * loss + self.mse_weight * tf.reduce_mean(loss2)

        self.loss = loss + l2_loss
        
        grads = tf.gradients(self.loss, self.var_list)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_global_norm)
        grad_var_pairs = zip(grads, self.var_list)
        optim = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train = optim.apply_gradients(grad_var_pairs)

    def run_ops(self, sess, loader, training = True, pretrain = False):

        tot_loss_epoch = 0
        frames = 0
        start_time = time.time()
        self.feed_dict[self.training] = training

        # Iterate dataset
        for batch in loader.batchify(pretrain):

            self.feed_dict[self.inputs] = batch['frame']
            self.feed_dict[self.labels] = batch['label']

            if self.mse_decay > 0:
                self.feed_dict[self.clean] = batch['clean']
                self.feed_dict[self.mse_weight] = self.current_mse_weight
            
            if training:
                batch_loss, _ = sess.run([self.loss, self.train], feed_dict = self.feed_dict)
            else:
                batch_loss = sess.run(self.loss, feed_dict = self.feed_dict)

            frames += batch['frame'].shape[0]
            update_progressbar(frames / loader.frame_count)
            tot_loss_epoch += batch['frame'].shape[0] * batch_loss

        # Compute loss
        avg_loss_epoch = float(tot_loss_epoch) / frames
        duration = time.time() - start_time

        loader.reset()
        if self.mse_decay > 0:
            self.current_mse_weight *= self.mse_decay

        return avg_loss_epoch, duration

