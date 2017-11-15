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
            min_mse   = 0.,
            optim = 'adam',
            match_all = False,
            critic = None,
            actor = None,
            output_critic = None):
        """ 
        Params:
         * learning_rate : float
            Rate of gradient descent
         * max_global_norm : float
            For clipping norm
         * l2_weight : float
            Amount of l2 loss to include
         * mse_decay : float
            How much to decay the ratio of mse to mimic loss, 0 for no decay
         * min_mse : float
            The minimum proportion of mse relative to mimic loss
         * optim : 'adam', 'adam_decay', or 'sgd'
            Optimization algorithm to use
         * match_all : boolean
            Whether to match all layers or just output layer for mimic loss
         * critic : Critic
            model to train. If None, pretrain actor
         * actor : Actor
            (optional) model to train. If passed, critic is frozen.
         * output_critic : Critic
            critic for generating posteriors as labels
        """
        
        self.feed_dict = {}

        # Critic is none if we're pretraining actor
        self.pretrain = critic is None
        
        # Set this to a placeholder if clean speech is input
        self.clean = None

        # Actor is none if we're training critic
        if actor is None:
            self.inputs = critic.inputs
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
            self.training = critic.training
            self.labels = critic.labels

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=critic.outputs, labels=critic.labels)
            self.loss = tf.reduce_mean(loss)

        # Training actor
        else:
            self.inputs = actor.inputs
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
            self.training = actor.training
            
            if self.pretrain:
                self.labels = tf.placeholder(tf.float32, shape=actor.outputs.shape)
                loss = tf.losses.mean_squared_error(labels=self.labels, predictions=actor.outputs)
                self.loss = tf.reduce_mean(loss)
            else:
                self.feed_dict[critic.training] = False
                self.feed_dict[output_critic.training] = False
                
                # We're going to try to match the posteriors of clean speech
                self.labels = critic.labels
                self.clean = output_critic.inputs

                loss = tf.losses.mean_squared_error(
                        labels      = output_critic.outputs,
                        predictions = critic.outputs)
                self.mimic_loss = tf.reduce_mean(loss) / 10

                if match_all:
                    for i in range(len(critic.layers)):
                        loss = tf.losses.mean_squared_error(
                                labels      = output_critic.layers[i],
                                predictions = critic.layers[i])
                        self.mimic_loss += tf.reduce_mean(loss) / 20

                # This checks whether or not we're including mse loss
                if mse_decay > 0 or min_mse > 0:
                    self.mse_weight = tf.placeholder(tf.float32)
                    self.current_mse_weight = 1.0 if mse_decay > 0 else 0.0

                    loss = tf.losses.mean_squared_error(labels=self.clean, predictions=actor.outputs)
                    self.mse_loss = tf.reduce_mean(loss)

                    self.loss = (1-self.mse_weight) * (1-min_mse) * self.mimic_loss + \
                        (self.mse_weight * (1-min_mse) + min_mse) * self.mse_loss
                else:
                    self.loss = self.mimic_loss

        l2_reg = l2_weight * tf.reduce_sum([tf.nn.l2_loss(var) for var in self.var_list])
        self.loss += l2_reg

        self.learning_rate = learning_rate
        self.max_global_norm = max_global_norm
        self.mse_decay = mse_decay
        self.min_mse = min_mse
        self.optim = optim

        self._create_train_op()

    def _create_train_op(self):
        """ Define the training op """
        
        grads = tf.gradients(self.loss, self.var_list)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_global_norm)
        grad_var_pairs = zip(grads, self.var_list)
        if self.optim == 'adam':
            optim = tf.train.AdamOptimizer(self.learning_rate)
            self.train = optim.apply_gradients(grad_var_pairs)
        elif self.optim == 'adam_decay':
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 1e4, 0.95)
            optim = tf.train.AdamOptimizer(learning_rate)
            self.train = optim.apply_gradients(grad_var_pairs, global_step=global_step)
        else:
            optim = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
            self.train = optim.apply_gradients(grad_var_pairs)

    def run_ops(self, sess, loader, training = True):

        tot_loss = 0
        tot_mse_loss = 0
        tot_mimic_loss = 0
        frames = 0
        start_time = time.time()
        self.feed_dict[self.training] = training

        # Iterate dataset
        for batch in loader.batchify(shuffle_batches = training):

            self.feed_dict[self.inputs] = batch['frame']
            self.feed_dict[self.labels] = batch['label']

            # Count the frames in the batch
            batch_frames = batch['frame'].shape[0]

            if self.clean is not None:
                self.feed_dict[self.clean] = batch['clean']

            # If we're combining mse and critic loss, report both independently
            if self.mse_decay > 0 or self.min_mse > 0:
                self.feed_dict[self.mse_weight] = self.current_mse_weight

                ops = [self.mse_loss, self.mimic_loss, self.loss]

                if training:
                    mse_loss, mimic_loss, batch_loss, _ = sess.run(ops + [self.train], self.feed_dict)
                else:
                    mse_loss, mimic_loss, batch_loss = sess.run(ops, self.feed_dict)

                tot_mse_loss += batch_frames * mse_loss
                tot_mimic_loss += batch_frames * mimic_loss 
            
            # Just mimic loss
            elif training:
                batch_loss, _ = sess.run([self.loss, self.train], feed_dict = self.feed_dict)
            else:
                batch_loss = sess.run(self.loss, feed_dict = self.feed_dict)

            tot_loss += batch_frames * batch_loss

            # Update the progressbar
            frames += batch_frames
            update_progressbar(frames / loader.frame_count)

        # Compute loss
        avg_loss = float(tot_loss) / frames
        duration = time.time() - start_time

        if self.mse_decay > 0 or self.min_mse > 0:
            avg_mse_loss = tot_mse_loss / frames
            avg_mimic_loss = tot_mimic_loss / frames
            return avg_mse_loss, avg_mimic_loss, avg_loss, duration

        return avg_loss, duration

