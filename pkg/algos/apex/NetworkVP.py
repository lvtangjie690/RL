# Copyright (c) 2016, hzlvtangjie. All rights reserved.

import os
import re
import numpy as np
import tensorflow as tf

from .Config import Config

class NetworkVP(object):
    def __init__(self, device, model_name, state_space_size, action_space_size):
        self.device = device
        self.model_name = model_name
        self.state_space_size = state_space_size
        self.num_actions = action_space_size

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    self.vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in self.vars}, max_to_keep=0)
                    self.var_list = []
                    self.update_op = []
                    for id, var in enumerate(self.vars):    
                        self.var_list.append(tf.placeholder(var.dtype, shape = var.get_shape()))
                        self.update_op.append(tf.assign(self.vars[id], self.var_list[id]))


    def _create_graph(self):
        self.x = tf.placeholder(tf.float32, [None, self.state_space_size], name='X')
        self.q_label = tf.placeholder(tf.float32, [None], name='q_label')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        self.weights = tf.placeholder(tf.float32, [None], name='weights')

        self.actions = tf.placeholder(tf.int32, [None], name='actions') 

        # As implemented in A3C paper
        #self.n1 = self.conv2d_layer(self.x, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        #self.n2 = self.conv2d_layer(self.n1, 4, 32, 'conv12', strides=[1, 2, 2, 1])

        # _input = self.n2

        # flatten_input_shape = _input.get_shape()
        # nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]

        # self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.fc1 = self.dense_layer(self.x, 256, 'fc1')
        #self.fc2 = self.dense_layer(self.fc1, 16, 'fc2')
        self.d1 = self.dense_layer(self.fc1, 128, 'dense1')

        self.q_value = self.dense_layer(self.d1, self.num_actions, 'q_value', func=None)

        #self.cost_all = tf.reduce_sum(self.weights*tf.reduce_sum(tf.square(self.q_value-self.q_label), axis=1), axis=0)
        self.cost_all = tf.reduce_sum(self.weights*tf.square(tf.reduce_sum(self.q_value*tf.one_hot(self.actions, self.num_actions), axis=1)-self.q_label))

        if Config.OPTIMIZER == 'Adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate = self.var_learning_rate,
                beta1 = Config.ADAM_BETA1,
                beta2 = Config.ADAM_BETA2,
                epsilon = Config.ADAM_EPSILON
                )
        elif Config.OPTIMIZER == 'RMSProp':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

        self.train_op = self.opt.minimize(self.cost_all)

        self.max_action = tf.argmax(self.q_value, axis=1)

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_n1", self.n1))
        summaries.append(tf.summary.histogram("activation_n2", self.n2))
        summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def train(self, x, q_label, actions, weights):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x:x, self.q_label:q_label, self.actions:actions, self.weights:weights})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def predict(self, x):
        return self.sess.run(self.max_action, feed_dict={self.x:x})

    def calc_q(self, x):
        return self.sess.run(self.q_value, feed_dict={self.x:x})

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))

    def dumps(self):
        return self.sess.run(self.vars)

    def update(self, vars):
        feed_dict = {}
        for id, var in enumerate(vars):
            feed_dict[self.var_list[id]] = var
        self.sess.run(self.update_op, feed_dict = feed_dict)


class DqnNetworks(object):

    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model

    def replace_target(self):
        self.target_model.update(self.model.dumps())

    def update(self, model, is_target=False):
        if is_target:
            self.target_model.update(model)
        else:
            self.model.update(model)

    def predict(self, state):
        return self.model.predict(state)

    def calc_target_q(self, exps):
        next_states = np.array([exp.next_state for exp in exps])
        label_actions = self.model.predict(next_states)
        q_target_values = self.target_model.calc_q(next_states)
        q_target_values = np.array([q_target_values[idx][label_actions[idx]] for idx in range(len(exps))])
        for idx, exp in enumerate(exps):
            if exp.done: q_target_values[idx] = 0
        return q_target_values

    def calc_priority(self, exps):
        states = np.array([exp.state for exp in exps])
        q_values = self.model.calc_q(states)
        # compute q_values
        q_values = np.array([q_values[idx][exps[idx].action] \
            for idx in range(len(exps))])
        # compute q_labels
        q_target_values = self.calc_target_q(exps)
        # compute TD-error
        rewards = np.array([exp.reward for exp in exps])
        td_errors = rewards + Config.DISCOUNT*q_target_values - q_values
        for idx, td_error in enumerate(td_errors):
            exps[idx].priority = abs(td_error)

    def calc_q_labels(self, exps):
        # compute q_target_values
        q_target_values = self.calc_target_q(exps)
        # compute q_labels
        rewards = np.array([exp.reward for exp in exps])
        return rewards + Config.DISCOUNT*q_target_values

    def train(self, exps):
        states = np.array([exp.state for exp in exps])
        q_labels = self.calc_q_labels(exps)
        actions = np.array([exp.action for exp in exps])
        weights = np.array([exp.weight for exp in exps])
        # do train
        self.model.train(states, q_labels, actions, weights)

    def dumps(self, is_target=False):
        if is_target:
            return self.target_model.dumps()
        return self.model.dumps()
