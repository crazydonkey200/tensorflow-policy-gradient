import logging
import tensorflow as tf
import numpy as np
import tf_util


class NNAgent(object):
    # An reinforcement learning agent using vanilla policy gradient.
    def __init__(self, action_space, observation_space,
                 use_rnn=False, use_fnn=False,
                 max_steps=100, discount=0.9, learning_rate=0.01,
                 use_softmax_bias=True,
                 rnn_model='rnn', rnn_hidden_size=32, rnn_num_layers=1,
                 fnn_hidden_sizes=[32, 32],
                 fnn_activation_fns=[tf.nn.relu, tf.nn.relu],
                 fnn_l2_scale=0.0):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('Training'):
                self.train_graph = NNGraph(action_space, observation_space,
                                           learning_rate=learning_rate,
                                           use_softmax_bias=use_softmax_bias,
                                           rnn_model=rnn_model,
                                           rnn_hidden_size=rnn_hidden_size,
                                           rnn_num_layers=rnn_num_layers,
                                           fnn_hidden_sizes=fnn_hidden_sizes,
                                           fnn_activation_fns=fnn_activation_fns,
                                           fnn_l2_scale=fnn_l2_scale,
                                           use_rnn=use_rnn, use_fnn=use_fnn)
                self.inference_graph = self.train_graph
                saver = tf.train.Saver(name='checkpoint_saver')
            init_op = tf.initialize_all_variables()
        self.session = tf.Session(graph=self.graph)
        self.session.run(init_op)
        self.n_actions = self.inference_graph.n_actions
        self.use_rnn = use_rnn
        self.max_steps = max_steps
        self.discount = discount
        if self.use_rnn:
            self.last_state = None
            self.need_reset = False

    def reset(self):
        if self.use_rnn:
            self.need_reset = True

    def get_actions(self, obs):
        "Given a batch of observations, produce a batch of actions."
        if self.use_rnn:
            if self.need_reset:
                self.last_state = self.session.run(
                    self.inference_graph.zero_state,
                    feed_dict={self.inference_graph.obs: obs})
                self.need_reset = False

            probs, self.last_state = self.session.run(
                [self.inference_graph.probs,
                 self.inference_graph.final_state],
                feed_dict={self.inference_graph.obs: obs,
                           self.inference_graph.initial_state:
                           self.last_state,
                           self.inference_graph.seq_lens: [1] * obs.shape[1]})
        else:
            probs = self.session.run(self.inference_graph.probs,
                                     feed_dict={self.inference_graph.obs:
                                                obs})
        
        actions = []
        for prob in probs:
            actions.append(np.random.choice(self.n_actions, 1, p=prob)[0])
        return actions

    def get_action(self, ob):
        "Given one observation, produce one action."
        return self.get_actions(np.array([[ob]]))[0]

    def get_batch(self, env, batch_size=None,
                  total_steps=2000):
        paths = []
        if batch_size is None:
            batch_size = np.inf
        if total_steps is None:
            total_steps = np.inf
        if ((batch_size is None) and
            (total_steps is None)):
            raise ValueError("batch_size and total_steps can't all be None.")
            
        steps = 0
        i = 0
        # for _ in xrange(batch_size):
        while True:
            obs = []
            actions = []
            rewards = []
            paddings = []
            ob = env.reset()
            self.reset()
            for _ in xrange(self.max_steps):
                if isinstance(ob, np.ndarray):
                    ob = np.reshape(ob, [-1])
                action = self.get_action(ob)
                next_ob, reward, done, _ = env.step(action)
                obs.append(ob)
                actions.append(action)
                rewards.append(reward)
                ob = next_ob
                if done:
                    break
            # We need to compute the empirical return for each
            # time step along the trajectory.
            returns = []
            return_so_far = 0.0
            for t in xrange(len(rewards) - 1, -1, -1):
                return_so_far = rewards[t] + self.discount * return_so_far
                returns.append(return_so_far)            
            # The returns are stored backwards in time, so we need to revert it.
            returns = returns[::-1]

            steps += len(actions)
            i += 1
            if ((steps > total_steps) or
                (i > batch_size)):
                break

            paths.append(dict(
                observations=np.array(obs),
                actions=np.array(actions),
                rewards=np.array(rewards),
                returns=np.array(returns),
                ep_len=len(actions)))

        mean_return=np.mean([np.sum(path['rewards']) for path in paths])
        mean_ep_len=np.mean([path['ep_len'] for path in paths])
        return paths, mean_return, mean_ep_len

    def train_batch(self, env, batch_size=None,
                    total_steps=2000):
        paths, mean_return, mean_ep_len = self.get_batch(env, batch_size=batch_size,
                                                         total_steps=total_steps)
        obs_list = [path['observations'] for path in paths]
        actions_list = [path['actions'] for path in paths]
        returns_list = [path['returns'] for path in paths]
        
        if self.use_rnn:
            seq_lens = [path['ep_len'] for path in paths]
            max_ep_len = np.max(seq_lens)
            obs = pad_batch(obs_list, max_ep_len)
            actions = pad_batch(actions_list, max_ep_len)
            returns = pad_batch(returns_list, max_ep_len)
            
            self.last_state = self.session.run(
                self.inference_graph.zero_state,
                feed_dict={self.inference_graph.obs: obs})

            # print actions.shape
            # print obs.shape
            # print self.train_graph.actions.get_shape()
            _, outputs = self.session.run(
                [self.train_graph.train_op, self.train_graph.outputs],
                feed_dict={self.train_graph.obs: obs,
                           self.train_graph.initial_state:
                           self.last_state,
                           self.train_graph.seq_lens: seq_lens,
                           self.train_graph.returns: returns,
                           self.train_graph.actions: actions})
            # print outputs.shape
        else:
            # If not useing RNN, just concatenate every
            # steps into one large list.
            obs = np.array([np.concatenate(obs_list)])
            actions = np.array([np.concatenate(actions_list)])
            returns = np.array([np.concatenate(returns_list)])

            feed_dict = {self.train_graph.actions: actions,
                         self.train_graph.returns: returns,
                         self.train_graph.obs: obs}

            self.session.run([self.train_graph.train_op],
                             feed_dict=feed_dict)

        return mean_return, mean_ep_len


def pad_batch(batch, max_ep_len):
    num_dim = len(batch[0].shape)
    new_batch = []
    for ep in batch:
        # the first dimension, number of steps in the
        # episode is padded to be the same as max_ep_len,
        # the rest dimensions are not touched.
        padded_ep = np.pad(ep, ([(0, max_ep_len - ep.shape[0])] +
                                [(0, 0)] * (num_dim - 1)),
                           'constant', constant_values=0)
        new_batch.append(padded_ep)
    new_batch = np.array(new_batch)
    time_major_batch = np.swapaxes(new_batch, 0, 1)
    return time_major_batch
        

class NNGraph(object):
    def __init__(self, action_space, observation_space,
                 learning_rate=0.001, use_rnn=False, use_fnn=False,
                 max_grad_norm=5.0, rnn_model='lstm',
                 rnn_hidden_size=128,  rnn_num_layers=2,
                 fnn_hidden_sizes=[128, 128],
                 fnn_activation_fns=[tf.nn.relu, tf.nn.relu],
                 fnn_l2_scale=0.0,
                 use_softmax_bias=True,
                 is_training=True):

        self.n_actions = action_space.n
        
        try:
            # observation is an instance of Box.
            self.ob_dim = np.product(observation_space.shape)
            self.is_discrete_ob = False
        except AttributeError:
            # observation space is an instance of Discrete.
            self.ob_dim = observation_space.n
            self.is_discrete_ob = True
            
        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0.0),
            trainable=False)

        if use_rnn:
            shape = [None, None]
        else:
            shape = [1, None]
        
        # Placeholder to feed in observations, actions and returns.
        if self.is_discrete_ob:
            # if observation_space is an instance of Discrete, then
            # should use embeddings to expand it.
            self.obs = tf.placeholder(tf.int64, shape,
                                      name='Observation')
            # Embeddings layers.
            with tf.name_scope('Embeddings'):
                self.embedding = tf.constant(np.eye(self.ob_dim), dtype=tf.float32)
            self.inputs = tf.nn.embedding_lookup(self.embedding, self.obs)
            input_size = self.ob_dim
        else:
            # if observation_space is an instance of Box,
            # then just use itself.
            self.obs = tf.placeholder(tf.float32,
                                      shape + [self.ob_dim], # list(observation_space.shape),
                                      name='Observation')

            self.inputs = self.obs # tf.reshape(self.obs, [-1, ])
            input_size = self.ob_dim

        if use_fnn:
            self.processed_inputs = tf_util.create_fnn_ops(self.inputs, input_size,
                                                        hidden_sizes=fnn_hidden_sizes,
                                                        activation_fns=fnn_activation_fns,
                                                        l2_scale=fnn_l2_scale)
            self.processed_input_size = fnn_hidden_sizes[-1]
        else:
            self.processed_inputs = self.inputs
            self.processed_input_size = self.ob_dim

        if use_rnn:
            with tf.name_scope('Dynamic_RNN'):
                in_ops, out_ops = tf_util.create_rnn_ops(self.processed_inputs,
                                                      self.processed_input_size,
                                                      rnn_model=rnn_model,
                                                      hidden_size=rnn_hidden_size,
                                                      num_layers=rnn_num_layers)
            self.zero_state = in_ops[0]
            self.initial_state = in_ops[1]
            self.seq_lens = in_ops[2]
            self.outputs, self.final_state = out_ops
            output_dim = rnn_hidden_size
        else:
            self.outputs = self.processed_inputs
            output_dim = self.processed_input_size

        flat_outputs = tf.reshape(self.outputs, [-1, output_dim])

        self.logits, self.probs = tf_util.create_softmax_ops(flat_outputs,
                                                          output_dim, self.n_actions,
                                                          use_softmax_bias=use_softmax_bias)

        with tf.name_scope('Training'):
            # actions and returns.
            self.actions = tf.placeholder(tf.int64,
                                          [None, None],
                                          name='actions')

            self.returns = tf.placeholder(tf.float32,
                                          [None, None],
                                          name='returns')

            flat_actions = tf.reshape(self.actions, [-1])
            flat_returns = tf.reshape(self.returns, [-1])

            if use_rnn:
                self.mean_weighted_neg_ll = tf_util.rnn_weighted_neg_ll(
                    self.logits, flat_actions, flat_returns, self.seq_lens)
            else:
                self.mean_weighted_neg_ll = tf_util.weighted_neg_ll(
                    self.logits, flat_actions, flat_returns)

            with tf.name_scope('Optimization'):
                # self.learning_rate = tf.constant(learning_rate)
                self.learning_rate = tf.train.exponential_decay(
                    learning_rate, self.global_step, 100, 1.0, staircase=True)

                tvars = tf.trainable_variables()
                
                # print [tvar.name for tvar in tvars]
                self.model_size = np.sum([np.product(tvar.get_shape().as_list())
                                          for tvar in tvars])
                print('model size is %s' % self.model_size)
                grads = tf.gradients(self.mean_weighted_neg_ll, tvars)
                self.grads = grads

                if use_rnn:
                    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

                # self.grad_1 = grads[0]
                # self.grad_2 = grads[1]

                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                    
                # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate)
                # optimizer = tf.train.AdamOptimizer(self.learning_rate)

                self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                          global_step=self.global_step)


