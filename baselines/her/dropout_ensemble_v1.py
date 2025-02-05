from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines.her.util import (
    store_args, flatten_grads, transitions_in_episode_batch)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import ReplayBuffer
from baselines.her.q_function import DoubleQFunctionDropout
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DropoutEnsemble:
    @store_args
    def __init__(self, *, input_dims, size_ensemble, use_Q, use_double_network,
                 buffer_size, hidden, layers, batch_size, lr, norm_eps, norm_clip, polyak,
                 max_u, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False, **kwargs):
        """Implementation of value function ensemble.

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            size_ensemble (int): number of value functions in the ensemble
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            batch_size (int): batch size for training
            lr (float): learning rate for the Q (critic) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped in Bellman update
            inference_clip_pos_returns (boolean): whether or not output of the value output used for disagreement should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """

        if self.clip_return is None:
            self.clip_return = np.inf

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['u_2'] = stage_shapes['u']
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.buffer_ph_tf = []
            
            staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            stage_op = staging_tf.put(buffer_ph_tf)

            # store in attribute list
            self.staging_tf = [staging_tf]
            self.buffer_ph_tf.extend(buffer_ph_tf)
            self.stage_ops = [stage_op]

            self._create_double_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['ag'] = (self.T, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def get_values(self, o, ag, g, u=None):
        if u is not None:
            u = self._preprocess_u(u)
        o, g = self._preprocess_og(o, ag, g)
        # values to compute
        vars = [v_function.V_tf for v_function in self.V_fun]
        # feed
        ret_list = []
        for _ in range(self.size_ensemble):
            feed = {}
            feed[self.V_fun[0].o_tf] = o.reshape(-1, self.dimo)
            feed[self.V_fun[0].g_tf] = g.reshape(-1, self.dimg)
            feed[self.V_fun[0].u_tf] = u.reshape(-1, self.dimu)
            ret = self.sess.run(vars, feed_dict=feed)
            # value prediction postprocessing
            ret = np.clip(ret, -self.clip_return, 0. if self.clip_pos_returns else np.inf)
            ret_list.append(ret)
        return np.array(ret_list)

    def _sample_batch(self, policy):
        batch_size_in_transitions = self.batch_size
        transitions = self.buffer.sample(batch_size_in_transitions)

        # label policy
        u = transitions['u']
        u_2 = policy.get_actions(o=transitions['o_2'], ag=transitions['ag_2'], g=transitions['g'])
        transitions['u'] = self._preprocess_u(u)
        transitions['u_2'] = self._preprocess_u(u_2)

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions_batches = [transitions[key][:self.batch_size] for key in self.stage_shapes.keys()]

        return transitions_batches

    def _stage_batch(self, policy):
        batches = self._sample_batch(policy=policy)
        assert len(self.buffer_ph_tf) == len(batches)
        self.sess.run(self.stage_ops, feed_dict=dict(zip(self.buffer_ph_tf, batches)))

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def train(self, policy):
        self._stage_batch(policy=policy)
        V_loss, V_grad = self._grads()
        self._update(V_grad)
        return np.mean(V_loss)

    def _update(self, V_grad):
        self.V_adam[0].update(V_grad[0], self.lr)

    def _create_double_network(self, reuse=False):
        self.sess = tf_util.get_session()

        # running averages, separate from alg (this is within a different scope)
        with tf.variable_scope('o_stats') as vs:
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats'):
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        self.V_loss_tf = [None]
        self.V_fun = [None]
        self.V_target_fun = [None]
        self.V_grads_vars_tf = [None]
        self.V_grad_tf = [None]
        self.V_adam = [None]

        self.init_target_net_op = [None]
        self.update_target_net_op = [None]

        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else self.clip_return)

        # mini-batch sampling
        batch = self.staging_tf[0].get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks (no target network for now)
        with tf.variable_scope(f've_0') as vs:
            v_function = DoubleQFunctionDropout(batch_tf, **self.__dict__)
            vs.reuse_variables()

        with tf.variable_scope(f've_0_target') as vs:
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            target_batch_tf['u'] = batch_tf['u_2']
            v_target_function = DoubleQFunctionDropout(target_batch_tf, **self.__dict__)
            vs.reuse_variables()

        # loss functions
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * v_target_function.V_tf, *clip_range)
        V_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - v_function.V_tf))

        V_scope = f've_0/V'
        V_grads_tf = tf.gradients(V_loss_tf, self._vars(V_scope))
        assert len(self._vars(V_scope)) == len(V_grads_tf)
        V_grads_vars_tf = zip(V_grads_tf, self._vars(V_scope))
        V_grad_tf = flatten_grads(grads=V_grads_tf, var_list=self._vars(V_scope))

        # optimizers
        V_adam = MpiAdam(self._vars(V_scope), scale_grad_by_procs=False)

        # store in attribute lists
        self.V_loss_tf[0] = V_loss_tf
        self.V_fun[0] = v_function
        self.V_target_fun[0] = v_target_function
        self.V_grads_vars_tf[0] = V_grads_vars_tf
        self.V_grad_tf[0] = V_grad_tf
        self.V_adam[0] = V_adam

        # polyak averaging
        main_vars = sum([self._vars(f've_0/V')], [])
        target_vars = sum([self._vars(f've_0_target/V')], [])
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(target_vars, main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]),
                zip(target_vars, main_vars)))

        assert len(main_vars) == len(target_vars)

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _sync_optimizers(self):
        self.V_adam[0].sync()

    def _grads(self):
        """
        returns:
            V_loss (scalar)
            V_grad (list)
        """
        V_loss, V_grad = self.sess.run([
            self.V_loss_tf,
            self.V_grad_tf,
        ])
        return V_loss, V_grad

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch)

        if update_stats:

            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _preprocess_u(self, u):
        return np.clip(u, -self.max_u, self.max_u)

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'V_fun', 'V_target_fun', 'lock', 'env', 'sample_transitions', 'stage_shapes', 'create_v_function']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None
        if 'use_Q' not in state:
            state['use_Q'] = False  # a hack to accomendate old data
        if 'create_v_function' in state:
            del state['create_v_function']

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert (len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path)

