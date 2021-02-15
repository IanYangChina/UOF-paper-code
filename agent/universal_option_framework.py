import os
import json
import numpy as np
import torch as T
import torch.nn.functional as F
import multigoal_env as mg
from copy import deepcopy as dcp
from gym.error import UnregisteredEnv
from torch.optim.adam import Adam
from agent.utils.normalizer import GoalEnvNormalizer
from agent.utils.networks import Actor, Critic, Mlp
from agent.utils.replay_buffer import HighLevelHindsightReplayBuffer, LowLevelHindsightReplayBuffer
from agent.utils.exploration_strategy import ExpDecayGreedy, ConstantChance, AutoAdjustingConstantChance
from agent.utils.plot import smoothed_plot, smoothed_plot_multi_line
from config.config_uof import Params
from collections import namedtuple

OPT_Tr = namedtuple("transition",
                    ('state', 'desired_goal', 'option', 'next_state', 'achieved_goal', 'option_done', 'reward', 'done'))
ACT_Tr = namedtuple("transition",
                    ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))


class UniversalOptionFramework(object):
    def __init__(self):
        T.manual_seed(Params.SEED)
        self.path = Params.PATH
        self.ckpt_path = Params.CKPT_PATH
        self.data_path = Params.DATA_PATH
        for path in [self.path, self.ckpt_path, self.data_path]:
            os.makedirs(path, exist_ok=True)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        try:
            self.env = mg.make(Params.ENV_ID)
        except UnregisteredEnv:
            raise UnregisteredEnv("Make sure the env id: {} is correct\nExisting id: {}".format(Params.ENV_ID, mg.ids))
        self.env.seed(Params.SEED)

        self.training_time_steps = dcp(self.env._max_episode_steps)
        self.testing_time_steps = Params.TESTING_TIMESTEP

        self.training_epoch = Params.TRAINING_EPOCH
        self.training_cycle = Params.TRAINING_CYCLE
        self.training_episode = Params.TRAINING_EPISODE
        self.testing_episode_per_goal = Params.TESTING_EPISODE
        self.testing_gap = Params.TESTING_GAP
        self.saving_gap = Params.SAVING_GAP

        self.use_abstraction_demonstration = Params.ABSTRACT_DEMONSTRATION
        self.num_demonstrated_episode = int(Params.ABSTRACT_DEMONSTRATION_PROPORTION*self.training_episode)

        self.statistic_dict = {
            'high_level_train_avg_return_per_epoch': [],
            'high_level_train_avg_goal_specific_return_per_epoch': [],
            'high_level_test_avg_return': [],
            'high_level_test_avg_success_rate': [],
            'high_level_test_avg_goal_specific_return': [],
            'high_level_test_avg_goal_specific_success_rate': [],

            'low_level_train_avg_return_per_epoch': [],
            'low_level_train_avg_goal_specific_return_per_epoch': [],
            'low_level_test_avg_return': [],
            'low_level_test_avg_success_rate': [],
            'low_level_test_avg_goal_specific_return': [],
            'low_level_test_avg_goal_specific_success_rate': [],
        }

        obs = self.env.reset()
        self.obs_dim = obs['observation'].shape[0]
        self.final_goal_dim = obs['desired_goal'].shape[0]
        self.sub_goal_dim = obs['sub_goal'].shape[0]
        self.normalizer = GoalEnvNormalizer(self.obs_dim, self.sub_goal_dim, self.final_goal_dim,
                                            self.env.env.binary_final_goal)

        """Inter-Option/High-Level policies - (Double-) DIOL/SMDPDQN"""
        # opt, optor refer to high-level policy
        # Learning params
        self.opt_batch_size = Params.HIGH_LEVEL_BATCH_SIZE
        self.opt_gamma = Params.HIGH_LEVEL_GAMMA
        self.opt_tau = Params.HIGH_LEVEL_TAU
        self.opt_clip_value = Params.HIGH_LEVEL_CLIP_VALUE
        self.opt_optim_steps = Params.HIGH_LEVEL_OPTIMIZATION_STEP
        # Get optors
        self.multi_inter = Params.MULTI_INTER_POLICY
        self.option_num = self.env.option_space.n
        if not self.multi_inter:
            self.optor = self._get_optor(Params.HIGH_LEVEL_MEM_CAPACITY,
                                         Params.SEED,
                                         Params.HIGH_LEVEL_LEARNING_RATE)
            self.current_opt_memory = self.optor['optor_memory']
        else:
            self.num_inter_option_policies = self.option_num
            self.optors = []
            for _ in range(self.num_inter_option_policies):
                self.optors.append(self._get_optor(Params.HIGH_LEVEL_MEM_CAPACITY,
                                                   Params.SEED,
                                                   Params.HIGH_LEVEL_LEARNING_RATE))
            self.current_opt_memory = self.optors[0]['optor_memory']
        self.current_optor_dict = None
        # Exploration
        self.optor_exploration = ExpDecayGreedy(start=Params.HIGH_LEVEL_EXPLORATION_START,
                                                end=Params.HIGH_LEVEL_EXPLORATION_END,
                                                decay=Params.HIGH_LEVEL_EXPLORATION_DECAY)

        """Intra-Option/Low-Level policies - (Double-) DDPG"""
        # act, actor refer to low-level policy
        # Learning Params
        self.act_batch_size = Params.LOW_LEVEL_BATCH_SIZE
        self.act_gamma = Params.LOW_LEVEL_GAMMA
        self.act_tau = Params.LOW_LEVEL_TAU
        self.act_clip_value = Params.LOW_LEVEL_CLIP_VALUE
        self.act_optim_steps = Params.LOW_LEVEL_OPTIMIZATION_STEP
        # Get actors
        self.multi_intra = Params.MULTI_INTRA_POLICY
        self.action_dim = self.env.action_space.shape[0]
        self.action_max = self.env.action_space.high
        if not self.multi_intra:
            self.actor = self._get_actor(Params.LOW_LEVEL_MEM_CAPACITY,
                                         Params.SEED,
                                         Params.LOW_LEVEL_LEARNING_RATE)
            self.current_act_memory = self.actor['actor_memory']
        else:
            self.num_intra_option_policies = self.option_num
            self.actors = []
            for _ in range(self.num_intra_option_policies):
                self.actors.append(self._get_actor(Params.LOW_LEVEL_MEM_CAPACITY,
                                                   Params.SEED,
                                                   Params.LOW_LEVEL_LEARNING_RATE))
            self.current_act_memory = self.actors[0]['actor_memory']
        self.current_actor_dict = None
        # Exploration
        self.aaes_exploration = Params.LOW_LEVEL_EXPLORATION_AAES
        if not self.aaes_exploration:
            self.actor_exploration = ConstantChance(chance=Params.LOW_LEVEL_EXPLORATION_ALPHA)
        else:
            self.actor_exploration = AutoAdjustingConstantChance(goal_num=self.option_num,
                                                                 chance=Params.LOW_LEVEL_EXPLORATION_ALPHA,
                                                                 tau=Params.LOW_LEVEL_EXPLORATION_AAES_TAU)
        self.noise_deviation = Params.LOW_LEVEL_EXPLORATION_SIGMA * self.action_max

    def run(self):
        pass

    def _train(self, test=False):
        pass

    def _test_actor(self):
        pass

    def _test_optor(self):
        pass

    def _select_action(self, state, low_level_goal, step, test=False):
        """
        Parameters
        ----------
        state : np.ndarray
            a env state vector as a numpy array
        low_level_goal : np.ndarray
            a task goal vector as a numpy array
        step : int
            an index of the task step selected by the optor (high-level policy)
        test : bool
            a boolean flag to indicate whether to add noise in actions

        Returns
        -------
        res : np.ndarray
            an action produced by the actor or at random
        """
        if self.multi_intra:
            # use a different actor for different step if the it is a multiple actor experiment
            self.current_actor_dict = self.actors[step]
            current_actor = self.actors[step]['actor_target']
            self.current_act_memory = self.actors[step]['actor_memory']
        else:
            self.current_actor_dict = self.actor
            current_actor = self.actor['actor_target']

        inputs = np.concatenate((state, low_level_goal), axis=0)
        inputs = self.normalizer(inputs, level='low')
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        with T.no_grad():
            action = current_actor(inputs).cpu().detach().numpy()
        if test:
            action = np.clip(action, -self.action_max, self.action_max)
        else:
            eps = self.actor_exploration(step)
            if self.env.np_random.uniform(0, 1) < eps:
                action = self.env.np_random.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
            else:
                if self.aaes_exploration:
                    # noise deviation depends on the last testing success rates of the actor when using aaes
                    deviation = self.noise_deviation * (1 - self.actor_exploration.success_rates[step])
                else:
                    deviation = self.noise_deviation
                action += deviation * self.env.np_random.randn(self.action_dim)
                action = np.clip(action, -self.action_max, self.action_max)
        return action

    def _select_option(self, state, high_level_goal, desired_goal_id, ep=0, test=False):
        """
        Parameters
        ----------
        state : np.ndarray
            a env state vector as a numpy array
        high_level_goal : np.ndarray
            a task final goal vector as a numpy array
        desired_goal_id : int
            the index of the final-goal
        ep : int
            the index of the current episode for decaying exploration
        test : bool
            a boolean flag to indicate whether to add noise in actions

        Returns
        -------
        res : int
            an option produced by the optor or at random
        """
        if self.multi_inter:
            self.current_optor_dict = self.optors[desired_goal_id]
            current_optor = self.optors[desired_goal_id]['optor_target']
            self.current_opt_memory = self.optors[desired_goal_id]['optor_memory']
        else:
            self.current_optor_dict = self.optor
            current_optor = self.optor['optor_target']

        inputs = np.concatenate((state, high_level_goal), axis=0)
        inputs = self.normalizer(inputs, level='high')
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        with T.no_grad():
            option_values = current_optor(inputs)
        if test:
            option = T.argmax(option_values).item()
        else:
            if self.env.np_random.uniform(0, 1) < self.optor_exploration(ep):
                option = self.env.np_random.randint(0, self.option_num - 1)
            else:
                option = T.argmax(option_values).item()
        return option

    def _update(self, act_update=True, opt_update=True, act_hindsight=True):
        if act_update:
            if not self.multi_intra:
                self._act_learn(self.actor, act_hindsight)
            else:
                for actor_dict in self.actors:
                    self.current_act_memory = actor_dict['actor_memory']
                    self._act_learn(actor_dict, act_hindsight)

        if opt_update:
            if not self.multi_inter:
                self._opt_learn(self.optor)
            else:
                for optor_dict in self.optors:
                    self.current_opt_memory = optor_dict['optor_memory']
                    self._opt_learn(optor_dict)

    def _act_learn(self, actor_dict, hindsight=True):
        if len(self.current_act_memory.episodes) == 0:
            return
        if hindsight:
            self.current_act_memory.modify_experiences()
        self.current_act_memory.store_episode()

        batch_size = self.act_batch_size
        if len(self.current_act_memory) < batch_size:
            return

        steps = self.act_optim_steps
        for i in range(steps):
            batch = self.current_act_memory.sample(batch_size)
            actor_inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
            actor_inputs = self.normalizer(actor_inputs, level='low')
            actor_inputs = T.tensor(actor_inputs, dtype=T.float32).to(self.device)
            actions = T.tensor(batch.action, dtype=T.float32).to(self.device)
            actor_inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            actor_inputs_ = self.normalizer(actor_inputs_, level='low')
            actor_inputs_ = T.tensor(actor_inputs_, dtype=T.float32).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
            done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

            with T.no_grad():
                actions_ = actor_dict['actor_target'](actor_inputs_)
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1).to(self.device)
                value_ = actor_dict['critic_target'](critic_inputs_)
                value_2_ = actor_dict['critic_2_target'](critic_inputs_)

            value_ = T.min(value_, value_2_)
            value_target = rewards + done * self.act_gamma * value_
            value_target = T.clamp(value_target, self.act_clip_value, -0.0)

            critic_inputs = T.cat((actor_inputs, actions), dim=1).to(self.device)
            value_estimate = actor_dict['critic'](critic_inputs)
            actor_dict['critic_optimizer'].zero_grad()
            critic_loss = F.mse_loss(value_estimate, value_target.detach())
            critic_loss.backward()
            actor_dict['critic_optimizer'].step()

            value_estimate_2 = actor_dict['critic_2'](critic_inputs)
            actor_dict['critic_2_optimizer'].zero_grad()
            critic_2_loss = F.mse_loss(value_estimate_2, value_target.detach())
            critic_2_loss.backward()
            actor_dict['critic_2_optimizer'].step()

            actor_dict['actor_optimizer'].zero_grad()
            new_actions = actor_dict['actor'](actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
            new_value = actor_dict['critic'](critic_eval_inputs)
            new_value_2 = actor_dict['critic_2'](critic_eval_inputs)
            new_value = T.min(new_value, new_value_2)

            actor_loss = -new_value.mean()
            actor_loss.backward()
            actor_dict['actor_optimizer'].step()

            self._act_soft_update(actor_dict)

    def _opt_learn(self, optor_dict):
        if len(self.current_opt_memory.episodes) == 0:
            return
        self.current_opt_memory.store_episode()

        batch_size = self.opt_batch_size
        if len(self.current_opt_memory) < batch_size:
            return

        steps = self.opt_optim_steps
        for i in range(steps):
            batch = self.current_opt_memory.sample(batch_size)
            inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
            inputs = self.normalizer(inputs, level='high')
            inputs = T.tensor(inputs, dtype=T.float).to(self.device)
            inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            inputs_ = self.normalizer(inputs_, level='high')
            inputs_ = T.tensor(inputs_, dtype=T.float).to(self.device)
            options = T.tensor(batch.option, dtype=T.long).unsqueeze(1).to(self.device)
            option_done = T.tensor(batch.option_done, dtype=T.float).unsqueeze(1).to(self.device)
            episode_done = T.tensor(batch.done, dtype=T.float).unsqueeze(1).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)

            # calculate "option value upon arrival"
            with T.no_grad():
                # the option values in cases when the option is not terminated at the next state
                unchanged_next_option_values = optor_dict['optor_target'](inputs_).gather(1, options)
                # the maximal option value of the next state
                maximal_next_option_values = optor_dict['optor_target'](inputs_).max(1)[0].view(batch_size, 1)
                # option value upon arrival
                next_option_values = option_done * unchanged_next_option_values + (
                        1 - option_done) * maximal_next_option_values

                unchanged_next_option_values_2 = optor_dict['optor_2_target'](inputs_).gather(1, options)
                maximal_next_option_values_2 = optor_dict['optor_2_target'](inputs_).max(1)[0].view(batch_size, 1)
                next_option_values_2 = option_done * unchanged_next_option_values_2 + (
                        1 - option_done) * maximal_next_option_values_2

            next_option_values = T.min(next_option_values, next_option_values_2)
            target_option_values = rewards + episode_done * self.opt_gamma * next_option_values
            target_option_values = T.clamp(target_option_values, self.opt_clip_value, -0.0)

            optor_dict['optor_optimizer'].zero_grad()
            estimated_option_values = optor_dict['optor'](inputs).gather(1, options)
            loss = F.smooth_l1_loss(estimated_option_values, target_option_values.detach())
            loss.backward()
            optor_dict['optor_optimizer'].step()

            optor_dict['optor_2_optimizer'].zero_grad()
            estimated_option_values_2 = optor_dict['optor_2'](inputs).gather(1, options)
            loss_2 = F.smooth_l1_loss(estimated_option_values_2, target_option_values.detach())
            loss_2.backward()
            optor_dict['optor_2_optimizer'].step()

            self._opt_soft_update(optor_dict)

    def _act_soft_update(self, actor_dict, act_tau=None):
        if act_tau is None:
            act_tau = self.act_tau

        for target_param, param in zip(actor_dict['critic_target'].parameters(), actor_dict['critic'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - act_tau) + param.data * act_tau
            )
        for target_param, param in zip(actor_dict['critic_2_target'].parameters(), actor_dict['critic_2'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - act_tau) + param.data * act_tau
            )
        for target_param, param in zip(actor_dict['actor_target'].parameters(), actor_dict['actor'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - act_tau) + param.data * act_tau
            )

    def _opt_soft_update(self, optor_dict, opt_tau=None):
        if opt_tau is None:
            opt_tau = self.opt_tau

        for target_param, param in zip(optor_dict['optor_target'].parameters(), optor_dict['optor'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - opt_tau) + param.data * opt_tau
            )

        for target_param, param in zip(optor_dict['optor_2_target'].parameters(), optor_dict['optor_2'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - opt_tau) + param.data * opt_tau
            )

    def _get_optor(self, opt_mem_capacity, seed, option_lr):
        optor_dict = dict()
        optor_dict['optor_memory'] = HighLevelHindsightReplayBuffer(opt_mem_capacity, OPT_Tr, seed=seed)

        optor_dict['optor'] = Mlp(self.obs_dim + self.final_goal_dim, self.option_num).to(self.device)
        optor_dict['optor_target'] = Mlp(self.obs_dim + self.final_goal_dim, self.option_num).to(self.device)
        optor_dict['optor_optimizer'] = Adam(optor_dict['optor'].parameters(), lr=option_lr)

        optor_dict['optor_2'] = Mlp(self.obs_dim + self.final_goal_dim, self.option_num).to(self.device)
        optor_dict['optor_2_target'] = Mlp(self.obs_dim + self.final_goal_dim, self.option_num).to(self.device)
        optor_dict['optor_2_optimizer'] = Adam(optor_dict['optor_2'].parameters(), lr=option_lr)

        self._opt_soft_update(optor_dict, opt_tau=1.0)

        return optor_dict

    def _get_actor(self, act_mem_capacity, seed, action_lr):
        actor_dict = dict()
        actor_dict['actor_memory'] = LowLevelHindsightReplayBuffer(act_mem_capacity, ACT_Tr, seed=seed)

        actor_dict['actor'] = Actor(self.obs_dim + self.sub_goal_dim, self.action_dim).to(self.device)
        actor_dict['actor_target'] = Actor(self.obs_dim + self.sub_goal_dim, self.action_dim).to(self.device)
        actor_dict['actor_optimizer'] = Adam(actor_dict['actor'].parameters(), lr=action_lr)

        actor_dict['critic'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
        actor_dict['critic_target'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
        actor_dict['critic_optimizer'] = Adam(actor_dict['critic'].parameters(), lr=action_lr)

        actor_dict['critic_2'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
        actor_dict['critic_2_target'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
        actor_dict['critic_2_optimizer'] = Adam(actor_dict['critic_2'].parameters(), lr=action_lr)

        self._act_soft_update(actor_dict, act_tau=1.0)

        return actor_dict

    def _save_ckpts(self, epoch, intra=True, inter=True):
        if intra:
            if not self.multi_intra:
                self._save_networks(actor_dict=self.actor, optor_dict=None, epoch=epoch, ind=None)
            else:
                for _ in range(len(self.actors)):
                    self._save_networks(actor_dict=self.actors[_], optor_dict=None, epoch=epoch, ind=_)
        if inter:
            if not self.multi_inter:
                self._save_networks(actor_dict=None, optor_dict=self.optor, epoch=epoch, ind=None)
            else:
                for _ in range(len(self.optors)):
                    self._save_networks(actor_dict=None, optor_dict=self.optors[_], epoch=epoch, ind=_)

    def _save_networks(self, actor_dict, optor_dict, epoch, ind=None):
        if ind is None:
            ckpt_mark = str(epoch) + '.pt'
        else:
            ckpt_mark = str(epoch) + 'No' + str(ind) + '.pt'
        if optor_dict is not None:
            T.save(optor_dict['optor_target'].state_dict(), self.ckpt_path + '/ckpt_optor_target_epoch' + ckpt_mark)

        if actor_dict is not None:
            T.save(actor_dict['actor_target'].state_dict(), self.ckpt_path + '/ckpt_actor_target_epoch' + ckpt_mark)

    def _load_ckpts(self, epoch, intra=True, inter=True):
        self.normalizer.set_statistics(
            mean=np.load(os.path.join(self.data_path, "act_input_means.npy")),
            var=np.load(os.path.join(self.data_path, "act_input_vars.npy"))
        )
        if intra:
            if not self.multi_intra:
                self._load_network(actor_dict=self.actor, optor_dict=None, epoch=epoch, ind=None)
            else:
                for _ in range(len(self.actors)):
                    self._load_network(actor_dict=self.actors[_], optor_dict=None, epoch=epoch, ind=_)
        if inter:
            if not self.multi_inter:
                self._load_network(actor_dict=None, optor_dict=self.optor, epoch=epoch, ind=None)
            else:
                for _ in range(len(self.optors)):
                    self._load_network(actor_dict=None, optor_dict=self.optors[_], epoch=epoch, ind=_)

    def _load_network(self, actor_dict, optor_dict, epoch, ind=None, specified_path=None):
        if ind is None:
            ckpt_mark = str(epoch) + '.pt'
        else:
            ckpt_mark = str(epoch) + 'No' + str(ind) + '.pt'
        if specified_path is None:
            path = self.ckpt_path
        else:
            path = specified_path
        if optor_dict is not None:
            optor_dict['optor_target'].load_state_dict(
                T.load(path + '/ckpt_optor_target_epoch' + ckpt_mark, map_location=self.device))

        if actor_dict is not None:
            actor_dict['actor_target'].load_state_dict(
                T.load(path + '/ckpt_actor_target_epoch' + ckpt_mark, map_location=self.device))

    def _save_statistics(self):
        np.save(os.path.join(self.data_path, 'act_input_means'), self.normalizer.input_mean_low)
        np.save(os.path.join(self.data_path, 'act_input_vars'), self.normalizer.input_var_low)
        json.dump(self.statistic_dict, open(os.path.join(self.data_path, 'statistics.json'), 'w'))

    def _plot_statistics(self, keys=None, x_labels=None, y_labels=None, window=5):
        legend = dcp(self.env.sub_goal_strs)
        if y_labels is None:
            y_labels = {}
        for key in list(self.statistic_dict.keys()):
            if key not in y_labels.keys():
                if 'loss' in key:
                    label = 'Loss'
                elif 'return' in key:
                    label = 'Return'
                elif 'success' in key:
                    label = 'Success'
                else:
                    label = key
                y_labels.update({key: label})

        if x_labels is None:
            x_labels = {}
        for key in list(self.statistic_dict.keys()):
            if key not in x_labels.keys():
                if ('loss' in key) or ('alpha' in key) or ('entropy' in key) or ('step' in key):
                    label = 'Optimization step'
                elif 'cycle' in key:
                    label = 'Cycle'
                elif 'epoch' in key:
                    label = 'Epoch'
                else:
                    label = 'Episode'
                x_labels.update({key: label})

        if keys is None:
            for key in list(self.statistic_dict.keys()):
                if 'goal_specific' in key:
                    data = np.transpose(np.array(self.statistic_dict[key]))
                    smoothed_plot_multi_line(os.path.join(self.path, key + '.png'), data,
                                             legend=legend, x_label=x_labels[key], y_label=y_labels[key])
                else:
                    smoothed_plot(os.path.join(self.path, key + '.png'), self.statistic_dict[key],
                                  x_label=x_labels[key], y_label=y_labels[key], window=window)
        else:
            for key in keys:
                if 'goal_specific' in key:
                    data = np.transpose(np.array(self.statistic_dict[key]))
                    smoothed_plot_multi_line(os.path.join(self.path, key + '.png'), data,
                                             legend=legend, x_label=x_labels[key], y_label=y_labels[key])
                else:
                    smoothed_plot(os.path.join(self.path, key + '.png'), self.statistic_dict[key],
                                  x_label=x_labels[key], y_label=y_labels[key], window=window)
