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
from collections import namedtuple

np.set_printoptions(precision=3)
OPT_Tr = namedtuple("transition",
                    ('state', 'desired_goal', 'option', 'next_state', 'achieved_goal', 'option_done', 'reward', 'done'))
ACT_Tr = namedtuple("transition",
                    ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))


class UniversalOptionFramework(object):
    def __init__(self, params):
        T.manual_seed(params.SEED)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.path = params.PATH
        self.ckpt_path = params.CKPT_PATH
        self.data_path = params.DATA_PATH
        for path in [self.path, self.ckpt_path, self.data_path]:
            os.makedirs(path, exist_ok=True)

        self.load_pre_trained_policy = params.LOAD_PER_TRAIN_POLICY
        self.pre_trained_path = params.PRE_TRAIN_PATH
        self.pre_trained_ckpt_path = params.PRE_TRAIN_CKPT_PATH
        self.pre_trained_data_path = params.PRE_TRAIN_DATA_PATH

        try:
            self.env = mg.make(params.ENV_NAME)
        except UnregisteredEnv:
            raise UnregisteredEnv("Make sure the env name: {} is correct\nExisting env: {}".format(params.ENV_NAME, mg.ids))
        self.env.seed(params.SEED)

        self.training_time_steps = dcp(self.env._max_episode_steps)
        self.testing_time_steps = params.TESTING_TIMESTEP

        self.training_episode_count = 0
        self.training_epoch = params.TRAINING_EPOCH
        self.training_cycle = params.TRAINING_CYCLE
        self.training_episode = params.TRAINING_EPISODE
        self.testing_episode_per_goal = params.TESTING_EPISODE
        self.testing_gap = params.TESTING_GAP
        self.saving_gap = params.SAVING_GAP

        self.use_abstraction_demonstration = params.ABSTRACT_DEMONSTRATION
        self.num_demonstrated_episode = int(params.ABSTRACT_DEMONSTRATION_PROPORTION * self.training_episode)

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
            'low_level_goal_specific_exploration_probability': []
        }

        obs = self.env.reset()
        self.obs_dim = obs['observation'].shape[0]
        self.final_goal_dim = obs['desired_goal'].shape[0]
        self.sub_goal_dim = obs['sub_goal'].shape[0]
        self.normalizer = GoalEnvNormalizer(self.obs_dim, self.sub_goal_dim, self.final_goal_dim,
                                            different_goals=self.env.env.binary_final_goal)

        """Inter-Option/High-Level policies - DIOL"""
        # opt, optor refer to high-level policy
        # Learning params
        self.opt_train = params.HIGH_LEVEL_TRAIN
        self.opt_batch_size = params.HIGH_LEVEL_BATCH_SIZE
        self.opt_gamma = params.HIGH_LEVEL_GAMMA
        self.opt_tau = params.HIGH_LEVEL_TAU
        self.opt_clip_value = params.HIGH_LEVEL_CLIP_VALUE
        self.opt_optim_steps = params.HIGH_LEVEL_OPTIMIZATION_STEP
        # Get optors
        self.multi_inter = params.MULTI_INTER_POLICY
        self.option_num = self.env.option_space.n
        if not self.multi_inter:
            self.optor = self._get_optor(params.HIGH_LEVEL_MEM_CAPACITY,
                                         params.SEED,
                                         params.HIGH_LEVEL_LEARNING_RATE)
            self.current_opt_memory = self.optor['optor_memory']
        else:
            self.num_inter_option_policies = self.option_num
            self.optors = []
            for _ in range(self.num_inter_option_policies):
                self.optors.append(self._get_optor(params.HIGH_LEVEL_MEM_CAPACITY,
                                                   params.SEED,
                                                   params.HIGH_LEVEL_LEARNING_RATE))
            self.current_opt_memory = self.optors[0]['optor_memory']
        self.current_optor_dict = None
        # Exploration
        self.optor_exploration = ExpDecayGreedy(start=params.HIGH_LEVEL_EXPLORATION_START,
                                                end=params.HIGH_LEVEL_EXPLORATION_END,
                                                decay=params.HIGH_LEVEL_EXPLORATION_DECAY)

        """Intra-Option/Low-Level policies - Double-DDPG"""
        # act, actor refer to low-level policy
        # Learning params
        self.act_train = params.LOW_LEVEL_TRAIN
        self.act_batch_size = params.LOW_LEVEL_BATCH_SIZE
        self.act_gamma = params.LOW_LEVEL_GAMMA
        self.act_tau = params.LOW_LEVEL_TAU
        self.act_clip_value = -self.training_time_steps
        self.act_optim_steps = params.LOW_LEVEL_OPTIMIZATION_STEP
        self.act_hindsight = params.LOW_LEVEL_HINDSIGHT_REPLAY
        # Get actors
        self.action_dim = self.env.action_space.shape[0]
        self.action_max = self.env.action_space.high
        self.actor = self._get_actor(params.LOW_LEVEL_MEM_CAPACITY,
                                     params.SEED,
                                     params.LOW_LEVEL_LEARNING_RATE)
        # Exploration
        self.aaes_exploration = params.LOW_LEVEL_EXPLORATION_AAES
        if not self.aaes_exploration:
            self.actor_exploration = ConstantChance(chance=params.LOW_LEVEL_EXPLORATION_ALPHA)
        else:
            self.actor_exploration = AutoAdjustingConstantChance(goal_num=self.option_num,
                                                                 chance=params.LOW_LEVEL_EXPLORATION_ALPHA,
                                                                 tau=params.LOW_LEVEL_EXPLORATION_AAES_TAU)
        self.noise_deviation = params.LOW_LEVEL_EXPLORATION_SIGMA

    def run(self, render=False):
        for epo in range(self.training_epoch):
            self.env._max_episode_steps = self.training_time_steps
            self._train(epo=epo, render=render)

            print("Epoch: %i Low-level train avg return: " % epo,
                  self.statistic_dict['low_level_train_avg_goal_specific_return_per_epoch'][-1])
            print("Epoch: %i High-level train avg return: " % epo,
                  self.statistic_dict['high_level_train_avg_goal_specific_return_per_epoch'][-1])

            if epo % self.testing_gap == 0:
                if self.act_train:
                    self.test_actor(render=render)
                    print("Epoch: %i Low-level test success rates: " % epo,
                          self.statistic_dict['low_level_test_avg_goal_specific_success_rate'][-1])

                    # update aaes parameters if demanded
                    if self.aaes_exploration:
                        self.actor_exploration.update_success_rates(
                            self.statistic_dict['low_level_test_avg_goal_specific_success_rate'][-1])
                        self.statistic_dict['low_level_goal_specific_exploration_probability'].append(
                            self.actor_exploration.chance.copy())
                        print("Epoch: %i Current chances: " % epo,
                              self.statistic_dict['low_level_goal_specific_exploration_probability'][-1])

                if self.opt_train:
                    self.test_optor(render=render)
                    print("Epoch: %i High-level test success rates: " % epo,
                          self.statistic_dict['high_level_test_avg_goal_specific_success_rate'][-1])

            if (epo % self.saving_gap == 0) and (epo != 0):
                self._save_ckpts(epo)

        self._plot_statistics()
        self._save_statistics()

    def _train(self, epo, render):
        low_level_goal_specific_avg_return = []
        high_level_goal_specific_avg_return = []
        for cyc in range(self.training_cycle):
            for ep in range(self.training_episode):
                low_level_goal_specific_return = np.zeros(self.option_num)
                high_level_goal_specific_return = np.zeros(self.option_num)
                new_episode = True
                time_done = False
                demon = False
                obs = self.env.reset()
                op_obs = dcp(obs)
                # reset the demonstrator with the final goal index
                final_goal_ind = self.env.final_goal_strs.index(self.env.goal_str)
                self.env.demonstrator.reset_with_final_goal_index(final_goal_ind)
                # decide whether to use demonstrations in the current episode
                if self.use_abstraction_demonstration and (self.num_demonstrated_episode + ep) >= self.training_episode:
                    demon = True
                while not time_done:
                    sub_goal_done = False
                    new_option = True
                    # sample an option from demonstration or the high-level policy
                    if demon:
                        option = self.env.demonstrator.get_next_goal()
                    else:
                        option = self._select_option(op_obs['observation'], op_obs['desired_goal'], final_goal_ind,
                                                     ep=self.training_episode_count)
                    self.env.set_goals(sub_goal_ind=option)
                    obs['sub_goal'] = self.env.sub_goal.copy()
                    op_obs['sub_goal'] = self.env.sub_goal.copy()
                    while (not sub_goal_done) and (not time_done):
                        if render:
                            self.env.render()
                        action = self._select_action(obs['observation'], obs['sub_goal'], option,
                                                     test=(not self.act_train))
                        obs_, act_reward, time_done, _ = self.env.step(action)
                        low_level_goal_specific_return[option] += int(act_reward + 1)
                        opt_reward, sub_goal_done, _, _ = self.env.option_step(obs_, option, act_reward)
                        high_level_goal_specific_return[final_goal_ind] += int(opt_reward + 1)
                        if self.act_train:
                            self.actor['actor_memory'].store_experience(new_option,
                                                                        obs['observation'], obs['sub_goal'], action,
                                                                        obs_['observation'], obs_['achieved_sub_goal'],
                                                                        act_reward, 1 - int(sub_goal_done))
                        if self.opt_train:
                            self.current_opt_memory.store_experience(new_episode,
                                                                     op_obs['observation'],
                                                                     op_obs['desired_goal'], option,
                                                                     obs_['observation'],
                                                                     obs_['achieved_goal'],
                                                                     1 - int(sub_goal_done),
                                                                     opt_reward, 1 - int(time_done))
                        obs = dcp(obs_)
                        new_option = False
                        op_obs = dcp(obs_)
                        new_episode = False
                        # store observation data for normalization, this is updated in the self._update() method
                        self.normalizer.store_history(obs_['observation'], obs_['achieved_sub_goal'],
                                                      obs_['achieved_goal'])
                self.training_episode_count += 1
                low_level_goal_specific_avg_return.append(low_level_goal_specific_return)
                high_level_goal_specific_avg_return.append(high_level_goal_specific_return)
            self._update()
            print('Epoch: %i, cycle: %i, avg returns:\n' % (epo, cyc),
                  'Low-level: {}\nHigh-level: {}'.format(
                      np.mean(low_level_goal_specific_avg_return[-self.training_episode:], axis=0),
                      np.mean(high_level_goal_specific_avg_return[-self.training_episode:], axis=0)
                  ))
        # save return statistics
        self.statistic_dict['low_level_train_avg_goal_specific_return_per_epoch'].append(
            np.mean(low_level_goal_specific_avg_return, axis=0))
        self.statistic_dict['high_level_train_avg_goal_specific_return_per_epoch'].append(
            np.mean(high_level_goal_specific_avg_return, axis=0))
        self.statistic_dict['low_level_train_avg_return_per_epoch'].append(
            np.mean(self.statistic_dict['low_level_train_avg_goal_specific_return_per_epoch'][-1]))
        self.statistic_dict['high_level_train_avg_return_per_epoch'].append(
            np.mean(self.statistic_dict['high_level_train_avg_goal_specific_return_per_epoch'][-1]))

    def test_actor(self, render, testing_episode_per_goal=None, load_network_epoch=None):
        self.env._max_episode_steps = self.testing_time_steps
        if load_network_epoch is not None:
            self._load_ckpts(epoch=load_network_epoch, intra=True)
        if testing_episode_per_goal is None:
            testing_episode_per_goal = self.testing_episode_per_goal
        avg_return = np.zeros(self.option_num)
        avg_success = np.zeros(self.option_num)
        for goal_ind in range(self.option_num):
            for ep in range(testing_episode_per_goal):
                ep_return = 0
                obs = self.env.reset()
                time_done = False
                self.env.demonstrator.reset_with_final_goal_index(goal_ind)
                self.env.set_goals(desired_goal_ind=goal_ind)
                obs['desired_goal'] = self.env.goal.copy()
                while not time_done:
                    sub_goal_ind = self.env.demonstrator.get_next_goal()
                    self.env.set_goals(sub_goal_ind=sub_goal_ind)
                    obs['sub_goal'] = self.env.sub_goal.copy()
                    goal_done = False
                    while (not goal_done) and (not time_done):
                        if render:
                            self.env.render()
                        action = self._select_action(obs['observation'], obs['sub_goal'], sub_goal_ind, test=True)
                        obs_, act_reward, time_done, act_info = self.env.step(action)
                        opt_reward, goal_done, opt_info, _ = self.env.option_step(obs_, sub_goal_ind, act_reward)
                        obs = dcp(obs_)
                        avg_return[goal_ind] += int(opt_reward + 1)
                        ep_return += int(opt_reward + 1)
                if ep_return > 0:
                    # refer the episode as a success if the demanded goal was achieved for even just one timestep
                    avg_success[goal_ind] += 1

        if load_network_epoch is not None:
            print("After %i test episodes per goal:" % testing_episode_per_goal)
            print("----Low-level policy average returns of each goal: ", avg_return / testing_episode_per_goal)
            print("----Low-level policy average success rates of each goal: ", avg_success / testing_episode_per_goal)

        # save return statistics
        self.statistic_dict['low_level_test_avg_goal_specific_return'].append(
            avg_return / testing_episode_per_goal)
        self.statistic_dict['low_level_test_avg_goal_specific_success_rate'].append(
            avg_success / testing_episode_per_goal)
        self.statistic_dict['low_level_test_avg_return'].append(
            self.statistic_dict['low_level_test_avg_goal_specific_return'][-1].mean())
        self.statistic_dict['low_level_test_avg_success_rate'].append(
            self.statistic_dict['low_level_test_avg_goal_specific_success_rate'][-1].mean())

    def test_optor(self, render, testing_episode_per_goal=None, load_network_epoch=None, intervention=False):
        self.env._max_episode_steps = self.testing_time_steps
        if load_network_epoch is not None:
            self._load_ckpts(epoch=load_network_epoch, intra=True, inter=True)
        if testing_episode_per_goal is None:
            testing_episode_per_goal = self.testing_episode_per_goal
        avg_return = np.zeros(self.option_num)
        avg_success = np.zeros(self.option_num)
        for goal_ind in range(self.option_num):
            for ep in range(testing_episode_per_goal):
                ep_return = 0
                obs = self.env.reset()
                time_done = False
                self.env.set_goals(desired_goal_ind=goal_ind)
                obs['desired_goal'] = self.env.goal.copy()
                op_obs = dcp(obs)
                while not time_done:
                    option = self._select_option(op_obs['observation'], op_obs['desired_goal'], goal_ind, test=True)
                    self.env.set_goals(sub_goal_ind=option)
                    obs['sub_goal'] = self.env.sub_goal.copy()
                    sub_goal_done = False
                    while (not sub_goal_done) and (not time_done):
                        if render:
                            self.env.render()
                        action = self._select_action(obs['observation'], obs['sub_goal'], option, test=True)
                        obs_, act_reward, time_done, act_info = self.env.step(action)
                        opt_reward, sub_goal_done, opt_info, _ = self.env.option_step(obs_, option, act_reward)
                        obs = dcp(obs_)
                        op_obs = dcp(obs_)
                        avg_return[goal_ind] += int(opt_reward + 1)
                        ep_return += int(opt_reward + 1)
                        if intervention:
                            sub_goal_done = True
                if ep_return > 0:
                    # refer the episode as a success if the demanded goal was achieved for even just one timestep
                    avg_success[goal_ind] += 1

        if load_network_epoch is not None:
            print("After %i test episodes per goal:" % testing_episode_per_goal)
            print("----High-level policy average returns of each goal: ", avg_return / testing_episode_per_goal)
            print("----High-level policy average success rates of each goal: ", avg_success / testing_episode_per_goal)

        # save return statistics
        self.statistic_dict['high_level_test_avg_goal_specific_return'].append(
            avg_return / testing_episode_per_goal)
        self.statistic_dict['high_level_test_avg_goal_specific_success_rate'].append(
            avg_success / testing_episode_per_goal)
        self.statistic_dict['high_level_test_avg_return'].append(
            self.statistic_dict['high_level_test_avg_goal_specific_return'][-1].mean())
        self.statistic_dict['high_level_test_avg_success_rate'].append(
            self.statistic_dict['high_level_test_avg_goal_specific_success_rate'][-1].mean())

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
        inputs = np.concatenate((state, low_level_goal), axis=0)
        inputs = self.normalizer(inputs, level='low')
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        with T.no_grad():
            action = self.actor['actor_target'](inputs).cpu().detach().numpy()
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

    def _update(self):
        if self.act_train:
            self.normalizer.update()
            self._act_learn(self.actor)

        if self.opt_train:
            self.normalizer.update()
            if not self.multi_inter:
                self._opt_learn(self.optor)
            else:
                for optor_dict in self.optors:
                    self.current_opt_memory = optor_dict['optor_memory']
                    self._opt_learn(optor_dict)

    def _act_learn(self, actor_dict):
        if len(self.actor['actor_memory'].episodes) == 0:
            return
        if self.act_hindsight:
            self.actor['actor_memory'].modify_experiences()
        self.actor['actor_memory'].store_episode()

        batch_size = self.act_batch_size
        if len(self.actor['actor_memory']) < batch_size:
            return

        steps = self.act_optim_steps
        for i in range(steps):
            batch = self.actor['actor_memory'].sample(batch_size)
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
            self._save_networks(actor_dict=self.actor, optor_dict=None, epoch=epoch, ind=None)

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
            T.save(optor_dict['optor_target'].state_dict(), os.path.join(self.ckpt_path, 'ckpt_optor_target_epoch' + ckpt_mark))

        if actor_dict is not None:
            T.save(actor_dict['actor_target'].state_dict(), os.path.join(self.ckpt_path, 'ckpt_actor_target_epoch' + ckpt_mark))

    def _load_ckpts(self, epoch, intra=False, inter=False):
        if self.load_pre_trained_policy:
            path = self.pre_trained_data_path
        else:
            path = self.data_path
        self.normalizer.set_statistics(
            mean=np.loadtxt(os.path.join(path, "act_input_means.dat")),
            var=np.loadtxt(os.path.join(path, "act_input_vars.dat"))
        )
        if intra:
            self._load_network(actor_dict=self.actor, optor_dict=None, epoch=epoch, ind=None)

        if inter:
            if not self.multi_inter:
                self._load_network(actor_dict=None, optor_dict=self.optor, epoch=epoch, ind=None)
            else:
                for _ in range(len(self.optors)):
                    self._load_network(actor_dict=None, optor_dict=self.optors[_], epoch=epoch, ind=_)

    def _load_network(self, actor_dict, optor_dict, epoch, ind=None):
        if self.load_pre_trained_policy:
            path = self.pre_trained_ckpt_path
        else:
            path = self.ckpt_path
        if ind is None:
            ckpt_mark = str(epoch) + '.pt'
        else:
            ckpt_mark = str(epoch) + 'No' + str(ind) + '.pt'
        if optor_dict is not None:
            optor_dict['optor_target'].load_state_dict(
                T.load(os.path.join(path, 'ckpt_optor_target_epoch'+ckpt_mark), map_location=self.device))

        if actor_dict is not None:
            actor_dict['actor_target'].load_state_dict(
                T.load(os.path.join(path, 'ckpt_actor_target_epoch'+ckpt_mark), map_location=self.device))

    def _save_statistics(self):
        np.savetxt(os.path.join(self.data_path, 'act_input_means.dat'), self.normalizer.input_mean_low)
        np.savetxt(os.path.join(self.data_path, 'act_input_vars.dat'), self.normalizer.input_var_low)
        for key in self.statistic_dict.keys():
            self.statistic_dict[key] = np.array(self.statistic_dict[key]).tolist()
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
                elif 'episode' in key:
                    label = 'Episode'
                else:
                    label = 'Epoch'
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
