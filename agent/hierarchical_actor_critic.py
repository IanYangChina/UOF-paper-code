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
from agent.utils.networks import Actor, Critic
from agent.utils.replay_buffer import LowLevelHindsightReplayBuffer, HACReplayBuffer
from agent.utils.exploration_strategy import ConstantChance
from agent.utils.plot import smoothed_plot, smoothed_plot_multi_line
from collections import namedtuple

OPT_Tr = namedtuple("transition",
                    ('state', 'desired_goal', 'option', 'next_state', 'achieved_goal', 'option_done', 'reward', 'done'))
ACT_Tr = namedtuple("transition",
                    ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))


class HierarchicalActorCritic(object):
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
            raise UnregisteredEnv("Make sure the env id: {} is correct\nExisting id: {}".format(params.ENV_ID, mg.ids))
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
            'high_level_test_avg_return': [],
            'high_level_test_avg_success_rate': [],
            'high_level_test_avg_goal_specific_return': [],
            'high_level_test_avg_goal_specific_success_rate': [],

            'low_level_train_avg_return_per_epoch': [],
            'low_level_test_avg_return': [],
            'low_level_test_avg_success_rate': [],
        }

        obs = self.env.reset()
        self.obs_dim = obs['observation'].shape[0]
        self.final_goal_dim = obs['desired_goal'].shape[0]
        self.sub_goal_dim = obs['sub_goal'].shape[0]
        # The option space of the env class is not used in this HAC codes
        self.num_sub_goal_set = self.env.option_space.n
        self.normalizer = GoalEnvNormalizer(self.obs_dim, self.sub_goal_dim, self.final_goal_dim,
                                            different_goals=self.env.env.binary_final_goal)
        self.train = params.TRAIN

        """High-Level policies - DDPG"""
        # opt, optor refer to high-level policy
        # In HAC, an option is a sub-goal vector
        # Learning params
        self.opt_batch_size = params.HIGH_LEVEL_BATCH_SIZE
        self.opt_gamma = params.HIGH_LEVEL_GAMMA
        self.opt_tau = params.HIGH_LEVEL_TAU
        self.opt_clip_value = params.HIGH_LEVEL_CLIP_VALUE
        self.opt_optim_steps = params.HIGH_LEVEL_OPTIMIZATION_STEP
        self.option_min = -0.25 * np.ones(self.sub_goal_dim)
        self.option_max = 1.60 * np.ones(self.sub_goal_dim)
        # Get optors
        self.optor = self._get_optor(params.HIGH_LEVEL_MEM_CAPACITY,
                                     params.SEED,
                                     params.HIGH_LEVEL_LEARNING_RATE)

        """Low-Level policies - DDPG"""
        # act, actor refer to low-level policy
        # Learning params
        self.act_batch_size = params.LOW_LEVEL_BATCH_SIZE
        self.act_gamma = params.LOW_LEVEL_GAMMA
        self.act_tau = params.LOW_LEVEL_TAU
        self.act_clip_value = -self.training_time_steps
        self.act_optim_steps = params.LOW_LEVEL_OPTIMIZATION_STEP
        # Get actors
        self.action_dim = self.env.action_space.shape[0]
        self.action_max = self.env.action_space.high
        self.actor = self._get_actor(params.LOW_LEVEL_MEM_CAPACITY,
                                     params.SEED,
                                     params.LOW_LEVEL_LEARNING_RATE)

        # The maximal timesteps allowed for attempting to achieve a sub-goal
        self.H = 10
        self.sub_goal_testing_rate = 0.3
        # Hindsight
        self.hindsight = params.HINDSIGHT_REPLAY
        # Exploration. In HAC, the both levels use the same exploration strategy
        self.exploration = ConstantChance(chance=params.EXPLORATION_ALPHA)
        self.noise_deviation = params.EXPLORATION_SIGMA

    def run(self, render=False):
        for epo in range(self.training_epoch):
            self.env._max_episode_steps = self.training_time_steps
            self._train(epo=epo, render=render)

            print("Epoch: %i Low-level train avg return: " % epo,
                  self.statistic_dict['low_level_train_avg_return_per_epoch'][-1])
            print("Epoch: %i High-level train avg return: " % epo,
                  self.statistic_dict['high_level_train_avg_return_per_epoch'][-1])

            if epo % self.testing_gap == 0:
                self.env._max_episode_steps = self.testing_time_steps
                self.test(render=render)
                print("Epoch: %i Low-level test success rates: " % epo,
                      self.statistic_dict['low_level_test_avg_success_rate'][-1])
                print("Epoch: %i High-level test success rates: " % epo,
                      self.statistic_dict['high_level_test_avg_success_rate'][-1])

            if (epo % self.saving_gap == 0) and (epo != 0):
                self._save_ckpts(epo)

        self._plot_statistics()
        self._save_statistics()

    def _train(self, epo, render):
        low_level_return_cyc = []
        high_level_return_cyc = []
        for cyc in range(self.training_cycle):
            low_level_return = 0
            high_level_return = 0
            for ep in range(self.training_episode):
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
                    opt_reward = -1
                    step_count = 0
                    # sample an option from demonstration or the high-level policy
                    if demon:
                        option = self.env.demonstrator.get_next_goal()
                        self.env.set_goals(sub_goal_ind=option)
                    else:
                        option = self._select_option(op_obs['observation'], op_obs['desired_goal'])
                        self.env.set_goals(sub_goal=option)
                    obs['sub_goal'] = self.env.sub_goal.copy()
                    op_obs['sub_goal'] = self.env.sub_goal.copy()

                    sub_goal_test = False
                    if self.env.np_random.uniform(0, 1) < self.sub_goal_testing_rate:
                        sub_goal_test = True

                    while (step_count < self.H) and (not sub_goal_done) and (not time_done):
                        if render:
                            self.env.render()
                        step_count += 1
                        action = self._select_action(obs['observation'], obs['sub_goal'], test=False)
                        obs_, act_reward, time_done, _ = self.env.step(action)
                        low_level_return += int(act_reward + 1)
                        opt_reward, sub_goal_done, _, _ = self.env.option_step(obs_, option, act_reward,
                                                                               option_is_goal=True)
                        if self.train:
                            self.actor['actor_memory'].store_experience(new_option,
                                                                        obs['observation'], obs['sub_goal'], action,
                                                                        obs_['observation'], obs_['achieved_sub_goal'],
                                                                        act_reward, 1 - int(sub_goal_done))
                        obs = dcp(obs_)
                        new_option = False
                        # store observation data for normalization, this is updated in the self._update() method
                        self.normalizer.store_history(obs_['observation'], obs_['achieved_sub_goal'],
                                                      obs_['achieved_goal'])

                    # True system reward
                    high_level_return += int(opt_reward + 1)
                    # Penalise high level with -H reward if sub goal test fails
                    if sub_goal_test and (not sub_goal_done):
                        opt_reward = int(-self.H)

                    if self.train:
                        self.optor['optor_memory'].store_experience(new_episode,
                                                                    op_obs['observation'],
                                                                    op_obs['desired_goal'],
                                                                    op_obs['sub_goal'],  # sub-goal is option
                                                                    obs['observation'],
                                                                    obs['achieved_goal'],
                                                                    opt_reward,
                                                                    1 - int(time_done))
                    op_obs = dcp(obs)
                    new_episode = False

                self.training_episode_count += 1

            self._update()
            print('Epoch: %i, cycle: %i, avg returns:' % (epo, cyc),
                  'Low-level: {}, High-level: {}'.format(low_level_return / self.training_episode,
                                                         high_level_return / self.training_episode))
            low_level_return_cyc.append(low_level_return / self.training_episode)
            high_level_return_cyc.append(high_level_return / self.training_episode)
        # save return statistics
        self.statistic_dict['low_level_train_avg_return_per_epoch'].append(np.mean(low_level_return_cyc))
        self.statistic_dict['high_level_train_avg_return_per_epoch'].append(np.mean(high_level_return_cyc))

    def test(self, render, testing_episode_per_goal=None, load_network_epoch=None):
        if load_network_epoch is not None:
            self._load_ckpts(epoch=load_network_epoch, intra=True, inter=True)
        if testing_episode_per_goal is None:
            testing_episode_per_goal = self.testing_episode_per_goal
        sub_goal_count = 0
        low_level_avg_return = 0
        low_level_avg_success = 0
        high_level_avg_return = np.zeros(self.num_sub_goal_set)
        high_level_avg_success = np.zeros(self.num_sub_goal_set)
        for goal_ind in range(self.num_sub_goal_set):
            for ep in range(testing_episode_per_goal):
                high_level_return = 0
                obs = self.env.reset()
                time_done = False
                self.env.set_goals(desired_goal_ind=goal_ind)
                obs['desired_goal'] = self.env.goal.copy()
                op_obs = dcp(obs)
                while not time_done:
                    option = self._select_option(op_obs['observation'], op_obs['desired_goal'], test=True)
                    self.env.set_goals(sub_goal=option)
                    obs['sub_goal'] = self.env.sub_goal.copy()
                    op_obs['sub_goal'] = self.env.sub_goal.copy()
                    sub_goal_count += 1
                    step_count = 0
                    sub_goal_done = False
                    low_level_return = 0
                    while (step_count < self.H) and (not sub_goal_done) and (not time_done):
                        if render:
                            self.env.render()
                        step_count += 1
                        action = self._select_action(obs['observation'], obs['sub_goal'], test=True)
                        obs_, act_reward, time_done, act_info = self.env.step(action)
                        opt_reward, sub_goal_done, opt_info, _ = self.env.option_step(obs_, option, act_reward,
                                                                                      option_is_goal=True)
                        obs = dcp(obs_)
                        op_obs = dcp(obs_)
                        # record returns
                        low_level_return += int(act_reward + 1)
                        low_level_avg_return += int(act_reward + 1)
                        high_level_return += int(opt_reward + 1)
                        high_level_avg_return[goal_ind] += int(opt_reward + 1)

                    if low_level_return > 0:
                        low_level_avg_success += 1

                if high_level_return > 0:
                    # refer the episode as a success if the demanded goal was achieved for even just one timestep
                    high_level_avg_success[goal_ind] += 1

        if load_network_epoch is not None:
            print("After %i test episodes per goal:" % testing_episode_per_goal)
            print("----Low-level policy average return: ",
                  low_level_avg_return / sub_goal_count)
            print("----Low-level policy average success rate: ",
                  low_level_avg_success / sub_goal_count)
            print("----High-level policy average returns of each goal: ",
                  high_level_avg_return / testing_episode_per_goal)
            print("----High-level policy average success rates of each goal: ",
                  high_level_avg_success / testing_episode_per_goal)

        # save return statistics
        self.statistic_dict['low_level_test_avg_return'].append(low_level_avg_return / sub_goal_count)
        self.statistic_dict['low_level_test_avg_success_rate'].append(high_level_avg_success / sub_goal_count)
        self.statistic_dict['high_level_test_avg_goal_specific_return'].append(
            high_level_avg_return / testing_episode_per_goal)
        self.statistic_dict['high_level_test_avg_goal_specific_success_rate'].append(
            high_level_avg_success / testing_episode_per_goal)
        self.statistic_dict['high_level_test_avg_return'].append(
            self.statistic_dict['high_level_test_avg_goal_specific_return'][-1].mean())
        self.statistic_dict['high_level_test_avg_success_rate'].append(
            self.statistic_dict['high_level_test_avg_goal_specific_success_rate'][-1].mean())

    def _select_action(self, state, low_level_goal, test=False):
        """
        Parameters
        ----------
        state : np.ndarray
            a env state vector as a numpy array
        low_level_goal : np.ndarray
            a task goal vector as a numpy array
        test : bool
            a boolean flag to indicate whether to add noise in actions

        Returns
        -------
        res : np.ndarray
            an action produced by the actor or at random
        """
        inputs = np.concatenate((state, low_level_goal), axis=0)
        inputs = self.normalizer(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        with T.no_grad():
            action = self.actor['actor_target'](inputs).cpu().detach().numpy()
        if test:
            action = np.clip(action, -self.action_max, self.action_max)
        else:
            if self.env.np_random.uniform(0, 1) < self.exploration():
                action = self.env.np_random.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
            else:
                deviation = self.noise_deviation
                action += deviation * self.env.np_random.randn(self.action_dim)
                action = np.clip(action, -self.action_max, self.action_max)
        return action

    def _select_option(self, state, high_level_goal, test=False):
        """
        Parameters
        ----------
        state : np.ndarray
            a env state vector as a numpy array
        high_level_goal : np.ndarray
            a task final goal vector as a numpy array
        test : bool
            a boolean flag to indicate whether to add noise in actions

        Returns
        -------
        res : np.ndarray
            an option produced by the optor or at random
        """
        inputs = np.concatenate((state, high_level_goal), axis=0)
        inputs = self.normalizer(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        with T.no_grad():
            option = self.optor['optor_target'](inputs).cpu().detach().numpy()
        if test:
            option = np.clip(option, -self.option_max, self.option_max)
        else:
            if self.env.np_random.uniform(0, 1) < self.exploration():
                option = np.random.uniform(-self.option_max, self.option_max, size=(self.sub_goal_dim,))
            else:
                deviation = self.noise_deviation
                option += deviation * np.random.randn(self.sub_goal_dim)
                option = np.clip(option, self.option_min, self.option_max)
        return option

    def _update(self):
        if not self.train:
            return
        self.normalizer.update()
        self._act_learn(self.actor)
        self._opt_learn(self.optor)

    def _act_learn(self, actor_dict):
        if len(self.actor['actor_memory'].episodes) == 0:
            return
        if self.hindsight:
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

            value_target = rewards + done * self.act_gamma * value_
            value_target = T.clamp(value_target, self.act_clip_value, -0.0)

            critic_inputs = T.cat((actor_inputs, actions), dim=1).to(self.device)
            value_estimate = actor_dict['critic'](critic_inputs)
            actor_dict['critic_optimizer'].zero_grad()
            critic_loss = F.mse_loss(value_estimate, value_target.detach())
            critic_loss.backward()
            actor_dict['critic_optimizer'].step()

            actor_dict['actor_optimizer'].zero_grad()
            new_actions = actor_dict['actor'](actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
            new_value = actor_dict['critic'](critic_eval_inputs)

            actor_loss = -new_value.mean()
            actor_loss.backward()
            actor_dict['actor_optimizer'].step()

            self._act_soft_update(actor_dict)

    def _opt_learn(self, optor_dict):
        if len(self.optor['optor_memory'].episodes) == 0:
            return

        if self.hindsight:
            self.optor['optor_memory'].modify_experiences()
        self.optor['optor_memory'].store_episode()

        batch_size = self.opt_batch_size
        if len(self.optor['optor_memory']) < batch_size:
            return

        steps = self.opt_optim_steps
        for i in range(steps):
            batch = self.optor['optor_memory'].sample(batch_size)
            optor_inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
            optor_inputs = self.normalizer(optor_inputs, level='low')
            optor_inputs = T.tensor(optor_inputs, dtype=T.float32).to(self.device)
            options = T.tensor(batch.action, dtype=T.float32).to(self.device)
            # scale the stored options from [-0.25, 1.60] to be within [-1, 1]
            options = (options + 0.25) * 2 / 1.85 - 1
            optor_inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            optor_inputs_ = self.normalizer(optor_inputs_, level='low')
            optor_inputs_ = T.tensor(optor_inputs_, dtype=T.float32).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
            done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

            with T.no_grad():
                options_ = optor_dict['optor'](optor_inputs_)
                critic_inputs_ = T.cat((optor_inputs_, options_), dim=1).to(self.device)
                value_ = optor_dict['critic'](critic_inputs_)

            value_target = rewards + done * self.opt_gamma * value_
            value_target = T.clamp(value_target, self.opt_clip_value, -0.0)

            critic_inputs = T.cat((optor_inputs, options), dim=1).to(self.device)
            value_estimate = optor_dict['critic'](critic_inputs)
            optor_dict['critic_optimizer'].zero_grad()
            critic_loss = F.mse_loss(value_estimate, value_target.detach())
            critic_loss.backward()
            optor_dict['critic_optimizer'].step()

            optor_dict['optor_optimizer'].zero_grad()
            new_options = optor_dict['optor'](optor_inputs)
            critic_eval_inputs = T.cat((optor_inputs, new_options), dim=1).to(self.device)
            new_value = optor_dict['critic'](critic_eval_inputs)
            optor_loss = -new_value
            optor_loss = optor_loss.mean()
            optor_loss.backward()
            optor_dict['optor_optimizer'].step()

    def _act_soft_update(self, actor_dict, act_tau=None):
        if act_tau is None:
            act_tau = self.act_tau

        for target_param, param in zip(actor_dict['critic_target'].parameters(), actor_dict['critic'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - act_tau) + param.data * act_tau
            )
        for target_param, param in zip(actor_dict['actor_target'].parameters(), actor_dict['actor'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - act_tau) + param.data * act_tau
            )

    def _get_optor(self, opt_mem_capacity, seed, option_lr):
        optor_dict = dict()
        optor_dict['optor_memory'] = HACReplayBuffer(opt_mem_capacity, OPT_Tr, seed=seed)

        optor_dict['optor'] = Actor(self.obs_dim + self.sub_goal_dim, self.sub_goal_dim).to(self.device)
        optor_dict['critic'] = Critic(self.obs_dim + self.sub_goal_dim + self.sub_goal_dim).to(self.device)
        optor_dict['optor_optimizer'] = Adam(optor_dict['optor'].parameters(), lr=option_lr)
        optor_dict['critic_optimizer'] = Adam(optor_dict['critic'].parameters(), lr=option_lr)

        return optor_dict

    def _get_actor(self, act_mem_capacity, seed, action_lr):
        actor_dict = dict()
        actor_dict['actor_memory'] = LowLevelHindsightReplayBuffer(act_mem_capacity, ACT_Tr, seed=seed)

        actor_dict['actor'] = Actor(self.obs_dim + self.sub_goal_dim, self.action_dim).to(self.device)
        actor_dict['actor_target'] = Actor(self.obs_dim + self.sub_goal_dim, self.action_dim).to(self.device)
        actor_dict['critic'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
        actor_dict['critic_target'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
        actor_dict['actor_optimizer'] = Adam(actor_dict['actor'].parameters(), lr=action_lr)
        actor_dict['critic_optimizer'] = Adam(actor_dict['critic'].parameters(), lr=action_lr)

        self._act_soft_update(actor_dict, act_tau=1.0)

        return actor_dict

    def _save_ckpts(self, epoch, intra=True, inter=True):
        if intra:
            self._save_networks(actor_dict=self.actor, optor_dict=None, epoch=epoch, ind=None)

        if inter:
            self._save_networks(actor_dict=None, optor_dict=self.optor, epoch=epoch, ind=None)

    def _save_networks(self, actor_dict, optor_dict, epoch, ind=None):
        if ind is None:
            ckpt_mark = str(epoch) + '.pt'
        else:
            ckpt_mark = str(epoch) + 'No' + str(ind) + '.pt'
        if optor_dict is not None:
            T.save(optor_dict['optor'].state_dict(), self.ckpt_path + '/ckpt_optor_epoch' + ckpt_mark)

        if actor_dict is not None:
            T.save(actor_dict['actor_target'].state_dict(), self.ckpt_path + '/ckpt_actor_target_epoch' + ckpt_mark)

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
            self._load_network(actor_dict=None, optor_dict=self.optor, epoch=epoch, ind=None)

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
            optor_dict['optor'].load_state_dict(
                T.load(os.path.join(path, '/ckpt_optor_epoch' + ckpt_mark), map_location=self.device))

        if actor_dict is not None:
            actor_dict['actor_target'].load_state_dict(
                T.load(os.path.join(path, '/ckpt_actor_target_epoch' + ckpt_mark), map_location=self.device))

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
