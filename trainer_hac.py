import os
import numpy as np
from copy import deepcopy as dcp
from collections import namedtuple
import multigoal_env  # import this one for env registration
from gym import make
from gym.error import UnregisteredEnv
from utils.plot import smoothed_plot, smoothed_plot_multi_line
from agent.hac import HierarchicalActorCritic
from agent.utils.demonstrator import Demonstrator


class Trainer(object):
    def __init__(self, path, env_name='TwoObjectOneOrderBinaryHighLvGoal-v0', seed=0,
                 load_act_init_input=False, load_opt_init_input=False, init_input_path=None,
                 demonstrations=None, use_demonstration_in_training=False, demonstrated_episode_proportion=0.75,
                 act_hindsight=True, act_clip_value=None,
                 opt_hindsight=True, opt_clip_value=None,
                 training_epoch=201, training_cycle=50, training_episode=16,
                 testing_episode_per_goal=30, testing_time_steps=50, testing_gap=1, saving_gap=50):
        np.set_printoptions(precision=3)
        self.path = path
        try:
            self.env = make(env_name)
        except UnregisteredEnv:
            raise UnregisteredEnv(
                "Make sure the env id: {} is correct\nExisting id: {}".format(env_name, multigoal_env.ids))
        self.env.seed(seed)

        Tr = namedtuple("transition",
                        ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))

        self.training_time_steps = dcp(self.env._max_episode_steps)
        self.testing_time_steps = testing_time_steps

        self.training_epoch = training_epoch
        self.training_cycle = training_cycle
        self.training_episode = training_episode
        self.testing_episode_per_goal = testing_episode_per_goal
        self.testing_gap = testing_gap
        self.saving_gap = saving_gap

        # the maximal timesteps allowed for attempting to achieve a sub-goal
        self.H = 10
        self.sub_goal_testing_rate = 0.3

        obs = self.env.reset()
        if load_act_init_input:
            if init_input_path is None:
                init_input_path = self.path
            act_init_input_means = np.loadtxt(init_input_path + "/data/act_input_means.dat")
            act_init_input_vars = np.loadtxt(init_input_path + "/data/act_input_vars.dat")
        else:
            act_init_input_means = None
            act_init_input_vars = None
        if load_opt_init_input:
            if init_input_path is None:
                init_input_path = self.path
            opt_init_input_means = np.loadtxt(init_input_path + "/data/opt_input_means.dat")
            opt_init_input_vars = np.loadtxt(init_input_path + "/data/opt_input_vars.dat")
        else:
            opt_init_input_means = None
            opt_init_input_vars = None

        env_params = {'act_init_input_means': act_init_input_means,
                      'act_init_input_vars': act_init_input_vars,
                      'opt_init_input_means': opt_init_input_means,
                      'opt_init_input_vars': opt_init_input_vars,
                      'obs_dims': obs['observation'].shape[0],
                      'goal_dims': obs['desired_goal'].shape[0],
                      'sub_goal_dims': obs['sub_goal'].shape[0],
                      'action_dims': self.env.action_space.shape[0],
                      'action_max': self.env.action_space.high,
                      'option_num': self.env.option_space.n,
                      'sub_goals': self.env.sub_goal_strs,
                      'different_goals': self.env.env.binary_final_goal
                      }
        self.option_num = self.env.option_space.n
        self.act_hindsight = act_hindsight
        self.opt_hindsight = opt_hindsight
        if act_clip_value is not None:
            self.act_clip_value = act_clip_value
        else:
            self.act_clip_value = -self.H
        if opt_clip_value is not None:
            self.opt_clip_value = opt_clip_value
        else:
            self.opt_clip_value = -self.H
        self.agent = HierarchicalActorCritic(env_params=env_params, tr=Tr, path=self.path, seed=seed,
                                             act_clip_value=self.act_clip_value,
                                             opt_clip_value=self.opt_clip_value)

        if demonstrations is None:
            """Example sequences of steps, e.g.: 
               demonstrations = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]
            """
            raise ValueError("Demonstrations need to be specified")
        self.demonstrator = Demonstrator(demonstrations)
        self.use_demonstration_in_training = use_demonstration_in_training
        self.demonstrated_episode = int(demonstrated_episode_proportion*self.training_episode)

        self.final_goal_train_mean_normalized_score = []
        self.final_goal_test_mean_avg_return = []
        self.final_goal_test_mean_success_rate = []
        self.final_goal_test_inpd_avg_return = []
        self.final_goal_test_inpd_success_rate = []

        self.sub_goal_train_mean_normalized_score = []
        self.sub_goal_test_mean_avg_return = []
        self.sub_goal_test_mean_success_rate = []

    def run(self, training_render=False, testing_render=False):
        for epo in range(self.training_epoch):
            self.env._max_episode_steps = self.training_time_steps
            for cyc in range(self.training_cycle):
                self.train(epo=epo, cyc=cyc, render=training_render)

            if epo % self.testing_gap == 0:
                self.test(render=testing_render)
                print("Low-level test success rates: ", self.sub_goal_test_mean_success_rate[-1])
                print("High-level test inpd success rates: ", self.final_goal_test_inpd_success_rate[-1])
                print("High-level test mean success rates: ", self.final_goal_test_mean_success_rate[-1])

            if (epo % self.saving_gap == 0) and (epo != 0):
                self._save_ckpts(epo)

        self._plot()
        self._save_numpy_to_txt()

    def train(self, epo=0, cyc=0, render=False):
        high_level_score = 0
        low_level_score = 0
        for ep in range(self.training_episode):
            new_episode = True
            obs = self.env.reset()
            time_done = False

            demon = False
            final_goal_ind = self.env.final_goal_strs.index(self.env.goal_str)
            self.demonstrator.reset_with_final_goal_index(final_goal_ind)
            if self.use_demonstration_in_training and (self.demonstrated_episode + ep) >= self.training_episode:
                demon = True

            op_obs = dcp(obs)
            while not time_done:
                sub_goal_done = False
                new_option = True
                opt_reward = -1
                step_count = 0

                if demon:
                    sub_goal_ind = self.demonstrator.get_next_goal()
                    self.env.set_goals(sub_goal_ind=sub_goal_ind)
                else:
                    sub_goal = self.agent.select_option(op_obs['observation'], op_obs['desired_goal'])
                    self.env.set_goals(sub_goal=sub_goal)
                obs['sub_goal'] = self.env.sub_goal.copy()
                op_obs['sub_goal'] = self.env.sub_goal.copy()
                sub_goal_test = False
                if np.random.uniform(0, 1) < self.sub_goal_testing_rate:
                    sub_goal_test = True

                while (step_count < self.H) and (not sub_goal_done) and (not time_done):
                    step_count += 1
                    action = self.agent.select_action(obs['observation'], obs['sub_goal'], test=sub_goal_test)
                    obs_, act_reward, time_done, _ = self.env.step(action)
                    low_level_score += int(act_reward + 1)
                    opt_reward, sub_goal_done, _, _ = self.env.option_step(obs_, sub_goal, act_reward, option_is_goal=True)
                    if render:
                        self.env.render()
                    self.agent.current_act_memory.store_experience(new_option,
                                                                   obs['observation'], obs['sub_goal'], action,
                                                                   obs_['observation'], obs_['achieved_sub_goal'],
                                                                   act_reward, 1 - int(sub_goal_done))
                    obs = dcp(obs_)
                    new_option = False
                    self.agent.normalizer.store_history(obs_['observation'], obs_['achieved_sub_goal'],
                                                        obs_['achieved_goal'])
                high_level_score += int(opt_reward + 1)

                if sub_goal_test and (not sub_goal_done):
                    # penalise high level with -H reward if sub goal test fails
                    opt_reward = int(-self.H)
                self.agent.current_opt_memory.store_experience(new_episode,
                                                               op_obs['observation'],
                                                               op_obs['desired_goal'], sub_goal,
                                                               obs['observation'],
                                                               obs['achieved_goal'],
                                                               opt_reward, 1 - int(time_done))
                op_obs = dcp(obs)
                new_episode = False

            self.agent.normalizer.update()

        self.agent.update(act_hindsight=self.act_hindsight, opt_hindsight=self.opt_hindsight)

        self.sub_goal_train_mean_normalized_score.append(low_level_score/self.training_episode)
        self.final_goal_train_mean_normalized_score.append(high_level_score/self.training_episode)
        print("Epo: {}, Cycle: {}, Actor_Return: {}, Optor_return: {}".format(epo, cyc, low_level_score, high_level_score))

    def test(self, render=False):
        self.env._max_episode_steps = self.testing_time_steps
        goal_num = len(self.env.final_goal_strs)
        episodes = goal_num * self.testing_episode_per_goal
        goal_ind = 0

        high_level_score = np.zeros(self.option_num)
        high_level_success = np.zeros(self.option_num)

        low_level_score = 0
        low_level_success = 0
        sub_goal_count = 0

        for ep_t in range(episodes):
            obs = self.env.reset()
            if render:
                self.env.render()
            time_done = False
            self.env.set_goals(desired_goal_ind=goal_ind)
            obs['desired_goal'] = self.env.goal.copy()
            op_obs = dcp(obs)
            high_level_return = 0
            while not time_done:
                if render:
                    self.env.render()
                sub_goal_done = False
                step_count = 0
                low_level_return = 0
                sub_goal = self.agent.select_option(op_obs['observation'], op_obs['desired_goal'], test=True)
                sub_goal_count += 1
                self.env.set_goals(sub_goal=sub_goal)
                obs['sub_goal'] = self.env.sub_goal.copy()
                op_obs['sub_goal'] = self.env.sub_goal.copy()
                while (step_count < self.H) and (not sub_goal_done) and (not time_done):
                    step_count += 1
                    action = self.agent.select_action(obs['observation'], obs['sub_goal'], test=True)
                    obs_, act_reward, time_done, act_info = self.env.step(action)
                    low_level_score += int(act_reward + 1)
                    low_level_return += int(act_reward + 1)
                    opt_reward, sub_goal_done, _, _ = self.env.option_step(obs_, sub_goal, act_reward, option_is_goal=True)
                    high_level_score[goal_ind] += int(opt_reward + 1)
                    high_level_return += int(opt_reward + 1)
                    obs = dcp(obs_)
                    if render:
                        self.env.render()
                op_obs = dcp(obs)
                if low_level_return > 0:
                    low_level_success += 1
            if high_level_return > 0:
                high_level_success[goal_ind] += 1
            goal_ind = (goal_ind + 1) % goal_num

        self.sub_goal_test_mean_success_rate.append(low_level_success / sub_goal_count)
        self.sub_goal_test_mean_avg_return.append(low_level_score / self.testing_episode_per_goal)

        self.final_goal_test_mean_success_rate.append(high_level_success.mean() / self.testing_episode_per_goal)
        self.final_goal_test_mean_avg_return.append(high_level_score.mean() / self.testing_episode_per_goal)
        self.final_goal_test_inpd_success_rate.append(high_level_success / self.testing_episode_per_goal)
        self.final_goal_test_inpd_avg_return.append(high_level_score / self.testing_episode_per_goal)

    def _save_ckpts(self, epo):
        self.agent.save_ckpts(epo)

    def _plot(self):
        smoothed_plot(self.path + "/sub_goal_train_mean_normalized_score.png",
                      self.sub_goal_train_mean_normalized_score,
                      x_label="Cycle", y_label="Normalized score")
        smoothed_plot(self.path + "/sub_goal_test_mean_success_rate.png", self.sub_goal_test_mean_success_rate,
                      x_label="Epoch", y_label="Success rate")
        smoothed_plot(self.path + "/sub_goal_test_mean_avg_return.png", self.sub_goal_test_mean_avg_return,
                      x_label="Epoch", y_label="Normalized score")

        smoothed_plot(self.path + "/final_goal_train_mean_normalized_score.png",
                      self.final_goal_train_mean_normalized_score,
                      x_label="Cycle", y_label="Normalized score")
        smoothed_plot(self.path + "/final_goal_test_mean_success_rate.png", self.final_goal_test_mean_success_rate,
                      x_label="Epoch", y_label="Success rate")
        smoothed_plot(self.path + "/final_goal_test_mean_avg_return.png", self.final_goal_test_mean_avg_return,
                      x_label="Epoch", y_label="Normalized score")
        self._plot_sub_goal()

    def _plot_sub_goal(self):
        legend = dcp(self.env.sub_goal_strs)

        final_goal_test_inpd_avg_return = np.array(self.final_goal_test_inpd_avg_return)
        final_goal_test_inpd_avg_return = np.transpose(final_goal_test_inpd_avg_return)
        final_goal_test_inpd_success_rate = np.array(self.final_goal_test_inpd_success_rate)
        final_goal_test_inpd_success_rate = np.transpose(final_goal_test_inpd_success_rate)
        smoothed_plot_multi_line(self.path + "/final_goal_test_inpd_avg_return.png", final_goal_test_inpd_avg_return,
                                 legend=legend, x_label="Epoch", y_label="Normalized score")
        smoothed_plot_multi_line(self.path + "/final_goal_test_inpd_success_rate.png", final_goal_test_inpd_success_rate,
                                 legend=legend, x_label="Epoch", y_label="Success rate")

    def _save_numpy_to_txt(self):
        path = self.path + "/data"
        if not os.path.isdir(path):
            os.mkdir(path)

        np.savetxt(path + "/act_input_means.dat", self.agent.normalizer.input_mean_low)
        np.savetxt(path + "/act_input_vars.dat", self.agent.normalizer.input_var_low)

        np.savetxt(path + "/tr_sg_mean_score.dat", np.array(self.sub_goal_train_mean_normalized_score))
        np.savetxt(path + "/te_sg_mean_avg_return.dat", np.array(self.sub_goal_test_mean_avg_return))
        np.savetxt(path + "/te_sg_mean_success.dat", np.array(self.sub_goal_test_mean_success_rate))

        np.savetxt(path + "/opt_input_means.dat", self.agent.normalizer.input_mean_high)
        np.savetxt(path + "/opt_input_vars.dat", self.agent.normalizer.input_var_high)

        np.savetxt(path + "/tr_fg_mean_score.dat", np.array(self.final_goal_train_mean_normalized_score))
        np.savetxt(path + "/te_fg_mean_avg_return.dat", np.array(self.final_goal_test_mean_avg_return))
        np.savetxt(path + "/te_fg_mean_success.dat", np.array(self.final_goal_test_mean_success_rate))
        np.savetxt(path + "/te_fg_inpd_avg_return.dat", np.array(self.final_goal_test_inpd_avg_return))
        np.savetxt(path + "/te_fg_inpd_success.dat", np.array(self.final_goal_test_inpd_success_rate))