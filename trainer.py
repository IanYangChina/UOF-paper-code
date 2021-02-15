import os
import numpy as np
from copy import deepcopy as dcp
from collections import namedtuple
import multigoal_env  # import this one for env registration
from gym import make
from gym.error import UnregisteredEnv
from agent.utils.plot import smoothed_plot, smoothed_plot_multi_line
from agent.universal_option_framework import UniversalOptionFramework as UOF
from agent.utils.demonstrator import Demonstrator


class Trainer(object):
    def __init__(self, path, env_name='TwoObjectOneOrderBinaryHighLvGoal-v0', seed=0,
                 load_act_init_input=False, load_opt_init_input=False, init_input_path=None,
                 demonstrations=None, use_demonstration_in_training=False, demonstrated_episode_proportion=0.75,
                 act_exploration=None, chance=0.2, deviation=0.05, aaes_tau=0.05, act_train=True, opt_train=True,
                 act_hindsight=True, act_clip_value=-25,
                 opt_hindsight=False, opt_eps_decay=30000, opt_clip_value=None, intra_option_learning=True,
                 double_q=True, multi_inter=False,
                 training_epoch=201, training_cycle=50, training_episode=16,
                 testing_episode_per_goal=30, testing_time_steps=50, testing_gap=1, saving_gap=50):
        np.set_printoptions(precision=3)
        self.path = path
        try:
            self.env = make(env_name)
        except UnregisteredEnv:
            raise UnregisteredEnv("Make sure the env id: {} is correct\nExisting id: {}".format(env_name, multigoal_env.ids))
        self.env.seed(seed)

        OPT_Tr = namedtuple("transition",
                            ('state', 'desired_goal', 'option', 'next_state', 'achieved_goal', 'option_done', 'timesteps', 'reward', 'done'))
        ACT_Tr = namedtuple("transition",
                            ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))

        self.training_time_steps = dcp(self.env._max_episode_steps)
        self.testing_time_steps = testing_time_steps

        self.training_epoch = training_epoch
        self.training_cycle = training_cycle
        self.training_episode = training_episode
        self.testing_episode_per_goal = testing_episode_per_goal
        self.testing_gap = testing_gap
        self.saving_gap = saving_gap

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
        self.act_train = act_train
        self.act_exploration = act_exploration
        self.act_hindsight = act_hindsight
        self.opt_train = opt_train
        self.opt_hindsight = opt_hindsight
        self.intra_option_learning = intra_option_learning
        self.act_clip_value = act_clip_value
        if opt_clip_value is not None:
            self.opt_clip_value = opt_clip_value
        else:
            self.opt_clip_value = -self.training_time_steps
        self.agent = UOF(env_params, OPT_Tr, ACT_Tr, path=self.path, seed=seed,
                         double_q=double_q,
                         multi_inter=multi_inter,
                         act_clip_value=self.act_clip_value,
                         act_exploration=self.act_exploration, chance=chance, deviation=deviation, aaes_tau=aaes_tau,
                         opt_clip_value=self.opt_clip_value, intra_option_learning=self.intra_option_learning,
                         opt_eps_decay=opt_eps_decay)

        if demonstrations is None:
            """Example sequences of steps, e.g.: 
               demonstrations = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]
            """
            raise ValueError("Demonstrations need to be specified")
        self.demonstrator = Demonstrator(demonstrations)
        self.use_demonstration_in_training = use_demonstration_in_training
        self.demonstrated_episode = int(demonstrated_episode_proportion*self.training_episode)

        self.final_goal_train_mean_normalized_score = []
        self.final_goal_train_inpd_normalized_score_cyc = np.zeros(self.env.option_space.n)
        self.final_goal_train_inpd_normalized_score = []
        self.final_goal_test_mean_avg_return = []
        self.final_goal_test_mean_success_rate = []
        self.final_goal_test_inpd_avg_return = []
        self.final_goal_test_inpd_success_rate = []

        self.sub_goal_train_mean_normalized_score = []
        self.sub_goal_train_inpd_normalized_score_cyc = np.zeros(self.env.option_space.n)
        self.sub_goal_train_inpd_normalized_score = []
        self.sub_goal_test_mean_avg_return = []
        self.sub_goal_test_mean_success_rate = []
        self.sub_goal_test_inpd_avg_return = []
        self.sub_goal_test_inpd_success_rate = []
        self.sub_goal_chances = []

    def run(self, training_render=False, testing_render=False):
        for epo in range(self.training_epoch):
            self.sub_goal_train_inpd_normalized_score_cyc = np.zeros(self.env.option_space.n)
            self.final_goal_train_inpd_normalized_score_cyc = np.zeros(self.env.option_space.n)
            self.env._max_episode_steps = self.training_time_steps
            for cyc in range(self.training_cycle):
                self.train(epo=epo, cyc=cyc, render=training_render, act_update=self.act_train, opt_update=self.opt_train)

            self.sub_goal_train_inpd_normalized_score_cyc /= self.training_cycle
            self.sub_goal_train_inpd_normalized_score.append(dcp(self.sub_goal_train_inpd_normalized_score_cyc))
            self.final_goal_train_inpd_normalized_score_cyc /= self.training_cycle
            self.final_goal_train_inpd_normalized_score.append(dcp(self.final_goal_train_inpd_normalized_score_cyc))

            if epo % self.testing_gap == 0:
                print("Subgoal train normalized scores: ", self.sub_goal_train_inpd_normalized_score[-1])
                print("Finalgoal train normalized scores", self.final_goal_train_inpd_normalized_score[-1])
                if self.act_train:
                    self.sub_goal_test(render=testing_render)
                    print("Subgoal test success rates: ", self.sub_goal_test_inpd_success_rate[-1])
                    if self.act_exploration is not None:
                        self.agent.actor_exploration.update_success_rates(self.sub_goal_test_inpd_success_rate[-1])
                        self.sub_goal_chances.append(self.agent.actor_exploration.chance.copy())
                        print("Current chances: ", self.sub_goal_chances[-1])
                if self.opt_train:
                    self.final_goal_test(render=testing_render)
                    print("Finalgoal test success rates: ", self.final_goal_test_inpd_success_rate[-1])
            if (epo % self.saving_gap == 0) and (epo != 0):
                self._save_ckpts(epo)

        self._plot()
        self._save_numpy_to_txt()

    def train(self, epo=0, cyc=0, render=False, act_update=True, opt_update=True):
        high_level_score = np.zeros(self.option_num)
        low_level_score = np.zeros(self.option_num)
        for ep in range(self.training_episode):
            new_episode = True
            obs = self.env.reset()
            time_done = False
            demon = False

            op_obs = dcp(obs)
            final_goal_ind = self.env.final_goal_strs.index(self.env.goal_str)
            self.demonstrator.reset_with_final_goal_index(final_goal_ind)

            if self.use_demonstration_in_training and (self.demonstrated_episode + ep) >= self.training_episode:
                demon = True

            while not time_done:
                sub_goal_done = False
                new_option = True
                if demon:
                    option = self.demonstrator.get_next_goal()
                else:
                    option = self.agent.select_option(op_obs['observation'], op_obs['desired_goal'], final_goal_ind,
                                                      ep=((ep+1)*(cyc+1)*(epo+1)))

                self.env.set_goals(sub_goal_ind=option)
                obs['sub_goal'] = self.env.sub_goal.copy()
                op_obs['sub_goal'] = self.env.sub_goal.copy()
                option_time_steps = 0
                option_discounted_reward = 0
                while (not sub_goal_done) and (not time_done):
                    action = self.agent.select_action(obs['observation'], obs['sub_goal'], option, test=(not act_update))
                    obs_, act_reward, time_done, _ = self.env.step(action)
                    low_level_score[option] += int(act_reward+1)
                    opt_reward, sub_goal_done, _, _ = self.env.option_step(obs_, option, act_reward)
                    high_level_score[final_goal_ind] += int(opt_reward+1)
                    option_time_steps += 1
                    option_discounted_reward += (self.agent.opt_gamma**(option_time_steps-1))*opt_reward
                    if render:
                        self.env.render()
                    if act_update:
                        self.agent.current_act_memory.store_experience(new_option,
                                                                       obs['observation'], obs['sub_goal'], action,
                                                                       obs_['observation'], obs_['achieved_sub_goal'],
                                                                       act_reward, 1-int(sub_goal_done))
                    obs = dcp(obs_)
                    new_option = False
                    if opt_update:
                        if not self.intra_option_learning:
                            # SMDP-Q only uses transitions upon terminations
                            if sub_goal_done or time_done:
                                self.agent.current_opt_memory.store_experience(new_episode,
                                                                               op_obs['observation'],
                                                                               op_obs['desired_goal'], option,
                                                                               obs_['observation'],
                                                                               obs_['achieved_goal'],
                                                                               1-int(sub_goal_done), option_time_steps,
                                                                               option_discounted_reward, 1-int(time_done))
                                op_obs = dcp(obs_)
                                new_episode = False
                        else:
                            self.agent.current_opt_memory.store_experience(new_episode,
                                                                           op_obs['observation'],
                                                                           op_obs['desired_goal'], option,
                                                                           obs_['observation'],
                                                                           obs_['achieved_goal'],
                                                                           1-int(sub_goal_done), option_time_steps,
                                                                           opt_reward, 1-int(time_done))
                            op_obs = dcp(obs_)
                            new_episode = False
                    self.agent.normalizer.store_history(obs_['observation'], obs_['achieved_sub_goal'], obs_['achieved_goal'])
            if act_update or opt_update:
                self.agent.normalizer.update()

        self.agent.update(act_update=act_update, opt_update=opt_update,
                          act_hindsight=self.act_hindsight, opt_hindsight=self.opt_hindsight)

        self.sub_goal_train_inpd_normalized_score_cyc += low_level_score
        self.sub_goal_train_mean_normalized_score.append(low_level_score.mean())
        self.final_goal_train_inpd_normalized_score_cyc += high_level_score
        self.final_goal_train_mean_normalized_score.append(high_level_score.mean())
        print("Epo: {}, Cycle: {}, Actor_Return: {}, Optor_return: {}".format(epo, cyc, low_level_score, high_level_score.sum()))

    def sub_goal_test(self, render=False, episode_per_demon_to_test=None, test_noise=False, print_result=False):
        self.env._max_episode_steps = self.testing_time_steps
        if episode_per_demon_to_test is None:
            episode_per_demon_to_test = self.testing_episode_per_goal
        episodes = self.demonstrator.demon_num * episode_per_demon_to_test
        score = np.zeros(self.option_num)
        success = np.zeros(self.option_num)
        demon_ind = 0
        for ep_t in range(episodes):
            obs = self.env.reset()
            time_done = False
            self.demonstrator.manual_reset(demon_ind)
            demon_goal_ind = self.demonstrator.current_final_goal
            self.env.set_goals(desired_goal_ind=demon_goal_ind)
            obs['desired_goal'] = self.env.goal.copy()
            demon_goal_return = 0
            while not time_done:
                sub_goal_ind = self.demonstrator.get_next_goal()
                self.env.set_goals(sub_goal_ind=sub_goal_ind)
                obs['sub_goal'] = self.env.sub_goal.copy()
                goal_done = False
                while (not goal_done) and (not time_done):
                    if render:
                        self.env.render()
                    action = self.agent.select_action(obs['observation'], obs['sub_goal'], sub_goal_ind, test=True, test_noise=test_noise)
                    obs_, act_reward, time_done, act_info = self.env.step(action)
                    opt_reward, goal_done, opt_info, _ = self.env.option_step(obs_, sub_goal_ind, act_reward)
                    obs = dcp(obs_)
                    score[demon_goal_ind] += int(opt_reward+1)
                    demon_goal_return += int(opt_reward+1)
            if demon_goal_return > 0:
                success[demon_goal_ind] += 1
            demon_ind = (demon_ind + 1) % self.demonstrator.demon_num
        if print_result:
            print(score)
        self.sub_goal_test_inpd_success_rate.append(success / episode_per_demon_to_test)
        self.sub_goal_test_mean_success_rate.append((success / episode_per_demon_to_test).mean())
        self.sub_goal_test_inpd_avg_return.append(score)
        self.sub_goal_test_mean_avg_return.append(score.mean())

    def final_goal_test(self, render=False, testing_episode_final_goal=None, given_goals=None):
        self.env._max_episode_steps = self.testing_time_steps
        if testing_episode_final_goal is None:
            testing_episode_final_goal = self.testing_episode_per_goal
        goal_num = len(self.env.final_goal_strs)
        if given_goals is not None:
            given_goal_num = len(given_goals)
        ind = 0
        goal_ind = 0
        episodes = goal_num * testing_episode_final_goal
        score = np.zeros(self.option_num)
        success = np.zeros(self.option_num)
        for ep_t in range(episodes):
            if given_goals is not None:
                goal_ind = given_goals[ind]
            obs = self.env.reset()
            if render:
                self.env.render()
            time_done = False
            self.env.set_goals(desired_goal_ind=goal_ind)
            obs['desired_goal'] = self.env.goal.copy()
            op_obs = dcp(obs)
            goal_return = 0
            while not time_done:
                if render:
                    self.env.render()
                sub_goal_done = False
                option = self.agent.select_option(op_obs['observation'], op_obs['desired_goal'], goal_ind, test=True)
                self.env.set_goals(sub_goal_ind=option)
                obs['sub_goal'] = self.env.sub_goal.copy()
                op_obs['sub_goal'] = self.env.sub_goal.copy()
                # print(self.env.sub_goal_str)
                while (not sub_goal_done) and (not time_done):
                    action = self.agent.select_action(obs['observation'], obs['sub_goal'], option, test=True)
                    obs_, act_reward, time_done, act_info = self.env.step(action)
                    opt_reward, sub_goal_done, opt_info, final_goal_done = self.env.option_step(obs_, option, act_reward)
                    score[goal_ind] += int(opt_reward+1)
                    goal_return += int(opt_reward+1)
                    obs = dcp(obs_)
                    if not self.intra_option_learning:
                        if sub_goal_done:
                            op_obs = dcp(obs_)
                    else:
                        op_obs = dcp(obs_)
                    if render:
                        self.env.render()
            if goal_return > 0:
                success[goal_ind] += 1
            goal_ind = (goal_ind + 1) % goal_num
            if given_goals is not None:
                ind = (ind + 1) % given_goal_num
        print(score, score.mean(), success, success.mean())
        self.final_goal_test_inpd_success_rate.append(success / testing_episode_final_goal)
        self.final_goal_test_mean_success_rate.append((success / testing_episode_final_goal).mean())
        self.final_goal_test_inpd_avg_return.append(score)
        self.final_goal_test_mean_avg_return.append(score.mean())

    def _save_ckpts(self, epo):
        self.agent.save_ckpts(epo, intra=self.act_train, inter=self.opt_train)

    def _plot(self):
        if self.act_train:
            smoothed_plot(self.path + "/sub_goal_train_mean_normalized_score.png", self.sub_goal_train_mean_normalized_score,
                          x_label="Cycle", y_label="Normalized score")
            smoothed_plot(self.path + "/sub_goal_test_mean_success_rate.png", self.sub_goal_test_mean_success_rate,
                          x_label="Epoch", y_label="Success rate")
            smoothed_plot(self.path + "/sub_goal_test_mean_avg_return.png", self.sub_goal_test_mean_avg_return,
                          x_label="Epoch", y_label="Normalized score")

        if self.opt_train:
            smoothed_plot(self.path + "/final_goal_train_mean_normalized_score.png", self.final_goal_train_mean_normalized_score,
                          x_label="Cycle", y_label="Normalized score")
            smoothed_plot(self.path + "/final_goal_test_mean_success_rate.png", self.final_goal_test_mean_success_rate,
                          x_label="Epoch", y_label="Success rate")
            smoothed_plot(self.path + "/final_goal_test_mean_avg_return.png", self.final_goal_test_mean_avg_return,
                          x_label="Epoch", y_label="Normalized score")

        self._plot_sub_goal()

    def _plot_sub_goal(self):
        legend = dcp(self.env.sub_goal_strs)
        if self.act_train:
            sub_goal_train_normalized_score = np.array(self.sub_goal_train_inpd_normalized_score)
            sub_goal_train_normalized_score = np.transpose(sub_goal_train_normalized_score)
            sub_goal_test_normalized_score = np.array(self.sub_goal_test_inpd_avg_return)
            sub_goal_test_normalized_score = np.transpose(sub_goal_test_normalized_score)
            sub_goal_test_inpd_success_rate = np.array(self.sub_goal_test_inpd_success_rate)
            sub_goal_test_inpd_success_rate = np.transpose(sub_goal_test_inpd_success_rate)
            smoothed_plot_multi_line(self.path + "/sub_goal_test_inpd_avg_return.png", sub_goal_test_normalized_score, legend=legend,
                                     x_label="Epoch", y_label="Normalized score")
            smoothed_plot_multi_line(self.path + "/sub_goal_test_inpd_success_rate.png", sub_goal_test_inpd_success_rate, legend=legend,
                                     x_label="Epoch", y_label="Success rate")
            smoothed_plot_multi_line(self.path + "/sub_goal_train_inpd_normalized_score.png", sub_goal_train_normalized_score, legend=legend,
                                     x_label="Epoch", y_label="Normalized score")
            if self.act_exploration is not None:
                sub_goal_chances = np.array(self.sub_goal_chances)
                sub_goal_chances = np.transpose(sub_goal_chances)
                smoothed_plot_multi_line(self.path + "/sub_goal_chances.png", sub_goal_chances, legend=legend,
                                         x_label="Epoch", y_label="Chance")
        if self.opt_train:
            final_goal_train_inpd_normalized_score = np.array(self.final_goal_train_inpd_normalized_score)
            final_goal_train_inpd_normalized_score = np.transpose(final_goal_train_inpd_normalized_score)
            final_goal_test_inpd_avg_return = np.array(self.final_goal_test_inpd_avg_return)
            final_goal_test_inpd_avg_return = np.transpose(final_goal_test_inpd_avg_return)
            final_goal_test_inpd_success_rate = np.array(self.final_goal_test_inpd_success_rate)
            final_goal_test_inpd_success_rate = np.transpose(final_goal_test_inpd_success_rate)
            smoothed_plot_multi_line(self.path + "/final_goal_test_inpd_avg_return.png", final_goal_test_inpd_avg_return,
                                     legend=legend, x_label="Epoch", y_label="Normalized score")
            smoothed_plot_multi_line(self.path + "/final_goal_test_inpd_success_rate.png", final_goal_test_inpd_success_rate,
                                     legend=legend, x_label="Epoch", y_label="Success rate")
            smoothed_plot_multi_line(self.path + "/final_goal_train_inpd_normalized_score.png", final_goal_train_inpd_normalized_score,
                                     legend=legend, x_label="Epoch", y_label="Normalized score")

    def _save_numpy_to_txt(self):
        path = self.path + "/data"
        if not os.path.isdir(path):
            os.mkdir(path)

        if self.act_train:
            np.savetxt(path + "/act_input_means.dat", self.agent.normalizer.input_mean_low)
            np.savetxt(path + "/act_input_vars.dat", self.agent.normalizer.input_var_low)

            np.savetxt(path + "/tr_sg_mean_score.dat", np.array(self.sub_goal_train_mean_normalized_score))
            np.savetxt(path + "/tr_sg_inpd_score.dat", np.array(self.sub_goal_train_inpd_normalized_score))

            np.savetxt(path + "/te_sg_mean_avg_return.dat", np.array(self.sub_goal_test_mean_avg_return))
            np.savetxt(path + "/te_sg_mean_success.dat", np.array(self.sub_goal_test_mean_success_rate))
            np.savetxt(path + "/te_sg_inpd_avg_return.dat", np.array(self.sub_goal_test_inpd_avg_return))
            np.savetxt(path + "/te_sg_inpd_success.dat", np.array(self.sub_goal_test_inpd_success_rate))

            if self.act_exploration is not None:
                np.savetxt(path + '/sub_goal_chances.dat', np.array(self.sub_goal_chances))

        if self.opt_train:
            np.savetxt(path + "/opt_input_means.dat", self.agent.normalizer.input_mean_high)
            np.savetxt(path + "/opt_input_vars.dat", self.agent.normalizer.input_var_high)

            np.savetxt(path + "/tr_fg_mean_score.dat", np.array(self.final_goal_train_mean_normalized_score))
            np.savetxt(path + "/tr_fg_inpd_score.dat", np.array(self.final_goal_train_inpd_normalized_score))

            np.savetxt(path + "/te_fg_mean_avg_return.dat", np.array(self.final_goal_test_mean_avg_return))
            np.savetxt(path + "/te_fg_mean_success.dat", np.array(self.final_goal_test_mean_success_rate))
            np.savetxt(path + "/te_fg_inpd_avg_return.dat", np.array(self.final_goal_test_inpd_avg_return))
            np.savetxt(path + "/te_fg_inpd_success.dat", np.array(self.final_goal_test_inpd_success_rate))
