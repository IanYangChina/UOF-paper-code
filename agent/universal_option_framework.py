import os
import random as R
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.normalizer import GoalEnvNormalizer
from agent.utils.networks import Actor, Critic, Mlp
from agent.utils.replay_buffer import HighLevelHindsightReplayBuffer, LowLevelHindsightReplayBuffer
from agent.utils.exploration_strategy import ExpDecayGreedy, ConstantChance, AutoAdjustingConstantChance


class UniversalOptionFramework(object):
    def __init__(self, env_params, opt_tr, act_tr, path=None, seed=0, double_q=False, intra_option_learning=True,
                 multi_inter=False, multi_intra=False,
                 option_lr=0.001, opt_tau=0.1, opt_batch_size=128, opt_mem_capacity=int(1e6), opt_clip_value=-50.0,
                 opt_gamma=0.98, opt_optim_steps=40, opt_eps_decay=30000,
                 action_lr=0.001, act_tau=0.1, act_batch_size=128, act_mem_capacity=int(1e6), act_clip_value=-50.0,
                 act_gamma=0.98, act_optim_steps=40,
                 act_exploration=None, chance=0.2, deviation=0.05, aaes_tau=0.05):
        T.manual_seed(seed)
        R.seed(seed)
        np.random.seed(seed)
        if path is None:
            self.ckpt_path = "ckpts"
        else:
            self.ckpt_path = path + "/ckpts"
        if not os.path.isdir(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        use_cuda = T.cuda.is_available()
        self.device = T.device("cuda" if use_cuda else "cpu")

        self.obs_dim = env_params['obs_dims']
        self.goal_dim = env_params['goal_dims']
        self.sub_goal_dim = env_params['sub_goal_dims']
        self.normalizer = GoalEnvNormalizer(self.obs_dim, self.sub_goal_dim, self.goal_dim,
                                            env_params['act_init_input_means'], env_params['act_init_input_vars'],
                                            env_params['opt_init_input_means'], env_params['opt_init_input_vars'],
                                            env_params['different_goals']
                                            )
        self.double_q = double_q

        """Inter-Option/High-Level policies - (Double-) DIOL/SMDPDQN"""
        # Learning params
        self.intra_option_learning = intra_option_learning
        self.opt_batch_size = opt_batch_size
        self.opt_gamma = opt_gamma
        self.opt_tau = opt_tau
        self.opt_clip_value = opt_clip_value
        self.opt_optim_steps = opt_optim_steps
        # Get optors
        self.multi_inter = multi_inter
        self.option_num = env_params['option_num']
        if not self.multi_inter:
            self.optor = self.get_optor(opt_mem_capacity, opt_tr, seed, option_lr)
            self.opt_soft_update(self.optor, opt_tau=1.0)
            self.current_opt_memory = self.optor['optor_memory']
        else:
            self.num_inter_option_policies = self.option_num
            self.optors = []
            for _ in range(self.num_inter_option_policies):
                self.optors.append(self.get_optor(opt_mem_capacity, opt_tr, seed, option_lr))
                self.opt_soft_update(self.optors[-1], opt_tau=1.0)
            self.current_opt_memory = self.optors[0]['optor_memory']
        self.current_optor_dict = None
        # Exploration
        self.optor_exploration = ExpDecayGreedy(decay=opt_eps_decay)

        """Intra-Option/Low-Level policies - (Double-) DDPG"""
        # Learning Params
        self.act_batch_size = act_batch_size
        self.act_gamma = act_gamma
        self.act_tau = act_tau
        self.act_clip_value = act_clip_value
        self.act_optim_steps = act_optim_steps
        # Get actors
        self.multi_intra = multi_intra
        self.action_dim = env_params['action_dims']
        self.action_max = env_params['action_max']
        if not self.multi_intra:
            self.actor = self.get_actor(act_mem_capacity, act_tr, seed, action_lr)
            self.act_soft_update(self.actor, act_tau=1.0)
            self.current_act_memory = self.actor['actor_memory']
        else:
            self.num_intra_option_policies = self.option_num
            self.actors = []
            for _ in range(self.num_intra_option_policies):
                self.actors.append(self.get_actor(act_mem_capacity, act_tr, seed, action_lr))
                self.act_soft_update(self.actors[-1], act_tau=1.0)
            self.current_act_memory = self.actors[0]['actor_memory']
        self.current_actor_dict = None
        # Exploration
        if act_exploration is None:
            self.actor_exploration = ConstantChance(chance=chance)
        else:
            self.actor_exploration = AutoAdjustingConstantChance(goal_num=self.option_num, chance=chance, tau=aaes_tau)
        self.noise_deviation = deviation * self.action_max

    def select_action(self, state, desired_goal, option, test=False, test_noise=False):
        if self.multi_intra:
            self.current_actor_dict = self.actors[option]
            current_actor = self.actors[option]['actor_target']
            self.current_act_memory = self.actors[option]['actor_memory']
        else:
            self.current_actor_dict = self.actor
            current_actor = self.actor['actor_target']

        current_actor.eval()
        inputs = np.concatenate((state, desired_goal), axis=0)
        inputs = self.normalizer(inputs, level='low')
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        action = current_actor(inputs).cpu().detach().numpy()
        if test:
            action = np.clip(action, -self.action_max, self.action_max)
        else:
            _ = R.uniform(0, 1)
            eps = self.actor_exploration(option)
            if _ < eps:
                action = np.random.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
            else:
                if isinstance(self.actor_exploration, AutoAdjustingConstantChance):
                    deviation = self.noise_deviation * (1-self.actor_exploration.success_rates[option])
                else:
                    deviation = self.noise_deviation
                action += deviation * np.random.randn(self.action_dim)
                action = np.clip(action, -self.action_max, self.action_max)
        return action

    def select_option(self, state, desired_goal, desired_goal_id, ep=0, test=False):
        if self.multi_inter:
            self.current_optor_dict = self.optors[desired_goal_id]
            current_optor = self.optors[desired_goal_id]['optor_target']
            self.current_opt_memory = self.optors[desired_goal_id]['optor_memory']
        else:
            self.current_optor_dict = self.optor
            current_optor = self.optor['optor_target']

        current_optor.eval()
        inputs = np.concatenate((state, desired_goal), axis=0)
        inputs = self.normalizer(inputs, level='high')
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        option_values = current_optor(inputs)
        if test:
            option = T.argmax(option_values).item()
        else:
            _ = R.uniform(0, 1)
            if _ < self.optor_exploration(ep):
                option = R.randint(0, self.option_num - 1)
            else:
                option = T.argmax(option_values).item()
        return option

    def update(self, act_update=True, opt_update=True, act_hindsight=True, opt_hindsight=True):
        if act_update:
            if not self.multi_intra:
                self.act_learn(self.actor, act_hindsight)
            else:
                for actor_dict in self.actors:
                    self.current_act_memory = actor_dict['actor_memory']
                    self.act_learn(actor_dict, act_hindsight)

        if opt_update:
            if not self.multi_inter:
                self.opt_learn(self.optor, opt_hindsight)
            else:
                for optor_dict in self.optors:
                    self.current_opt_memory = optor_dict['optor_memory']
                    self.opt_learn(optor_dict, opt_hindsight)

    def act_learn(self, actor_dict, hindsight=True):
        if len(self.current_act_memory.episodes) == 0:
            return
        else:
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

            actor_dict['actor'].eval()
            actor_dict['actor_target'].eval()
            actor_dict['critic'].train()
            actor_dict['critic_target'].eval()
            actions_ = actor_dict['actor_target'](actor_inputs_)
            critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1).to(self.device)
            value_ = actor_dict['critic_target'](critic_inputs_)
            if self.double_q:
                actor_dict['critic_2'].train()
                actor_dict['critic_2_target'].eval()
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
            if self.double_q:
                value_estimate_2 = actor_dict['critic_2'](critic_inputs)
                actor_dict['critic_2_optimizer'].zero_grad()
                critic_2_loss = F.mse_loss(value_estimate_2, value_target.detach())
                critic_2_loss.backward()
                actor_dict['critic_2_optimizer'].step()

            actor_dict['critic'].eval()
            actor_dict['actor'].train()
            actor_dict['actor_optimizer'].zero_grad()
            new_actions = actor_dict['actor'](actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
            new_value = actor_dict['critic'](critic_eval_inputs)
            if self.double_q:
                actor_dict['critic_2'].eval()
                new_value_2 = actor_dict['critic_2'](critic_eval_inputs)
                new_value = T.min(new_value, new_value_2)
            actor_loss = -new_value
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            actor_dict['actor_optimizer'].step()
            actor_dict['actor'].eval()
            self.act_soft_update(actor_dict)
        return

    def opt_learn(self, optor_dict, hindsight=True):
        if len(self.current_opt_memory.episodes) == 0:
            return
        else:
            if hindsight:
                self.current_opt_memory.modify_experiences()
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
            timesteps = T.tensor(batch.timesteps, dtype=T.float).unsqueeze(1).to(self.device)
            episode_done = T.tensor(batch.done, dtype=T.float).unsqueeze(1).to(self.device)

            optor_dict['optor'].train()
            optor_dict['optor_target'].eval()
            if self.intra_option_learning:
                # calculate "option value upon arrival" with immediate rewards
                rewards = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)
                unchanged_next_option_values = optor_dict['optor_target'](inputs_).gather(1, options)
                maximal_next_option_values = optor_dict['optor_target'](inputs_).max(1)[0].view(batch_size, 1)
                next_option_values = option_done * unchanged_next_option_values + (
                            1 - option_done) * maximal_next_option_values
                if self.double_q:
                    optor_dict['optor_2'].train()
                    optor_dict['optor_2_target'].eval()
                    unchanged_next_option_values_2 = optor_dict['optor_2_target'](inputs_).gather(1, options)
                    maximal_next_option_values_2 = optor_dict['optor_2_target'](inputs_).max(1)[0].view(batch_size, 1)
                    next_option_values_2 = option_done * unchanged_next_option_values_2 + (
                                1 - option_done) * maximal_next_option_values_2
                    next_option_values = T.min(next_option_values, next_option_values_2)
                target_option_values = rewards + episode_done * self.opt_gamma * next_option_values
            else:
                # calculate option values with multi-step returns
                discounted_returns = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)
                next_option_values = optor_dict['optor_target'](inputs_).max(1)[0].view(batch_size, 1)
                if self.double_q:
                    optor_dict['optor_2'].train()
                    optor_dict['optor_2_target'].eval()
                    next_option_values_2 = optor_dict['optor_2_target'](inputs_).max(1)[0].view(batch_size, 1)
                    next_option_values = T.min(next_option_values, next_option_values_2)
                gamma = T.pow(self.opt_gamma, timesteps+1).to(self.device)
                target_option_values = discounted_returns + episode_done * gamma * next_option_values

            target_option_values = T.clamp(target_option_values, self.opt_clip_value, -0.0)

            optor_dict['optor_optimizer'].zero_grad()
            estimated_option_values = optor_dict['optor'](inputs).gather(1, options)
            loss = F.smooth_l1_loss(estimated_option_values, target_option_values.detach())
            loss.backward()
            optor_dict['optor_optimizer'].step()
            optor_dict['optor'].eval()
            if self.double_q:
                optor_dict['optor_2_optimizer'].zero_grad()
                estimated_option_values_2 = optor_dict['optor_2'](inputs).gather(1, options)
                loss_2 = F.smooth_l1_loss(estimated_option_values_2, target_option_values.detach())
                loss_2.backward()
                optor_dict['optor_2_optimizer'].step()
                optor_dict['optor_2'].eval()
            self.opt_soft_update(optor_dict)
        return

    def act_soft_update(self, actor_dict, act_tau=None):
        if act_tau is None:
            act_tau = self.act_tau

        for target_param, param in zip(actor_dict['critic_target'].parameters(), actor_dict['critic'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - act_tau) + param.data * act_tau
            )
        if self.double_q:
            for target_param, param in zip(actor_dict['critic_2_target'].parameters(), actor_dict['critic_2'].parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - act_tau) + param.data * act_tau
                )
        for target_param, param in zip(actor_dict['actor_target'].parameters(), actor_dict['actor'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - act_tau) + param.data * act_tau
            )

    def opt_soft_update(self, optor_dict, opt_tau=None):
        if opt_tau is None:
            opt_tau = self.opt_tau

        for target_param, param in zip(optor_dict['optor_target'].parameters(), optor_dict['optor'].parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - opt_tau) + param.data * opt_tau
            )
        if self.double_q:
            for target_param, param in zip(optor_dict['optor_2_target'].parameters(), optor_dict['optor_2'].parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - opt_tau) + param.data * opt_tau
                )

    def get_optor(self, opt_mem_capacity, opt_tr, seed, option_lr):
        optor_dict = dict()
        optor_dict['optor_memory'] = HighLevelHindsightReplayBuffer(opt_mem_capacity, opt_tr, seed=seed)
        optor_dict['optor'] = Mlp(self.obs_dim + self.goal_dim, self.option_num).to(self.device)
        optor_dict['optor_target'] = Mlp(self.obs_dim + self.goal_dim, self.option_num).to(self.device)
        optor_dict['optor_optimizer'] = Adam(optor_dict['optor'].parameters(), lr=option_lr)
        if self.double_q:
            optor_dict['optor_2'] = Mlp(self.obs_dim + self.goal_dim, self.option_num).to(self.device)
            optor_dict['optor_2_target'] = Mlp(self.obs_dim + self.goal_dim, self.option_num).to(self.device)
            optor_dict['optor_2_optimizer'] = Adam(optor_dict['optor_2'].parameters(), lr=option_lr)
        return optor_dict
    
    def get_actor(self, act_mem_capacity, act_tr, seed, action_lr):
        actor_dict = dict()
        actor_dict['actor_memory'] = LowLevelHindsightReplayBuffer(act_mem_capacity, act_tr, seed=seed)
        actor_dict['actor'] = Actor(self.obs_dim + self.sub_goal_dim, self.action_dim).to(self.device)
        actor_dict['actor_target'] = Actor(self.obs_dim + self.sub_goal_dim, self.action_dim).to(self.device)
        actor_dict['critic'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
        actor_dict['critic_target'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
        actor_dict['actor_optimizer'] = Adam(actor_dict['actor'].parameters(), lr=action_lr)
        actor_dict['critic_optimizer'] = Adam(actor_dict['critic'].parameters(), lr=action_lr)
        if self.double_q:
            actor_dict['critic_2'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
            actor_dict['critic_2_target'] = Critic(self.obs_dim + self.sub_goal_dim + self.action_dim).to(self.device)
            actor_dict['critic_2_optimizer'] = Adam(actor_dict['critic_2'].parameters(), lr=action_lr)
        return actor_dict

    def save_ckpts(self, epoch, intra=True, inter=True):
        if intra:
            if not self.multi_intra:
                self.save_networks(actor_dict=self.actor, optor_dict=None, epoch=epoch, ind=None)
            else:
                for _ in range(len(self.actors)):
                    self.save_networks(actor_dict=self.actors[_], optor_dict=None, epoch=epoch, ind=_)
        if inter:
            if not self.multi_inter:
                self.save_networks(actor_dict=None, optor_dict=self.optor, epoch=epoch, ind=None)
            else:
                for _ in range(len(self.optors)):
                    self.save_networks(actor_dict=None, optor_dict=self.optors[_], epoch=epoch, ind=_)

    def save_networks(self, actor_dict, optor_dict, epoch, ind=None):
        if ind is None:
            ckpt_mark = str(epoch) + '.pt'
        else:
            ckpt_mark = str(epoch) + 'No' + str(ind) + '.pt'
        if optor_dict is not None:
            T.save(optor_dict['optor_target'].state_dict(), self.ckpt_path + '/ckpt_optor_target_epoch' + ckpt_mark)
            if self.double_q:
                T.save(optor_dict['optor_2_target'].state_dict(), self.ckpt_path + '/ckpt_optor_2_target_epoch' + ckpt_mark)
        if actor_dict is not None:
            T.save(actor_dict['actor_target'].state_dict(), self.ckpt_path + '/ckpt_actor_target_epoch' + ckpt_mark)
            T.save(actor_dict['critic_target'].state_dict(), self.ckpt_path + '/ckpt_critic_target_epoch' + ckpt_mark)
            if self.double_q:
                T.save(actor_dict['critic_2_target'].state_dict(), self.ckpt_path + '/ckpt_critic_2_target_epoch' + ckpt_mark)

    def load_ckpts(self, epoch, intra=True, inter=True):
        if intra:
            if not self.multi_intra:
                self.load_network(actor_dict=self.actor, optor_dict=None, epoch=epoch, ind=None)
            else:
                for _ in range(len(self.actors)):
                    self.load_network(actor_dict=self.actors[_], optor_dict=None, epoch=epoch, ind=_)
        if inter:
            if not self.multi_inter:
                self.load_network(actor_dict=None, optor_dict=self.optor, epoch=epoch, ind=None)
            else:
                for _ in range(len(self.optors)):
                    self.load_network(actor_dict=None, optor_dict=self.optors[_], epoch=epoch, ind=_)

    def load_network(self, actor_dict, optor_dict, epoch, ind=None, specified_path=None):
        if ind is None:
            ckpt_mark = str(epoch) + '.pt'
        else:
            ckpt_mark = str(epoch) + 'No' + str(ind) + '.pt'
        if specified_path is None:
            path = self.ckpt_path
        else:
            path = specified_path
        if optor_dict is not None:
            optor_dict['optor'].load_state_dict(T.load(path + '/ckpt_optor_target_epoch' + ckpt_mark, map_location=self.device))
            optor_dict['optor_target'].load_state_dict(T.load(path + '/ckpt_optor_target_epoch' + ckpt_mark, map_location=self.device))
            if self.double_q:
                optor_dict['optor_2'].load_state_dict(T.load(path + '/ckpt_optor_2_target_epoch' + ckpt_mark, map_location=self.device))
                optor_dict['optor_2_target'].load_state_dict(T.load(path + '/ckpt_optor_2_target_epoch' + ckpt_mark, map_location=self.device))
        if actor_dict is not None:
            actor_dict['actor'].load_state_dict(T.load(path + '/ckpt_actor_target_epoch' + ckpt_mark, map_location=self.device))
            actor_dict['actor_target'].load_state_dict(T.load(path + '/ckpt_actor_target_epoch' + ckpt_mark, map_location=self.device))
            actor_dict['critic'].load_state_dict(T.load(path + '/ckpt_critic_target_epoch' + ckpt_mark, map_location=self.device))
            actor_dict['critic_target'].load_state_dict(T.load(path + '/ckpt_critic_target_epoch' + ckpt_mark, map_location=self.device))
            if self.double_q:
                actor_dict['critic_2'].load_state_dict(T.load(path + '/ckpt_critic_2_target_epoch' + ckpt_mark, map_location=self.device))
                actor_dict['critic_2_target'].load_state_dict(T.load(path + '/ckpt_critic_2_target_epoch' + ckpt_mark, map_location=self.device))
