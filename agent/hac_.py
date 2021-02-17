import os
import random as R
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.normalizer import GoalEnvNormalizer
from agent.utils.networks import *
from agent.utils.replay_buffer import *
from agent.utils.exploration_strategy import *


class HierarchicalActorCritic(object):
    def __init__(self, params, env_params, tr, path=None, seed=0,
                 option_lr=0.001, opt_tau=0.1, opt_batch_size=128, opt_mem_capacity=int(1e6), opt_clip_value=-50.0,
                 opt_gamma=0.98, opt_optim_steps=40,
                 action_lr=0.001, act_tau=0.1, act_batch_size=128, act_mem_capacity=int(1e6), act_clip_value=-50.0,
                 act_gamma=0.98, act_optim_steps=40,
                 chance=0.2, deviation=0.05):
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
        # sub goal dimension is also the option dimension
        self.sub_goal_dim = env_params['sub_goal_dims']
        self.normalizer = GoalEnvNormalizer(self.obs_dim, self.sub_goal_dim, self.goal_dim,
                                            env_params['act_init_input_means'], env_params['act_init_input_vars'],
                                            env_params['opt_init_input_means'], env_params['opt_init_input_vars'],
                                            env_params['different_goals']
                                            )

        """High-Level policies - DDPG"""
        # Learning params
        self.opt_batch_size = opt_batch_size
        self.opt_gamma = opt_gamma
        self.opt_tau = opt_tau
        self.opt_clip_value = opt_clip_value
        self.opt_optim_steps = opt_optim_steps
        # Get optors
        self.optor = self.get_optor(opt_mem_capacity, tr, seed, option_lr)
        self.current_opt_memory = self.optor['optor_memory']
        self.current_optor_dict = None
        self.option_min = -0.25 * np.ones(self.sub_goal_dim)
        self.option_max = 1.60 * np.ones(self.sub_goal_dim)

        """Low-Level policies - DDPG"""
        # Learning Params
        self.act_batch_size = act_batch_size
        self.act_gamma = act_gamma
        self.act_tau = act_tau
        self.act_clip_value = act_clip_value
        self.act_optim_steps = act_optim_steps
        # Get actors
        self.action_dim = env_params['action_dims']
        self.action_max = env_params['action_max']

        self.actor = self.get_actor(act_mem_capacity, tr, seed, action_lr)
        self.act_soft_update(self.actor, act_tau=1.0)
        self.current_act_memory = self.actor['actor_memory']

        self.current_actor_dict = None

        """Exploration"""
        self.exploration = ConstantChance(chance=chance)
        self.noise_deviation = deviation

    def run(self):
        pass

    def select_action(self, state, desired_goal, test=False):
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
            eps = self.exploration()
            if _ < eps:
                action = np.random.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
            else:
                deviation = self.noise_deviation
                action += deviation * np.random.randn(self.action_dim)
                action = np.clip(action, -self.action_max, self.action_max)
        return action

    def select_option(self, state, desired_goal, test=False):
        self.current_optor_dict = self.optor
        current_optor = self.optor['optor']
        current_optor.eval()
        inputs = np.concatenate((state, desired_goal), axis=0)
        inputs = self.normalizer(inputs, level='low')
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        option = current_optor(inputs).cpu().detach().numpy()

        if test:
            option = np.clip(option, -self.option_max, self.option_max)
        else:
            _ = R.uniform(0, 1)
            eps = self.exploration()
            if _ < eps:
                option = np.random.uniform(-self.option_max, self.option_max, size=(self.sub_goal_dim,))
            else:
                deviation = self.noise_deviation
                option += deviation * np.random.randn(self.sub_goal_dim)
                option = np.clip(option, self.option_min, self.option_max)

        # network output is within [-1, 1], rescale it to be within [-0.25, 1.60]
        option = (option+1)*1.85/2 - 0.25
        return option

    def update(self, act_hindsight=True, opt_hindsight=True):
        self.act_learn(self.actor, act_hindsight)
        self.opt_learn(self.optor, opt_hindsight)

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
            value_target = rewards + done * self.act_gamma * value_
            value_target = T.clamp(value_target, self.act_clip_value, -0.0)

            critic_inputs = T.cat((actor_inputs, actions), dim=1).to(self.device)
            value_estimate = actor_dict['critic'](critic_inputs)
            actor_dict['critic_optimizer'].zero_grad()
            critic_loss = F.mse_loss(value_estimate, value_target.detach())
            critic_loss.backward()
            actor_dict['critic_optimizer'].step()

            actor_dict['critic'].eval()
            actor_dict['actor'].train()
            actor_dict['actor_optimizer'].zero_grad()
            new_actions = actor_dict['actor'](actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
            new_value = actor_dict['critic'](critic_eval_inputs)
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
            optor_inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
            optor_inputs = self.normalizer(optor_inputs, level='low')
            optor_inputs = T.tensor(optor_inputs, dtype=T.float32).to(self.device)
            options = T.tensor(batch.action, dtype=T.float32).to(self.device)
            # scale the stored options from [-0.25, 1.60] to be within [-1, 1]
            options = (options+0.25)*2/1.85 - 1
            optor_inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            optor_inputs_ = self.normalizer(optor_inputs_, level='low')
            optor_inputs_ = T.tensor(optor_inputs_, dtype=T.float32).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
            done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

            optor_dict['optor'].eval()
            optor_dict['critic'].train()
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

            optor_dict['critic'].eval()
            optor_dict['optor'].train()
            optor_dict['optor_optimizer'].zero_grad()
            new_options = optor_dict['optor'](optor_inputs)
            critic_eval_inputs = T.cat((optor_inputs, new_options), dim=1).to(self.device)
            new_value = optor_dict['critic'](critic_eval_inputs)
            optor_loss = -new_value
            optor_loss = optor_loss.mean()
            optor_loss.backward()
            optor_dict['optor_optimizer'].step()
            optor_dict['optor'].eval()
        return

    def act_soft_update(self, actor_dict, act_tau=None):
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

    def get_optor(self, opt_mem_capacity, opt_tr, seed, option_lr):
        optor_dict = dict()
        optor_dict['optor_memory'] = HACReplayBuffer(opt_mem_capacity, opt_tr, seed=seed)
        optor_dict['optor'] = Actor(self.obs_dim + self.sub_goal_dim, self.sub_goal_dim).to(self.device)
        optor_dict['critic'] = Critic(self.obs_dim + self.sub_goal_dim + self.sub_goal_dim).to(self.device)
        optor_dict['optor_optimizer'] = Adam(optor_dict['optor'].parameters(), lr=option_lr)
        optor_dict['critic_optimizer'] = Adam(optor_dict['critic'].parameters(), lr=option_lr)
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
        return actor_dict

    def save_ckpts(self, epoch):
        self.save_networks(actor_dict=self.actor, optor_dict=self.optor, epoch=epoch, ind=None)

    def save_networks(self, actor_dict, optor_dict, epoch, ind=None):
        if ind is None:
            ckpt_mark = str(epoch) + '.pt'
        else:
            ckpt_mark = str(epoch) + 'No' + str(ind) + '.pt'
        if optor_dict is not None:
            T.save(optor_dict['optor'].state_dict(), self.ckpt_path + '/ckpt_optor_epoch' + ckpt_mark)
            T.save(optor_dict['critic'].state_dict(), self.ckpt_path + '/ckpt_optor_critic_epoch' + ckpt_mark)

        if actor_dict is not None:
            T.save(actor_dict['actor_target'].state_dict(), self.ckpt_path + '/ckpt_actor_target_epoch' + ckpt_mark)
            T.save(actor_dict['critic_target'].state_dict(), self.ckpt_path + '/ckpt_critic_target_epoch' + ckpt_mark)

    def load_ckpts(self, epoch):
        self.load_network(actor_dict=self.actor, optor_dict=self.optor, epoch=epoch, ind=None)

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
            optor_dict['optor'].load_state_dict(T.load(path + '/ckpt_optor_epoch' + ckpt_mark, map_location=self.device))
            optor_dict['critic'].load_state_dict(T.load(path + '/ckpt_optor_critic_epoch' + ckpt_mark, map_location=self.device))

        if actor_dict is not None:
            actor_dict['actor'].load_state_dict(T.load(path + '/ckpt_actor_target_epoch' + ckpt_mark, map_location=self.device))
            actor_dict['actor_target'].load_state_dict(T.load(path + '/ckpt_actor_target_epoch' + ckpt_mark, map_location=self.device))
            actor_dict['critic'].load_state_dict(T.load(path + '/ckpt_critic_target_epoch' + ckpt_mark, map_location=self.device))
            actor_dict['critic_target'].load_state_dict(T.load(path + '/ckpt_critic_target_epoch' + ckpt_mark, map_location=self.device))
