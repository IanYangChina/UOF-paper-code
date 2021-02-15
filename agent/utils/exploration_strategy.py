import math as M
import numpy as np


class ExpDecayGreedy(object):
    def __init__(self,  start=1, end=0.02, decay=30000, decay_start=None):
        self.start = start
        self.end = end
        self.decay = decay
        self.decay_start = decay_start

    def __call__(self, ep):
        if self.decay_start is None:
            epsilon = self.end + (self.start - self.end) * M.exp(-1. * ep / self.decay)
        else:
            ep -= self.decay_start
            if ep < 0:
                ep = 0
            epsilon = self.end + (self.start - self.end) * M.exp(-1. * ep / self.decay)
        return epsilon


class ConstantChance(object):
    def __init__(self, chance=0.2):
        self.chance = chance

    def __call__(self, _=None):
        return self.chance


class AutoAdjustingConstantChance(object):
    def __init__(self, goal_num, tau=0.05, chance=0.2):
        self.base_chance = chance
        self.goal_num = goal_num
        self.tau = tau
        self.success_rates = np.zeros(self.goal_num)
        self.chance = np.ones(self.goal_num) * chance

    def update_success_rates(self, new_tet_suc_rate):
        old_tet_suc_rate = self.success_rates.copy()
        self.success_rates = (1-self.tau)*old_tet_suc_rate + self.tau*new_tet_suc_rate
        self.chance = self.base_chance*(1-self.success_rates)

    def __call__(self, goal_ind):
        return self.chance[goal_ind]
