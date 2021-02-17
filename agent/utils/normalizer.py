import numpy as np


def calculate_mean_var(sample_count, new_sample_num, new_history, old_mean, old_var):
    new_mean = np.mean(new_history, axis=0)

    new_var = np.sum(np.square(new_history - new_mean), axis=0)
    new_var = (sample_count * old_var + new_var)
    new_var /= (new_sample_num + sample_count)
    new_var = np.sqrt(new_var)

    new_mean = (sample_count * old_mean + new_sample_num * new_mean)
    new_mean /= (new_sample_num + sample_count)

    return new_mean, new_var


class Normalizer(object):
    def __init__(self, input_dims, init_mean, init_var, scale_factor=1, epsilon=1e-2, clip_range=3):
        self.input_dims = input_dims
        self.sample_count = 0
        self.obs_history = []
        if init_mean is None:
            self.obs_history_mean = np.zeros(self.input_dims)
        else:
            self.obs_history_mean = init_mean
        if init_var is None:
            self.obs_history_var = np.zeros(self.input_dims)
        else:
            self.obs_history_var = init_var
        self.epsilon = epsilon*np.ones(self.input_dims)
        self.input_clip_range = (-clip_range*np.ones(self.input_dims), clip_range*np.ones(self.input_dims))
        self.scale_factor = scale_factor

    def store_history(self, *args):
        self.obs_history.append(*args)

    # update mean and var for z-score normalization
    def update_mean(self):
        if len(self.obs_history) == 0:
            return
        new_sample_num = len(self.obs_history)

        self.obs_history_mean, self.obs_history_var = calculate_mean_var(
            self.sample_count, new_sample_num,
            np.array(self.obs_history, dtype=np.float),
            self.obs_history_mean.copy(), self.obs_history_var.copy()
        )
        self.obs_history.clear()

        self.sample_count += new_sample_num
        self.obs_history.clear()

    # pre-process inputs, currently using max-min-normalization
    def __call__(self, inputs):
        inputs = (inputs - self.obs_history_mean) / (self.obs_history_var+self.epsilon)
        inputs = np.clip(inputs, self.input_clip_range[0], self.input_clip_range[1])
        return self.scale_factor*inputs


class GoalEnvNormalizer(object):
    def __init__(self, obs_dims, goal_dims_low, goal_dims_high,
                 input_mean_low=None, input_var_low=None, different_goals=True,
                 scale_factor=1, epsilon=1e-2, clip_range=3):
        self.obs_dims = obs_dims
        self.obs_history = []
        self.goal_dims_low = goal_dims_low
        self.goal_history_low = []
        self.goal_dims_high = goal_dims_high
        self.goal_history_high = []

        self.different_goals = different_goals
        self.set_statistics(input_mean_low, input_var_low)

        self.sample_count = 0
        self.epsilon_low = epsilon * np.ones(self.obs_dims+self.goal_dims_low)
        self.input_clip_range_low = (-clip_range*np.ones(self.obs_dims+self.goal_dims_low),
                                     clip_range*np.ones(self.obs_dims+self.goal_dims_low))
        self.epsilon_high = epsilon * np.ones(self.obs_dims+self.goal_dims_high)
        self.input_clip_range_high = (-clip_range*np.ones(self.obs_dims+self.goal_dims_high),
                                      clip_range*np.ones(self.obs_dims+self.goal_dims_high))
        self.scale_factor = scale_factor

    def set_statistics(self, mean=None, var=None):
        if mean is None:
            assert var is None
            self.obs_history_mean = np.zeros(self.obs_dims)
            self.goal_history_mean_low = np.zeros(self.goal_dims_low)
            self.obs_history_var = np.zeros(self.obs_dims)
            self.goal_history_var_low = np.zeros(self.goal_dims_low)
        else:
            self.obs_history_mean = mean[:self.obs_dims]
            self.goal_history_mean_low = mean[self.obs_dims:]
            self.obs_history_var = var[:self.obs_dims]
            self.goal_history_var_low = var[self.obs_dims:]

        if self.different_goals:
            self.goal_history_mean_high = np.zeros(self.goal_dims_high)
            self.goal_history_var_high = np.ones(self.goal_dims_high)
        else:
            self.goal_history_mean_high = self.goal_history_mean_low.copy()
            self.goal_history_var_high = self.goal_history_var_low.copy()

        self.input_mean_low = np.concatenate((self.obs_history_mean, self.goal_history_mean_low), axis=0)
        self.input_var_low = np.concatenate((self.obs_history_var, self.goal_history_var_low), axis=0)
        self.input_mean_high = np.concatenate((self.obs_history_mean, self.goal_history_mean_high), axis=0)
        self.input_var_high = np.concatenate((self.obs_history_var, self.goal_history_var_high), axis=0)

    def store_history(self, obs, goal_low, goal_high):
        self.obs_history.append(obs)
        self.goal_history_low.append(goal_low)
        if self.different_goals:
            self.goal_history_high.append(goal_high)

    # update mean and var for z-score normalization
    def update(self):
        if len(self.obs_history) == 0:
            return
        new_sample_num = len(self.obs_history)

        self.obs_history_mean, self.obs_history_var = calculate_mean_var(
            self.sample_count, new_sample_num,
            np.array(self.obs_history, dtype=np.float),
            self.obs_history_mean.copy(), self.obs_history_var.copy()
        )
        self.obs_history.clear()

        self.goal_history_mean_low, self.goal_history_var_low = calculate_mean_var(
            self.sample_count, new_sample_num,
            np.array(self.goal_history_low, dtype=np.float),
            self.goal_history_mean_low.copy(), self.goal_history_var_low.copy()
        )
        self.goal_history_low.clear()

        if self.different_goals:
            # high level goals are binary vectors as default if differ from the low level
            # it is unnecessary to normalise binary vectors
            self.goal_history_high.clear()
        else:
            self.goal_history_mean_high = self.goal_history_mean_low.copy()
            self.goal_history_var_high = self.goal_history_var_low.copy()

        self.input_mean_low = np.concatenate((self.obs_history_mean, self.goal_history_mean_low), axis=0)
        self.input_var_low = np.concatenate((self.obs_history_var, self.goal_history_var_low), axis=0)
        self.input_mean_high = np.concatenate((self.obs_history_mean, self.goal_history_mean_high), axis=0)
        self.input_var_high = np.concatenate((self.obs_history_var, self.goal_history_var_high), axis=0)
        self.sample_count += new_sample_num

    # pre-process inputs, currently using mean-variance-normalization
    def __call__(self, inputs, level='low'):
        if level == 'low':
            inputs = (inputs - self.input_mean_low) / (self.input_var_low + self.epsilon_low)
            inputs = np.clip(inputs, self.input_clip_range_low[0], self.input_clip_range_low[1])
        elif level == 'high':
            inputs = (inputs - self.input_mean_high) / (self.input_var_high + self.epsilon_high)
            inputs = np.clip(inputs, self.input_clip_range_high[0], self.input_clip_range_high[1])
        else:
            raise ValueError('goal level should be low or high')
        return self.scale_factor*inputs
