import os
import copy
import numpy as np

from gym import error, spaces, GoalEnv
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))

DEFAULT_SIZE = 500


class MultiGoalFetchBaseEnv(GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        # initialize goals
        self.sub_goal_space, self.final_goal_space, self.grip_targets, self.object_targets = self._renew_goals()
        self.sub_goal_strs = [key for key in self.sub_goal_space]
        self.final_goal_strs = [key for key in self.final_goal_space]
        # sample a final goal
        self.goal_str = self.sub_goal_str = self._sample_goal()
        self.goal = self.final_goal_space[self.goal_str].copy()
        self.sub_goal = self.sub_goal_space[self.sub_goal_str].copy()

        obs = self._get_obs()
        self.option_space = spaces.Discrete(len(self.sub_goal_strs))
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            sub_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_sub_goal'].shape, dtype='float32'),
            achieved_sub_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_sub_goal'].shape, dtype='float32')
        ))

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()

        self.sub_goal_space, self.final_goal_space, self.grip_targets, self.object_targets = self._renew_goals()
        self.goal = self.final_goal_space[self.goal_str].copy()
        self.sub_goal = self.sub_goal_space[self.sub_goal_str].copy()
        obs = self._get_obs()

        time_done = False
        info = {
            'is_sub_goal_reached': self._is_success(obs['achieved_sub_goal'], self.sub_goal),
        }
        act_reward = self.compute_reward(obs['achieved_sub_goal'], self.sub_goal, info)

        return obs, act_reward, time_done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(MultiGoalFetchBaseEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.sub_goal_space, self.final_goal_space, self.grip_targets, self.object_targets = self._renew_goals()
        self.goal_str = self.sub_goal_str = self._sample_goal()
        self.goal = self.final_goal_space[self.goal_str].copy()
        self.sub_goal = self.sub_goal_space[self.sub_goal_str].copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new final goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    # Additional methods
    # ----------------------------

    def option_step(self, obs_, option, act_reward, option_is_goal=False):
        opt_info = {}
        if not option_is_goal:
            opt_reward = self.compute_extrinsic_reward(obs_['achieved_goal'], self.goal, self.sub_goal_strs[option], opt_info)
        else:
            opt_reward = self.compute_reward(obs_['achieved_goal'], self.goal, opt_info)
        assert act_reward in [-1.0, 0.0]
        sub_goal_done = bool(1+act_reward)
        assert opt_reward in [-1.0, 0.0]
        final_goal_done = bool(1+opt_reward)
        return opt_reward, sub_goal_done, opt_info, final_goal_done

    def set_goals(self, desired_goal_ind=None, sub_goal_ind=None, sub_goal=None):
        if sub_goal_ind is not None:
            self.sub_goal_str = self.sub_goal_strs[sub_goal_ind]
            self.sub_goal = self.sub_goal_space[self.sub_goal_str].copy()
        if sub_goal is not None:
            self.sub_goal = sub_goal.copy()
        if desired_goal_ind is not None:
            self.goal_str = self.final_goal_strs[desired_goal_ind]
            self.goal = self.final_goal_space[self.goal_str].copy()

    def compute_extrinsic_reward(self, achieved_goal, goal, high_level_action, info):
        """This method computes extrinsic reward for the higher level policy
        """
        raise NotImplementedError

    def _renew_goals(self):
        """Initializes sub-goal (option) space
        """
        raise NotImplementedError()
