import numpy as np
from scipy.spatial.transform import Rotation
from gym.envs.robotics.utils import ctrl_set_action, mocap_set_action, robot_get_obs, reset_mocap_welds
from multigoal_env import multigoal_fetch_base_env


class MultiGoalFetch(multigoal_fetch_base_env.MultiGoalFetchBaseEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
            self, model_path, n_substeps, initial_qpos,
            gripper_extra_height, block_gripper,
            has_object, has_g_block, r_block_fixed, randomized_block_pos,
            target_offset, obj_range, target_range,
            goal_has_gripper_pos,
            reward_type, distance_threshold, binary_final_goal,
            move_block_target=False, block_ground_pos_z=0.425, rotational_control_z=False,
            block_size=0.025, randomized_block_size=False
    ):
        """Initializes a new Fetch environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.block_ground_pos_z = block_ground_pos_z
        self.block_size = block_size
        self.block_height = 2*self.block_size
        self.grasping_gripper_state = np.array([self.block_size - 0.002, self.block_size - 0.002])
        self.randomized_block_size = randomized_block_size

        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.rotational_control_z = rotational_control_z
        if self.rotational_control_z:
            n_actions = 5
        else:
            n_actions = 4

        self.has_object = has_object
        self.has_g_block = has_g_block
        self.r_block_fixed = r_block_fixed
        self.randomized_block_pos = randomized_block_pos

        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range

        self.move_block_target = move_block_target
        self.block_targets_ground = {}
        self.block_targets_air = {}
        self.goal_has_gripper_pos = goal_has_gripper_pos

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.binary_final_goal = binary_final_goal
        super(MultiGoalFetch, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions,
            initial_qpos=initial_qpos)

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_extrinsic_reward(self, achieved_goal, goal, high_level_action, info):
        if high_level_action != self.goal_str:
            return -1.0
        else:
            if self.binary_final_goal:
                return np.array_equal(achieved_goal, goal).__float__() - 1.0
            else:
                d = self.goal_distance(achieved_goal, goal)
                return -(d > self.distance_threshold).astype(np.float32)
    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        if not self.rotational_control_z:
            assert action.shape == (4,)
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        else:
            assert action.shape == (5,)
            pos_ctrl, rot_z, gripper_ctrl = action[:3], action[3], action[4]
            rot_ctrl = Rotation.from_euler('xyz', [180, -90, rot_z*90], degrees=True).as_quat()

        pos_ctrl *= 0.05  # limit maximum change in position
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        # gripper finger control (symmetric)
        ctrl_set_action(self.sim, action)
        # gripper xyz position control
        mocap_set_action(self.sim, action)

    def _get_obs(self):
        # gripper position
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # gripper linear velocities
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        # gripper finger states & linear velocities
        finger_state = robot_qpos[-2:]
        finger_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        if not self.rotational_control_z:
            gripper_state = np.concatenate([
                grip_pos, grip_velp, finger_vel, finger_state
            ])
        else:
            grip_quat = self.sim.data.get_body_xquat('robot0:gripper_link')
            gripper_state = np.concatenate([
                grip_pos, grip_velp, grip_quat, finger_vel, finger_state
            ])

        # quat = self.sim.data.get_body_xquat('robot0:mocap')
        # euler = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
        # print('mocap quat: {}, euler: {}'.format(quat, euler))
        # grip_quat = self.sim.data.get_body_xquat('robot0:gripper_link')
        # grip_euler = Rotation.from_quat(grip_quat).as_euler('xyz', degrees=True)
        # print('grip quat: {}, euler: {}'.format(grip_quat, grip_euler))

        # object positions
        block_r_pos = self.sim.data.get_site_xpos('block_r')
        block_b_pos = self.sim.data.get_site_xpos('block_b')
        # object linear & angular velocities
        block_r_velp = self.sim.data.get_site_xvelp('block_r') * dt
        block_r_velr = self.sim.data.get_site_xvelr('block_r') * dt
        block_b_velp = self.sim.data.get_site_xvelp('block_b') * dt
        block_b_velr = self.sim.data.get_site_xvelr('block_b') * dt
        # object relative (to gripper) position & linear velocities
        block_r_rel_pos = grip_pos - block_r_pos
        block_r_rel_velp = grip_velp - block_r_velp
        block_b_rel_pos = grip_pos - block_b_pos
        block_b_rel_velp = grip_velp - block_b_velp
        # quaternions of blocks
        block_r_quat = self.sim.data.get_body_xquat('block_r')
        block_b_quat = self.sim.data.get_body_xquat('block_b')

        if self.goal_has_gripper_pos:
            achieved_sub_goal = np.concatenate([
                # gripper state
                finger_state.copy(), grip_pos.ravel(),
                # absolute position of blocks
                block_r_pos.ravel(), block_b_pos.ravel(),
            ])
        else:
            achieved_sub_goal = np.concatenate([
                # absolute position of blocks
                block_r_pos.ravel(), block_b_pos.ravel(),
            ])
        achieved_final_goal = achieved_sub_goal.copy()

        obs = np.concatenate([
            # linear position, velocities of gripper; linear velocities and width of fingers
            gripper_state,
            # absolution & relative linear positions of blocks
            block_r_pos.ravel(), block_r_rel_pos.ravel(),
            block_b_pos.ravel(), block_b_rel_pos.ravel(),
            # relative linear & angular velocities of blocks
            block_r_rel_velp.ravel(), block_r_velr.ravel(),
            block_b_rel_velp.ravel(), block_b_velr.ravel(),
        ])
        if self.rotational_control_z:
            obs = np.concatenate([
                obs,
                # quaternions of blocks
                block_r_quat.ravel(), block_b_quat.ravel(),
            ])

        if self.has_g_block:
            block_g_pos = self.sim.data.get_site_xpos('block_g')
            block_g_velp = self.sim.data.get_site_xvelp('block_g') * dt
            block_g_velr = self.sim.data.get_site_xvelr('block_g') * dt
            block_g_rel_pos = grip_pos - block_g_pos
            block_g_rel_velp = grip_velp - block_g_velp
            block_g_quat = self.sim.data.get_body_xquat('block_g')

            if self.goal_has_gripper_pos:
                achieved_sub_goal = np.concatenate([
                    # gripper state
                    finger_state.copy(), grip_pos.ravel(),
                    # absolute position of blocks
                    block_r_pos.ravel(), block_b_pos.ravel(), block_g_pos.ravel(),
                ])
            else:
                achieved_sub_goal = np.concatenate([
                    # absolute position of blocks
                    block_r_pos.ravel(), block_b_pos.ravel(), block_g_pos.ravel(),
                ])
            achieved_final_goal = achieved_sub_goal.copy()

            obs = np.concatenate([
                # linear position, velocities of gripper; linear velocities and width of fingers
                gripper_state,
                # absolution & relative linear positions of blocks
                block_r_pos.ravel(), block_r_rel_pos.ravel(),
                block_b_pos.ravel(), block_b_rel_pos.ravel(),
                block_g_pos.ravel(), block_g_rel_pos.ravel(),
                # relative linear & angular velocities of blocks
                block_r_rel_velp.ravel(), block_r_velr.ravel(),
                block_b_rel_velp.ravel(), block_b_velr.ravel(),
                block_g_rel_velp.ravel(), block_g_velr.ravel(),
            ])
            if self.rotational_control_z:
                obs = np.concatenate([
                    obs,
                    # quaternions of blocks
                    block_r_quat.ravel(), block_b_quat.ravel(), block_g_quat.ravel()
                ])

        # get a binary vector representing done flags of subgoals, as a final goal
        bin_vector = np.zeros(len(self.sub_goal_space))
        for i, (key, value) in enumerate(self.sub_goal_space.items()):
            bin_vector[i] = self._is_success(achieved_sub_goal, value)

        if self.binary_final_goal:
            achieved_final_goal = bin_vector.copy()

        return {
            'observation': obs.copy(),
            'desired_goal': self.goal.copy(),
            'achieved_goal': achieved_final_goal.copy(),
            'sub_goal': self.sub_goal.copy(),
            'achieved_sub_goal': achieved_sub_goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        if self.goal_has_gripper_pos:
            site_id = self.sim.model.site_name2id('target_gripper')
            self.sim.model.site_pos[site_id] = self.grip_targets[self.sub_goal_str] - sites_offset[0]

        if not self.r_block_fixed:
            site_id_r = self.sim.model.site_name2id('target_r')
            self.sim.model.site_pos[site_id_r] = self.object_targets[self.sub_goal_str][:3] - sites_offset[0]
        site_id_b = self.sim.model.site_name2id('target_b')
        self.sim.model.site_pos[site_id_b] = self.object_targets[self.sub_goal_str][3:6] - sites_offset[0]
        if self.has_g_block:
            site_id_g = self.sim.model.site_name2id('target_g')
            self.sim.model.site_pos[site_id_g] = self.object_targets[self.sub_goal_str][6:9] - sites_offset[0]

        self.sim.forward()

    def _reset_sim(self):
        # reset robot pose
        self.sim.set_state(self.initial_state)

        if self.randomized_block_size:
            self.block_size = self.np_random.uniform(0.015, 0.035)
            self.block_height = 2 * self.block_size
            self.grasping_gripper_state = np.array([self.block_size - 0.002, self.block_size - 0.002])
            objs = ['r', 'b', 'g']
            for color in objs:
                if (color == 'g') and (not self.has_g_block):
                    break
                obj_id = self.sim.model.geom_name2id('block_'+color)
                self.sim.model.geom_size[obj_id] = np.array([self.block_size, self.block_size, self.block_size])
                # print('obj id {}, size {}'.format(obj_id, self.block_size))

        if self.randomized_block_pos:
            container_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(container_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                container_xpos = self.initial_gripper_xpos[:2] + \
                                 self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            try:
                # this is used for bin packing environment only
                object_qpos = self.sim.data.get_joint_qpos('container:joint')
                assert object_qpos.shape == (7,)
                object_qpos[:2] = container_xpos
                self.sim.data.set_joint_qpos('container:joint', object_qpos)
            except:
                pass
            objs = ['r', 'b', 'g']
            for color in objs:
                if (color == 'g') and (not self.has_g_block):
                    break
                # randomize the position of blue block
                object_xpos = self.initial_gripper_xpos[:2]
                while (np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1) or \
                        (np.linalg.norm(object_xpos - container_xpos) < 0.1):
                    object_xpos = self.initial_gripper_xpos[:2] + \
                                  self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                object_qpos = self.sim.data.get_joint_qpos('block_'+color+':joint')
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                object_qpos[2] = self.block_ground_pos_z
                self.sim.data.set_joint_qpos('block_'+color+':joint', object_qpos)
                if self.move_block_target:
                    self.block_targets_ground[color] = object_qpos[:3].copy()
                    self.block_targets_ground[color][:2] = self.initial_gripper_xpos[:2] + \
                                                    self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                    self.block_targets_air[color] = self.block_targets_ground[color].copy()
                    self.block_targets_air[color][2] += self.np_random.uniform(0, 0.2)

        self.sim.forward()
        return True

    def _sample_goal(self):
        # this function will return a final goal
        ind = self.np_random.random_integers(0, len(self.final_goal_space)-1)
        goal_str = self.final_goal_strs[ind]
        return goal_str

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('block_r')[2]

    def render(self, mode='human', width=300, height=300):
        return super(MultiGoalFetch, self).render(mode, width, height)
