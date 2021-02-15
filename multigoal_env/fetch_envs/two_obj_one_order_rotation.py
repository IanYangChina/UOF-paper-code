import os
import numpy as np
from copy import deepcopy as dcp
from gym import utils
from multigoal_env import multigoal_fetch_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'two_obj_pick_and_place_rotation.xml')
BLOCK_HEIGHT = 0.05
BLOCK_SIZE = 0.025
GRASPING_HEIGHT_OFFSET = np.array([0.0, 0.0, 0.005])
GRIP_END_POS = np.array([1.25, 0.75, 0.65])
# gripper finger state of grasping and placing
grasping_gripper_state = np.array([BLOCK_SIZE - 0.002, BLOCK_SIZE - 0.002])
ending_gripper_state = np.array([0.005, 0.005])


class MGPickAndPlaceEnv(multigoal_fetch_env.MultiGoalFetch, utils.EzPickle):
    def __init__(self,
                 reward_type='sparse',
                 binary_final_goal=True):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            # 'name':[x, y, z, quaternion]
            'block_r:joint': [1.25, 0.75, 0.4, 1., 0., 0., 0.],
            'block_b:joint': [1.25, 0.9, 0.4, 1., 0., 0., 0.],
        }
        multigoal_fetch_env.MultiGoalFetch.__init__(
            self, model_path=MODEL_XML_PATH,
            has_object=True, has_g_block=False, r_block_fixed=True, randomized_block_pos=True,
            goal_has_gripper_pos=True,
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.02,
            initial_qpos=initial_qpos, reward_type=reward_type, binary_final_goal=binary_final_goal, rotational_control_z=True)
        utils.EzPickle.__init__(self)

    # Additional RobotEnv methods
    # ----------------------------

    def _renew_goals(self):
        # this function refreshes goals
        block_r_pos = self.sim.data.get_site_xpos('block_r')
        block_b_pos = self.sim.data.get_site_xpos('block_b')

        block_r_ground_pos = dcp(block_r_pos)
        block_r_ground_pos[2] = self.block_ground_pos_z
        block_b_ground_pos = dcp(block_b_pos)
        block_b_ground_pos[2] = self.block_ground_pos_z
        # absolute position of blue block on top of red block
        block_b_pos_on_red = dcp(block_r_pos)
        block_b_pos_on_red[2] += BLOCK_HEIGHT

        sub_goals = {
            "pick_blue": np.concatenate([
                # gripper state & position
                grasping_gripper_state, (block_b_ground_pos+GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                block_r_ground_pos.ravel(), block_b_ground_pos.ravel(),
            ]),
            "blue_on_red": np.concatenate([
                # gripper state & position
                grasping_gripper_state, (block_b_pos_on_red+GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                block_r_ground_pos.ravel(), block_b_pos_on_red.ravel(),
            ]),
            "ending_br": np.concatenate([
                # gripper state & position
                ending_gripper_state, GRIP_END_POS.ravel(),
                # absolute positions of blocks
                block_r_ground_pos.ravel(), block_b_pos_on_red.ravel(),
            ]),
        }

        # a final goal is a binary vector which represents the completion of subgoals
        final_goals = {}
        gripper_target_positions = {}
        object_target_positions = {}
        _ = np.zeros(len(sub_goals))
        for i, (k, v) in enumerate(sub_goals.items()):
            final_goals[k] = _.copy()
            final_goals[k][i] += 1.0
            gripper_target_positions[k] = v[2:5].copy()
            object_target_positions[k] = v[-6:].copy()
        assert len(sub_goals) == len(final_goals) == len(gripper_target_positions) == len(object_target_positions)
        return dcp(sub_goals), dcp(final_goals), dcp(gripper_target_positions), dcp(object_target_positions)
