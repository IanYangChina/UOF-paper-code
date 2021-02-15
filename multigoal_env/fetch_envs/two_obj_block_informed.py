import os
import numpy as np
from copy import deepcopy as dcp
from gym import utils
from multigoal_env import multigoal_fetch_env
from multigoal_env.demonstrator import StepDemonstrator

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'two_obj_pick_and_place.xml')
BLOCK_HEIGHT = 0.05


class MGPickAndPlaceEnv(multigoal_fetch_env.MultiGoalFetch, utils.EzPickle):
    def __init__(self,
                 reward_type='sparse',
                 binary_final_goal=True):
        self.demonstrator = StepDemonstrator([
            [0],
            [1],
            [2]
        ])
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
            goal_has_gripper_pos=False, move_block_target=True,
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.02,
            initial_qpos=initial_qpos, reward_type=reward_type, binary_final_goal=binary_final_goal)
        utils.EzPickle.__init__(self)

    # Additional RobotEnv methods
    # ----------------------------

    def _renew_goals(self):
        # this function refreshes goals
        block_r_pos = self.sim.data.get_site_xpos('block_r')
        # absolute position of blue block on top of red block
        block_b_pos_on_red = dcp(block_r_pos)
        block_b_pos_on_red[2] += BLOCK_HEIGHT

        # goals of moving block generate target locations sometimes on the table sometimes one the air
        # this is necessary to train a policy without gripper-informed goals,
        #                                                       as pointed out by the original HER paper
        sub_goals = {
            "move_blue_ground": np.concatenate([
                # absolute positions of blocks
                block_r_pos.ravel(), self.block_targets_ground['b'].ravel()
            ]),
            "move_blue_air": np.concatenate([
                # absolute positions of blocks
                block_r_pos.ravel(), self.block_targets_air['b'].ravel()
            ]),
            "blue_on_red": np.concatenate([
                # absolute positions of blocks
                block_r_pos.ravel(), block_b_pos_on_red.ravel()
            ]),
        }

        # a final goal is a binary vector which represents the completion of subgoals
        # without gripper-information, we can still train a high-level policy to select subgoals,
        #                                                                 although somewhat meaningless...
        final_goals = {}
        gripper_target_positions = {}
        object_target_positions = {}
        _ = np.zeros(len(sub_goals))
        for i, (k, v) in enumerate(sub_goals.items()):
            final_goals[k] = _.copy()
            final_goals[k][i] += 1.0
            gripper_target_positions[k] = v[-3:].copy()
            object_target_positions[k] = v[-6:].copy()
        assert len(sub_goals) == len(final_goals) == len(gripper_target_positions) == len(object_target_positions)
        return dcp(sub_goals), dcp(final_goals), dcp(gripper_target_positions), dcp(object_target_positions)
