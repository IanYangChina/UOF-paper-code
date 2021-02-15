import os
import numpy as np
from copy import deepcopy as dcp
from gym import utils
from multigoal_env import multigoal_fetch_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'three_obj_bin_packing.xml')
BLOCK_HEIGHT = 0.05
GRASPING_HEIGHT_OFFSET = np.array([0.0, 0.0, 0.005])
GRIP_END_POS = np.array([1.25, 0.75, 0.65])
BLOCK_SIZES = {
    'r': 0.025,
    'b': 0.025,
    'g': 0.025
}
TRAY_HEIGHT = 0.02
# gripper finger state of grasping and placing
grasping_gripper_state_r = np.array([BLOCK_SIZES['r'] - 0.002, BLOCK_SIZES['r'] - 0.002])
grasping_gripper_state_b = np.array([BLOCK_SIZES['b'] - 0.002, BLOCK_SIZES['b'] - 0.002])
grasping_gripper_state_g = np.array([BLOCK_SIZES['g'] - 0.002, BLOCK_SIZES['g'] - 0.002])
ending_gripper_state = np.array([0.005, 0.005])


class MGPickAndPlaceEnv(multigoal_fetch_env.MultiGoalFetch, utils.EzPickle):
    def __init__(self,
                 reward_type='sparse',
                 binary_final_goal=True,
                 distance_threshold=0.02,
                 rotational_control_z=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            # 'name':[x, y, z, quaternion]
            'container:joint': [1.3, 0.5, 0.405, 1., 0., 0., 0.],
            'block_r:joint': [1.3, 0.5, 0.474, 1., 0., 0., 0.],
            'block_b:joint': [1.265, 0.5, 0.425, 1., 0., 0., 0.],
            'block_g:joint': [1.325, 0.5, 0.425, 1., 0., 0., 0.],
        }
        multigoal_fetch_env.MultiGoalFetch.__init__(
            self, model_path=MODEL_XML_PATH,
            has_object=True, has_g_block=True, r_block_fixed=False, randomized_block_pos=True,
            goal_has_gripper_pos=True,
            block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=distance_threshold,
            initial_qpos=initial_qpos, reward_type=reward_type, binary_final_goal=binary_final_goal,
            rotational_control_z=rotational_control_z)
        utils.EzPickle.__init__(self)
    # Additional methods
    # ----------------------------

    def _renew_goals(self):
        # this function refreshes goals
        block_r_pos = self.sim.data.get_site_xpos('block_r')
        block_b_pos = self.sim.data.get_site_xpos('block_b')
        block_g_pos = self.sim.data.get_site_xpos('block_g')

        block_r_ground_pos = dcp(block_r_pos)
        block_r_ground_pos[2] = self.block_ground_pos_z
        block_b_ground_pos = dcp(block_b_pos)
        block_b_ground_pos[2] = self.block_ground_pos_z
        block_g_ground_pos = dcp(block_g_pos)
        block_g_ground_pos[2] = self.block_ground_pos_z

        container_bottom_center = self.sim.data.get_site_xpos('container_center')
        container_bottom_center[2] = self.block_ground_pos_z + TRAY_HEIGHT
        container_bottom_front = self.sim.data.get_site_xpos('container_front')
        container_bottom_front[2] = self.block_ground_pos_z + TRAY_HEIGHT
        container_bottom_tail = self.sim.data.get_site_xpos('container_tail')
        container_bottom_tail[2] = self.block_ground_pos_z + TRAY_HEIGHT

        container_top_center = dcp(container_bottom_center)
        container_top_center[2] += BLOCK_HEIGHT
        container_top_front = dcp(container_bottom_front)
        container_top_front[2] += BLOCK_HEIGHT
        container_top_tail = dcp(container_bottom_tail)
        container_top_tail[2] += BLOCK_HEIGHT

        sub_goals = {
            "pick_red": np.concatenate([
                # gripper state & position
                grasping_gripper_state_r, (block_r_ground_pos + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                block_r_ground_pos.ravel(), block_b_ground_pos.ravel(), block_g_ground_pos.ravel()
            ]),
            "place_red_on_bottom": np.concatenate([
                # gripper state & position
                grasping_gripper_state_r, (container_bottom_center + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                container_bottom_center.ravel(), block_b_ground_pos.ravel(), block_g_ground_pos.ravel()
            ]),
            "pick_green_r_on_bottom": np.concatenate([
                # gripper state & position
                grasping_gripper_state_g, (block_g_ground_pos + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                container_bottom_center.ravel(), block_b_ground_pos.ravel(), block_g_ground_pos.ravel()
            ]),
            "place_green_on_top": np.concatenate([
                # gripper state & position
                grasping_gripper_state_g, (container_top_front + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                container_bottom_center.ravel(), block_b_ground_pos.ravel(), container_top_front.ravel()
            ]),
            "pick_blue_r_on_bottom": np.concatenate([
                # gripper state & position
                grasping_gripper_state_b, (block_b_ground_pos + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                container_bottom_center.ravel(), block_b_ground_pos.ravel(), container_top_front.ravel()
            ]),
            "place_blue_on_top": np.concatenate([
                # gripper state & position
                grasping_gripper_state_g, (container_top_tail + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                container_bottom_center.ravel(), container_top_tail.ravel(), container_top_front.ravel()
            ]),
            "end_gb_r": np.concatenate([
                # gripper state & position
                ending_gripper_state, GRIP_END_POS.ravel(),
                # absolute positions of blocks
                container_bottom_center.ravel(), container_top_tail.ravel(), container_top_front.ravel()
            ]),

            "pick_green": np.concatenate([
                # gripper state & position
                grasping_gripper_state_g, (block_g_ground_pos + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                block_r_ground_pos.ravel(), block_b_ground_pos.ravel(), block_g_ground_pos.ravel()
            ]),
            "place_green_on_bottom": np.concatenate([
                # gripper state & position
                grasping_gripper_state_g, (container_bottom_front + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                block_r_ground_pos.ravel(), block_b_ground_pos.ravel(), container_bottom_front.ravel()
            ]),
            "pick_blue": np.concatenate([
                # gripper state & position
                grasping_gripper_state_b, (block_b_ground_pos + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                block_r_ground_pos.ravel(), block_b_ground_pos.ravel(), container_bottom_front.ravel()
            ]),
            "place_blue_on_bottom": np.concatenate([
                # gripper state & position
                grasping_gripper_state_g, (container_bottom_tail + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                block_r_ground_pos.ravel(), container_bottom_tail.ravel(), container_bottom_front.ravel()
            ]),
            "pick_red_gb_on_bottom": np.concatenate([
                # gripper state & position
                grasping_gripper_state_r, (block_r_ground_pos + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                block_r_ground_pos.ravel(), container_bottom_tail.ravel(), container_bottom_front.ravel()
            ]),
            "place_red_on_top": np.concatenate([
                # gripper state & position
                grasping_gripper_state_r, (container_top_center + GRASPING_HEIGHT_OFFSET).ravel(),
                # absolute positions of blocks
                container_top_center.ravel(), container_bottom_tail.ravel(), container_bottom_front.ravel()
            ]),
            "end_r_gb": np.concatenate([
                # gripper state & position
                ending_gripper_state, GRIP_END_POS.ravel(),
                # absolute positions of blocks
                container_top_center.ravel(), container_bottom_tail.ravel(), container_bottom_front.ravel()
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
            object_target_positions[k] = v[-9:].copy()
        final_goals['pick_green_r_on_bottom'][7] = 1.0
        final_goals['pick_blue_r_on_bottom'][9] = 1.0
        final_goals['pick_red_gb_on_bottom'][0] = 1.0
        assert len(sub_goals) == len(final_goals) == len(gripper_target_positions) == len(object_target_positions)
        return dcp(sub_goals), dcp(final_goals), dcp(gripper_target_positions), dcp(object_target_positions)
