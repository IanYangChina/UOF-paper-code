from gym.envs.registration import register, make


# Fetch Pick and Place
# ----------------------------------------
ids = ['TwoObjectOneOrderBlockInformedLowLvGoal-v0']
register(
    id='TwoObjectOneOrderBlockInformed-v0',
    entry_point='multigoal_env.fetch_envs.two_obj_block_informed:MGPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse',
            'binary_final_goal': True},
    max_episode_steps=25,
)

ids.append('TwoObjectOneOrderRotation-v0')
register(
    id='TwoObjectOneOrderRotation-v0',
    entry_point='multigoal_env.fetch_envs.two_obj_one_order_rotation:MGPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse',
            'binary_final_goal': True},
    max_episode_steps=25,
)

ids.append('TwoObjectOneOrderHAC-v0')
register(
    id='TwoObjectOneOrderHAC-v0',
    entry_point='multigoal_env.fetch_envs.two_obj_one_order:MGPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse',
            'binary_final_goal': False},
    max_episode_steps=25,
)

ids.append('ThreeObjectOneStageTwoOrderHAC-v0')
register(
    id='ThreeObjectOneStageTwoOrderHAC-v0',
    entry_point='multigoal_env.fetch_envs.three_obj_one_stage_two_order:MGPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse',
            'binary_final_goal': False},
    max_episode_steps=25,
)

ids.append('TwoObjectRandomSize-v0')
register(
    id='TwoObjectRandomSize-v0',
    entry_point='multigoal_env.fetch_envs.two_obj_one_order:MGPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse',
            'binary_final_goal': True,
            'randomized_block_size': True},
    max_episode_steps=25,
)

ids.append('ThreeObjectOneStageTwoOrderRandomSize-v0')
register(
    id='ThreeObjectOneStageTwoOrderRandomSize-v0',
    entry_point='multigoal_env.fetch_envs.three_obj_one_stage_two_order:MGPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse',
            'binary_final_goal': True,
            'randomized_block_size': True},
    max_episode_steps=25,
)

for high_level_goal_type in ['BlockInformed', 'Binary']:
    if high_level_goal_type == 'BlockInformed':
        binary_final_goal = False
    else:
        binary_final_goal = True

    ids.append('TwoObjectOneOrder{}HighLvGoal-v0'.format(high_level_goal_type))
    register(
        id='TwoObjectOneOrder{}HighLvGoal-v0'.format(high_level_goal_type),
        entry_point='multigoal_env.fetch_envs.two_obj_one_order:MGPickAndPlaceEnv',
        kwargs={'reward_type': 'sparse',
                'binary_final_goal': binary_final_goal},
        max_episode_steps=25,
                    )

    ids.append('ThreeObjectOneStageTwoOrder{}HighLvGoal-v0'.format(high_level_goal_type))
    register(
        id='ThreeObjectOneStageTwoOrder{}HighLvGoal-v0'.format(high_level_goal_type),
        entry_point='multigoal_env.fetch_envs.three_obj_one_stage_two_order:MGPickAndPlaceEnv',
        kwargs={'reward_type': 'sparse',
                'binary_final_goal': binary_final_goal},
        max_episode_steps=25,
                    )

ids.append('ThreeObjectOneStageSixOrder-v0')
register(
    id='ThreeObjectOneStageSixOrder-v0',
    entry_point='multigoal_env.fetch_envs.three_obj_one_stage_six_order:MGPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse',
            'binary_final_goal': True},
    max_episode_steps=25,
                )

ids.append('ThreeObjectTwoStageTwoOrder-v0'.format(high_level_goal_type))
register(
    id='ThreeObjectTwoStageTwoOrder-v0'.format(high_level_goal_type),
    entry_point='multigoal_env.fetch_envs.three_obj_two_stage_two_order:MGPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse',
            'binary_final_goal': True},
    max_episode_steps=40,
                )

ids.append('Pyramid-v0')
register(
    id='Pyramid-v0',
    entry_point='multigoal_env.fetch_envs.three_obj_bin_packing_two_stage:MGPickAndPlaceEnv',
    kwargs={'reward_type': 'sparse',
            'binary_final_goal': True},
    max_episode_steps=60,
)
