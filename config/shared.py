# tasks, see Table II, III in the paper
env_ids = [
    # basic tasks
    'TwoObjectOneOrderBinaryHighLvGoal-v0',
    'ThreeObjectOneStageTwoOrderBinaryHighLvGoal-v0',
    'ThreeObjectOneStageSixOrder-v0',
    'ThreeObjectTwoStageTwoOrder-v0',
    # additional tasks
    'Pyramid-v0',
    'TwoObjectOneOrderRotation-v0',
    'TwoObjectRandomSize-v0',
    'ThreeObjectOneStageTwoOrderRandomSize-v0',
]

training_epochs = [301, 801, 1001, 1501, 2000, 300, 301, 801]
testing_timesteps = [50, 50, 50, 60, 80, 50, 50, 50]