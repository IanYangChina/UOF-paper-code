import os


class Params:
    # training params
    ENV_ID = 0
    ENV_NAME = 'TwoObjectOneOrderBlockInformedLowLvGoal-v0'
    TRAINING_EPOCH = 301
    TRAINING_CYCLE = 50
    TRAINING_EPISODE = 16
    TESTING_EPISODE = 30
    TESTING_GAP = 1
    TESTING_TIMESTEP = 50
    SAVING_GAP = 50
    SEED = 0
    PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'result', 'task_'+str(ENV_ID))
    CKPT_PATH = os.path.join(PATH, "ckpts")
    DATA_PATH = os.path.join(PATH, "data")
    LOAD_PER_TRAIN_POLICY = True
    PRE_TRAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'pretrained_policy', 'task_'+str(ENV_ID))
    PRE_TRAIN_CKPT_PATH = os.path.join(PRE_TRAIN_PATH, "ckpts")
    PRE_TRAIN_DATA_PATH = os.path.join(PRE_TRAIN_PATH, "data")

    # high-level policy params
    HIGH_LEVEL_TRAIN = False
    HIGH_LEVEL_LEARNING_RATE = 0.001
    HIGH_LEVEL_GAMMA = 0.98
    HIGH_LEVEL_TAU = 0.1
    HIGH_LEVEL_BATCH_SIZE = 128
    HIGH_LEVEL_MEM_CAPACITY = int(1e6)
    HIGH_LEVEL_CLIP_VALUE = -50
    HIGH_LEVEL_OPTIMIZATION_STEP = 40
    HIGH_LEVEL_EXPLORATION_START = 1.0
    HIGH_LEVEL_EXPLORATION_END = 0.02
    HIGH_LEVEL_EXPLORATION_DECAY = 30000
    LOAD_HIGH_LEVEL_INPUT_NORMALIZATION_STATISTICS = False
    MULTI_INTER_POLICY = False

    # low-level policy params
    LOW_LEVEL_TRAIN = False
    LOW_LEVEL_LEARNING_RATE = 0.001
    LOW_LEVEL_GAMMA = 0.98
    LOW_LEVEL_TAU = 0.1
    LOW_LEVEL_BATCH_SIZE = 128
    LOW_LEVEL_MEM_CAPACITY = int(1e6)
    LOW_LEVEL_CLIP_VALUE = -25
    LOW_LEVEL_OPTIMIZATION_STEP = 40
    LOW_LEVEL_HINDSIGHT_REPLAY = True
    LOW_LEVEL_EXPLORATION_ALPHA = 0.2
    LOW_LEVEL_EXPLORATION_SIGMA = 0.05
    LOAD_LOW_LEVEL_INPUT_NORMALIZATION_STATISTICS = False

    # AAES & demonstration
    LOW_LEVEL_EXPLORATION_AAES = True
    LOW_LEVEL_EXPLORATION_AAES_TAU = 0.05
    ABSTRACT_DEMONSTRATION = True
    ABSTRACT_DEMONSTRATION_PROPORTION = 0.75
