import os
from trainer_to_be_published import Trainer

algo_params = {
    'demonstrations': [
        [0], [0, 1], [0, 1, 2]
    ],
    'use_demonstration_in_training': True,
    'demonstrated_episode_proportion': 0.75,
    # double critics/q-functions for both levels
    'double_q': True,
    # parameters for the intra-option/low-level/control policy
    'act_train': True,
    'act_separate_policy': False,
    'act_exploration': 'AAES',
    'act_exploration_C_alpha': 0.2,
    'act_exploration_C_sigma': 0.05,
    'act_hindsight': True,
    'act_clip_value': -25,
    'act_batch_size': 128,
    'act_gamma': 0.98,
    'act_tau': 0.1,
    'act_optim_steps': 40,
    'act_mem_capacity': int(1e6),
    'act_learning_rate': 0.001,
    # parameters for the inter-option/high-level/planning policy
    'opt_train': True,
    'opt_separate_policy': False,
    'opt_exploration_decay_parameter': 30000,
    'opt_hindsight': False,
    'opt_clip_value': None,
    'opt_intra_option_learning': False,
    'opt_batch_size': 128,
    'opt_gamma': 0.98,
    'opt_tau': 0.1,
    'opt_optim_steps': 40,
    'opt_mem_capacity': int(1e6),
    'opt_learning_rate': 0.001,
}

exp_params = {
    'path': None,  # to be specified later
    'seed': None,  # to be specified later
    'load_init_input': False,
    'init_input_path': None,
    'training_epoch': 151,
    'training_cycle': 50,
    'training_episode': 16,
    'testing_episode_per_step': 30,
    'testing_time_steps': None,
    'testing_gap': 1,  # unit: epoch
    'saving_gap': 50,  # unit: epoch
}

env_id = 'TwoObjectOneOrderBinaryHighLvGoal-v0'

result_dir = os.path.dirname(os.path.realpath(__file__))+'/'+env_id[:-3]+'Results'
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
for seed in [0, 1, 2, 3]:
    exp_params['path'] = result_dir+'/seed'+str(seed)
    if not os.path.isdir(exp_params['path']):
        os.mkdir(exp_params['path'])
    exp_params['seed'] = seed

    trainer = Trainer(env_id=env_id, algo_params=algo_params, exp_params=exp_params)
    trainer.run()
    del trainer