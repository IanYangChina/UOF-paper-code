import os
import argparse
from config.config_hac import Params
from config.shared import *
from agent.hierarchical_actor_critic import HierarchicalActorCritic as HAC

parser = argparse.ArgumentParser()
parser.add_argument('--task-id', dest='task_id', type=int,
                    help='Index of the target task id, default: 0', default=0, choices=[0, 1])
parser.add_argument('--no-demo', dest='no_abstract_demonstration',
                    help='Whether to NOT use abstract demonstrations, default: False', default=False, action='store_true')
parser.add_argument('--demo-proportion', dest='demonstration_proportion', type=float,
                    help='The proportion of episodes that are demonstrated, default: 0.75', default=0.75, choices=[0.0, 0.25, 0.5, 0.75, 1.0])
parser.add_argument('--train', dest='train',
                    help='Whether to train policies from scratch, default: False', default=False, action='store_true')
parser.add_argument('--render', dest='render',
                    help='Whether to render the task, default: False', default=False, action='store_true')

if __name__ == '__main__':
    args = vars(parser.parse_args())
    # task/environment setup
    print("Task id %i, env name %s " % (args['task_id'], env_ids[args['task_id']]))
    Params.ENV_ID = args['task_id']
    Params.ENV_NAME = env_ids[args['task_id']]
    Params.TRAINING_EPOCH = training_epochs[args['task_id']]
    Params.TESTING_TIMESTEP = testing_timesteps[args['task_id']]
    # paths for training from scratch
    Params.PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'result', 'task_'+str(Params.ENV_ID))
    Params.CKPT_PATH = os.path.join(Params.PATH, "ckpts")
    Params.DATA_PATH = os.path.join(Params.PATH, "data")
    # training flags
    Params.TRAIN = args['train']
    Params.LOAD_PER_TRAIN_POLICY = not args['train']
    # paths to load pre-trained policies
    Params.PRE_TRAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'pretrained_policy', 'hac_task_'+str(Params.ENV_ID))
    Params.PRE_TRAIN_CKPT_PATH = os.path.join(Params.PRE_TRAIN_PATH, "ckpts")
    Params.PRE_TRAIN_DATA_PATH = os.path.join(Params.PRE_TRAIN_PATH, "data")
    # algorithm setup
    Params.ABSTRACT_DEMONSTRATION = not args['no_abstract_demonstration']
    Params.ABSTRACT_DEMONSTRATION_PROPORTION = args['demonstration_proportion']
    # build agent
    agent = HAC(params=Params)
    if args['train']:
        print("Start training from scratch...")
        agent.run(render=args['render'])
    else:
        print("Evaluate the agent pre-trained for %i epochs" % Params.TRAINING_EPOCH-1)
        agent.test(render=args['render'], load_network_epoch=Params.TRAINING_EPOCH-1)
