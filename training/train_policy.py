import numpy as np
import torch as th
import wandb


import argparse
import pickle
import random
import sys
import os
import pathlib
import json

from vpt.torch_util import set_default_torch_device
from vpt.policy import HearthstoneAgentPolicy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(experiment_name, variant):
    device = variant.get('device', 'cuda')
    set_default_torch_device(device)

    # Load data (no state or action norm needed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(dir_path, "..", 'data')
    data_file_prefix = os.path.join(data_folder, args.data_name)
    state_file = os.path.join(data_file_prefix, 'states.npy')
    actions_file = os.path.join(data_file_prefix, 'actions.npy')

    states = np.load(state_file, mmap_mode='r')
    actions = np.load(actions_file, mmap_mode='r')

    assert (states.shape[0] == actions.shape[0]), "unequal number of states and actions"

    n_transitions = states.shape[0]
    print(n_transitions, "total transitions")

    img_size = 128
    # Original img shape 99,128,3
    assert (states[2] == 128), "width expected to match"
    original_width = states.shape[1]
    width_padding = img_size - original_width

    action_dim = actions.shape[-1]

    # Load policy architecture config
    config_file = os.path.join(dir_path, 'config', variant['policy_config'] + '.json')
    policy_kwargs = json.load(open())


    # Set up model
    policy = HearthstoneAgentPolicy(policy_kwargs=policy_kwargs).to(device=device)
    print("Trainable model params", count_parameters(policy))

    # Set up optimizer and loss

    # Start training

    # Evaluation? 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--policy_config', required=True, type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', required=True, type=str)
    args = parser.parse_args()

    main('gym-experiment', variant=vars(args))
