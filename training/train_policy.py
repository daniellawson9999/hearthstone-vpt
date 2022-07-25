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

    log_to_wandb = variant.get('log_to_wandb', False)

    # Load data (no state or action norm needed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(dir_path, "..", 'data')
    data_file_prefix = os.path.join(data_folder, variant['dataset'])
    state_file = os.path.join(data_file_prefix, 'states.npy')
    actions_file = os.path.join(data_file_prefix, 'actions.npy')
    start_file = os.path.join(data_file_prefix, 'starts.npy')


    states = np.load(state_file, mmap_mode='r')
    actions = np.load(actions_file, mmap_mode='r')
    starts = np.load(start_file, mmap_mode='r')


    assert (states.shape[0] == actions.shape[0]), "unequal number of states and actions"

    num_trajectories = starts.sum()
    num_transitions = states.shape[0]
    print(num_transitions, "total transitions")

    img_size = 128
    # Original img shape 99,128,3
    assert (states[2] == 128), "width expected to match"
    original_height = states.shape[1]
    height_padding = img_size - original_height

    action_dim = actions.shape[-1]

    # Store trajectory starts, ends, lengths
    start_indexes = np.arange(num_transitions)[starts]
    end_indexes = np.array([*start_indexes[1:], num_transitions - 1])
    trajectory_lengths = end_indexes - start_indexes + 1
    p_sample = trajectory_lengths / trajectory_lengths.sum()

    # Load policy architecture config
    config_file = os.path.join(dir_path, 'config', variant['policy_config'] + '.json')
    policy_kwargs = json.load(open())

    # create batch sampler
    def get_batch(batch_size=64, max_len=config_file['timesteps']):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        # Get states, actions, and mask
        # Mask removes padded transitions from loss calculations
        s, a, mask = [], [], []
        for i in range(batch_size):
            # Get start index
            si = random.randint(start_indexes[i], end_indexes[i])

            # Get states
            ei = min(si+max_len, end_indexes[i] + 1)
            new_states = states[si:ei]

            # Get timestep padding info
            num_included_steps = new_states.shape[0]
            timestep_padding_value = max_len - num_included_steps

            # Pad height
            padding = np.zeros(num_included_steps, height_padding, img_size, 3)
            new_states = np.concatenate([new_states, padding], axis=1)

            # Get actions
            new_actions = states[si:ei]
            
            # Pad states and actions to max_len
            new_states = np.concatenate([new_states, np.zeros(timestep_padding_value, img_size, img_size, 3)],axis=0)
            new_actions = np.concatenate([new_actions, np.zeros(timestep_padding_value, action_dim)])
            
            new_states = new_states.expand_dims(new_states, axis=0)
            new_actions = new_actions.expand_dims(new_actions, axis=0)
            new_mask = np.concatenate([np.ones(1, num_included_steps), np.zeros(1, timestep_padding_value)], axis=1)

            s.append(new_states)
            a.append(new_actions)
            mask.append(new_mask)


        s = th.from_numpy(np.concatenate(s, axis=0)).to(dtype=th.float32, device=device)
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=device)
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, mask

    # Set up model
    policy = HearthstoneAgentPolicy(policy_kwargs=policy_kwargs).to(device=device)
    print("Trainable model params", count_parameters(policy))

    # Set up optimizer and loss
    optimizer = th.optim.Adam(
        policy.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay']
    )

    dataset=variant['dataset'] 
    group_name = f'{dataset}-policy'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
    # Start training
    if log_to_wandb:
        variant.update(policy_kwargs)
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='vpt',
            config=variant
        )

    for i in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=i+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)
            
    # Optionally save model
    if variant['save_model']:
        raise NotImplementedError("Saving not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--policy_config', required=True, type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_iters', default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    args = parser.parse_args()

    main('gym-experiment', variant=vars(args))
