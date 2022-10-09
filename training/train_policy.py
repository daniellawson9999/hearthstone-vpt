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
from vpt.policy import HearthstoneAgentPolicy, InverseActionPolicy

from training.trainer import Trainer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(variant):
    device = variant.get('device', 'cuda')
    set_default_torch_device(device)

    log_to_wandb = variant.get('log_to_wandb', False)

    # Load data (no state or action norm needed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_folder = os.path.join(dir_path, '..', 'models')
    data_folder = os.path.join(dir_path, "..", 'data')
    data_file_prefix = os.path.join(data_folder, variant['dataset'])

    if not variant['use_npz']:
        state_ending = '.npy'
    else:
        state_ending = '.npz'
    state_file = os.path.join(data_file_prefix, 'states' + state_ending)

    if variant['action_type'] == 'relative':
        action_file_end = 'actions_relative'
    elif variant['action_type'] == 'absolute':
        action_file_end = 'actions_absolute'
    else:
        raise Exception("invalid action type")
    actions_file = os.path.join(data_file_prefix, action_file_end + '.npy')


    start_file = os.path.join(data_file_prefix, 'starts.npy')


    states = np.load(state_file, mmap_mode='r')
    actions = np.load(actions_file, mmap_mode='r')
    starts = np.load(start_file, mmap_mode='r')


    assert (states.shape[0] == actions.shape[0]), "unequal number of states and actions"

    num_trajectories = starts.sum()
    num_transitions = states.shape[0]
    print(num_transitions, "total transitions")

    # filter nulls
    if variant['filter_nulls']:
        # Find repeated actions
        good_index = []
        last_action = np.zeros_like(actions[0])
        for i in range(len(actions)):
            if not np.array_equal(actions[i],last_action):
                good_index.append(i)
            last_action = actions[i]
        # Remove
        actions = actions[good_index]
        states = states[good_index]
        starts = starts[good_index]

        # After removing repeated actions
        num_transitions = states.shape[0]
        print(num_transitions, "new total transitions")


    img_size = 128
    # Original img shape 99,128,3
    assert (states.shape[2] == 128), "width expected to match"
    original_height = states.shape[1]
    height_padding = img_size - original_height

    action_dim = actions.shape[-1]

    # Store trajectory starts, ends, lengths
    start_indexes = np.arange(num_transitions)[starts]
    end_indexes = np.array([*start_indexes[1:] - 1, num_transitions - 1])
    trajectory_lengths = end_indexes - start_indexes + 1
    p_sample = trajectory_lengths / trajectory_lengths.sum()

    # Load policy architecture config
    config_file = os.path.join(dir_path, 'config', variant['policy_config'] + '.json')
    policy_kwargs = json.load(open(config_file,'r'))
    variant.update(policy_kwargs)

    # create batch sampler
    def get_batch(batch_size=64, max_len=variant['timesteps']):
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
            index = batch_inds[i]
            si = random.randint(start_indexes[index], end_indexes[index])

            # Get states
            ei = min(si+max_len, end_indexes[index] + 1)
            new_states = states[si:ei]

            # Get timestep padding info
            num_included_steps = new_states.shape[0]
            timestep_padding_value = max_len - num_included_steps

            # Pad height
            padding = np.zeros((num_included_steps, height_padding, img_size, 3))
            new_states = np.concatenate([new_states, padding], axis=1)

            # Get actions
            new_actions = actions[si:ei]
            
            # Pad states and actions to max_len
            new_states = np.concatenate([new_states, np.zeros((timestep_padding_value, img_size, img_size, 3))],axis=0)
            new_actions = np.concatenate([new_actions, np.zeros((timestep_padding_value, action_dim))])
            
            new_states = np.expand_dims(new_states, axis=0)
            new_actions = np.expand_dims(new_actions, axis=0)
            new_mask = np.concatenate([np.ones((1, num_included_steps)), np.zeros((1, timestep_padding_value))], axis=1)

            s.append(new_states)
            a.append(new_actions)
            mask.append(new_mask)


        s = th.from_numpy(np.concatenate(s, axis=0)).to(dtype=th.float32, device=device)
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=device)
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, mask

    # Set up model
    if variant['train_idm']:
        policy =  InverseActionPolicy(idm_net_kwargs=policy_kwargs, pi_head_kwargs={}).to(device=device)
    else:
        policy = HearthstoneAgentPolicy(policy_kwargs=policy_kwargs, pi_head_kwargs={}).to(device=device)
        
    print("Trainable model params", count_parameters(policy))

    # Set up optimizer and loss
    optimizer = th.optim.Adam(
        policy.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay']
    )

    def loss_fn(translation_actions, target_translation_actions, logp_actions): 
        translation_loss = th.mean((target_translation_actions - translation_actions)**2)
        click_loss = -th.mean(logp_actions)
        loss = translation_loss + click_loss
        return loss, translation_loss, click_loss


    dataset=variant['dataset'] 
    group_name = f'{dataset}-policy'
    if variant['train_idm']:
        group_name += '-idm'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'



    # setup WANDB
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='vpt',
            config=variant
        )

    # Set up trainer
    trainer = Trainer(policy, optimizer, variant['batch_size'], get_batch, loss_fn, device)

    # Start Training
    for i in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=i+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)
            
    # Optionally save model
    if variant['save_model']:
        model_folder = os.path.join(models_folder, exp_prefix)
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        model_path =  os.path.join(model_folder, exp_prefix + '.pt')
        th.save(policy.state_dict(), model_path)
        # write config 
        policy_kwargs['action_type'] = variant['action_type']
        config_path = os.path.join(model_folder, 'config.json')
        with open(config_path, 'w') as config_file:
            json.dump(policy_kwargs, config_file, ensure_ascii=False, indent=4)


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
    parser.add_argument('--max_iters', default=10, type=int)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--action_type', default='absolute', choices=['relative','absolute'])
    parser.add_argument('--use_npz', default=False, action='store_true')
    parser.add_argument('--filter_nulls', default=False, action='store_true')
    parser.add_argument('--train_idm', default=False, action='store_true')
    args = parser.parse_args()

    main(variant=vars(args))
