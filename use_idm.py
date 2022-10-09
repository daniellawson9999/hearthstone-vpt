import json
import argparse
import os
import time
import numpy as np

import torch as th

from vpt.policy import InverseActionPolicy
from vpt.torch_util import set_default_torch_device

def main(args):
    device = args.device
    set_default_torch_device(device)

    if args.stochastic:
        raise Exception("Not yet implemented, currently lables determinsitcally")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    policy_dir = os.path.join(dir_path, 'models', args.policy)
    policy_weight_file = os.path.join(policy_dir, args.policy + '.pt')
    policy_config_file = os.path.join(policy_dir, 'config.json')

    data_folder = os.path.join(dir_path, 'data')
    data_file = os.path.join(data_folder, args.data_file)

    # Load policy
    config = json.load(open(policy_config_file,'r'))
    policy = InverseActionPolicy(idm_net_kwargs=config, pi_head_kwargs={}).to(device=device)
    policy.load_state_dict(th.load(policy_weight_file, map_location=device))

    # Load data, should be (T, 99, 128, 3), height by not be 99, but width should be 128, pad height as necessary
    raw_states = np.load(data_file, mmap_mode='r')

    # get timesteps that idm handles
    timesteps = config['timesteps']
    stride = timesteps // 2
    
    current_frame = 0

    hidden_state = policy.initial_state(1)
    dummy_first = th.zeros((timesteps, 1)).to(device=device)

    while current_frame < raw_states.shape[0]:
        # get states
        np_states = raw_states[current_frame:current_frame + timesteps]

        # pad height
        img_width = np_states.shape[-2]
        assert (img_width == 128), "width should be 128"
        img_height = np_states.shape[-3]
        height_padding = img_width - img_height

        padding = np.zeros((timesteps, height_padding, img_width, 3))
        padded_states = np.concatenate([np_states, padding], axis=1)


        states = th.tensor(padded_states, device=device, dtype=th.float32).unsqueeze(0)
        (translation_actions, click_dists, click_logits, logp_actions), hidden_state = policy.forward(
            states, dummy_first, state_in=hidden_state
        )
        translation_actions = translation_actions.detach().cpu().squeeze().numpy()

        click_action_index = click_dists.logits.argmax(dim=-1).detach().cpu().numpy()
        click_actions = np.zeros((timesteps, 4))
        click_actions[np.arange(timesteps), click_action_index] = 1

        # combine
        actions = np.concatenate([translation_actions, click_actions], axis=-1).astype(np.float32)

        # store  np_states and actions, only store non-causal chunks, append ot file, etc



        current_frame += stride




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--data_file', required=True, type=str)
    parser.add_argument('--stochastic', default=False, action='store_true')

    args = parser.parse_args()
    main(args)

