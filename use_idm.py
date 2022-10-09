import json
import argparse
import os
import time
import numpy as np
from npy_append_array import NpyAppendArray

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
    data_file = os.path.join(data_folder, 'youtube', args.input_data_file)


    run_dir = os.path.join(data_folder, 'labeled', args.run_name)
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    assert (len(os.listdir(run_dir)) == 0), "must write to empty directory"

    output_actions_absolute_file = os.path.join(run_dir, 'actions_absolute.npy')
    output_state_file = os.path.join(run_dir, 'states.npy')
    starts_file = os.path.join(run_dir, 'starts.npy')

    # Load policy
    config = json.load(open(policy_config_file,'r'))
    policy = InverseActionPolicy(idm_net_kwargs=config, pi_head_kwargs={}).to(device=device)
    policy.load_state_dict(th.load(policy_weight_file, map_location=device))

    # Load data, should be (T, 99, 128, 3), height by not be 99, but width should be 128, pad height as necessary
    raw_states = np.load(data_file, mmap_mode='r')

    total_timesteps = raw_states.shape[0]
    print("total timesteps", total_timesteps)

    # get timesteps that idm handles
    timesteps = config['timesteps']
    print("training timesteps", timesteps)
    if args.custom_timesteps is not None:
        timesteps = args.custom_timesteps
        print("setting timestep to ", timesteps)


    offset = timesteps // 4
    stride = timesteps // 2

    print("offset:", offset, "stride:", stride)
    
    current_frame = 0

    hidden_state = policy.initial_state(1)
    dummy_first = th.zeros((timesteps, 1)).to(device=device)


    #while current_frame < raw_states.shape[0]:
    total_processed_frames = 0

    done = False
    while not done:
        print("Current window starting frame", current_frame)

        # get states
        np_states = raw_states[current_frame:current_frame + timesteps]

        # pad height
        img_width = np_states.shape[-2]
        assert (img_width == 128), "width should be 128"
        img_height = np_states.shape[-3]
        height_padding = img_width - img_height

        padding = np.zeros((np_states.shape[0], height_padding, img_width, 3))
        padded_states = np.concatenate([np_states, padding], axis=1)


        states = th.tensor(padded_states, device=device, dtype=th.float32).unsqueeze(0)
        (translation_actions, click_dists, click_logits, logp_actions), hidden_state = policy.forward(
            states, dummy_first, state_in=hidden_state
        )


        translation_actions = translation_actions.detach().cpu().squeeze().numpy()

        click_action_index = click_dists.logits.argmax(dim=-1).detach().cpu().numpy()
        click_actions = np.zeros((np_states.shape[0], 4))
        click_actions[np.arange(np_states.shape[0]), click_action_index] = 1

        # combine
        actions = np.concatenate([translation_actions, click_actions], axis=-1).astype(np.float32)

        # Save memory
        del states,translation_actions, click_actions, click_logits, logp_actions


        # store  np_states and actions, only store non-causal chunks, append ot file, etc

        # Tremove outer and inner
        if current_frame != 0:
            np_states = np_states[offset:]
            actions = actions[offset:]
        
        if current_frame + timesteps + 1 >= total_timesteps:
            done = True
        else:
            np_states = np_states[:-offset]
            actions = actions[:-offset]
        
        processed_frames = np_states.shape[0]
        total_processed_frames += processed_frames

        # append to file
        with NpyAppendArray(output_state_file) as npaa:
            npaa.append(np_states)

        with NpyAppendArray(output_actions_absolute_file) as npaa:
            npaa.append(actions)
        
        with NpyAppendArray(starts_file) as npaa:
            starts = np.zeros(processed_frames, dtype=bool)
            if current_frame == 0:
                starts[0] = 1
            npaa.append(starts)

        current_frame += stride

        #th.cuda.empty_cache()
        hidden_state = policy.initial_state(1)

    print("processed frames", total_processed_frames)
    print("total timesteps", total_timesteps)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True, type=str)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--input_data_file', required=True, type=str)
    parser.add_argument('--stochastic', default=False, action='store_true')
    parser.add_argument('--custom_timesteps',default=None, type=int)

    args = parser.parse_args()
    main(args)

