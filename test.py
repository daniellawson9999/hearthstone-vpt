import numpy as np
import torch as th
from torch import nn
from vpt.torch_util import set_default_torch_device

from vpt.policy import HearthstoneAgentPolicy


device = 'cuda'
set_default_torch_device(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# policy_kwargs = {
#     'recurrence_type': 'transformer' ,
#     'img_shape': (3,128,128),
#     'timesteps': 128,
#     'memory_size': 256,
#     'input_size': 1024,
#     'heads': 8,
#     'hidsize': 1024,
#     'n_recurrence_layers': 4
#     }

policy_kwargs = {
    "attention_heads": 8,
    "attention_mask_style": "clipped_causal",
    "attention_memory_size": 256,
    "diff_mlp_embedding": False,
    "hidsize": 1024,
    "img_shape": [
        128,
        128,
        3
    ],
    "impala_chans": [
        16,
        32,
        32
    ],
    "impala_kwargs": {
        "post_pool_groups": 1
    },
    "impala_width": 4,
    "init_norm_kwargs": {
        "batch_norm": False,
        "group_norm_groups": 1
    },
    "n_recurrence_layers": 4,
    "only_img_input": True,
    "pointwise_ratio": 4,
    "pointwise_use_activation": False,
    "recurrence_is_residual": True,
    "recurrence_type": "transformer",
    "timesteps": 128,
    "use_pointwise_layer": True,
    "use_pre_lstm_ln": False
}



agent = HearthstoneAgentPolicy(policy_kwargs=policy_kwargs, pi_head_kwargs={}).to(device=device)
print("Number of params", count_parameters(agent))

# Test get action
obs = th.zeros((1,128,128,3), dtype=th.float32, device=device)
first = th.from_numpy(np.zeros((1),dtype=bool)).to(device)
state = agent.initial_state(1)
#import pdb; pdb.set_trace()
action = agent.act(obs, first, state, stochastic=False)

# Test larger 
batch_size = 4
timesteps = 10
obs = th.zeros((batch_size,timesteps,128,128,3), dtype=th.float32, device=device)
first = th.from_numpy(np.zeros((batch_size, timesteps),dtype=bool)).to(device)
state = agent.initial_state(batch_size)

(translation_actions, click_dists, click_logits), state_out = agent.forward(obs, first, state)
import pdb; pdb.set_trace()
