import numpy as np
import torch as th
from torch import nn
from vpt.torch_util import set_default_torch_device

from vpt.policy import HearthstoneAgentPolicy, InverseActionPolicy



device = 'cuda'
set_default_torch_device(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# policy_kwargs = {
#    "attention_heads":32,
#    "attention_mask_style":"none",
#    "attention_memory_size":128,
#    "conv3d_params":{
#       "inchan":3,
#       "kernel_size":[
#          5,
#          1,
#          1
#       ],
#       "outchan":128,
#       "padding":[
#          2,
#          0,
#          0
#       ]
#    },
#    "hidsize":4096,
#    "img_shape":[
#       128,
#       128,
#       128
#    ],
#    "impala_kwargs":{
#       "post_pool_groups":1
#    },
#    "impala_width":16,
#    "init_norm_kwargs":{
#       "batch_norm":False,
#       "group_norm_groups":1
#    },
#    "n_recurrence_layers":2,
#    "only_img_input":True,
#    "pointwise_ratio":4,
#    "pointwise_use_activation":False,
#    "recurrence_is_residual":True,
#    "recurrence_type":"transformer",
#    "single_output":True,
#    "timesteps":128,
#    "use_pointwise_layer":True,
#    "use_pre_lstm_ln":False
# }
timesteps = 128
timesteps = 32
policy_kwargs = {
   "attention_heads":4,
   "attention_mask_style":"none",
   "attention_memory_size":timesteps,
   "conv3d_params":{
      "inchan":3,
      "kernel_size":[
         5,
         1,
         1
      ],
      "outchan":128,
      "padding":[
         2,
         0,
         0
      ]
   },
   #"hidsize":4096,
   'hidsize': 512,
   "img_shape":[
      128,
      128,
      128
   ],
   "impala_kwargs":{
      "post_pool_groups":1
   },
   "impala_width":16,
   "init_norm_kwargs":{
      "batch_norm":False,
      "group_norm_groups":1
   },
   "n_recurrence_layers":2,
   "only_img_input":True,
   "pointwise_ratio":4,
   "pointwise_use_activation":False,
   "recurrence_is_residual":True,
   "recurrence_type":"transformer",
   "single_output":True,
   "timesteps":timesteps,
   "use_pointwise_layer":True,
   "use_pre_lstm_ln":False
}

# current policy has no temp
policy = InverseActionPolicy(idm_net_kwargs = policy_kwargs, pi_head_kwargs={}).to(device=device)

dummy_first = th.zeros((timesteps, 1)).to(device=device)
hidden_state = policy.initial_state(1)
agent_input = th.zeros((1, timesteps, 128,128,3)).to(device=device)

import pdb; pdb.set_trace()
# test forward
(translation_actions, click_dists, click_logits, logp_actions), hidden_state = policy.forward(
    agent_input, dummy_first, state_in=hidden_state
)


# test predict

import pdb; pdb.set_trace()