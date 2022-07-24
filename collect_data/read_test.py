import numpy as np
import os 

specific_data = 'debug'

dir_path = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(dir_path, "..", 'data', specific_data)

action_data = data_folder + '/actions.npy'
state_data = data_folder + './states.npy'

actions = np.load(action_data, mmap_mode='r')
states = np.load(state_data, mmap_mode='r')

import pdb; pdb.set_trace()