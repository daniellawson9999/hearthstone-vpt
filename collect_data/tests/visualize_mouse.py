import numpy as np
import os 
import matplotlib.pyplot as plt

specific_data = 'record'

dir_path = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(dir_path, "..", "..", 'data', specific_data)

action_data = data_folder + '/actions_relative.npy'

actions = np.load(action_data, mmap_mode='r')


#import pdb; pdb.set_trace()
plt.scatter(actions[:,0], actions[:, 1])
plt.show()