import os
import argparse
import numpy as np

def main(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(dir_path, "..", 'data')
    data_file_prefix = os.path.join(data_folder, args.data_name)
    state_file = os.path.join(data_file_prefix, 'states.npy')
    state_data  = np.load(state_file, mmap_mode='r')

    new_file_name = os.path.join(data_file_prefix, 'states.npz')
    np.savez_compressed(new_file_name, states = state_data)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='test', type=str)
    args = parser.parse_args()
    main(args)