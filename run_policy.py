import json
import argparse
import os
import time
import numpy as np

import pyautogui
import win32gui
from win32gui import GetWindowText, GetForegroundWindow, FindWindow, GetWindowRect
from pynput import mouse, keyboard
from pynput.keyboard import Key
from pynput.mouse import Button, Controller


import torch as th

from vpt.policy import HearthstoneAgentPolicy
from vpt.torch_util import set_default_torch_device

from collect_data.mouse_screen_record import crop_resize_img

done = False
model_enabled = True

# Todo, change to configurable 
def on_press(key):
    if key == Key.esc:
        # Exit program
        global done
        done = True
        return False
    elif key == Key.space:
        # Toggles model enabled
        global model_enabled
        model_enabled = not model_enabled

def on_release(key):
    pass

def apply_action(mouse, action, relative_action=True):
    
    x_diff = action[0] * 1920
    y_diff = action[1] * 1080

    pressed = action[2]
    released = action[3]
    press_and_release = action[4]
    nothing = action[5]

    #print(mouse)
    print(x_diff,y_diff)
    if relative_action:
        mouse.move(x_diff, y_diff)
    else:
        mouse.position = (x_diff, y_diff)
    
    if pressed:
        mouse.press(Button.left)
    elif released:
        mouse.release(Button.left)
    elif press_and_release:
        mouse.press(Button.left)
        time.sleep(1/10)
        mouse.release(Button.left)


def main(args):
    device = args.device
    set_default_torch_device(device)

    policy_dir = os.path.join('.', 'hearthstone', 'models', args.policy)
    policy_weight_file = os.path.join(policy_dir, args.policy + '.pt')
    policy_config_file = os.path.join(policy_dir, 'config.json')

    # Load policy
    config = json.load(open(policy_config_file,'r'))
    policy = HearthstoneAgentPolicy(policy_kwargs=config, pi_head_kwargs={}).to(device=device)
    policy.load_state_dict(th.load(policy_weight_file, map_location=device))

    relative_action = config['action_type'] != 'absolute'
    print("Relative action:", relative_action)

    dummy_first = th.from_numpy(np.zeros((1),dtype=bool)).to(device)
    

    mouse = Controller()

    window_handle = FindWindow(None, "Hearthstone")

    # Setup thread for getting exit and enabled/disabled signals
    key_listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    key_listener.start()

    try:
    #while True:
        # Main loop, run until input to terminate
        while not done:
            # Reset hidden state
            hidden_state = policy.initial_state(1)

            # Control Loop, repeat while enabled, want to run at period >= target_target
            # target_delay
            target_delay = 1/4

            while model_enabled:
                start_time = time.perf_counter_ns()

                # capture image, and transform to 128 by 128
                if not args.any_window:
                    win32gui.SetForegroundWindow(window_handle)
                screenshot = pyautogui.screenshot()
                img = crop_resize_img(screenshot)

                # Convert to numpy
                img = np.array(img) / 255
                img = np.expand_dims(img, axis=0)

                # Determine and add padding
                img_size = img.shape[2]
                assert img_size == 128, "incorrect size"
                original_height = img.shape[1]
                height_padding = img_size - original_height
                img = np.concatenate([img, np.zeros((1,height_padding, img_size,3))],axis=1)

                # conver to tensor
                state = th.from_numpy(img).to(dtype=th.float32, device=device)

                # get prediction, update state
                action, hidden_state, result = policy.act(state, dummy_first, hidden_state)
                print(action)

                # apply action
                if not args.any_window:
                    win32gui.SetForegroundWindow(window_handle)

                # apply translation, and mouse actions 
                apply_action(mouse, action.squeeze().cpu().numpy(), relative_action=relative_action)
                
                # delay
                end_time = time.perf_counter_ns()
                time_taken = ((start_time - end_time) / 1e9)
                extra_time = target_delay - time_taken
                if extra_time > 0: time.sleep(extra_time)

            # sleep, reduce interval for checking if enabled
            time.sleep(.1)
    except Exception as e:
        print(e)
        key_listener.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--any_window', default=False, action='store_true')
    #parser.add_argument('--absolute_actions', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
