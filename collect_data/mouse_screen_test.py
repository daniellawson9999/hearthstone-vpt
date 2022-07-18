import pyautogui
import win32gui
from win32gui import GetWindowText, GetForegroundWindow, FindWindow, GetWindowRect
import time
from enum import Enum, auto
from pynput import mouse, keyboard
from pynput.keyboard import Key
import threading
import numpy as np
from npy_append_array import NpyAppendArray
import os 
import argparse


class EventType(Enum):
    move = auto()
    press = auto()
    release = auto()

target_window = "Hearthstone"
events = []
screenshots = []

done = False
any_window = False

"Mouse"

def on_move(x, y):
    if done: return False
    if GetWindowText(GetForegroundWindow()) == target_window or any_window:
        event_time = time.time_ns()
        #print('Pointer moved to {0}'.format((x, y)))
        events.append([EventType.move, event_time, (x,y)])
    

def on_click(x, y, button, pressed):
    if done: return False
    if GetWindowText(GetForegroundWindow()) == target_window or any_window:
        event_time = time.time_ns()
        #print('{0} at {1}'.format('Pressed' if pressed else 'Released',(x, y)))
        if pressed: 
            event_type = EventType.press
        else: 
            event_type = EventType.release
        events.append([event_type, event_time])

def on_scroll(x, y, dx, dy):
    pass

"Keys"

def on_press(key):
    if key == Key.esc:
        global done
        done = True
        print(done)
        return False

def on_release(key):
    pass



def collect_screenshots():
    target_delay = 1/10
    while True:
        if done: return
        if GetWindowText(GetForegroundWindow()) == target_window or any_window:
            start_time = time.time_ns()
            screenshot = pyautogui.screenshot()
            screenshots.append([start_time, screenshot])
            end_time = time.time_ns()
            time_taken = ((start_time - end_time) / 1e9)
            extra_time = target_delay - time_taken
            if extra_time > 0: time.sleep(extra_time)

def crop_resize_img(img):
    left = 280
    width = 1400
    height = 1080
    img = img.crop((left, 0, left + width, height))

    scale = 10.9375
    img = img.resize((round(img.size[0]/scale), round(img.size[1]/scale)))

    return img


def main(args):
    # Create directory and state/action files
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(dir_path, "..", 'data')
    data_file_prefix = os.path.join(data_folder, args.data_name)
    if not os.path.isdir(data_file_prefix):
        os.mkdir(data_file_prefix)
    state_file = os.path.join(data_file_prefix, 'states.npy')
    actions_file = os.path.join(data_file_prefix, 'actions.npy')

    global any_window
    any_window = args.any_window

    window_handle = FindWindow(None, "Hearthstone")
    win32gui.SetForegroundWindow(window_handle)

    listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll
    )
    imager = threading.Thread(target=collect_screenshots)

    key_listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )

    #listener.daemon = True
    listener.start()
    #imager.daemon = True
    imager.start()
    key_listener.start()


 


    '''
    Action:
    [x_diff, y_diff, press, release]
    x_diff: change in x axis
    y_diff: change in y axis
    press: 1 if press, 0 if not press
    release: 1 if release, 0 if not release
    '''

    # Merge two streams of mouse events and screenshots
    last_screenshot_len = 0
    next_event_index = 0
    prev_x_y = None
    trajectory = []
    #import pdb; pdb.set_trace()
    try:
        while True:
            if done:
                listener.join()
                imager.join()
                break
            num_screenshots = len(screenshots)
            behind = num_screenshots - last_screenshot_len
            # if behind > 0:
            #     print("Behind", num_screenshots - last_screenshot_len, "screenshots")
            if num_screenshots > last_screenshot_len:
                #print(num_screenshots)
                screenshot_time, raw_screenshot = screenshots[last_screenshot_len]
                # Remove old screenshot data from list
                screenshots[last_screenshot_len] = None
                # scale = 1/8
                # raw_screenshot  = raw_screenshot.resize((round(raw_screenshot.size[0]*scale), round(raw_screenshot.size[1]*scale)))
                raw_screenshot = crop_resize_img(raw_screenshot)
                # Convert to numpy
                screenshot = np.array(raw_screenshot) / 255
                screenshot = np.expand_dims(screenshot, axis=0)
                del raw_screenshot


                # If no events taken place, empty action
                if next_event_index < len(events):
                    # otherwise gather events taken
                    last_x = 0
                    last_y = 0
                    has_pressed = 0
                    has_released = 0 
                    current_num_events = len(events)
                    for i in range(next_event_index, current_num_events):
                        event = events[i]
                        event_type = event[0]
                        event_time = event[1]
                        if event_time < screenshot_time:
                            if event_type == EventType.move:
                                last_x = event[2][0]
                                last_y = event[2][1]
                            elif event_type == EventType.press:
                                has_pressed = has_pressed or 1
                            elif event_type == EventType.release:
                                has_released == has_released or 1
                        else:
                            next_event_index = i
                            break
                    if prev_x_y is not None:
                        x_diff = last_x - prev_x_y[0]
                        y_diff = last_y - prev_x_y[1]
                    else:
                        x_diff = 0
                        y_diff = 0
                    prev_x_y = [last_x, last_y]
                    if has_pressed and has_released:
                        has_pressed = 0
                        has_released = 0
                    action = [x_diff, y_diff, has_pressed, has_released]
                else:
                    action = [0,0,0,0]
                #trajectory.append([screenshot, action])
                if not args.remove_nulls or (args.remove_nulls and sum(action) != 0):
                    with NpyAppendArray(state_file) as npaa:
                        npaa.append(screenshot)
                    with NpyAppendArray(actions_file) as npaa:
                        npaa.append(np.array([action]))
                    print("Logged to trajectory with action", action)
                last_screenshot_len += 1
    except KeyboardInterrupt:
        listener.join()
        imager.join()
    except Exception:
        listener.join()
        imager.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='test', type=str)
    parser.add_argument('--any_window', default=False, action='store_true', help="records any window, useful for debugging")
    parser.add_argument('--remove_nulls', default=False, action='store_true')
    args = parser.parse_args()
    main(args)