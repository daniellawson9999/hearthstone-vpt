import pyautogui
import win32gui
from win32gui import FindWindow, GetWindowRect
from time import sleep

window_handle = FindWindow(None, "Hearthstone")
win32gui.SetForegroundWindow(window_handle)
sleep(1)
img = pyautogui.screenshot()
# scale = 1/8
# resized_im = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
left = 280
width = 1400
height = 1080
img = img.crop((left, 0, left + width, height))

scale = 10.9375
img = img.resize((round(img.size[0]/scale), round(img.size[1]/scale)))
img.show()