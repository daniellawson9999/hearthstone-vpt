from win32gui import FindWindow, GetWindowRect

window_handle = FindWindow(None, "Hearthstone")
while True:
    window_rect   = GetWindowRect(window_handle)
    x, y = window_rect[0], window_rect[1]
    w = window_rect[2] - x
    h = window_rect[3] - y
    print("position: ", (x,y), "width,height", (w,h))
