import sys
import time
import tkinter as tk

_MARKERS = ("Can't find a usable init.tcl", "couldn't read file", ": No error")


def create_tk_root():
    for attempt in range(1, 6):
        try:
            return tk.Tk()
        except tk.TclError as error:
            text = str(error)
            retriable = (sys.platform == "win32"
                         and all(marker in text for marker in _MARKERS))
            if not retriable or attempt == 5:
                raise
            print(f"tk.Tk attempt {attempt} failed: {text}", file=sys.stderr)
            time.sleep(0.3)
