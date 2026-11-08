"""Input backends play the gestures the Bridge hands them (see humanize.py); they draw nothing themselves."""

from pymobiledevice3.remote.core_device.hid_service import ASCII_TO_HID

BACKSPACE = 0x2A
KEYS = {**ASCII_TO_HID, "\b": (BACKSPACE, False)}  # char -> (HID usage, shift); "\n" is Return
