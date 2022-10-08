# util_funcs
import win32gui, win32ui, win32con, win32api
import time
import cv2
import numpy as np
import ctypes
import random


def sortByHSV(img, h1, s1, v1, h2, s2, v2): 
  minimap_blurred = cv2.medianBlur(img,3)
  hsv = cv2.cvtColor(minimap_blurred, cv2.COLOR_BGR2HSV )

  # формируем начальный и конечный цвет фильтра
  h_min = np.array((h1, s1, v1), np.uint8)
  h_max = np.array((h2, s2, v2), np.uint8)

  # накладываем фильтр на кадр в модели HSV
  thresh = cv2.inRange(hsv, h_min, h_max)
  kernel = np.ones((5,5),np.uint8)
  thresh = cv2.dilate(thresh,kernel,iterations = 2)
  return thresh

# grabScreen
def grabScreen(region=None, convert_colors = 'BGR'):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    if convert_colors == "RGB": img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif convert_colors == "GRAY": img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif convert_colors == "BGR": img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else: pass # NO CONVERT IMAGE IS BGRA 
    return img


# direct inputs
# source to this solution and code:
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
# https://pastebin.com/Qy3E0qwj



class Controller():

    def __init__(self):
        self.keyboard = Keyboard()
        self.mouse = Mouse()

class Mouse():

    def getMousePosition(self):
        tup = win32api.GetCursorPos()
        return tup

    def setCursorPos(self,x,y):
        win32api.SetCursorPos((x,y))

    def press(self,x,y, button='left'):
        self.setCursorPos(x,y)
        time.sleep(0.01)
        if button == 'left':
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
        self.left_mouse_stance = 1
    
    def release(self,x=None,y=None, button='left'):
        if x is None:
            x, y = self.getMousePosition()
        if button == 'left':
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)
        self.left_mouse_stance = 0

    def click(self,x,y,button='left'):
        # TODO add slight move!
        self.setCursorPos(x,y)
        time.sleep(0.02)
        if button == 'left':
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
            time.sleep(0.01)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
            time.sleep(0.01)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)

class Keyboard():
    keymap = {
        'DIK_ESCAPE' : 0x01,
        'DIK_1' : 0x02,
        'DIK_2' : 0x03,
        'DIK_3' : 0x04,
        'DIK_4' : 0x05,
        'DIK_5' : 0x06,
        'DIK_6' : 0x07,
        'DIK_7' : 0x08,
        'DIK_8' : 0x09,
        'DIK_9' : 0x0A,
        'DIK_0' : 0x0B,
        'DIK_MINUS' : 0x0C,
        'DIK_EQUALS' : 0x0D,
        'DIK_BACK' : 0x0E,
        'DIK_TAB' : 0x0F,
        'DIK_Q' : 0x10,
        'DIK_W' : 0x11,
        'DIK_E' : 0x12,
        'DIK_R' : 0x13,
        'DIK_T' : 0x14,
        'DIK_Y' : 0x15,
        'DIK_U' : 0x16,
        'DIK_I' : 0x17,
        'DIK_O' : 0x18,
        'DIK_P' : 0x19,
        'DIK_LBRACKET' : 0x1A,
        'DIK_RBRACKET' : 0x1B,
        'DIK_RETURN' : 0x1C,
        'DIK_LCONTROL' : 0x1D,
        'DIK_A' : 0x1E,
        'DIK_S' : 0x1F,
        'DIK_D' : 0x20,
        'DIK_F' : 0x21,
        'DIK_G' : 0x22,
        'DIK_H' : 0x23,
        'DIK_J' : 0x24,
        'DIK_K' : 0x25,
        'DIK_L' : 0x26,
        'DIK_SEMICOLON' : 0x27,
        'DIK_APOSTROPHE' : 0x28,
        'DIK_GRAVE' : 0x29,
        'DIK_LSHIFT' : 0x2A,
        'DIK_BACKSLASH' : 0x2B,
        'DIK_Z' : 0x2C,
        'DIK_X' : 0x2D,
        'DIK_C' : 0x2E,
        'DIK_V' : 0x2F,
        'DIK_B' : 0x30,
        'DIK_N' : 0x31,
        'DIK_M' : 0x32,
        'DIK_COMMA' : 0x33,
        'DIK_PERIOD' : 0x34,
        'DIK_SLASH' : 0x35,
        'DIK_RSHIFT' : 0x36,
        'DIK_MULTIPLY' : 0x37,
        'DIK_LMENU' : 0x38,
        'DIK_SPACE' : 0x39,
        'DIK_CAPITAL' : 0x3A,
        'DIK_F1' : 0x3B,
        'DIK_F2' : 0x3C,
        'DIK_F3' : 0x3D,
        'DIK_F4' : 0x3E,
        'DIK_F5' : 0x3F,
        'DIK_F6' : 0x40,
        'DIK_F7' : 0x41,
        'DIK_F8' : 0x42,
        'DIK_F9' : 0x43,
        'DIK_F10' : 0x44,
        'DIK_NUMLOCK' : 0x45,
        'DIK_SCROLL' : 0x46,
        'DIK_NUMPAD7' : 0x47,
        'DIK_NUMPAD8' : 0x48,
        'DIK_NUMPAD9' : 0x49,
        'DIK_SUBTRACT' : 0x4A,
        'DIK_NUMPAD4' : 0x4B,
        'DIK_NUMPAD5' : 0x4C,
        'DIK_NUMPAD6' : 0x4D,
        'DIK_ADD' : 0x4E,
        'DIK_NUMPAD1' : 0x4F,
        'DIK_NUMPAD2' : 0x50,
        'DIK_NUMPAD3' : 0x51,
        'DIK_NUMPAD0' : 0x52,
        'DIK_DECIMAL' : 0x53,
        'DIK_OEM_102' : 0x56,
        'DIK_F11' : 0x57,
        'DIK_F12' : 0x58,
        'DIK_F13' : 0x64,
        'DIK_F14' : 0x65,
        'DIK_F15' : 0x66,
        'DIK_KANA' : 0x70,
        'DIK_ABNT_C1' : 0x73,
        'DIK_CONVERT' : 0x79,
        'DIK_NOCONVERT' : 0x7B,
        'DIK_YEN' : 0x7D,
        'DIK_ABNT_C2' : 0x7E,
        'DIK_NUMPADEQUALS' : 0x8D,
        'DIK_PREVTRACK' : 0x90,
        'DIK_AT' : 0x91,
        'DIK_COLON' : 0x92,
        'DIK_UNDERLINE' : 0x93,
        'DIK_KANJI' : 0x94,
        'DIK_STOP' : 0x95,
        'DIK_AX' : 0x96,
        'DIK_UNLABELED' : 0x97,
        'DIK_NEXTTRACK' : 0x99,
        'DIK_NUMPADENTER' : 0x9C,
        'DIK_RCONTROL' : 0x9D,
        'DIK_MUTE' : 0xA0,
        'DIK_CALCULATOR' : 0xA1,
        'DIK_PLAYPAUSE' : 0xA2,
        'DIK_MEDIASTOP' : 0xA4,
        'DIK_VOLUMEDOWN' : 0xAE,
        'DIK_VOLUMEUP' : 0xB0,
        'DIK_WEBHOME' : 0xB2,
        'DIK_NUMPADCOMMA' : 0xB3,
        'DIK_DIVIDE' : 0xB5,
        'DIK_SYSRQ' : 0xB7,
        'DIK_RMENU' : 0xB8,
        'DIK_PAUSE' : 0xC5,
        'DIK_HOME' : 0xC7,
        'DIK_UP' : 0xC8,
        'DIK_PRIOR' : 0xC9,
        'DIK_LEFT' : 0xCB,
        'DIK_RIGHT' : 0xCD,
        'DIK_END' : 0xCF,
        'DIK_DOWN' : 0xD0,
        'DIK_NEXT' : 0xD1,
        'DIK_INSERT' : 0xD2,
        'DIK_DELETE' : 0xD3,
        'DIK_LWIN' : 0xDB,
        'DIK_RWIN' : 0xDC,
        'DIK_APPS' : 0xDD,
        'DIK_POWER' : 0xDE,
        'DIK_SLEEP' : 0xDF,
        'DIK_WAKE' : 0xE3,
        'DIK_WEBSEARCH' : 0xE5,
        'DIK_WEBFAVORITES' : 0xE6,
        'DIK_WEBREFRESH' : 0xE7,
        'DIK_WEBSTOP' : 0xE8,
        'DIK_WEBFORWARD' : 0xE9,
        'DIK_WEBBACK' : 0xEA,
        'DIK_MYCOMPUTER' : 0xEB,
        'DIK_MAIL' : 0xEC,
        'DIK_MEDIASELECT' : 0xED
    }

    def getValidButtons(self):
        return [button for button in self.keymap.keys()]

    def isValidButton(self, button):
        return button in self.keymap.keys()

    def buttonToKey(self, button):
        return self.keymap[button]

    def pushButton(self, button):
        butt = self.buttonToKey(button)
        pressKey(butt)
        time.sleep(0.01)
        releaseKey(butt)

    def pressKey(self, button):
        butt = self.buttonToKey(button)
        pressKey(butt)

    def releaseKey(self,button):
        butt = self.buttonToKey(button)
        releaseKey(butt)

SendInput = ctypes.windll.user32.SendInput
# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def pressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def releaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

if __name__ == '__main__':
    pressKey(0x11)
    time.sleep(1)
    releaseKey(0x11)
    time.sleep(1)