# INSTRUCTIONS
# 1 - Press 'P' or 'p' for play or pause, no matter if upper or lower
# 2 - Color Picker = hold left in some green area, then drag up to the end of region to pick, then release
# 3 - Change the Tolerance and Softness if you want, i put the best values


import cv2
import numpy as np

# Names of image/video and background
VID_NAME  = 'greenscreen-demo.mp4'
IMG_NAME  = 'greenimg.png'
IMG_BG    = 'bg.jpg'
ESC       = 27 
PLAY      = 'P'
# Window Name
WIN_NAME       = "Chroma Keying"
# Trackbar settings
TOLERANCE      = 'Tolerance'
MAX_TOL        = 65535
vTol           = 10000
SOFTNESS       = 'Softness'
MAX_SOFT       = 19
vSoft          = 5
# Define boundaries
start = []
stop  = []
chromastart = False
cmdPlay     = False
h = 0

# Format correctly to save
def fixPoint(pt1, pt2):
  dx = pt2[0][1] - pt1[0][1]  # distance x
  dy = pt2[0][0] - pt1[0][0]  # distance y
  start = pt1
  stop  = pt2
  if dx == dy:              # if are equal
    start = [(pt1[0][0] + 1, pt1[0][1] + 1)]
    stop  = [(pt1[0][0] + 2, pt1[0][1] + 2)]
  if (dx > 0 and dy > 0):   # traced from upper-left -> bottom-right
    start = pt1 
    stop  = pt2
  if (dx > 0 and dy < 0):   # traced from upper-right -> bottom-left
    start = [(pt2[0][0], pt1[0][1])]
    stop  = [(pt1[0][0], pt2[0][1])]
  if (dx < 0 and dy < 0):   # traced from bottom-right -> upper-left
    start = pt2
    stop  = pt1
  if (dx < 0 and dy > 0):   # traced from bottom-left to upper-right
    start = [(pt1[0][0], pt2[0][1])]
    stop  = [(pt2[0][0], pt1[0][1])]
  return start, stop


def onSelectSquare(action, x, y, flags, userdata):
  global start, stop, h, chromastart


  if action==cv2.EVENT_LBUTTONDOWN:   # Action to be taken when left mouse button is pressed
    start = [(x,y)]
  elif action==cv2.EVENT_LBUTTONUP:   # Action to be taken when left mouse button is released
    start, stop = fixPoint(start, [(x, y)])
    img = im[start[0][1]:stop[0][1],start[0][0]:stop[0][0]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H   = cv2.split(img)[0]
    meanH, _ = cv2.meanStdDev(H)
    h = meanH[0][0]
    chromastart = True

# Callback functions
def onTolerance(*args):
  global vTol
  vTol       = cv2.getTrackbarPos(TOLERANCE, WIN_NAME)


def onSoftness(*args):
  global vSoft
  vSoft      = cv2.getTrackbarPos(SOFTNESS, WIN_NAME)


# Trackbar Call
def createTrackbarsAndCallbacks(): 
  cv2.createTrackbar(TOLERANCE, WIN_NAME, vTol, MAX_TOL, onTolerance)
  cv2.createTrackbar(SOFTNESS, WIN_NAME, vSoft, MAX_SOFT, onSoftness)
  cv2.setMouseCallback(WIN_NAME, onSelectSquare)


def display(im): 
  if chromastart:
    bgmask, fgmask, _, _ = buildmasks(im)
    im = cv2.add(bgmask, fgmask)  
  return im


def buildmasks(im):
  lgreen, ugreen = getHBoundaries(im)
  hsv            = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  mask           = cv2.inRange(blur(hsv), lgreen, ugreen)
  bgmask         = cv2.bitwise_and(bg, bg, mask=mask)
  nmask          = cv2.bitwise_not(mask) 
  fgmask         = cv2.bitwise_and(im, im, mask=nmask)
  return bgmask, fgmask, mask, nmask

def blur(mask):
    ksize = vSoft // 2
    if (ksize % 2) == 0:
        ksize = ksize + 1
    size = (ksize, ksize)
    mask = cv2.GaussianBlur(mask, size, 0, 0)
    return mask


def getHBoundaries(im):
  green = h
  tolerance = green*vTol/65535
  hmin = green - tolerance
  hmax = green + tolerance
  lgreen = np.array([hmin, 55, 55])
  ugreen = np.array([hmax, 255, 255])
  return lgreen, ugreen



cap = cv2.VideoCapture(VID_NAME)# open capture 


# Create a window to display results
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object.
# The output is stored in 'outputChaplin.avi' file.
out = cv2.VideoWriter('out.avi',
                      cv2.VideoWriter_fourcc('M','J','P','G'), 
                      10, 
                      (frame_width,frame_height))


if (cap.isOpened() == False):   # was the capture opened?
  print("Error opening video")  # something ocurred
ret, im = cap.read()            # read only the first frame
dim = (im.shape[1], im.shape[0])# dimensions of background according to im
bg  = cv2.imread(IMG_BG,  -1)   # load bg
bg  = cv2.resize(bg, dim)       # resize
out.write(im)
while(cap.isOpened()):          # repeat until opened
  key = cv2.waitKey(25) & 0xFF  # get the key
  if key == ESC:                # if exit
    break                       # ESC key received
  elif chr(key).upper() == PLAY:# play or pause
    cmdPlay = not(cmdPlay)      # control play or pause
  if cmdPlay:                   # if play command 
    ret, im = cap.read()     # Capture the frame
  if ret == True:               # if success
    im = display(im)
    out.write(im)
    cv2.imshow(WIN_NAME, im) # display final results
  else:                         # 
    break                       # Break the loop
  createTrackbarsAndCallbacks() # Create all Trackbares each frame
cap.release()                   # release capture
out.write(im)
out.release()                   # release output stream
cv2.destroyAllWindows()         # finish all

# Create a window to display results
'''
im  = cv2.imread(IMG_NAME, 1)
dim = (im.shape[1], im.shape[0])
bg  = cv2.imread(IMG_BG,  -1)
bg  = cv2.resize(bg, dim)
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
while (True):
  cv2.imshow(WIN_NAME, display(im))  # final result
  key = cv2.waitKey(25) & 0xFF       # get the key
  if key == ESC:                     # if exit
    break                            # ESC key received
  createTrackbarsAndCallbacks()      # Create all Trackbares each frame
cv2.destroyAllWindows()              # finish all
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    #morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #morphed = cv2.erode(mask, kernel, iterations=1)

  dim = (230, 240)
  cv2.imshow('mask', cv2.resize(mask, dim))
  cv2.imshow('bgmask', cv2.resize(bgmask, dim))
  cv2.imshow('nmask', cv2.resize(nmask, dim))
  cv2.imshow('fgmask', cv2.resize(fgmask, dim))

'''