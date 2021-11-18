# Enter your code here
# Document Scanner
import cv2
import numpy as np
import sys

# 1 - Read image and make copies
img = cv2.imread('scanned-form.jpg')
w = 500
r = img.shape[0]/img.shape[1]
h = int(w*r)
dim = (w, h)
img = cv2.resize(img, dim)
img1 = img.copy()
img2 = img.copy()                 
output = np.zeros(img.shape, np.uint8)     

# 2 - Use grabcut to make image binary
x = 5
y = 80
start_point = (x,y)
end_point   = (img.shape[1] - 10, img.shape[0] - 10)
color       = (255, 0, 0)
thickness   = 2
cv2.rectangle(img, start_point, end_point, color, thickness)
rect = (x, y, end_point[1] + x, end_point[0] + y)
mask = np.zeros(img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
bgdmodel = np.zeros((1, 65), np.float64)
fgdmodel = np.zeros((1, 65), np.float64)
cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')

# 3 - Eliminate some blobs
size      = 5
kSize     = (size, size)
knl       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)
mask2     = cv2.dilate(mask2, knl, iterations = 2)
mask2     = cv2.erode(mask2, knl, iterations = 4)
mask2     = cv2.dilate(mask2, knl, iterations = 2)

# 4 - Find and draw contours
contours, hierarchy = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
epsilon = 0.1*cv2.arcLength(contours[0],True)
approx = cv2.approxPolyDP(contours[0],epsilon,True)
cv2.drawContours(img, contours, -1, (0,255,0), 3);

# 5 - Find Homography
srcPoints = approx.astype(np.float32())
dstPoints = np.float32([[img.shape[1], 0],[0, 0],[0, img.shape[0]], [img.shape[1], img.shape[0]]])
h, status = cv2.findHomography(srcPoints, dstPoints)
outDim = (mask2.shape[1], mask2.shape[0])
imH = cv2.warpPerspective(img1, h, outDim)

# 6 - Plot, waitkey and destroy windows
cv2.imshow("Input", img)
cv2.imshow("Output", imH)
cv2.waitKey(0)
cv2.destroyAllWindows()