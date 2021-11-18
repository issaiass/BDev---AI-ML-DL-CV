# Enter your code here
import cv2
import numpy as np

def onMouse(event, x, y, flags, param):
    global src
    # handle minimum patch in x
    c = (x, y)
    if (x - radius < 0 or x + radius > src.shape[1]):
        return
    # handle minimum patch in y
    if (y - radius < 0 or y + radius > src.shape[0]):
        return
    # left click
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, y)
        # get region of interest
        patch = src[y - radius: y + radius, x - radius: x + radius]
        #cv2.imshow("blemish", patch)
        # get best patch
        newpatch = makeBestPatch(patch)
        #cv2.imshow("newpatch", newpatch)
        # seamless clonning
        mask = np.zeros((2*radius, 2*radius), newpatch.dtype)
        cv2.circle(mask, (radius, radius), radius, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        #img[y - radius:y + radius, x - radius:x + radius] = newpatch
        src = cv2.seamlessClone(newpatch, src, mask, c, cv2.NORMAL_CLONE)

def makeBestPatch(img):
    # for each nxn square
    rows = img.shape[1]
    cols = img.shape[0]
    nsize = 3 # size of squares is nsize x nsize
    minVal = 100000
    #print("size of patch is {0},{1} with squares of {2}x{2}, total of {3} squares each row and each col".format(rows, cols, nsize, blocks))
    for row in range(0, rows, nsize):
        for col in range(0, cols, nsize):
            # get the mini patch of nsize x nsize
            minipatch = img[col:col+nsize, row:row+nsize]
            # calculate sobel and get the lowest
            meanSobelX = np.mean(np.abs(cv2.Sobel(minipatch, cv2.CV_32F, 1, 0 )))
            meanSobelY = np.mean(np.abs(cv2.Sobel(minipatch, cv2.CV_32F, 0, 1 )))
            if (meanSobelX + meanSobelY) < minVal:
                minVal = meanSobelX + meanSobelY
                bestpatch = cv2.resize(minipatch, (2*radius, 2*radius))
                #cv2.imshow("minipatch", bestpatch)
                #print("best patch ({0}, {1})".format(row, col))
            else:
                continue
    # return the patch with the best occurrence
    return bestpatch

radius = 15
src    = cv2.imread("blemish.png")
cv2.namedWindow("Remover")
cv2.setMouseCallback("Remover", onMouse)
k = 0
while k != 27:
    cv2.imshow("Remover", src)
    k = cv2.waitKey(20)
cv2.destroyAllWindows()