# Enter your code here
# 1 - Sunglass Filter
import cv2
import numpy as np

WIN_NAME = "GLASSES"

# 1 - Read images
glass = cv2.imread("sunglass.png", -1)
hcimg = cv2.imread("hcriver.jpg")

# 2 - open capture
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# 3 - Face Detection model
# HAAR Cascades
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while(cap.isOpened()):
    # Capture frame-by-frame
    ret, img = cap.read()
    if ret:
        # 3 - Get lens alpha mask and bgr
        glassMask1 = glass[:,:,3]
        glassBGR   = glass[:,:,0:3]
        hcimgcp    = hcimg.copy()

        # 5 - Convert to Gray and detect face
        imgGray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scaleFact  = 1.2
        minNeigh   = 7
        face       = faceCascade.detectMultiScale(imgGray, scaleFact, minNeigh)
        if face != tuple():    
            # 6 - mark the face region
            xf, yf, wf, hf = face[0]
            face       = img[yf:yf + hf, xf: xf + wf]

            Ky         = 2.4
            glasscp    = glassBGR.copy()
            glassdim   = (face.shape[0],int(face.shape[1]/Ky))
            sx         = face.shape[1]/glass.shape[1]
            sy         = face.shape[0]/(Ky*glass.shape[0])
            M          = np.float32([[sx,  0, 0],[0 , sy, 0]])
            glasscp    = cv2.warpAffine(glasscp, M, glassdim)
            glassMask1 = cv2.warpAffine(glassMask1, M, glassdim)
            hcimgcp    = cv2.warpAffine(hcimgcp, M, glassdim)


            # 7 - get Eye Position
            x     = 0
            y     = int(0.22*face.shape[0])
            w, h  = glasscp.shape[1], glasscp.shape[0]
            eye   = face[y:y + h,x:x + w].copy()


            # 8 - Get all masks (pos and neg)
            glassMask3  = cv2.merge((glassMask1, glassMask1, glassMask1)) 
            nglassMask3 = cv2.bitwise_not(glassMask3)
            eyeMask     = cv2.bitwise_and(eye, nglassMask3)
            faceMask    = cv2.bitwise_and(eye, glassMask3)
            glassMask   = cv2.bitwise_and(glasscp, glassMask3)
            hcMask      = cv2.bitwise_and(hcimgcp, glassMask3)

            # 9 - Add weights to masks
            alpha = 0.3
            cv2.addWeighted(faceMask, alpha, glassMask, 1 - alpha, 0, glassMask)
            alpha = 0.1
            cv2.addWeighted(hcMask, alpha, glassMask, 1 - alpha, 0, glassMask)

            # 10 - Position and replace
            eyewithglass = cv2.bitwise_or(eyeMask, glassMask)
            face[y :y  + h ,x :x  + w ] = eyewithglass
            img [yf:yf + hf,xf:xf + wf] = face

        cv2.imshow(WIN_NAME, img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cv2.destroyAllWindows()