import cv2
import numpy as np
import sys
#import coloredlogs, logging


# constants
MAX_DIST_ERROR   = 100
MAX_SKIP_FRAMES  = 5
MIN_OBJ_WSIZE    = 20
MIN_OBJ_HSIZE    = 20

# hyperparameter intialization
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


# get center position given bounding box
def getCenterPixel(bbox):
   x = int(bbox[0] + bbox[2]//2)
   y = int(bbox[1] + bbox[3]//2)
   return x, y


# Estimate the error
def getError(prev_bbox, curr_bbox):
    c_prev = getCenterPixel(prev_bbox)
    c_curr = getCenterPixel(curr_bbox)
    midpoint_delta = abs(np.array(c_prev) - np.array(c_curr))
    L1 = np.linalg.norm(midpoint_delta)
    abs_loc  = abs(np.array(prev_bbox) - np.array(curr_bbox))[:2]
    xloc    = (prev_bbox[0] == curr_bbox[0])
    yloc    = (prev_bbox[1] == curr_bbox[1])
    xbounds = curr_bbox[0] < 0  
    ybounds = curr_bbox[1] < 0
    xsize   = curr_bbox[2] <= MIN_OBJ_HSIZE
    ysize   = curr_bbox[3] <= MIN_OBJ_HSIZE
#    sizex   = curr_bbox[2]
#    sizey   = curr_bbox[3]
#    area    = sizex*sizey
    if L1 == 0 and xloc and yloc:
#        logging.warning("BBOX AT SAME POINT") 
        return 0
    if L1 > MAX_DIST_ERROR:
#        logging.critical("BBOX LOCATION CHANGED DRAMATICALLY")
        return 1
#    if area > 30000 or area < 200:
#        logging.critical("BBOX SIZE CHANGED DRAMATICALLY")
#        return 2
#    if xbounds or ybounds:
#        logging.critical("BBOX LOCATION OUT OF BOUNDS") 
#        return 3
    if xsize or ysize:
#        logging.critical("WIDTH OR HEIGHT OF BBOX TOO SMALL")
        return 4
    return -1


# Run CNN
def ObjectDetection(frame, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    # Remove the bounding boxes with low confidence
    return postprocess(frame, outs)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    return indices, boxes, confidences, classIds

# get the bounding box for the soccer ball
def getBox(indices, classIdS):
    for i in indices:
        i = i[0]
        if classIds[i] == 32:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            return box
    return np.array([-1, -1, -1, -1])

# draw a rectangle from a bounding box
def drawRect(frame, box):
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 0, 0), 3)

if __name__ == '__main__' :
    # Set up tracker.
    # Choose one tracker
#    coloredlogs.install()
    tracker = cv2.TrackerTLD_create()
    nFrameErrors     = 0

    # Load names of classes
    classesFile = 'coco.names'
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = 'yolov3.cfg'
    modelWeights       = 'yolov3.weights'
    filename           = 'soccer-ball.mp4'
    video              = cv2.VideoCapture(filename)
    
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Run first time to detect
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    indices, boxes, confidences, classIds = ObjectDetection(frame, net)
    box = getBox(indices, classIds)
    drawRect(frame, box)
    prev_bbox = tuple(box)
    print("Initial bounding box : {}".format(prev_bbox))
    
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, prev_bbox)
    cv2.putText(frame, "Press any key to start", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2, 1)
    cv2.imshow("Tracking", cv2.resize(frame,(640, 480)))
    cv2.waitKey(0)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # update the tracker and see if any error
        ok, bbox = tracker.update(frame)
        error = getError(prev_bbox, bbox)
        ok = False if error > 0 else True
#        logging.info("{}, {}".format(np.round(prev_bbox,2), np.round(bbox,2)))
        prev_bbox = bbox
        # Draw bounding box
        if ok:
            # Tracking success
            nFrameErrors = 0
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            for i in indices:
                i = i[0]
                if classIds[i] == 32:
                    cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
                    cv2.putText(frame, "Tracking", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv2.LINE_AA)
            nFrameErrors += 1
            if nFrameErrors >= MAX_SKIP_FRAMES or error == 1:
                cv2.putText(frame, "Detecting", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, 1)
                indices, boxes, confidences, classIds = ObjectDetection(frame, net)
                box = getBox(indices, classIds)
                prev_bbox = box
                drawRect(frame, box)
                ok = tracker.init(frame, tuple(prev_bbox))
                nFrameErrors = 0
        
        # Display result
        cv2.imshow("Tracking", cv2.resize(frame,(640, 480)))     
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: 
            break