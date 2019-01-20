import cv2
import numpy as np
import copy
import math
import pyautogui as pgi
import time
#from appscript import app

# Environment:
# OS    : Mac OS EL Capitan
# python: 3.5
# opencv: 2.4.13

# parameters
cap_region_x_begin=0.4  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
tabshift = False
bgSubThreshold = 50
learningRate = 0
window = 25
trigger=0
zoomTrigger = False

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works
hasReset = True
def removeExtras(arr):
    newArr = []
    start = arr[0]
    i = 0
    while i!=len(arr) :
        if(arr[i]!=start or i==len(arr)-1):
            newArr.append(start)
            start = arr[i]
        i=i+1
    return newArr

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))

def resetScreen():
    global bgModel
    global triggerSwitch
    global isBgCaptured
    global hasReset
    bgModel = None
    triggerSwitch = False
    isBgCaptured = 0
    print('!!!Reset BackGround!!!')
    for k in range(0,100000):
        k=k*1
    bgModel = cv2.createBackgroundSubtractorMOG2(0,bgSubThreshold)
    isBgCaptured = 1
    print('!!!Background Captured!!!')
    hasReset = True

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

startTime = time.time()
otherTime = time.time()
sizeQ = 32
Q = np.zeros((2,sizeQ))
arrC = np.zeros(32)
while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 100, 300)  # smoothing filter              #ORIGINAL PARAMETERS = (5,50,100)
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)


        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
            
            res = contours[ci]
            hull = cv2.convexHull(res)
            areaHull = cv2.contourArea(hull)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt = calculateFingers(res,drawing)
            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print (cnt)
                    #app('System Events').keystroke(' ')  # simulate pressing blank space
        if(cnt>=5):
            
            resetScreen()
            
        M = cv2.moments(res)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except ZeroDivisionError:
            cx=cx*1
        
        if(hasReset == True):
            hasReset = False
            Q[0,:] = cx
            Q[1,:] = cy 
            arrC[:] = 0
        else:
            Q[:,0:sizeQ-1] = Q[:,1:sizeQ]

            Q[0,sizeQ-1] = cx
            Q[1,sizeQ-1] = cy
            arrC[0:sizeQ-1] = arrC[1:sizeQ]
            arrC[sizeQ-1] = cnt
            
        pattern = removeExtras(arrC)
        if(len(pattern)>=4 and pattern[0]==0 and pattern[1]==1 and pattern[2]==0 and pattern[3]==1):
            #Zoom has been triggered
            zoomTrigger =True
            otherTime = time.time()
        if(len(pattern)>=4 and pattern[0]==0 and pattern[1]==1 and pattern[2]==0 and pattern[3]==1 and zoomTrigger == True):
            #Zoom has been triggered off
            zoomTrigger = False
            #otherTime = time.time()
        #print(Q)
        if(time.time()>otherTime+10 and zoomTrigger==True): #Triggering zoom off after every 10 seconds no matter what
            zoomTrigger = False
            
        
        cv2.imshow('output', drawing)
        #print(maxArea)
        #File Handling
        f = open("ourData.txt","w+")
        if(cy > 180 and cy < 280):
            valueWrite = 0
        elif(cy <= 180):
            valueWrite = 1
            #pgi.scroll(200)
            #time.sleep(1000)
        elif(cy >= 280):
            valueWrite = 2
            #pgi.scroll(-200)
            #time.sleep(1000)
        
        meany=np.mean(Q[1,:])
        meanx=np.mean(Q[0,:])

        if(Q[1,-1]>meany and abs(meany-Q[1,-1])>window and zoomTrigger==False):
            #is moving down
            y=1
            pgi.scroll(-50)
        elif((Q[1,-1]<meany and abs(meany-Q[1,-1])>window and zoomTrigger==False)):
            #is moving up
            y=2
            pgi.scroll(50)
        elif((Q[1,-1]>meany and abs(meany-Q[1,-1])>window and zoomTrigger==True)):
            #zoomin
            y=1
            pgi.hotkey('ctrl','+')
        elif((Q[1,-1]<meany and abs(meany-Q[1,-1])>window and zoomTrigger==True)):
            #zoomout
            y=2
            pgi.hotkey('ctrl','-')

        else:
            y=0
            trigger+=1
        
        if(Q[0,-1]>meanx and abs(meanx-Q[0,-1])>window+100):
            #is moving right
            x=1
        elif((Q[0,-1]<meanx and abs(meanx-Q[0,-1])>window+100)):
            #is moving up
            x=2
        else:
            x=0
            trigger+=1

        if(x==1 and y==0 and tabshift==False):
            pgi.hotkey('ctrl','tab')
            tabshift=True
        elif(x==2 and y==0 and tabshift==False):
            pgi.hotkey('ctrl','shift','tab')
            tabshift=True

        if(x==0 and y==0 and trigger>50):
            tabshift=False
            trigger=0
            #resetScreen()
        
        print('y=',y,'x=',x)
        
        f.write(str(valueWrite))
        f.close()
        #print(areaHull)
        '''if(((areaHull > 25000) and (time.time()>startTime+0.5))):
            startTime = time.time()
            otherTime = time.time()
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            print('!!!Reset BackGround!!!')
            bgModel = cv2.createBackgroundSubtractorMOG2(0,bgSubThreshold)
            isBgCaptured = 1
            print('!!!Background Captured!!!')
            hasReset = True'''
    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
        hasReset = True
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
        
        
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')