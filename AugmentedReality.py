import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

################################################################################################################
#Camera Calibration
################################################################################################################

#Create the aruco parameter var and dictionary
parameters = cv.aruco.DetectorParameters_create()
myDict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

#Markerlength = 3.7, marker separation = 1.1
myArucoGrid = cv.aruco.GridBoard_create(5, 7, 3.7, 1.1, myDict)

imboard = myArucoGrid.draw((2000, 2000))
cv.imwrite("./outputs/arucoboard.png", imboard)

# plt.axis('off') 
# plt.imshow(imboard)

############################
#Reading Images and getting marker locations
############################

imLoc = "".join([os.getcwd(),'/Images to Load/Calibration Images/'])
allCorners, allTagNums,counter = [], [], []

for idx,filename in enumerate(os.listdir(imLoc)):
    #Reading image from fodler and coverting to grayscale
    image = cv.imread("".join([imLoc,filename])) 
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    #detecting Aruco tags in curent image
    (corners, tagNumbers, _) = cv.aruco.detectMarkers(gray, myDict,parameters=parameters)

    #Appending the detected corner locaitons and tag numbers to the respective arrays
    if idx==0:
        allCorners=corners
        allTagNums = tagNumbers
    else:
        allCorners = np.vstack((allCorners, corners))
        allTagNums = np.vstack((allTagNums,tagNumbers))
    # print(allCorners)
    # print('---------------------')
    # print("APPLE")
    # print(allCorners[0])
    # print('---------------------')
    # print("APPLE")
    counter.append(len(tagNumbers))
counter = np.array(counter)

############################
#Performing Calibration
############################
ret, K_mat, distCoeffs, rvecs, Itvecs = cv.aruco.calibrateCameraAruco(allCorners,allTagNums,counter,myArucoGrid,gray.shape,None,None)
print('K')
print(K_mat)
print('Distortion Coefficients')
print(distCoeffs)


cap = cv.VideoCapture('tag.mp4')
parameters = cv.aruco.DetectorParameters_create()
myDict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

########################################################
#Drawing the AR cube using the 8 corner points passed into the function
########################################################
def drawCube(img, projectedPoints):   
    #Reshaping the projected to points to easier index them in the draw function below
    projectedPoints = np.int32(projectedPoints).reshape(-1,2)
    thickness = 10

    for i in range(4):
        if i<3:
            #Base
            img = cv.line(img, tuple(projectedPoints[i]), tuple(projectedPoints[i+1]),(255,0,0),thickness)
            #Top
            img = cv.line(img, tuple(projectedPoints[i+4]), tuple(projectedPoints[i+5]),(255,0,0),thickness)
        else:
            #Base
            img = cv.line(img, tuple(projectedPoints[i]), tuple(projectedPoints[i-3]),(255,0,0),thickness)
            #Top
            img = cv.line(img, tuple(projectedPoints[i+4]), tuple(projectedPoints[i+1]),(255,0,0),thickness)
        #Vertical
        img = cv.line(img, tuple(projectedPoints[i]), tuple(projectedPoints[i+4]),(255,0,0),thickness)  

    return img


################################################################################################################
#Augmented Reality Calibration
################################################################################################################
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
#This will read indefinitely if the video is a webcam
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    current = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #Create the aruco parameter var and dictionary
   
    #Size of aruco tag in real life in metres
    realSize = 0.11

    ########################################################
    #Getting the corners of the aruco tag and the pose
    ########################################################
    (corners, tagNumber, _) = cv.aruco.detectMarkers(current, myDict,parameters=parameters)
    [rot,trans,objPts] = cv.aruco.estimatePoseSingleMarkers(corners,realSize,K_mat,distCoeffs)


    #ObjPts from the pose function above returns the 4 corners of the aruco tag in the world frame
    #By copying these corners and changing their z values we get the 8 corners of a cube in the world frame
    #This can be passed into the projectPoints Function
    objPts = np.float32(objPts.reshape(4,3))
    objPts = np.vstack((objPts,objPts))
    
    for i in range(4):
        #Multiplying by 2 becuase each objpt is the distance from the centre
        #of the square to a corner. Which is 1/2 of the sidelength.
        objPts[i+4,2] = 2*objPts[0,1]
    if  ((rot is not None) and  (trans is not None) and (corners is not None)):
    #Taking the world frame coordinates objpts and projecting them onto the 2D image plane
      projectedPoints = cv.projectPoints(objPts,rot,trans,K_mat,distCoeffs)

    myImage = drawCube(current,projectedPoints[0])

    cv.imshow("Image",myImage)
    cv.waitKey(1)
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()