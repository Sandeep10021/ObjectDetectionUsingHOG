import sys
sys.path.append("C:/Users/Sandeep Saurav/HOG_Project/")
from hog_amsip import *
import numpy as np 
import cv2,joblib
from imutils.object_detection import non_max_suppression
import imutils
from skimage import color
from skimage.transform import pyramid_gaussian


#Define HOG Parameters
# change them if necessary to orientations = 8, pixels per cell = (16,16), cells per block to (1,1) for weaker HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2,2)
threshold = .3

# define the sliding window:
def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

# Upload the saved svm model:
model = joblib.load('C:/Users/Sandeep Saurav/HOG_Project/models/model_person.h5')

# Test the trained classifier on an image below!
scale = 0
detections = []
# read the image you want to detect the object in:
img= cv2.imread("C:/Users/Sandeep Saurav/HOG_Project/test/test.jpg")

# Try it with image resized if the image is too big
img= cv2.resize(img,(300,200)) # can change the size to default by commenting this code out our put in a random number

# defining the size of the sliding window (has to be, same as the size of the image in the training data)
(winW, winH)= (64,128)
windowSize=(winW,winH)
downscale=1.5
# Apply sliding window:
for resized in pyramid_gaussian(img, downscale=1.5): 
    for (x,y,window) in sliding_window(resized, stepSize=10, windowSize=(winW,winH)):
       
        if window.shape[0] != winH or window.shape[1] !=winW: 
            continue
        #window=color.rgb2gray(window)
        fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')  # extract HOG features from the window captured
        fds = fds.reshape(1, -1)
        pred = model.predict(fds) # use the SVM model to make a prediction on the HOG features extracted from the window
        
        if pred == 1:
            if model.decision_function(fds) > 0.4: 
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                   int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                      int(windowSize[1]*(downscale**scale))))
    scale+=1
    
clone = resized.copy()
for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
sc = [score[0] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

# the peice of code above creates a raw bounding box prior to using NMS
# the code below creates a bounding box after using nms on the detections
# you can choose which one you want to visualise, as you deem fit... simply use the following function:
# cv2.imshow in this right place (since python is procedural it will go through the code line by line).
        
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0,191,255), 2)
    cv2.putText(img,'Person',(xA-2,yA-2),1,0.75,(255,12,34),1)
cv2.imshow("Raw Detections after NMS", img)
#### Save the images below
k= cv2.waitKey(0) & 0xFF 
if k == 27:             #wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('E:/M-Tech/Projects 2021/AMSIP/HOG_Project/cache/test8.jpg',img)
    cv2.destroyAllWindows()