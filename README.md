# Object Detection using HOG
This is an application of Object detection using Histogram of Oriented Gradients (HOG) as features and Support Vector Machines (SVM) 
as the classifier. 

This process is implemented in python, the following libraries are required:
1. Scikit-learn (For implementing SVM)
2. OpenCV (for testing)
3. PIL (Image processing library)
4. Numpy (matrix multiplication)
5. Imutils for Non-maximum suppression

A training set should comprise of:
1. Positive images: these images should contain only the object you are trying to detect
2. Negative images: these images can contain anything except for the object you are detecting

Web link for the Inria dataset is shown below - the dataset is for pedestrian detection but this code can be adapted for other datasets too eg. car detection (dataset link: http://cogcomp.org/Data/Car/). Inria dataset link: http://pascal.inrialpes.fr/data/human/

The files are divided into the following:
Training & Testing (this is where you evaluate your trained classifier)
Visualise Hog: simply allows you to see what the gradients calculated look like on a given image (specified by the user).







