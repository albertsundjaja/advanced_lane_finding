import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import pickle

image_files = glob.glob('github_master/camera_cal/*.jpg')

# Arrays to store object points and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

for file in image_files:
	img = mpimg.imread(file)
	# Prepare object points (blank template)
	# we have 6 X and 8 Y corners combination, so 48 corners (points)
	objp = np.zeros((6*9,3), np.float32) #48 points with 3 elements denoting X Y Z for each point
	# each point will have Z = 0, since we are in 2D image plane

	objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) #x,y coordinates
	# this creates 0, 1 ,2, 3 ... 7 for X and Y combinations
	# e.g. 0,0,0 1,0,0 2,0,0 ... 0,1,0 1,1,0 ... 0,2,0 ...

	# convert image to grayscale, since the function below accept grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

	# if corners are found, add object points & image points
	if ret == True:
	    imgpoints.append(corners)
	    objpoints.append(objp)
	    
	    #draw and display corners
	    #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
	    #plt.imshow(img)
	    #plt.show()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

"""
# Perform undistortion to test images
images = glob.glob('github_master/camera_cal/*.jpg')
for i in images:
	image_name = i.split('/')[-1].split('.')[0]
	print(image_name)
	image = mpimg.imread(i)
	dst_image = cv2.undistort(image, mtx, dist, None, mtx)
	plt.imshow(dst_image)
	plt.savefig('calibrated/' + image_name + '_dst.jpg') """


#save our calibration parameters to pickle file
save_calibration = {
	'ret': ret,
	'mtx': mtx,
	'dist': dist,
	'rvecs': rvecs,
	'tvecs': tvecs
}

pickle.dump(save_calibration, open( "camera_calibration.p", "wb" ) )