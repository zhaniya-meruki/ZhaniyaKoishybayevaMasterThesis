# import the necessary packages
from imutils import paths
import numpy as np
import cv2 
import argparse
import os

def make_dir(dirName):
	# Create a target directory & all intermediate 
	# directories if they don't exists
	if not os.path.exists(dirName):
		os.makedirs(dirName, exist_ok = True)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
ap.add_argument("-i", "--session_info",  nargs='+', type=int,
	help="sessionID(1,...,500+)")
ap.add_argument("-y", "--dy",  nargs='+', type=int,
	help="a list of shifts in y axis for each position")
ap.add_argument("-x", "--dx",  nargs='+', type=int,
	help="a list of shifts in x axis for each position")
args = vars(ap.parse_args())

session_start = args["session_info"][0]
session_end = args["session_info"][1]

# initialize a path to our dataset
path_to_dataset = args["dataset"]

# initialize lists of shifts
dy = args["dy"][0]
dx = args["dx"][0]

# construct arrays of matched features
# for the given position
ptsA = np.array([[399 + dx, 345 + dy], [423 + dx, 293 + dy], [293 + dx, 316 + dy], [269 + dx, 368 + dy]])
ptsB = np.array([[249, 237], [267, 196], [169, 214], [151, 254]])

# estimate a homography matrix
# for the given position 
(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 2.0)

for session in range(session_start, session_end+1):
    
	if session == 247 or session == 390 or session == 504:
		print("Skipping session "+str(session))
		continue
        
	print("Aligning session "+str(session))
    
	# construct a path to our visual images
	rgb_image_path = os.path.join(path_to_dataset, "session_{}/rgb_frames".format(session))

	# construct a path to our thermal images
	thr_image_path = os.path.join(path_to_dataset, "session_{}/thr_frames".format(session))

	# create a directory to save the aligned visual images
	rgb_image_aligned_path = os.path.join(path_to_dataset, "session_{}/rgb_frames_aligned".format(session))
	make_dir(rgb_image_aligned_path) 
    
	# grab the path to the visual images
	rgb_image_filepaths = list(paths.list_images(rgb_image_path))

	# loop over the visible images
	for rgb_image_filepath in rgb_image_filepaths:
		# extract the current image info
		frame = rgb_image_filepath.split(os.path.sep)[-1]

		# construct a filenames of the corresponding thermal images  
		thr_image_filepath = os.path.join(thr_image_path, frame)
    
		# load rgb and corresponding thermal image 
		rgb_image = cv2.imread(rgb_image_filepath)
		thr_image = cv2.imread(thr_image_filepath)

		# grab height and width of the thermal image 
		(h_thr, w_thr) = thr_image.shape[:2]
        
		# align rgb image with the thermal one
		rgb_image_aligned = cv2.warpPerspective(rgb_image, H, (w_thr, h_thr), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

		# construct paths to save images
		rgb_aligned_path = os.path.join(rgb_image_aligned_path, frame)

		# save the images
		cv2.imwrite(rgb_aligned_path, rgb_image_aligned)