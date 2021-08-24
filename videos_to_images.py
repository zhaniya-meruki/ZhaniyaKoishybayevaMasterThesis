import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import random

sessions = glob('/raid/madina_abdrakhmanova/datasets/SpeakingFacesLipReading/lip_reading/purified/*')

for session_dir in sessions:
	session = session_dir.split(os.path.sep)[-1]
	print('Processing '+session)
    
	if session == 'session_504' or session == '123':
		continue
    
	rgb_video_dir = glob(session_dir + '/*_rgb.avi')[0]
	thr_video_dir = glob(session_dir + '/*_thm.avi')[0]

	rgb_video_stream = cv2.VideoCapture(rgb_video_dir)
	thr_video_stream = cv2.VideoCapture(thr_video_dir)

	rgb_frames = []
	thr_frames = []
    
	while 1:
		still_reading, rgb_frame = rgb_video_stream.read()
		still_reading, thr_frame = thr_video_stream.read()
		if not still_reading:
			rgb_video_stream.release()
			thr_video_stream.release()
			break
		rgb_frames.append(rgb_frame)
		thr_frames.append(thr_frame)
    
	rgb_fulldir = 'Dataset/zhanat/preprocessed/' + session + '/rgb_frames/'
	thr_fulldir = 'Dataset/zhanat/preprocessed/' + session + '/thr_frames/'
	os.makedirs(rgb_fulldir, exist_ok=True)
	os.makedirs(thr_fulldir, exist_ok=True)

	for j in range(len(rgb_frames)):
		cv2.imwrite(path.join(rgb_fulldir, '{}.jpg'.format(j)), rgb_frames[j])
		cv2.imwrite(path.join(thr_fulldir, '{}.jpg'.format(j)), thr_frames[j])