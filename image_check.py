# import cv2 

# img_dir = '/raid/zhaniya_koishybayeva/Lip2Wav_thesis/Dataset/zhanat/preprocessed/session_1/both_roi/0.jpg'
# img = cv2.imread(img_dir)
# print(img.shape)

import cv2
import os
import numpy as np
from PIL import Image
import tensorflow as tf

ROOT_DIR = os.path.abspath("/raid/zhaniya_koishybayeva/Lip2Wav_thesis/Dataset/zhanat/preprocessed/session_1/")
 
rgb_img = cv2.imread(os.path.join(ROOT_DIR, 'rgb_roi/0.jpg'))
thr_img = cv2.imread(os.path.join(ROOT_DIR, 'thr_roi_gray/0.jpg'))
thr_img = cv2.cvtColor(thr_img, cv2.COLOR_BGR2GRAY)
thr_img = thr_img.reshape((thr_img.shape[0], thr_img.shape[1], 1))
both_img = cv2.imread(os.path.join(ROOT_DIR, 'both_roi/0.jpg'))

print(rgb_img.shape)
print(thr_img.shape)
print(both_img.shape)
img = np.concatenate((rgb_img, thr_img), 2)
print(img.shape)