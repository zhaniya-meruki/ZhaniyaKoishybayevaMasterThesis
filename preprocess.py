import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
from synthesizer import audio
from synthesizer.hparams import hparams as hp
import random
from shutil import copy
import face_detection
# import pynormalize

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=16, type=int)
parser.add_argument("--sessions_root", help="Root folder of all sessions", required=True)
parser.add_argument("--resize_factor", help="Resize the frames before face detection", default=1, type=int)

args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(args.ngpu)]

def process_video_file(session_dir, args, gpu_id, session):
    
	rgb_images = glob(session_dir + '/rgb_frames_aligned/*')    
	num_frames = len(rgb_images)
    
	rgb_frames = []
	thr_frames = []
    
	for i in range(num_frames):
		rgb_image = session_dir + '/rgb_frames_aligned/' + str(i) + '.jpg'
		thr_image = session_dir + '/thr_frames/' + str(i) + '.jpg'
        
		rgb_frame = cv2.imread(rgb_image)
		thr_frame = cv2.imread(thr_image)
        
		rgb_frame = cv2.resize(rgb_frame, (rgb_frame.shape[1]//args.resize_factor, rgb_frame.shape[0]//args.resize_factor))
		thr_frame = cv2.resize(thr_frame, (thr_frame.shape[1]//args.resize_factor, thr_frame.shape[0]//args.resize_factor))
        
		rgb_frames.append(rgb_frame)
		thr_frames.append(thr_frame)
    
	rgb_dir = 'Dataset/zhanat/preprocessed/' + session + '/rgb_roi'
	thr_dir = 'Dataset/zhanat/preprocessed/' + session + '/thr_roi'
    
	os.makedirs(rgb_dir, exist_ok=True)
	os.makedirs(thr_dir, exist_ok=True)
    
	batches = [rgb_frames[i:i + args.batch_size] for i in range(0, len(rgb_frames), args.batch_size)]

	i = -1
	for fb in batches:
		preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

		for j, f in enumerate(preds):
			i += 1
			if f is None:
				continue

			cv2.imwrite(path.join(rgb_dir, '{}.jpg'.format(i)), f[0])
            
			rgb_frame = cv2.imread(session_dir + '/rgb_frames_aligned/' + str(i) + '.jpg')
			thr_frame = cv2.imread(session_dir + '/thr_frames/' + str(i) + '.jpg')
            
			rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
			roi_gray = cv2.cvtColor(f[0], cv2.COLOR_BGR2GRAY)
            
			w, h = roi_gray.shape[::-1]
            
			res = cv2.matchTemplate(rgb_gray, roi_gray, cv2.TM_CCOEFF_NORMED)
			loc = np.where(res >= 0.9)
            
			for pt in zip(*loc[::-1]):
				thr_roi = thr_frame[pt[1]:(pt[1] + h), pt[0]:(pt[0] + w)]
				cv2.imwrite(path.join(thr_dir, '{}.jpg'.format(i)), thr_roi)


def process_audio_file(session_dir, args, gpu_id, session):   
	fulldir = 'Dataset/zhanat/preprocessed/' + session + '/'

	wavpath = path.join(fulldir, 'audio.wav')
	specpath = path.join(fulldir, 'mels.npz')
    
	wav = audio.load_wav(wavpath, hp.sample_rate)
	spec = audio.melspectrogram(wav, hp)
	lspec = audio.linearspectrogram(wav, hp)
	np.savez_compressed(specpath, spec=spec, lspec=lspec)
    
def process_roi(session_dir, args, gpu_id, session):
	both_dir = 'Dataset/zhanat/preprocessed/' + session + '/both_roi'
	gray_dir = 'Dataset/zhanat/preprocessed/' + session + '/thr_roi_gray'
    
	os.makedirs(both_dir, exist_ok=True)
	os.makedirs(gray_dir, exist_ok=True)
    
	rgb_roi = glob(session_dir + '/rgb_roi/*')
	num_frames = len(rgb_roi)
    
	for i in range(num_frames):
		rgb_roi_dir = session_dir + '/rgb_roi/' + str(i) + '.jpg'
		thr_roi_dir = session_dir + '/thr_roi/' + str(i) + '.jpg'
        
		rgb_roi = cv2.imread(rgb_roi_dir)
		thr_roi = cv2.imread(thr_roi_dir)
        
		thr_roi_gray = cv2.cvtColor(thr_roi, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(path.join(gray_dir, '{}.jpg'.format(i)), thr_roi_gray)
        
		thr_roi_gray = thr_roi_gray.reshape((thr_roi_gray.shape[0], thr_roi_gray.shape[1], 1))
		both_roi = np.concatenate((rgb_roi, thr_roi_gray), 2)
		cv2.imwrite(path.join(both_dir, '{}.jpg'.format(i)), both_roi)
        
def copy_normalize_audios(num_sessions):
	madina_folder = '/raid/madina_abdrakhmanova/datasets/SpeakingFacesLipReading/lip_reading/purified/'
	sessions = glob('Dataset/zhanat/preprocessed/*')
    
	for i in range(num_sessions):
		session = sessions[i].split(os.path.sep)[-1]
		audio_dir = glob(madina_folder+session+'/*_m1.wav')[0]
		fulldir = 'Dataset/zhanat/preprocessed/' + session + '/'
		wavpath = path.join(fulldir, 'audio.wav')
		copy(audio_dir, wavpath)
        
# 	Files = glob('Dataset/zhanat/preprocessed/*/audio.wav')
# 	target_dbfs = -13.5

# 	pynormalize.process_files(Files, target_dbfs)

		normalize = 'ffmpeg-normalize Dataset/zhanat/preprocessed/*/audio.wav -o Dataset/zhanat/preprocessed/*/audio.wav -f -v -ar 44100 -ext wav'
		subprocess.call(normalize, shell=True)
        
	
def mp_handler(job):
	session_dir, args, gpu_id = job
	session = session_dir.split(os.path.sep)[-1]
	try:
		#process_video_file(session_dir, args, gpu_id, session)
		process_audio_file(session_dir, args, gpu_id, session)
		#process_roi(session_dir, args, gpu_id, session)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def main(args):
	print('Started processing for {} with {} GPUs'.format(args.sessions_root, args.ngpu)) 
	
	sessions = glob(args.sessions_root + '*')
	num_sessions = len(sessions)
    
	test_size = num_sessions//20
	train_size = num_sessions - 2*test_size
	
	random.shuffle(sessions)

	t = open('Dataset/zhanat/train.txt', 'w')
	v = open('Dataset/zhanat/val.txt', 'w')
	s = open('Dataset/zhanat/test.txt', 'w')
    
	for i in range(num_sessions):
		txtdir = sessions[i].split(os.path.sep)[-1]
        
		if i < test_size:
			s.write(txtdir+"\n")
		elif i < 2*test_size:
			v.write(txtdir+"\n")
		else:
			t.write(txtdir+"\n")
            
	t.close()
	v.close()
	s.close()
    
# 	copy_normalize_audios(num_sessions)

	jobs = [(session_dir, args, i%args.ngpu) for i, session_dir in enumerate(sessions)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
    
# 	for session in range(1, num_sessions+1):
# 		rgb = len(glob(args.sessions_root + 'session_' + str(session) + '/rgb_frames/*'))
# 		rgb_aligned = len(glob(args.sessions_root + 'session_' + str(session) + '/rgb_frames_aligned/*'))
# 		rgb_roi = len(glob(args.sessions_root + 'session_' + str(session) + '/rgb_roi/*'))
# 		thr = len(glob(args.sessions_root + 'session_' + str(session) + '/thr_frames/*'))
# 		thr_roi = len(glob(args.sessions_root + 'session_' + str(session) + '/thr_roi/*'))
        
# 		if rgb == rgb_aligned and rgb == thr:
# 			if rgb == rgb_roi and rgb == thr_roi:
# 				continue
# 			else:
# 				print('session '+str(session)+': rgb '+str(rgb)+' rgb_roi '+str(rgb_roi)+' thr_roi '+str(thr_roi))
# 		else:
# 			print('session '+str(session)+' wrong, rgb: '+str(rgb)+' thr: '+str(thr)+' aligned: '+str(rgb_aligned))

if __name__ == '__main__':
	main(args)
