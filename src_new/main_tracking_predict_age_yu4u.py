import cv2
import copy
import numpy as np
import os
import time
import torch
import tensorflow as tf
import matplotlib.pyplot as plt 
import math
from keras.utils.data_utils import get_file
from detectors import YOLO
from keras import backend as K 
from model import Generator
from PIL import Image
from tracker import Tracker
from wide_resnet import WideResNet 

ix, iy, ex, ey = -1, -1, -1, -1
ix1, iy1, ix2, iy2 = -1, -1, -1, -1
cap_from_stream = False

path = '/home/neosai/Documents/dataset/camera_supervisor/1_02.avi'

def limit_mem():
	K.get_session().close()
	cfg = K.tf.ConfigProto()
	cfg.gpu_options.allow_growth = True
	K.set_session(K.tf.Session(config=cfg))

def draw_rec(event, x, y, flags, param):
	global ix, iy, ex, ey, drawing, mode
	if event == cv2.EVENT_LBUTTONDOWN:
		ix, iy = x, y 
	elif event == cv2.EVENT_LBUTTONUP:
		ex, ey = x, y
		cv2.rectangle(param, (ix, iy), (x, y), (0, 255, 0), 0)

def get_crop_size(video_path):
	cap = cv2.VideoCapture(video_path)
	while cap.isOpened():
		ret, frame = cap.read()
		if cap_from_stream:
			frame = cv2.resize(frame, (1280, 720))
		cv2.namedWindow('draw_rectangle')
		cv2.setMouseCallback('draw_rectangle', draw_rec, frame)
		# cv2.setMouseCallback("draw_rectangle", draw_line, frame)
		print("Choose your area of interest!")
		while 1:
			cv2.imshow('draw_rectangle', frame)
			k = cv2.waitKey(1) & 0xFF
			if k == ord('a'):
				cv2.destroyAllWindows()
				break
		break

def convert_bbox(bbox):
	a0 = int(bbox[0][0])
	a1 = int(bbox[1][0])
	a2 = int(bbox[2][0])
	a3 = int(bbox[3][0])

	return a0, a1, a2, a3

def main():
	limit_mem()
	aligned_images = []
	upscale_factor = 4
	model_name = "/home/neosai/Documents/projects/deep_face_recognition/weights/netG_epoch_4_100.pth"
	model = Generator(upscale_factor).eval()
	model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
	get_crop_size(path)
	print('Your area of interest: ', ix, ' ', iy, ' ', ex, ' ', ey)
	area = (ix, iy, ex, ey)

	# Create opencv video capture object
	cap = cv2.VideoCapture(path)
	w = int(cap.get(3))
	h = int(cap.get(4))
	if cap_from_stream:
		w = 1280
		h = 720
	# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('/home/neosai/Documents/projects/deep_face_recognition/video/1_024.avi', fourcc, 15, (w, h))

	# Create Object Detector
	detector = YOLO()
	tracker = Tracker(iou_thresh=0.3, max_frames_to_skip=5, max_trace_length=40, trackIdCount=0)
	track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(0, 255, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127)]  
	count_people_come_in_out = {'female_come_in': 0, 'female_come_out': 0, 'male_come_in': 0, 'male_come_out': 0}

	count_people = {'people_come_in': 0, 'people_come_out': 0}
	id_number = 0
	img_size = 64
	depth = 16
	k = 8
	weight_file = "/home/neosai/Documents/github/age-gender-estimation/utkface/weights.29-3.76_utk.hdf5"
	model_predict_age_and_gender = WideResNet(img_size, depth=depth, k=k)()
	model_predict_age_and_gender.load_weights(weight_file)

	while cap.isOpened():
		# Capture frame-by-frame
		ret, frame = cap.read()

		if cap_from_stream:
			frame = cv2.resize(frame, (1280, 720))
		frame = Image.fromarray(frame)

		# Detect and return centeroids of the objects in the frame
		result, centers, box_detected, obj_type = detector.detect_image(frame, area)
		result = np.asarray(result)
		frame = np.asarray(frame)

		#####
		for bbox in box_detected:
			a0, a1, a2, a3 = bbox[0], bbox[1], bbox[2], bbox[3]
			cv2.rectangle(result, (a0 - 10, a1 - 10), (a2 + 10, a3 + 10), (0, 255, 0), 3)
			if a1 < iy1 + 50 and a1 > iy1 -50:
				print(a1)
				image_crop = frame[a1:a3, a0:a2]
				# cv2.imwrite("image.jpg", image_crop)
				image_crop = Image.fromarray(image_crop, 'RGB')
				image_crop = super_resolution_image(image_crop, model)
				image_crop_array = np.asarray(image_crop)

				face_male_resize = image_crop.resize((img_size, img_size), Image.ANTIALIAS)
				face = np.array(face_male_resize)
				aligned_images.append(face)
				# faces[0, :, :, :] = face
				# age_predict, gender_predict = sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})
				# aligned_images = []

				# predict ages and genders of the detected faces
				results = model_predict_age_and_gender.predict(aligned_images)
				predicted_genders = results[0]
				ages = np.arange(0, 101).reshape(101, 1)
				predicted_ages = results[1].dot(ages).flatten()

				label = "{}, {}".format(int(predicted_ages[0]),
										"F" if predicted_genders[0][0] > 0.5 else "M")
				# print(gender_predict)
				# print(type(gender_predict))
				# label = "{}, {}".format(int(age_predict[0]), "Female" if gender_predict[0] == 0 else "Male")
				id_number += 1
				print(label)
				name = "../image/102/id_{}, {}".format(id_number, label)
				cv2.imwrite(name + ".jpg", image_crop_array)
				cv2.rectangle(result, (a0 - 5, a1 - 5), (a2 + 5, a3 + 5), color=(0, 0, 255),
							  thickness=3)
				cv2.putText(result, label, (a0 + 6, a1 - 6), font, 2, (0, 255, 0), 3, cv2.LINE_AA)

		#####

		# print('Number of detections: ', len(centers))
		# a = 0
		# If centroids are detected then track them
		if len(box_detected) > 0:

			# Track object using Kalman Filter
			tracker.Update(box_detected, obj_type)

			# For identified object tracks draw tracking line
			# Use various colors to indicate different track_id
			for i in range(len(tracker.tracks)):
				# print("trace of track i: ",len(tracker.tracks[i].trace))
				# print("len tracker: ", len(tracker.tracks[i].trace))
				# if len(tracker.tracks[i].trace) == 0:
				#	 bbox = tracker.tracks[i].ground_truth_box.reshape((4, 1))
				#	 a0, a1, a2, a3 = convert_bbox(bbox)
				#	 image_crop = frame[a1:a3, a0:a2]
				#	 cv2.imwrite("image.jpg", image_crop)
				#	 image_crop = Image.fromarray(image_crop, 'RGB')
				#	 # image_crop = super_resolution_image(image_crop, model)
				#	 image_crop_array = np.asarray(image_crop)

				#	 face_male_resize = image_crop.resize((160, 160), Image.ANTIALIAS)
				#	 face = np.array(face_male_resize)
				#	 aligned_images.append(face)
				#	 # faces[0, :, :, :] = face
				#	 age_predict, gender_predict = sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})
				#	 aligned_images = []
				#	 # print(gender_predict)
				#	 # print(type(gender_predict))
				#	 label = "{}, {}".format(int(age_predict[0]), "Female" if gender_predict[0] == 0 else "Male")
				#	 id_number += 1
				#	 print(label)
				#	 name = "../image/id_{}, {}".format(id_number, label)
				#	 cv2.imwrite(name + ".jpg", image_crop_array)
				#	 cv2.rectangle(result, (a0 - 5, a1 - 5), (a2 + 5, a3 + 5), color=(0, 0, 255),
				#				   thickness=3)
				#	 cv2.putText(result, label, (a0 + 6, a1 - 6), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
				if len(tracker.tracks[i].trace) >= 9:

					x_center_first = tracker.tracks[i].trace[0][0][0]
					y_center_first = tracker.tracks[i].trace[0][1][0]

					
					# cv2.circle(result, (int(x_center_first), int(y_center_first)), 5, (0, 0, 255), -1)
					for j in range(len(tracker.tracks[i].trace) - 1):
						# Draw trace line
						x1 = tracker.tracks[i].trace[j][0][0]
						y1 = tracker.tracks[i].trace[j][1][0]
						x2 = tracker.tracks[i].trace[j + 1][0][0]
						y2 = tracker.tracks[i].trace[j + 1][1][0]
						clr = tracker.tracks[i].track_id % 9
						cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)),
								 track_colors[clr], 2)
					classes = tracker.tracks[i].get_obj()
					if (len(tracker.tracks[i].trace) >= 9) and (not tracker.tracks[i].counted):
						bbox = tracker.tracks[i].ground_truth_box.reshape((4, 1))
						tracker.tracks[i].counted = True
						x_center_second = tracker.tracks[i].trace[8][0][0]
						y_center_second = tracker.tracks[i].trace[8][1][0]
						# if y_center_first > (a * x_center_first + b):
						#	 count_people["people_come_out"] += 1
						# if y_center_first < (a * x_center_first + b):
						#	 count_people["people_come_in"] += 1
						if y_center_second > y_center_first and x_center_second > x_center_first:
							count_people["people_come_in"] += 1
						if y_center_second < y_center_first and x_center_second < x_center_first:
							count_people["people_come_out"] += 1
						# a0, a1, a2, a3 = convert_bbox(bbox)


						# image_crop = frame[a1:a3, a0:a2]
						# cv2.imwrite("image.jpg", image_crop)
						# image_crop = Image.fromarray(image_crop, 'RGB')
						# image_crop = super_resolution_image(image_crop, model)
						# image_crop_array = np.asarray(image_crop)

						# face_male_resize = image_crop.resize((160, 160), Image.ANTIALIAS)
						# face = np.array(face_male_resize)
						# aligned_images.append(face)
						# # faces[0, :, :, :] = face
						# age_predict, gender_predict = sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})
						# aligned_images = []
						# # print(gender_predict)
						# # print(type(gender_predict))
						# label = "{}, {}".format(int(age_predict[0]), "Female" if gender_predict[0] == 0 else "Male")
						# id_number += 1
						# print(label)
						# name = "../image/id_{}, {}".format(id_number, label)
						# cv2.imwrite(name + ".jpg", image_crop_array)
						# cv2.rectangle(result, (a0, a1), (a2, a3), color=(255, 0, 0),
						#			   thickness=3)
						# cv2.putText(result, label, (a0 + 6, a1 - 6), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
						

		# Display the resulting tracking frame
		x = 30
		y = 30
		dy = 20
		i = 0
		font = cv2.FONT_HERSHEY_COMPLEX_SMALL

		for key, value in count_people.items():
			text = key + ':' + str(value)
			cv2.putText(result, text, (x, y + dy * i), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
			i += 1
		# cv2.line(result, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)
		cv2.circle(result, (ix1, iy1), 5, (0, 0, 255), 4)
		cv2.rectangle(result, (ix, iy), (ex, ey), (0, 255, 0), 0)
		cv2.imshow('Tracking', result)
		out.write(result)

		# Check for key strokes
		k = cv2.waitKey(1) & 0xff
		if k == ord('n'):
			continue
		elif k == 27:  # 'esc' key has been pressed, exit program.
			break

	# When everything done, release the capture
	out.release()
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	# execute main
	main()