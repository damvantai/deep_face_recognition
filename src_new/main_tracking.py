# detectors.py
# tracker.py

from detectors import YOLO
import cv2
from PIL import Image
import copy
from tracker import Tracker
import numpy as np
from keras import backend as K
from timeit import default_timer as timer

# Super resolution
from model import Generator
from PIL import Image
import os
import time
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import torch

import tensorflow as tf
import matplotlib.pyplot as plt
import math
import inception_resnet_v1

ix, iy, ex, ey = -1, -1, -1, -1
ix1, iy1, ix2, iy2 = -1, -1, -1, -1
cap_from_stream = False
# path = '/mnt/24eb92b8-4fc1-4f32-b05c-deaa999ce6cf/Documents/datasets/face/camera_supervisor/2.avi'
# path = '/mnt/24eb92b8-4fc1-4f32-b05c-deaa999ce6cf/Documents/datasets/face/camera_supervisor/1_02_cut.avi'
# path = '/home/neosai/Documents/dataset/camera_supervisor/1_03_H_082018120000.avi'
# path = '/home/neosai/Documents/dataset/camera_supervisor/1_02.avi'
path = '/home/neosai/Documents/dataset/camera_supervisor/1_02.avi'
# path = '/home/neosai/Documents/dataset/camera_supervisor/people counting camera [720p].mp4'
# path = '/home/neosai/Documents/dataset/camera_supervisor/People.avi'
#path = 'rtsp://192.168.10.16:554'


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

def draw_line(event, x, y, flags, param):
    global ix1, iy1, ix2, iy2, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        ix1, iy1 = x, y
        print(ix1, iy1)

    elif event == cv2.EVENT_LBUTTONUP:
        ix2, iy2 = x, y
        print(ix2, iy2)
        # cv2.line(param, (ix1, iy1), (x, y), (0, 0, 255), 2)
        cv2.circle(param, (ix1, iy1), 5, (0, 0, 255), 4)
        # cv2.circle(param, (x, y), 5, (0, 0, 255), 2)

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

def get_crop_size1(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if cap_from_stream:
            frame = cv2.resize(frame, (1280, 720))
        cv2.namedWindow('draw_line')
        cv2.setMouseCallback('draw_line', draw_line, frame)
        print("Choose your area of interest!")
        while 1:
            cv2.imshow('draw_line', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('a'):
                cv2.destroyAllWindows()
                break
        break

def super_resolution_image(image, model):
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    # image = image.cuda()
    out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())
    return out_img

def convert_bbox(bbox): # left top righ bottom
    # bbox[0][0] = bbox[0][0] - (bbox[2][0] - bbox[0][0]) / 4
    # bbox[1][0] = bbox[1][0] - (bbox[3][0] - bbox[1][0]) / 4
    # bbox[2][0] = bbox[2][0] + (bbox[2][0] - bbox[0][0]) / 4
    # bbox[3][0] = bbox[3][0] + (bbox[3][0] - bbox[1][0]) / 4

    # print(bbox)
    a0 = int(bbox[0][0])
    a1 = int(bbox[1][0])
    a2 = int(bbox[2][0])
    a3 = int(bbox[3][0])

    return a0, a1, a2, a3

# def loadmodel():
#     tf.reset_default_graph() 
#     sess = tf.InteractiveSession()
#     sess = tf.Session()
#     images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
#     images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
#     train_mode = tf.placeholder(tf.bool)
#     age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
#                                                                  phase_train=train_mode,
#                                                                  weight_decay=1e-5)

#     gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
#     age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
#     age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#     sess.run(init_op)
#     saver = tf.train.Saver()
#     ckpt = tf.train.get_checkpoint_state("/home/neosai/Documents/projects/deep_face_recognition/weights/models_gender_and_age/")
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print("restore model!")
#     else:
#         pass
#         print("error")

def main():
    # Load model and run graph inception resnet v1 from models and file resnetv1_inception.py
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
    train_mode = tf.placeholder(tf.bool)
    age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                 phase_train=train_mode,
                                                                 weight_decay=1e-5)

    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state("/home/neosai/Documents/projects/deep_face_recognition/weights/models_gender_and_age/")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore model!")
    else:
        pass
        print("error")

    # faces = np.empty((1,160, 160, 3))
    aligned_images = []

    upscale_factor = 4
    model_name = "/home/neosai/Documents/projects/deep_face_recognition/weights/netG_epoch_4_100.pth"
    # limit_mem()
    model = Generator(upscale_factor).eval()
    # model.cuda()
    model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
    # Choose area of interest
    get_crop_size(path)
    get_crop_size1(path)
    print('Your area of interest: ', ix, ' ', iy, ' ', ex, ' ', ey)
    area = (ix, iy, ex, ey)
    # mid_y = (iy + ey) / 2
    # print('point line: ', ix1, ' ', iy1, ' ', ix2, ' ', iy2)
    # point1 = (ix, ey)
    # point2 = (ex, iy)

    # y = ax + b
    # a = float((iy2 - iy1) / (ix2 - ix1))
    # b = iy2 - a * ix2
    # print("a,, b: ", a, b)

    # Create opencv video capture object
    cap = cv2.VideoCapture(path)
    w = int(cap.get(3))
    h = int(cap.get(4))
    if cap_from_stream:
        w = 1280
        h = 720
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('1_023.avi', fourcc, 15, (w, h))

    # Create Object Detector
    detector = YOLO()

    # Create Object Tracker
    # tracker = Tracker(iou_thresh=0.3, max_frames_to_skip=5, max_trace_length=20, trackIdCount=0)
    tracker = Tracker(iou_thresh=0.3, max_frames_to_skip=5, max_trace_length=40, trackIdCount=0)

    # Variables initialization
    # track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    #                 (0, 255, 255), (255, 0, 255), (255, 127, 255),
    #                 (127, 0, 255), (127, 0, 127)]
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]         
    # count_vehicle = {'person': 0, 'motorbike': 0, 'car': 0, 'truck': 0, 'bicycle': 0, 'bus': 0}
    # count_people = {'female_0_10':0, 'female_10_20': 0, 'female_20_30': 0,
    #         'female_30_40': 0, , 'female_40_50': 0, 'female_50_100': 0
    #         'male_0_10': 0, 'male_10_20': 0, 'male_20_30': 0, 'male_30_40': 0, 'male_40_50': 0, 'male_50_100': 0}
    count_people_come_in_out = {'female_come_in': 0, 'female_come_out': 0, 'male_come_in': 0, 'male_come_out': 0}

    count_people = {'people_come_in': 0, 'people_come_out': 0}
    id_number = 0
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
        # for bbox in box_detected:
        #     a0, a1, a2, a3 = bbox[0], bbox[1], bbox[2], bbox[3]
        #     cv2.rectangle(result, (a0 - 10, a1 - 10), (a2 + 10, a3 + 10), (0, 255, 0), 3)
        #     # if a1 < iy1 + 50 and a1 > iy1 -50:
        #     print(a1)
        #     image_crop = frame[a1:a3, a0:a2]
        #     cv2.imwrite("image.jpg", image_crop)
        #     image_crop = Image.fromarray(image_crop, 'RGB')
        #     image_crop = super_resolution_image(image_crop, model)
        #     image_crop_array = np.asarray(image_crop)

        #     face_male_resize = image_crop.resize((160, 160), Image.ANTIALIAS)
        #     face = np.array(face_male_resize)
        #     aligned_images.append(face)
        #     # faces[0, :, :, :] = face
        #     age_predict, gender_predict = sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})
        #     aligned_images = []
        #     # print(gender_predict)
        #     # print(type(gender_predict))
        #     label = "{}, {}".format(int(age_predict[0]), "Female" if gender_predict[0] == 0 else "Male")
        #     id_number += 1
        #     print(label)
        #     name = "../image/id_{}, {}".format(id_number, label)
        #     cv2.imwrite(name + ".jpg", image_crop_array)
        #     cv2.rectangle(result, (a0 - 5, a1 - 5), (a2 + 5, a3 + 5), color=(0, 0, 255),
        #                   thickness=3)
        #     cv2.putText(result, label, (a0 + 6, a1 - 6), font, 2, (0, 255, 0), 3, cv2.LINE_AA)

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
                if len(tracker.tracks[i].trace) == 0:
                    bbox = tracker.tracks[i].ground_truth_box.reshape((4, 1))
                    a0, a1, a2, a3 = convert_bbox(bbox)
                    image_crop = frame[a1:a3, a0:a2]
                    cv2.imwrite("image.jpg", image_crop)
                    image_crop = Image.fromarray(image_crop, 'RGB')
                    # image_crop = super_resolution_image(image_crop, model)
                    image_crop_array = np.asarray(image_crop)

                    face_male_resize = image_crop.resize((160, 160), Image.ANTIALIAS)
                    face = np.array(face_male_resize)
                    aligned_images.append(face)
                    # faces[0, :, :, :] = face
                    age_predict, gender_predict = sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})
                    aligned_images = []
                    # print(gender_predict)
                    # print(type(gender_predict))
                    label = "{}, {}".format(int(age_predict[0]), "Female" if gender_predict[0] == 0 else "Male")
                    id_number += 1
                    print(label)
                    name = "../image/id_{}, {}".format(id_number, label)
                    cv2.imwrite(name + ".jpg", image_crop_array)
                    cv2.rectangle(result, (a0 - 5, a1 - 5), (a2 + 5, a3 + 5), color=(0, 0, 255),
                                  thickness=3)
                    cv2.putText(result, label, (a0 + 6, a1 - 6), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
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
                        #     count_people["people_come_out"] += 1
                        # if y_center_first < (a * x_center_first + b):
                        #     count_people["people_come_in"] += 1
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
                        #               thickness=3)
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
