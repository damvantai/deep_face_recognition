#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
python yolo.py   OR   python yolo_video.py [video_path] [output_path(optional)]
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model

gpu_num = 1


class YOLO(object):
    def __init__(self):
        # self.model_path = 'model_data/yolo.h5'  # model path or trained weights path
        # self.anchors_path = 'model_data/yolo_anchors.txt'
        # self.classes_path = 'model_data/coco_classes.txt'        
        self.model_path = '/home/neosai/Documents/projects/deep_face_recognition/weights/ep069-loss46.542-val_loss45.218.h5'  # model path or trained weights path
        self.anchors_path = '/home/neosai/Documents/projects/deep_face_recognition/weights/yolo_anchors.txt'
        self.classes_path = '/home/neosai/Documents/projects/deep_face_recognition/weights/face.txt'
        self.score = 0.3
        self.iou = 0.2
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            print(self.yolo_model.layers[-1].output_shape[-1])
            print(len(self.yolo_model.output))
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        boxA = np.array(boxA).reshape((4,))
        boxB = np.array(boxB).reshape((4,))
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def detect_image(self, image, area):
        ix, iy, ex, ey = area
        start = timer()

        original_image = image.copy()

        image = image.crop((ix, iy, ex, ey))

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(2e-2 * original_image.size[1] + 0.5).astype('int32'))
        thickness = (original_image.size[0] + original_image.size[1]) // 300

        detection = []
        keep_detection = []

        # Remove overlapping

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            box = [left, top, right, bottom]

            detection.append((box, c, predicted_class, score))

        thresh_iou = 0.1
        thresh_dist_center = 10.
        thresh_score = 0.3
        # for i in range(len(detection)):
        #     # if detection[i][1] in [1, 2, 3, 5, 7]:
        #     if detection[i][1] in [1, 0]:
        #         # if detection[i][3] < thresh_score:
        #         #     keep_detection.append(False)
        #         if detection[i][1] == 0:  # Remove overlapping between person and motorbike, bicycle
        #             max_iou = 0
        #             for j in range(len(detection)):
        #                 # if detection[j][1] == 3 or detection[j][1] == 1:  # motorbike or bicycle
        #                 if detection[j][1] == 1:  # motorbike or bicycle
        #                     IoU = self.bb_intersection_over_union(detection[i][0], detection[j][0])
        #                     max_iou = max(max_iou, IoU)
        #             if max_iou > thresh_iou:
        #                 keep_detection.append(False)
        #             else:
        #                 keep_detection.append(True)
        #         elif detection[i][1] == 1:  # Remove overlapping between bicycle and motorbike
        #             max_iou = 0
        #             for j in range(len(detection)):
        #                 if detection[j][1] == 0:  # motorbike
        #                     IoU = self.bb_intersection_over_union(detection[i][0], detection[j][0])
        #                     max_iou = max(max_iou, IoU)
        #             if max_iou > thresh_iou:
        #                 keep_detection.append(False)
        #             else:
        #                 keep_detection.append(True)
                # elif detection[i][1] == 2:  # Remove overlapping between car and truck
                #     max_iou = 0
                #     for j in range(len(detection)):
                #         if detection[j][1] == 7:  # truck
                #             IoU = self.bb_intersection_over_union(detection[i][0], detection[j][0])
                #             max_iou = max(max_iou, IoU)
                #     if max_iou > thresh_iou:
                #         keep_detection.append(False)
                #     else:
                #         keep_detection.append(True)
                # elif detection[i][1] == 3:  # Remove motorbike overlapping motorbike
                #     max_iou = 0
                #     idx = -1
                #     for j in range(len(detection)):
                #         if detection[j][1] == 3:  # motorbike or bicycle
                #             IoU = self.bb_intersection_over_union(detection[i][0], detection[j][0])
                #             if IoU > max_iou:
                #                 max_iou = IoU
                #                 idx = j
                #     if max_iou > thresh_iou and detection[i][3] < detection[idx][3]:
                #         keep_detection.append(False)
                #     else:
                #         keep_detection.append(True)
            #     else:
            #         keep_detection.append(True)
            # else:
            #     keep_detection.append(False)

        center = []
        box_detect = []
        obj_type = []
        for i in range(len(detection)):
            # if not keep_detection[i]:
            #     continue
            c = detection[i][1]
            predicted_class = self.class_names[c]
            box = detection[i][0]
            score = detection[i][3]

            # label = '{} {:.2f}'.format(predicted_class, score)
            label = '{}'.format(predicted_class)
            draw = ImageDraw.Draw(original_image)
            label_size = draw.textsize(label, font)

            left, top, right, bottom = box
            left = int(left - (right - left) / 4)
            right = int(right + (right - left) / 4)
            top = int(top - (bottom - top) / 4)
            bottom = int(bottom + (bottom - top) / 4)


            left += ix
            right += ix
            top += iy
            bottom += iy
            box_detect.append(np.array([left, top, right, bottom]))
            obj_type.append(predicted_class)

            # print(label, (left, top), (right, bottom))

            x_center = (left + right) / 2
            y_center = (top + bottom) / 2

            centroid = np.array([[x_center], [y_center]])
            center.append(np.round(centroid))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print("time for detect", end - start)

        return original_image, center, box_detect, obj_type

    def close_session(self):
        self.sess.close()
