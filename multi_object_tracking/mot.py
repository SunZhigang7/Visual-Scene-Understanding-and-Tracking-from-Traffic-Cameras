import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import tensorflow as tf
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
import shutil
# Set GPU Memory-Usage
config = ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = InteractiveSession(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

import multi_object_tracking.core.utils as utils
from tensorflow.python.saved_model import tag_constants
from multi_object_tracking.core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .deep_sort import preprocessing, nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .tools import generate_detections as gdet


def mot(iou, score):


    feed_key_detection_input = []
    feed_speed_estimation_input = []

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    score_number = 0

    # initialize deep sort
    model_filename = 'weights_data/multi_object_tracking/deepsort/mars' \
                     '-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    input_size = 416
    video_path = 'data/current_processing_video/current.mp4'

    saved_model_loaded = tf.saved_model.load('weights_data'
                                             '/multi_object_tracking/yolo_v4/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/Traffic_Video_Results.mp4', codec, fps,
                          (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed')
            break
        frame_num += 1

        print('Frame #: ', frame_num)

        # initialize the feed key_detection input
        rois_list = []
        class_ids_list = []
        scores_list = []

        # initialize speed input
        rois_list_sp = []
        class_ids_list_sp = []
        tracker_id_list = []

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car', 'motorbike', 'bus', 'truck']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                    (0, 255, 0), 2)
        print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)

        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

            if score_number >= (len(scores)):
                continue

            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}， score: {}".format(
                str(track.track_id),
                class_name, (
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3]),
                scores[score_number]))

            certain_roi = [bbox[0], bbox[1], bbox[2], bbox[3]]

            tracker_id_list.append(track.track_id)
            rois_list_sp.append(certain_roi)
            class_ids_list_sp.append(class_name)


            if class_name == 'car' and scores[score_number] > 0.7:

                class_ids_list.append(2)
                scores_list.append(scores[score_number])

                rois_list.append(certain_roi)

            score_number = score_number + 1

        rois_array = np.array(rois_list, dtype=np.float32)
        class_ids_array = np.array(class_ids_list)
        scores_array = np.array(scores_list, dtype=np.float32)
        dic = {'rois': rois_array, 'class_ids': class_ids_array, 'scores': scores_array}
        dic_sp = {'rois': rois_list_sp, 'tracker_id': tracker_id_list, 'class': class_ids_list_sp}

        feed_key_detection_input.append(dic)
        feed_speed_estimation_input.append(dic_sp)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # cv2.imshow("Multi-Object Tracking", result)

        score_number = 0
        out.write(result)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    np.save('data/key_points_information/feed_key_detection_input.npy', feed_key_detection_input)

    np.save('data/speed_estimation'
            '/feed_speed_estimation_input.npy', feed_speed_estimation_input)

    cv2.destroyAllWindows()







# mot(0.4, 0.5)
