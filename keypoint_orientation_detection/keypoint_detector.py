import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np

from keypoint_orientation_detection.core.utils import visualize_keypoint, visualize_bbox
from keypoint_orientation_detection.core.keypoint import KeypointDetector
import shutil
import os
import torch



def keypoint_detector():

    checkpoint_path = 'weights_data/keypoint_orientation_detection' \
                      '/Vehicle_Key_Point_Orientation_Estimation/best_fine_kp_checkpoint.pth.tar'

    mean_std_path = 'weights_data/keypoint_orientation_detection' \
                    '/Vehicle_Key_Point_Orientation_Estimation/VeRi/mean.pth.tar'

    obj_list = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', '', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack',
        'umbrella', '', '', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '',
        'toilet', '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    feed_calibrator_input = []

    general_vehicle_data = [[0.115, 3.90, 0.32],  # 1
                            [0.115, 1.06, 0.32],  # 2
                            [1.685, 3.90, 0.32],  # 3
                            [1.685, 1.06, 0.32],  # 4
                            [0.20, 4.48, 0.35],  # 5
                            [1.60, 4.48, 0.35],  # 6
                            [0.17, 4.48, 0.65],  # 7
                            [1.63, 4.48, 0.65],  # 8
                            [0.90, 4.73, 0.57],  # 9
                            [0.90, 4.76, 0.43],  # 10
                            [0.00, 2.98, 1.00],  # 11
                            [1.80, 2.98, 1.00],  # 12
                            [0.33, 2.67, 1.30],  # 13
                            [1.47, 2.67, 1.30],  # 14
                            [0.36, 1.22, 1.30],  # 15
                            [1.44, 1.22, 1.30],  # 16
                            [0.24, 0.22, 0.83],  # 17
                            [1.56, 0.22, 0.83],  # 18
                            [0.90, 0.11, 0.92],  # 19
                            [0.90, 0.10, 0.78]  # 20
                            ]

    # initialize the folder to save the results
    shutil.rmtree('results/keypoints_results')
    os.mkdir('results/keypoints_results')

    detections_numpy = np.load(
        'data/key_points_information'
        '/feed_key_detection_input.npy', allow_pickle=True)

    detections = detections_numpy.tolist()

    cap_1 = cv2.VideoCapture('data/current_processing_video/current.mp4')

    ok_1, frame_1 = cap_1.read()
    [height, width, pixels] = frame_1.shape
    cap_1.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results/Key_Points_Traffic_Video_Results.mp4',
                          fourcc, 20.0, (width, height))

    cap = cv2.VideoCapture('data/current_processing_video/current.mp4')

    model = KeypointDetector(checkpoint_path=checkpoint_path,
                             mean_std_path=mean_std_path)

    frame_num = 0

    print('Per vehicle, per line')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        current_detections = detections[frame_num - 1]

        image = frame

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints, orientations = model.detect(frame_rgb,
                                               current_detections,
                                               visualize=True)

        keypoints = keypoints.tolist()



        for i in range(len(keypoints)):
            two_dimension_points = []
            three_dimension_points = []

            current_vehicle_keypoints = keypoints[i]
            for j in range(20):
                if current_vehicle_keypoints[j][0] != -1:
                    two_dimension_points.append(current_vehicle_keypoints[j])
                    three_dimension_points.append(general_vehicle_data[j])

            dic = {'2d': two_dimension_points, '3d': three_dimension_points}

            print(dic)
            feed_calibrator_input.append(dic)

        # print(orientations)

        result = image.copy()

        keypoints = torch.tensor(keypoints)

        for box_idx in range(keypoints.shape[0]):
            result = visualize_bbox(
                image=result,
                roi=current_detections['rois'][box_idx],
                class_id=current_detections['class_ids'][box_idx],
                score=current_detections['scores'][box_idx],
                obj_list=obj_list)
            for kp_idx in range(keypoints.shape[1]):
                result = visualize_keypoint(image=result,
                                            coord=keypoints[box_idx][kp_idx],
                                            kp_idx=kp_idx,
                                            orientation=orientations[box_idx])

        result_image_path = 'results/keypoints_results/keypoints_frame' \
                            + str(frame_num) + '.jpg'

        cv2.imwrite(result_image_path, result)

        result_image = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # cv2.imshow("Output Video", result_image)

        out.write(result_image)
        np.save('data/corresponding_points'
                '/feed_calibrator_input.npy', feed_calibrator_input)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break




# keypoint_detector()
