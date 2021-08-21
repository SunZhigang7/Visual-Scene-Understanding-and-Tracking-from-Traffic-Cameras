from multi_object_tracking import mot
from keypoint_orientation_detection import keypoint_detector
from auto_camera_calibrator import camera_calibrator
from transfer_coordinates import transfer_coordinates
from speed_estimation import speed_estimation
from crawl_videos import get_videos
from select_video import select_video, scale_video
import torch

# Procedure 0 : Hardware Detection
print('Procedure 0 : Hardware Detection')
print('----------------------------------------------------------------------')
print('The device using in this pipeline')
print(torch.cuda.get_device_name())
print()

# Procedure 1 : Get Videos from London Traffic Cameras
print('Procedure 1 : Get Videos from London Traffic Cameras')
print('----------------------------------------------------------------------')
get_videos_number = eval(input('please enter the number of video files that you want to get : '))
get_videos(get_videos_number)
print('Finished Getting Videos\n')

# Procedure 2 : Select the video that to be analysed
print('Procedure 2 : Select the video that to be analysed')
print('----------------------------------------------------------------------')
select_video_number = input('video : ')
select_video(select_video_number)

print('if there is no vehicle in the vide, please select other video.')
select_video_decision = input('whether to select other video (y/n): ')

while select_video_decision == 'y':
    select_video_number = input('video : ')
    select_video(select_video_number)
    select_video_decision = input('whether to select other video (y/n): ')

scale_decision = input('whether to scale the video (y/n): ')

if scale_decision == 'y':
    scale_number = input('scale number: ')
    scale_video(scale_number)
    print('Finished scaling')

print('Finished Selecting Video\n')

# Procedure 3 : Multi-Object Tracking
print('Procedure 3 : Multi-Object Tracking')
print('----------------------------------------------------------------------')
iou = eval(input('set iou threshold (0.4 recommended) :'))
score = eval(input('set score threshold (0.5 recommended) :'))
mot(0.4, 0.5)


print('Finished Multi-Object Tracking\n')


# Procedure 4 : Key_points and Orientation detection
print('Procedure 4 : Key_points and Orientation detection')
print('----------------------------------------------------------------------')

keypoint_detector()

print('Finished Key_points and Orientation detection\n')

# Procedure 5 : auto_camera_calibrator
print('Procedure 5 : auto_camera_calibrator')
print('----------------------------------------------------------------------')
camera_calibrator()

print('Finished auto_camera_calibrator\n')

# Procedure 6 : transfer_coordinates
print('Procedure 6 : transfer_coordinates')
print('----------------------------------------------------------------------')
transfer_coordinates()

print('Finished transfer_coordinates\n')

# Procedure 7 : speed_estimation
print('Procedure 7 : speed_estimation')
print('----------------------------------------------------------------------')
speed_estimation()

print('Finished speed_estimation\n')



