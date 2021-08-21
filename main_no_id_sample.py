from multi_object_tracking import mot
from keypoint_orientation_detection import keypoint_detector
from auto_camera_calibrator import camera_calibrator
from transfer_coordinates import transfer_coordinates
from speed_estimation import speed_estimation
import torch
import time

# Procedure 0 : Hardware Detection
print('Procedure 0 : Hardware Detection')
print('----------------------------------------------------------------------')
print('The device using in this pipeline')
print(torch.cuda.get_device_name())
print()


# Procedure 1 : Get Sample Video
print('Procedure 1 : Get Sample Video')
print('----------------------------------------------------------------------')

v_src = open('data/sample_videos/sample.mp4', 'rb')
content = v_src.read()
v_copy = open('data/current_processing_video/current.mp4', 'wb')
v_copy.write(content)
v_src.close()
v_copy.close()

print('Finished Getting Sample Video\n')

# Procedure 2 : Multi-Object Tracking
print('Procedure 2 : Multi-Object Tracking')
print('----------------------------------------------------------------------')
mot(0.4, 0.5)


print('Finished Multi-Object Tracking\n')


# Procedure 3 : Key_points and Orientation detection
print('Procedure 3 : Key_points and Orientation detection')
print('----------------------------------------------------------------------')

keypoint_detector()

print('Finished Key_points and Orientation detection\n')

# Procedure 4 : auto_camera_calibrator
print('Procedure 4 : auto_camera_calibrator')
print('----------------------------------------------------------------------')
camera_calibrator()

print('Finished auto_camera_calibrator\n')

# Procedure 5 : transfer_coordinates
print('Procedure 5 : transfer_coordinates')
print('----------------------------------------------------------------------')
transfer_coordinates()

print('Finished transfer_coordinates\n')

# Procedure 6 : speed_estimation
print('Procedure 6 : speed_estimation')
print('----------------------------------------------------------------------')
speed_estimation()

print('Finished speed_estimation\n')



