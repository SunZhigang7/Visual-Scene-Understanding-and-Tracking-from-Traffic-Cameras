import numpy as np


def pos_3d_from_2d_projection(pt, K, R, T, z):
    '''
        3D position from 2D projection with known calibration parameters and Z-coordinate - NumPy implementation
    '''
    pt2d = np.array((pt[0], pt[1], 1), dtype=np.float64).T
    kInv = np.linalg.inv(K)
    rInv = np.linalg.inv(R)

    rInv_kInv_pt = np.dot(np.dot(rInv, kInv), pt2d)
    rInv_t = np.dot(rInv, T)
    s = z + rInv_t[2]
    s /= rInv_kInv_pt[2]

    return np.dot(rInv, np.reshape(np.dot(s * kInv, pt2d), (3, 1)) - T)


def transfer_coordinates():
    speed_input_numpy = np.load(
        'data/speed_estimation/feed_speed_estimation_input.npy',
        allow_pickle=True)

    calibrator_information = np.load(
        'data/calibrator_information/calibrator.npy',
        allow_pickle=True)

    dic = calibrator_information[0]

    K = dic['K']
    R = dic['R']
    T = dic['T']
    z = dic['road_plane_z']

    file_path = 'speed_estimation/data/coords.txt'
    file = open(file_path, "w")

    print('points in road plane, object id and type')
    for i in range(len(speed_input_numpy)):
        if len(speed_input_numpy[i]['rois']) == 0:
            continue

        for j in range(len(speed_input_numpy[i]['rois'])):
            min_x = speed_input_numpy[i]['rois'][j][0]
            min_y = speed_input_numpy[i]['rois'][j][1]
            max_x = speed_input_numpy[i]['rois'][j][2]
            max_y = speed_input_numpy[i]['rois'][j][3]
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            center_point = (center_x, center_y)
            point_in_road_plane = pos_3d_from_2d_projection(center_point, K, R, T, z)

            x_road_plane = point_in_road_plane[0][0]
            y_road_plane = point_in_road_plane[1][0]

            print("%f,%f,%d,%s" % (x_road_plane, y_road_plane, speed_input_numpy[i]['tracker_id'][j],
                                     speed_input_numpy[i]['class'][j]))

            file.write("%f,%f,%d,%s\n" % (x_road_plane, y_road_plane, speed_input_numpy[i]['tracker_id'][j],
                                          speed_input_numpy[i]['class'][j]))

        print('NewFrame')
        file.write('NewFrame\n')


# transfer_coordinates()
