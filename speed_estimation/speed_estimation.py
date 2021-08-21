import cv2
import numpy as np
import statistics


def speed_estimation():
    video_path = 'data/chosen_video/sample.mp4'
    file_path = 'speed_estimation/data/coords.txt'
    speed_results_path = 'results/speed_estimation.txt'

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # get fps of video, we need that for the speed estimation
    print('the fps of video:' + str(fps))

    f = open(file_path, "r")
    fr = open(speed_results_path, "w")

    count = 1
    previous_frame = []
    current_frame = []
    speed_id_per_frame = []
    speed_ids = []

    delta = 0.65

    print("-----------------Instantaneous Speeds-----------------")
    fr.write("-----------------Instantaneous Speeds----------------- \n")

    while True:
        line = f.readline()
        currentline = line.split(",")
        if not line:
            break

        if line == "NewFrame\n":

            # vres ta speed edw
            ids_prev = np.array(previous_frame)[:, 2]
            ids_cur = np.array(current_frame)[:, 2]
            print("Frame {}".format(count))
            for i in range(len(ids_cur)):  # gia kathe current id
                if ids_cur[i] in ids_prev:  # an to id yparxei kai sto
                    index = np.where(ids_prev == ids_cur[i])
                    vector1 = [np.array(previous_frame)[index[0][0]][0], np.array(previous_frame)[index[0][0]][1]]
                    vector2 = [np.array(current_frame)[i][0], np.array(current_frame)[i][1]]
                    speed = np.linalg.norm(np.subtract(vector1, vector2)) * fps

                    if count > 1:
                        index = np.where(np.array(speed_ids)[:, 0] == ids_cur[i])
                        # index=np.amax(index)
                        check = index[0]
                        # print(index.size)
                        if len(check) > 0:
                            # print(index)
                            index = np.amax(index)
                            speed_prev = np.array(speed_ids)[index, 1]
                            speed = delta * speed_prev + (1 - delta) * speed
                        else:
                            speed = speed

                    speed_ids.append([ids_cur[i], speed])
                    fr.write("Frame: {:d}, Object ID: {:d} has instantaneous speed: {} m/s. \n".format(int(count),
                                                                                                       int(ids_cur[i]),
                                                                                                       speed))
                    print("Object ID: {:d} has instantaneous speed: {} m/s.".format(int(ids_cur[i]), speed))

            previous_frame = current_frame  # assign old frame to previous frame
            current_frame = []  # empty data from current frame
            count = count + 1  # auksanoume to count

        if count == 1 and line != "NewFrame\n":
            # to current kai to previous frame einai ta idia
            previous_frame.append([float(currentline[0]), float(currentline[1]), int(currentline[2])])
            current_frame.append([float(currentline[0]), float(currentline[1]), int(currentline[2])])
        if count > 1 and line != "NewFrame\n":
            current_frame.append([float(currentline[0]), float(currentline[1]), int(currentline[2])])

        # print("Line{}: {}".format(count, line.strip()))

    f.close()

    print("-----------------Average Speeds-----------------")
    fr.write("-----------------Average Speeds----------------- \n")

    differentIDs = np.array(speed_ids)[:, 0]
    unique_differentIDs = np.unique(differentIDs)
    for i in range(len(unique_differentIDs)):
        index = np.where(differentIDs == unique_differentIDs[i])
        index = index[0]
        speed = np.array(speed_ids)[index.astype(int), 1]
        speed_new = []
        for j in range(len(speed)):
            if j % 3 == 0 and len(speed) >= 3:
                continue
            speed_new.append(speed[j])
        avg_speed = statistics.median(speed_new)
        print("Object ID: {:d} has average speed: {} m/s or {} km/h.".format(int(unique_differentIDs[i]), avg_speed,
                                                                             avg_speed * 3.6))
        fr.write(
            "Object ID: {:d} has average speed: {} m/s or {} km/h. \n".format(int(unique_differentIDs[i]), avg_speed,
                                                                              avg_speed * 3.6))

    fr.close()
    f.close()


# speed_estimation()
