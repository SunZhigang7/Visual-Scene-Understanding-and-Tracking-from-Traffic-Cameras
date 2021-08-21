import cv2
import time


def select_video(number):
    v_src = open('data/all_london_traffic_videos/video' + str(number) + '.mp4', 'rb')
    content = v_src.read()
    v_copy = open('data/chosen_video/traffic_video.mp4', 'wb')
    v_copy.write(content)
    v_copy2 = open('data/current_processing_video/current.mp4', 'wb')
    v_copy2.write(content)
    v_src.close()
    v_copy.close()
    v_copy2.close()

    cap_1 = cv2.VideoCapture('data/chosen_video/traffic_video.mp4')

    while cap_1.isOpened():
        ret, frame = cap_1.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # cv2.imshow('frame', gray)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        #
        # time.sleep(0.02)

    cap_1.release()
    cv2.destroyAllWindows()


def scale_video(scale_number):
    cap = cv2.VideoCapture('data/chosen_video/traffic_video.mp4')
    ok, frame = cap.read()
    [height, width, pixels] = frame.shape
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/current_processing_video/current.mp4', fourcc, 20.0,
                          (int(width * eval(scale_number)), int(height * eval(scale_number))))

    cap = cv2.VideoCapture('data/chosen_video/traffic_video.mp4')

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # get img size
        [height, width, pixels] = frame.shape

        # print(width * 4, height * 4, pixels)
        new_img = cv2.resize(frame, (int(width * eval(scale_number)), int(height * eval(scale_number))),
                             interpolation=cv2.INTER_NEAREST)
        result = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        out.write(result)
        # cv2.imshow("new_video", new_img)
        # c = cv2.waitKey(25)
        # if c & 0xFF == ord('q'):
        #     break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# select_video(2)
