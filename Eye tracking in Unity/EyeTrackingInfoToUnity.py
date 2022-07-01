import socket
import time
import numpy as np
import cv2 as cv
import math
import mediapipe as mp

host, port = "127.0.0.1", 25001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))


mp_face_mesh = mp.solutions.face_mesh

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [130]
L_H_RIGHT = [133]
L_H_TOP = [159]
L_H_DOWN = [145]
# left 33, right 133, top 257, down 253

R_H_LEFT = [362]
R_H_RIGHT = [263]
R_H_TOP = [386]
R_H_DOWN = [374]
# left 362, right 263, top ? down 253


RIGHT_EYE_UP = [443]
RIGHT_EYE_DOWN = [450]


counter = 0
right_wink_count = 0
left_wink_count = 0


def euclaideanDistance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def iris_position(iric_center, right_point, left_point, top_point, down_point):
    center_to_right_dist = euclaideanDistance(iric_center, right_point)
    center_to_top_dist = euclaideanDistance(iric_center, top_point)
    total_distance_horizontal = euclaideanDistance(right_point, left_point)
    total_distance_vertical = euclaideanDistance(top_point, down_point)
    try:
        ratio_horizontal = center_to_right_dist / total_distance_horizontal
    except ZeroDivisionError:
        ratio_horizontal = 0.50

    try:
        ratio_vertical = center_to_top_dist / total_distance_vertical
    except ZeroDivisionError:
        ratio_vertical = 0.50

    iris_position = ""
    if ratio_horizontal <= 0.42:
        iris_position = "RIGHT"
    elif ratio_horizontal > 0.42 and ratio_horizontal <= 0.57:
        iris_position = "CENTER"
    else:
        iris_position = "LEFT"

    if ratio_vertical <= 0.47:
        iris_position += " TOP"
    elif ratio_vertical > 0.47 and ratio_vertical <= 0.55:
        iris_position += " MIDDLE"
    else:
        iris_position += " DOWN"

    return iris_position, ratio_vertical


def winking(
    right_eye_right_point,
    right_eye_left_point,
    right_eye_top_point,
    right_eye_down_point,
    left_eye_right_point,
    left_eye_left_point,
    left_eye_top_point,
    left_eye_down_point,
):

    right_total_distance_horizontal = euclaideanDistance(
        right_eye_right_point, right_eye_left_point
    )
    right_total_distance_vertical = euclaideanDistance(
        right_eye_top_point, right_eye_down_point
    )

    left_total_distance_horizontal = euclaideanDistance(
        left_eye_right_point, left_eye_left_point
    )
    left_total_distance_vertical = euclaideanDistance(
        left_eye_top_point, left_eye_down_point
    )

    try:
        ratio_right = right_total_distance_horizontal / right_total_distance_vertical
    except ZeroDivisionError:
        ratio_right = 0.50

    try:
        ratio_left = left_total_distance_horizontal / left_total_distance_vertical
    except ZeroDivisionError:
        ratio_left = 0.50

    winking = ""

    if ratio_right > 6 and ratio_left < 4:
        winking = "RIGHT_WINKING"
    elif ratio_right < 4 and ratio_left > 6:
        winking = "LEFT_WINKING"
    else:
        winking = "NO WINKING"

    return winking


cap = cv.VideoCapture(0)
startPos = [0, 0, 0]  # Vector3   x = 0, y = 0, z = 0
time.sleep(1)
while True:
    time.sleep(0.1)  # sleep 0.1 sec
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points = np.array(
                    [
                        np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                        for p in results.multi_face_landmarks[0].landmark
                    ]
                )
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                # iris
                cv.circle(
                    frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA
                )
                cv.circle(
                    frame, center_right, int(l_radius), (255, 0, 255), 1, cv.LINE_AA
                )

                # right eye
                # white
                cv.circle(
                    frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA
                )
                cv.circle(
                    frame, mesh_points[R_H_LEFT][0], 3, (255, 255, 255), -1, cv.LINE_AA
                )

                # right eye
                # yellow
                cv.circle(
                    frame, mesh_points[R_H_TOP][0], 3, (0, 255, 255), -1, cv.LINE_AA
                )
                cv.circle(
                    frame, mesh_points[R_H_DOWN][0], 3, (0, 255, 255), -1, cv.LINE_AA
                )

                # blue
                cv.circle(
                    frame, mesh_points[RIGHT_EYE_UP][0], 3, (0, 43, 255), -1, cv.LINE_AA
                )
                cv.circle(
                    frame,
                    mesh_points[RIGHT_EYE_DOWN][0],
                    3,
                    (0, 43, 255),
                    -1,
                    cv.LINE_AA,
                )

                #  left eye
                # brown
                cv.circle(
                    frame, mesh_points[L_H_TOP][0], 3, (93, 86, 155), -1, cv.LINE_AA
                )
                cv.circle(
                    frame, mesh_points[L_H_DOWN][0], 3, (93, 86, 155), -1, cv.LINE_AA
                )
                cv.circle(
                    frame, mesh_points[L_H_LEFT][0], 3, (93, 86, 155), -1, cv.LINE_AA
                )
                cv.circle(
                    frame, mesh_points[L_H_RIGHT][0], 3, (93, 86, 155), -1, cv.LINE_AA
                )

                iris_pos, ratio = iris_position(
                    center_right,
                    mesh_points[R_H_RIGHT],
                    mesh_points[R_H_LEFT][0],
                    mesh_points[RIGHT_EYE_UP],
                    mesh_points[RIGHT_EYE_DOWN],
                )
                # print()

                wink = winking(
                    mesh_points[R_H_RIGHT],
                    mesh_points[R_H_LEFT],
                    mesh_points[R_H_TOP],
                    mesh_points[R_H_DOWN],
                    mesh_points[L_H_RIGHT],
                    mesh_points[L_H_LEFT],
                    mesh_points[L_H_TOP],
                    mesh_points[L_H_DOWN],
                )

                if wink == "RIGHT_WINKING" and counter == 0:
                    counter = 1
                if counter != 0:
                    counter += 1
                    if counter > 10:
                        counter = 0
                        right_wink_count += 1
                        startPos[2] += 1

                if wink == "LEFT_WINKING" and counter == 0:
                    counter = 1
                if counter != 0:
                    counter += 1
                    if counter > 10:
                        counter = 0
                        left_wink_count += 1
                        startPos[2] += -1

                cv.putText(
                    frame,
                    f"!!Right wink count {right_wink_count} || Left wink count {left_wink_count}",
                    (30, 30),
                    cv.FONT_HERSHEY_PLAIN,
                    1.2,
                    (0, 255, 0),
                    1,
                    cv.LINE_AA,
                )

                cv.putText(
                    frame,
                    f"Horizontal  {iris_pos.split()[0]} Vertical:{iris_pos.split()[1]}",
                    (30, 50),
                    cv.FONT_HERSHEY_PLAIN,
                    1.2,
                    (0, 255, 0),
                    1,
                    cv.LINE_AA,
                )

                print(iris_pos)

            cv.imshow("img", frame)
            key = cv.waitKey(1)
            if key == ord("q"):
                break

            iris_pos_to_strings = iris_pos.split()
            if iris_pos_to_strings[0] == "LEFT":
                startPos[0] -= 1
            elif iris_pos_to_strings[0] == "RIGHT":
                startPos[0] += 1

            if iris_pos_to_strings[1] == "TOP":
                startPos[1] += 1
            elif iris_pos_to_strings[1] == "DOWN":
                startPos[1] -= 1

            posString = ",".join(
                map(str, startPos)
            )  # Converting Vector3 to a string, example "0,0,0"

            print(posString)

            sock.sendall(
                posString.encode("UTF-8")
            )  # Converting string to Byte, and sending it to C#
            receivedData = sock.recv(1024).decode(
                "UTF-8"
            )  # receiveing data in Byte fron C#, and converting it to String
            print(receivedData)

