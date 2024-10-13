import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import os
freq = 2500
dur = 2000
alert_cooldown = 10
last_alert_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode = False, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
cap = cv2.VideoCapture(0)
isCalibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
def calc_angle(a, b, c):
    a = np.array(a) #shouler joint
    b = np.array(b) #elbow joint
    c = np.array(c)  #wrist joint
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle>180.0:
        angle = 360-angle
    return angle
while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    res = pose.process(frame)
    frame.flags.writeable = True
    if res.pose_landmarks:
        landmarks = res.pose_landmarks.landmark
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        #print(left_shoulder)

        shoulder_angle = calc_angle(left_shoulder, right_shoulder, [right_shoulder[0],0])
        neck_angle = calc_angle(left_ear, left_shoulder, [left_shoulder[0],0])

        if not isCalibrated and calibration_frames <50:
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append(neck_angle)
            calibration_frames +=1
            txt = "Calibrating.........." + str(calibration_frames) + "/50"
            cv2.putText(frame, txt, (5,30), font, 0.5, (0,255,0),2, cv2.LINE_AA)
        elif not isCalibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles)-10
            neck_threshold = np.mean(calibration_neck_angles)-10
            isCalibrated = True
            print('Calibration complete')


        mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0]+right_shoulder[0])//2, (left_shoulder[1]+right_shoulder[1])//2)

        if isCalibrated:
            current_time = time.time()
            if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold:
                status = "Poor posture"
                color = (0,0,255)
                if current_time - last_alert_time > alert_cooldown:
                    print("POOR POSTURE DETECTED!! PLEASE SIT UP STRAIGHT")
                    winsound.Beep(freq, dur)
                    last_alert_time = current_time
                else:
                    status = "Good posture"
                    color = (0,255,0)
                cv2.putText(frame, status,(frame.shape[1], frame.shape[0]), font, 1.1, color, 2, cv2.LINE_AA)


    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Posture Corrector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
