import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import csv

from slr.model.classifier import KeyPointClassifier
from slr.utils.pre_process import (
    calc_landmark_list,
    pre_process_landmark,
)

st.set_page_config(page_title="Sign Language Detector", layout="wide")

st.title("🤟 Sign Language Recognition")
st.write("Real-time hand sign detection using MediaPipe + TensorFlow")

FRAME_WINDOW = st.image([])

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load classifier
keypoint_classifier = KeyPointClassifier()

# Load labels
with open("slr/model/label.csv", encoding="utf-8-sig") as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

cap = cv2.VideoCapture(0)

stop = st.button("Stop Camera")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        st.error("Cannot access camera")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    detected_text = ""

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            landmark_list = calc_landmark_list(frame, hand_landmarks)

            pre_processed_landmark_list = pre_process_landmark(
                landmark_list
            )

            hand_sign_id = keypoint_classifier(
                pre_processed_landmark_list
            )

            if hand_sign_id < len(keypoint_classifier_labels):
                detected_text = keypoint_classifier_labels[
                    hand_sign_id
                ]

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        f"Sign: {detected_text}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    FRAME_WINDOW.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    if stop:
        break

cap.release()