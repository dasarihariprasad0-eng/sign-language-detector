import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np
import csv

from slr.model.classifier import KeyPointClassifier
from slr.utils.pre_process import (
    calc_landmark_list,
    pre_process_landmark,
)

st.set_page_config(page_title="Sign Language Detector")

st.title("🤟 Sign Language Detector")
st.write("Show hand signs to the camera")

# Load classifier
keypoint_classifier = KeyPointClassifier()

# Load labels
with open("slr/model/label.csv", encoding="utf-8-sig") as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class SignProcessor(VideoProcessorBase):

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                landmark_list = calc_landmark_list(
                    image,
                    hand_landmarks
                )

                pre_processed_landmark_list = (
                    pre_process_landmark(
                        landmark_list
                    )
                )

                hand_sign_id = keypoint_classifier(
                    pre_processed_landmark_list
                )

                if hand_sign_id < len(
                    keypoint_classifier_labels
                ):
                    sign_text = (
                        keypoint_classifier_labels[
                            hand_sign_id
                        ]
                    )

                    cv2.putText(
                        image,
                        sign_text,
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


webrtc_streamer(
    key="sign-language",
    video_processor_factory=SignProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)
