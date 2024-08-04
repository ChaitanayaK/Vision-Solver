import os
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import streamlit as st

from scripts.solver import Solver

solver = Solver()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

st.set_page_config(layout="wide")


st.title("VisionSolver")

video_col, text_col = st.columns((3, 1))

video_placeholder = video_col.empty()

text_placeholder = text_col.empty()
text_placeholder.markdown("### Solutions Here")

canvas_height, canvas_width = 480, 640
canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")

drawing = False
prev_x, prev_y = None, None
color = (0, 255, 0)

rect_width, rect_height = 100, 40
top_left_x = canvas_width - rect_width
top_left_y = 0
bottom_right_x = canvas_width
bottom_right_y = rect_height

clear_rect_width, clear_rect_height = 100, 40
clear_top_left_x = 0
clear_top_left_y = 0
clear_bottom_right_x = clear_rect_width
clear_bottom_right_y = clear_rect_height

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
font_thickness = 2

def draw_solve_button():
    cv2.rectangle(canvas, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)
    (text_width, text_height), _ = cv2.getTextSize("Solve", font, font_scale, font_thickness)
    text_x = top_left_x + (rect_width - text_width) // 2
    text_y = top_left_y + (rect_height + text_height) // 2
    cv2.putText(canvas, "Solve", (text_x, text_y), font, font_scale, font_color, font_thickness)

def draw_clear_button():
    cv2.rectangle(canvas, (clear_top_left_x, clear_top_left_y), (clear_bottom_right_x, clear_bottom_right_y), (255, 0, 0), -1)
    (clear_text_width, clear_text_height), _ = cv2.getTextSize("Clear", font, font_scale, font_thickness)
    clear_text_x = clear_top_left_x + (clear_rect_width - clear_text_width) // 2
    clear_text_y = clear_top_left_y + (clear_rect_height + clear_text_height) // 2
    cv2.putText(canvas, "Clear", (clear_text_x, clear_text_y), font, font_scale, font_color, font_thickness)

def create_canvas():
    global canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
    draw_solve_button()
    draw_clear_button()

create_canvas()

def update_frame():
    global drawing, prev_x, prev_y, canvas

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if top_left_x <= x <= bottom_right_x and top_left_y <= y <= bottom_right_y:
                text_placeholder.markdown("### Loading.....")
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_image_path = os.path.join(temp_dir, 'question_image_solver.png')
                    cv2.imwrite(temp_image_path, canvas)
                    result_text = solver.solve(temp_image_path)
                    text_placeholder.markdown(f"### {result_text}")

            if clear_top_left_x <= x <= clear_bottom_right_x and clear_top_left_y <= bottom_right_y:
                create_canvas()

            if drawing:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), color, thickness=5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_x, middle_finger_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
            if abs(x - middle_finger_x) < 40 and abs(y - middle_finger_y) < 40:
                drawing = False
            else:
                drawing = True

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    frame = cv2.addWeighted(frame, 1, canvas, 1.0, 0)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

while True:
    frame_rgb = update_frame()
    if frame_rgb.any(): 
        video_placeholder.image(frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
