import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import math
from PIL import Image

# --- CONFIGURATION ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def calculate_distance(p1, p2, width, height):
    x1, y1 = int(p1.x * width), int(p1.y * height)
    x2, y2 = int(p2.x * width), int(p2.y * height)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2), (x1, y1), (x2, y2)

def calculate_angle(p1, p2, width, height):
    x1, y1 = int(p1.x * width), int(p1.y * height)
    x2, y2 = int(p2.x * width), int(p2.y * height)
    slope = (y2 - y1) / (x2 - x1 + 1e-6) # avoid zero division
    angle = math.degrees(math.atan(slope))
    return angle

st.title("Facial Aesthetics Analyzer (QOVES-Lite)")
st.write("Upload a front-facing photo to analyze geometric ratios.")

uploaded_file = st.file_uploader("Choose a face image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_height, img_width, _ = img_array.shape
    
    # Run MediaPipe Face Mesh
    results = face_mesh.process(img_array)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # --- METRIC 1: CANTHAL TILT (Eye Angle) ---
        # Left Eye: Inner corner (133), Outer corner (33)
        # Right Eye: Inner corner (362), Outer corner (263)
        # Note: We use right eye for simplicity in this demo (Landmarks are mirrored)
        inner_eye = landmarks[133]
        outer_eye = landmarks[33]
        
        tilt = calculate_angle(inner_eye, outer_eye, img_width, img_height)
        # Invert angle because image Y-coordinates go down
        tilt = -tilt 

        # --- METRIC 2: MIDFACE RATIO ---
        # IPD (Inter-pupillary distance) vs Midface Height
        # Pupil centers are roughly 468 and 473
        # Midface height: Nasion (168) to Upper Lip (0)
        
        pupil_l = landmarks[468]
        pupil_r = landmarks[473]
        nasion = landmarks[168]
        upper_lip = landmarks[0]

        ipd_dist, p1_coord, p2_coord = calculate_distance(pupil_l, pupil_r, img_width, img_height)
        midface_height, m1_coord, m2_coord = calculate_distance(nasion, upper_lip, img_width, img_height)
        
        midface_ratio = ipd_dist / midface_height

        # --- DRAWING ---
        # Draw Canthal Tilt Line
        cv2.line(img_array, (int(inner_eye.x * img_width), int(inner_eye.y * img_height)), 
                 (int(outer_eye.x * img_width), int(outer_eye.y * img_height)), (0, 255, 0), 2)
        
        # Draw Midface Height Line
        cv2.line(img_array, m1_coord, m2_coord, (255, 0, 0), 2)

        # --- DISPLAY RESULTS ---
        st.image(img_array, caption='Analyzed Face', use_column_width=True)
        
        st.header("Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Canthal Tilt", f"{tilt:.1f}°", delta="Positive is usually preferred" if tilt > 0 else "Negative tilt")
            if tilt > 4:
                st.success("Positive (Hunter) Tilt Detected")
            elif tilt > 0:
                st.info("Neutral Tilt")
            else:
                st.warning("Negative Tilt")

        with col2:
            st.metric("Midface Ratio", f"{midface_ratio:.2f}", help="Higher is often considered more compact/robust")
            if midface_ratio > 1.0:
                st.success("Compact Midface (High Ratio)")
            else:
                st.info("Longer Midface (Standard)")

    else:
        st.error("No face detected. Please use a clear, front-facing photo.")
