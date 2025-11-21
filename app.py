import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import requests
import json
from PIL import Image
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QOVES AI | Simplified Analysis",
    page_icon="🧬",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
    <style>
    .main {background-color: #0e1117; color: #c9d1d9;}
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; font-weight: 300;}
    .metric-card {background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d;}
    </style>
    """, unsafe_allow_html=True)

# --- DIRECT API FUNCTION ---
def call_gemini_api(api_key, prompt):
    model = "gemini-1.5-flash" 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        # Fallback logic
        if response.status_code != 200:
            model = "gemini-1.5-pro"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Connection Error: {e}"

# --- GEOMETRY ENGINE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def dist(p1, p2, w, h):
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle(p1, p2):
    return math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x)) * -1

def extract_comprehensive_biometrics(image):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    results = face_mesh.process(img_array)
    
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    
    # Raw Measurements
    face_width = dist(lm[454], lm[234], w, h)
    face_height = dist(lm[10], lm[152], w, h)
    jaw_width = dist(lm[132], lm[361], w, h)
    ipd = dist(lm[468], lm[473], w, h)
    eye_width_L = dist(lm[33], lm[133], w, h)
    intercanthal_dist = dist(lm[133], lm[362], w, h)
    nose_width = dist(lm[49], lm[279], w, h)
    nose_height = dist(lm[168], lm[2], w, h)
    mouth_width = dist(lm[61], lm[291], w, h)
    philtrum = dist(lm[2], lm[0], w, h)
    chin_height = dist(lm[17], lm[152], w, h)
    
    data = {
        "General": {
            "Face_Index": round(face_height / face_width, 2),
            "Jaw_Cheek_Ratio": round(jaw_width / face_width, 2),
        },
        "Eyes": {
            "Canthal_Tilt": round(angle(lm[33], lm[133]), 1),
            "Eye_Spacing_Ratio": round(intercanthal_dist / eye_width_L, 2),
        },
        "Midface": {
            "Midface_Compactness": round(ipd / dist(lm[168], lm[0], w, h), 2),
            "Nose_Ratio": round(nose_width / nose_height, 2)
        },
        "Lower_Third": {
            "Philtrum_Chin_Ratio": round(philtrum / chin_height, 2),
            "Mouth_Jaw_Ratio": round(mouth_width / jaw_width, 2)
        }
    }
    return data

# --- UI ---
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Google API Key", type="password")

st.title("🧬 QOVES AI | Simplified Report")
st.write("Upload your photo for an easy-to-understand aesthetic breakdown.")

uploaded_file = st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"])

if uploaded_file and api_key:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Input", use_column_width=True)
        
        if st.button("Analyze Face"):
            with st.spinner("Scanning..."):
                metrics = extract_comprehensive_biometrics(image)
                
            if metrics:
                st.success("Scan Complete")
                
                with st.spinner("Generating Simple Report..."):
                    # --- SIMPLIFIED PROMPT ---
                    prompt = f"""
                    You are a friendly aesthetic consultant. I will give you facial measurements.
                    Based on these numbers: {json.dumps(metrics)}
                    
                    Write a report for a beginner. Use very simple English. No complex medical jargon.
                    
                    REQUIREMENTS:
                    1. For every section (Eyes, Jaw, Midface), give a score out of 10 (e.g., "Score: 8/10").
                    2. Explain WHY you gave that score in 1 simple sentence (e.g., "Your jaw is wide, which looks masculine.").
                    3. At the very end, create a Markdown Table titled "My Glow-Up Routine".
                       - Columns: "Action Step", "Why do it?", "Timeline"
                       - Example Row: | Grow Stubble | Hides soft jawline | 2 Weeks |
                    
                    Keep it encouraging but honest.
                    """
                    
                    report = call_gemini_api(api_key, prompt)
                    st.session_state['report'] = report
            else:
                st.error("Face not found. Use a front-facing photo.")

    with col2:
        if 'report' in st.session_state:
            st.markdown("### 📑 Your Aesthetic Report")
            st.markdown(st.session_state['report'])

elif uploaded_file and not api_key:
    st.warning("Enter API Key in sidebar to start.")
