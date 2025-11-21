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
    page_title="QOVES AI | Direct API Mode",
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

# --- DIRECT API FUNCTION (Matches your JS Logic) ---
def call_gemini_api(api_key, prompt):
    """
    Uses raw REST API to bypass SDK version issues.
    Mirrors: https://generativelanguage.googleapis.com/v1beta/models/...
    """
    # We try Flash first, as it's the current standard
    model = "gemini-2.5-pro" 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Check if 1.5 Flash failed, try 1.5 Pro
        if response.status_code != 200:
            model = "gemini-2.5-flash"
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

def calc_dist(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_angle(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    slope = (y2 - y1) / (x2 - x1 + 1e-6)
    return math.degrees(math.atan(slope))

def extract_biometrics(image):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    results = face_mesh.process(img_array)
    
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # --- METRICS CALCULATION ---
    bizygomatic = calc_dist(landmarks[454], landmarks[234], w, h)
    face_height = calc_dist(landmarks[10], landmarks[152], w, h)
    bigonial = calc_dist(landmarks[149], landmarks[378], w, h)
    ipd = calc_dist(landmarks[468], landmarks[473], w, h)
    canthal_tilt = calc_angle(landmarks[133], landmarks[33], w, h) * -1
    midface_h = calc_dist(landmarks[168], landmarks[0], w, h)
    nose_w = calc_dist(landmarks[49], landmarks[279], w, h)
    
    metrics = {
        "Face Shape": {
            "Width-to-Height Ratio": round(bizygomatic / face_height, 2),
            "Jaw-to-Cheek Ratio": round(bigonial / bizygomatic, 2),
        },
        "Eyes": {
            "Canthal Tilt": f"{round(canthal_tilt, 1)} degrees",
            "Eye Spacing (IPD)": f"{int(ipd)} px (relative)"
        },
        "Midface": {
            "Compactness Ratio": round(ipd / midface_h, 2),
            "Nose Width Ratio": round(nose_w / bizygomatic, 2)
        }
    }
    return metrics

# --- UI ---
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Google API Key", type="password")
    st.caption("Using Direct REST API (No SDK)")

st.title("🧬 QOVES AI | Direct Mode")
st.write("Upload photo. We extract geometry and send raw JSON to Gemini 1.5.")

uploaded_file = st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"])

if uploaded_file and api_key:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Input", use_column_width=True)
        if st.button("Run Analysis"):
            with st.spinner("Measuring Face..."):
                metrics = extract_biometrics(image)
                
            if metrics:
                st.success("Measurements Complete")
                st.json(metrics)
                
                with st.spinner("Generating Report via API..."):
                    # PROMPT
                    prompt = f"""
                    Act as a world-class craniofacial aesthetician (QOVES Studio style).
                    Here is the biometric data for a client: {json.dumps(metrics)}
                    
                    Write a detailed, scientific, yet readable aesthetic report.
                    1. Analyze the Face Shape ratios (Is it Square, Oval, Oblong? What does the Jaw/Cheek ratio say about dimorphism?).
                    2. Analyze the Eyes (Discuss Canthal Tilt and spacing).
                    3. Analyze the Midface (Compactness and forward growth indicators).
                    4. Give a "Transformation Protocol" with specific advice (e.g., "To increase perceived jaw width...", "To optimize eye area...").
                    
                    Format in Markdown. Be objective and high-level.
                    """
                    
                    report = call_gemini_api(api_key, prompt)
                    st.session_state['report'] = report
            else:
                st.error("Could not detect face. Try a clearer photo.")

    with col2:
        if 'report' in st.session_state:
            st.markdown("### 📋 Analysis Report")
            st.markdown(st.session_state['report'])

elif uploaded_file and not api_key:
    st.warning("Enter API Key in sidebar to start.")
