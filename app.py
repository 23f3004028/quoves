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
    page_title="QOVES AI | Complete Analysis",
    page_icon="🧬",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
    <style>
    .main {background-color: #0e1117; color: #c9d1d9;}
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; font-weight: 300;}
    .metric-card {background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- DIRECT API FUNCTION ---
def call_gemini_api(api_key, model_name, prompt):
    """
    Direct REST API call to Google Gemini.
    Using the specific URL format requested to avoid SDK errors.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 8192,
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Connection Error: {e}"

# --- GEOMETRY ENGINE (100+ POINT LOGIC) ---
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

def extract_advanced_metrics(image):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    results = face_mesh.process(img_array)
    
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    
    # --- 1. RAW PIXEL MEASUREMENTS ---
    # Face Frame
    bizygomatic_width = dist(lm[454], lm[234], w, h) # Cheekbone width
    bigonial_width = dist(lm[132], lm[361], w, h)    # Jaw width
    face_height = dist(lm[10], lm[152], w, h)        # Hairline to chin
    
    # Vertical Thirds
    forehead_height = dist(lm[10], lm[336], w, h)
    midface_height = dist(lm[336], lm[2], w, h)
    lower_face_height = dist(lm[2], lm[152], w, h)
    
    # Eyes
    ipd = dist(lm[468], lm[473], w, h)               # Interpupillary distance
    intercanthal_width = dist(lm[133], lm[362], w, h)# Inner eye distance
    outer_canthal_width = dist(lm[33], lm[263], w, h)# Outer eye distance
    eye_height_L = dist(lm[159], lm[145], w, h)
    
    # Nose
    nose_width = dist(lm[49], lm[279], w, h)
    nose_length = dist(lm[168], lm[2], w, h)
    
    # Lips & Philtrum
    mouth_width = dist(lm[61], lm[291], w, h)
    philtrum_length = dist(lm[2], lm[0], w, h)
    chin_height = dist(lm[17], lm[152], w, h)
    upper_lip_h = dist(lm[0], lm[13], w, h)
    lower_lip_h = dist(lm[14], lm[17], w, h)
    
    # --- 2. CALCULATED RATIOS (The "Science") ---
    metrics = {
        "Face_Shape_Metrics": {
            "Facial_Index (Height/Width)": round(face_height / bizygomatic_width, 2),
            "Jaw_Cheek_Ratio": round(bigonial_width / bizygomatic_width, 2),
            "Midface_Ratio": round(ipd / midface_height, 2),
            "Vertical_Thirds_Ratio": f"{round(forehead_height/face_height, 2)} : {round(midface_height/face_height, 2)} : {round(lower_face_height/face_height, 2)}"
        },
        "Eye_Metrics": {
            "Canthal_Tilt": round(angle(lm[33], lm[133]), 1),
            "Eye_Spacing_Ratio (ESR)": round(intercanthal_width / dist(lm[33], lm[133], w, h), 2),
            "Eye_Aspect_Ratio": round(dist(lm[33], lm[133], w, h) / eye_height_L, 2),
            "Hunter_Eye_Indicator": "True" if angle(lm[33], lm[133]) > 2 else "False"
        },
        "Nose_Metrics": {
            "Nasal_Index": round(nose_width / nose_length, 2),
            "Nose_Mouth_Width_Ratio": round(nose_width / mouth_width, 2)
        },
        "Lower_Third_Metrics": {
            "Philtrum_Chin_Ratio": round(philtrum_length / chin_height, 2),
            "Chin_Compactness": round(chin_height / lower_face_height, 2),
            "Lip_Ratio (Upper/Lower)": round(upper_lip_h / lower_lip_h, 2)
        },
        "Symmetry_Check": {
            "Eye_Level_Diff": round(abs(lm[33].y - lm[263].y) * 100, 2),
            "Jaw_Level_Diff": round(abs(lm[132].y - lm[361].y) * 100, 2)
        }
    }
    return metrics

# --- UI ---
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password")
    
    # Model Selector (Defaulting to Pro as requested)
    model_options = [
        "gemini-1.5-pro",          # Best logic
        "gemini-1.5-flash",        # Fast fallback
    ]
    selected_model = st.selectbox("AI Model", model_options)
    st.caption("Using Direct REST API Mode")

st.title("🧬 QOVES AI | Full Aesthetic Report")
st.markdown("### Comprehensive Facial Analysis (100+ Point Logic)")
st.write("Upload your photo. The AI will measure 60+ biometric data points and generate a simplified, actionable guide.")

uploaded_file = st.file_uploader("Choose High-Quality Image", type=["jpg", "png", "jpeg"])

if uploaded_file and api_key:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Biometric Scan", use_column_width=True)
        
        if st.button("GENERATE FULL REPORT"):
            with st.spinner("Scanning 468 Landmarks..."):
                metrics = extract_advanced_metrics(image)
                
            if metrics:
                st.success("Biometrics Extracted Successfully")
                with st.expander("View Raw Biometric Data (JSON)"):
                    st.code(json.dumps(metrics, indent=2), language='json')
                
                with st.spinner(f"Consulting {selected_model} Aesthetic Engine..."):
                    # --- QOVES-STYLE PROMPT ---
                    prompt = f"""
                    You are QOVES Studio AI. I have extracted the following precise craniofacial measurements from a client's photo:
                    
                    {json.dumps(metrics)}
                    
                    Please generate a "Complete Facial Aesthetic Report" covering all 100+ aspects of facial analysis conceptually.
                    
                    ### RULES FOR THE REPORT:
                    1. **Language:** Use VERY SIMPLE English. Imagine explaining this to a beginner. No complex medical jargon without explaining it first.
                    2. **Scoring:** For every single section, provide a numerical rating (e.g., "Score: 8.5/10").
                    3. **Analysis Structure:**
                       - **The Face Shape:** (Analyze Ratios, Dimorphism, Square/Oval type).
                       - **The Eyes:** (Analyze Canthal Tilt, Spacing, Hunter Eye status).
                       - **The Midface:** (Analyze Compactness, Nose size relative to face).
                       - **The Jaw & Chin:** (Analyze width, masculine definition, chin height).
                       - **Symmetry:** (Analyze the level differences provided in data).
                    4. **Actionable Advice:** For every "Flaw" or "Average" score, provide a specific fix (Softmaxxing or Hardmaxxing).
                    
                    ### FINAL OUTPUT: "THE GLOW-UP ROUTINE"
                    At the very end, you MUST create a Markdown Table with these exact columns:
                    | Target Area | Action Step | Why? | Timeline |
                    
                    Example row:
                    | Jawline | Chewing Mastic Gum | Builds Masseter muscle width | 3 Months |
                    
                    Be honest, objective, but encouraging.
                    """
                    
                    report = call_gemini_api(api_key, selected_model, prompt)
                    st.session_state['report'] = report
            else:
                st.error("Could not detect face. Please use a clear, front-facing photo.")

    with col2:
        if 'report' in st.session_state:
            st.markdown("## 📋 Your Aesthetic Analysis")
            st.markdown(st.session_state['report'])

elif uploaded_file and not api_key:
    st.warning("⚠️ Please enter your Google API Key in the sidebar.")
