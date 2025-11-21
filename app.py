import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QOVES AI | Advanced Aesthetic Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
    <style>
    .main {background-color: #0e1117; color: #c9d1d9;}
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; font-weight: 300;}
    .stButton button {width: 100%; border-radius: 5px; font-weight: bold;}
    .metric-card {background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d;}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: API KEY ---
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key", type="password", help="Get your free key at makersuite.google.com")
    if api_key:
        genai.configure(api_key=api_key)
    
    st.divider()
    st.info("This app uses Computer Vision (MediaPipe) to extract face geometry, then sends that data to Gemini Pro for a dermatological & aesthetic assessment.")

# --- GEOMETRY ENGINE (MediaPipe) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_coords(p, w, h):
    return int(p.x * w), int(p.y * h)

def calc_dist(p1, p2, w, h):
    x1, y1 = get_coords(p1, w, h)
    x2, y2 = get_coords(p2, w, h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_angle(p1, p2, w, h):
    x1, y1 = get_coords(p1, w, h)
    x2, y2 = get_coords(p2, w, h)
    slope = (y2 - y1) / (x2 - x1 + 1e-6)
    return math.degrees(math.atan(slope))

def extract_biometrics(image):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    results = face_mesh.process(img_array)
    
    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    
    # --- RAW METRICS CALCULATION ---
    # 1. Face Dimensions
    bizygomatic = calc_dist(landmarks[454], landmarks[234], w, h) # Cheek width
    face_height = calc_dist(landmarks[10], landmarks[152], w, h) # Hairline to chin
    bigonial = calc_dist(landmarks[149], landmarks[378], w, h) # Jaw width
    
    # 2. Thirds (Vertical)
    forehead_h = calc_dist(landmarks[10], landmarks[336], w, h)
    midface_h = calc_dist(landmarks[336], landmarks[2], w, h)
    lowerface_h = calc_dist(landmarks[2], landmarks[152], w, h)
    
    # 3. Eyes
    ipd = calc_dist(landmarks[468], landmarks[473], w, h) # Interpupillary
    eye_width_l = calc_dist(landmarks[33], landmarks[133], w, h)
    canthal_tilt = calc_angle(landmarks[133], landmarks[33], w, h) * -1
    
    # 4. Nose & Lips
    nose_w = calc_dist(landmarks[49], landmarks[279], w, h)
    mouth_w = calc_dist(landmarks[61], landmarks[291], w, h)
    philtrum = calc_dist(landmarks[2], landmarks[0], w, h)
    chin_h = calc_dist(landmarks[17], landmarks[152], w, h)
    
    # Compile Data for Gemini
    metrics = {
        "Face Shape Ratios": {
            "Face Width/Height": round(bizygomatic / face_height, 2),
            "Jaw/Cheek Ratio": round(bigonial / bizygomatic, 2),
            "Facial Thirds": f"{forehead_h:.0f} : {midface_h:.0f} : {lowerface_h:.0f}"
        },
        "Eyes": {
            "Canthal Tilt (Degrees)": round(canthal_tilt, 1),
            "Eye Spacing Ratio (IPD/BiZyg)": round(ipd / bizygomatic, 2)
        },
        "Midface": {
            "Midface Ratio (Compactness)": round(ipd / midface_h, 2),
            "Nose Width Ratio": round(nose_w / bizygomatic, 2)
        },
        "Lower Third": {
            "Chin Height Ratio": round(chin_h / lowerface_h, 2),
            "Philtrum/Chin Ratio": round(philtrum / chin_h, 2),
            "Mouth/Jaw Width": round(mouth_w / bigonial, 2)
        }
    }
    
    return img_array, metrics

# --- GEMINI PROMPT GENERATOR ---
def generate_analysis(metrics):
    prompt = f"""
    Act as a world-renowned Facial Aesthetician and Craniofacial Specialist (like QOVES Studio).
    Analyze the user's face based STRICTLY on the following biometric data extracted from their image:
    
    {metrics}
    
    Please generate a highly detailed, clinical, and objective aesthetic report.
    
    REQUIREMENTS:
    1. Structure the report EXACTLY with these sections: 
       - 'First Impression & Facial Archetype'
       - 'The Eyes (Canthal Tilt, IPD, Support)'
       - 'The Midface (Forward Growth, Ratios)'
       - 'The Lower Third (Jaw, Chin, Dimorphism)'
       - 'Skin Quality (General Dermatological Advice)'
       - 'Transformation Protocol (Non-Surgical & Surgical options)'
       
    2. Use the provided data to make specific judgments (e.g., "A Jaw/Cheek ratio of {metrics['Face Shape Ratios']['Jaw/Cheek Ratio']} indicates...").
    3. If the Canthal Tilt is positive, mention "Hunter Eyes" or positive tilt benefits. If negative, suggest styling.
    4. For the 'Transformation Protocol', provide specific, actionable advice (Mewing, chewing, skincare ingredients like Retinol/Vitamin C, hairstyle changes) to optimize their specific ratios.
    5. Be honest but constructive. Use professional terminology (Bigonial width, Bizygomatic width, Sexual Dimorphism).
    
    Output Format: Clean Markdown.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- APP UI ---
st.title("🧬 AI Aesthetic Analysis Engine")
st.write("Upload your photo. The AI measures your face geometry and consults the LLM for a QOVES-style protocol.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("RUN FULL ANALYSIS"):
            if not api_key:
                st.error("Please enter a Gemini API Key in the sidebar to generate the text report.")
            else:
                with st.spinner("Extracting Biometrics..."):
                    processed_img, metrics = extract_biometrics(image)
                    
                if metrics:
                    st.success("Biometrics Extracted.")
                    st.json(metrics) # Show raw data
                    
                    with st.spinner("Consulting Aesthetic AI (Gemini)..."):
                        try:
                            analysis_text = generate_analysis(metrics)
                            st.session_state['analysis'] = analysis_text
                        except Exception as e:
                            st.error(f"API Error: {e}")
    
    with col2:
        if 'analysis' in st.session_state:
            st.markdown("### 📋 Detailed Aesthetic Report")
            st.markdown(st.session_state['analysis'])
            
            st.download_button(
                label="Download Report",
                data=st.session_state['analysis'],
                file_name="aesthetic_analysis.md",
                mime="text/markdown"
            )
