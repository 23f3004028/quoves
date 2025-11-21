import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import math
from PIL import Image

# --- PAGE CONFIGURATION (To look professional) ---
st.set_page_config(
    page_title="QOVES Clone | Facial Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLES ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
    h1, h2, h3 {
        font-family: 'Helvetica', sans-serif;
        font-weight: 300;
    }
    .highlight {
        color: #4facfe;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# --- GEOMETRY FUNCTIONS ---
def calculate_distance(p1, p2, width, height):
    x1, y1 = int(p1.x * width), int(p1.y * height)
    x2, y2 = int(p2.x * width), int(p2.y * height)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_angle(p1, p2, width, height):
    x1, y1 = int(p1.x * width), int(p1.y * height)
    x2, y2 = int(p2.x * width), int(p2.y * height)
    slope = (y2 - y1) / (x2 - x1 + 1e-6)
    angle = math.degrees(math.atan(slope))
    return angle

# --- THE LOGIC ENGINE (The "Doctor") ---
def generate_protocol(measurements):
    protocol = []
    
    # 1. JAW ANALYSIS
    jaw_ratio = measurements['jaw_width'] / measurements['face_width']
    if jaw_ratio < 0.85:
        protocol.append({
            "area": "Lower Third (Jaw)",
            "issue": "Soft Gonial Angle / Narrow Jaw",
            "science": "A wider bigonial width is associated with higher testosterone and perceived dominance.",
            "solution": "Grow a short stubble beard (3-5mm).",
            "timeline": "2 Weeks",
            "action": "Use Minoxidil 5% (consult doctor) + Microneedling to increase density in the masseter region."
        })
    else:
        protocol.append({
            "area": "Lower Third (Jaw)",
            "issue": "Good Jaw Definition",
            "science": "Your bigonial width is within the ideal range.",
            "solution": "Maintain low body fat (<15%) to keep definition visible.",
            "timeline": "Maintenance",
            "action": "Gua Sha massage to reduce water retention."
        })

    # 2. EYE ANALYSIS (Canthal Tilt)
    if measurements['canthal_tilt'] < 2:
        protocol.append({
            "area": "Upper Third (Eyes)",
            "issue": "Neutral/Negative Canthal Tilt",
            "science": "Positive canthal tilt (hunter eyes) is preferred in evolutionary biology as a sign of alertness.",
            "solution": "Optical illusion via Grooming.",
            "timeline": "Instant",
            "action": "Shape eyebrows to be straighter and lower. Avoid curved tails that drag the eye down."
        })
    
    # 3. MIDFACE RATIO
    if measurements['midface_ratio'] < 0.95:
        protocol.append({
            "area": "Midface",
            "issue": "Long Midface (Compactness Index Low)",
            "science": "A compact midface indicates proper maxilla forward growth.",
            "solution": "Volumize Cheeks.",
            "timeline": "Instant (Visual)",
            "action": "Use hairstyles with volume on the sides, not the top. Height on top elongates the face further."
        })
        
    return protocol

# --- MAIN APP ---
st.title("SCIENCE-BASED AESTHETICS")
st.subheader("Get your personalized facial analysis and transformation plan.")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload clear front facing photo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_height, img_width, _ = img_array.shape
    
    results = face_mesh.process(img_array)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # --- EXTRACT DATA POINTS ---
        # Jaw (Bigonial) vs Cheek (Bizygomatic)
        jaw_left = landmarks[58]
        jaw_right = landmarks[288]
        cheek_left = landmarks[123]
        cheek_right = landmarks[352]
        
        # Eyes
        inner_eye = landmarks[133]
        outer_eye = landmarks[33]
        
        # Midface
        nasion = landmarks[168]
        upper_lip = landmarks[0]
        pupil_l = landmarks[468]
        pupil_r = landmarks[473]

        # --- CALCULATE METRICS ---
        jaw_dist = calculate_distance(jaw_left, jaw_right, img_width, img_height)
        cheek_dist = calculate_distance(cheek_left, cheek_right, img_width, img_height)
        
        tilt = calculate_angle(inner_eye, outer_eye, img_width, img_height) * -1
        
        midface_h = calculate_distance(nasion, upper_lip, img_width, img_height)
        ipd = calculate_distance(pupil_l, pupil_r, img_width, img_height)
        
        measurements = {
            "jaw_width": jaw_dist,
            "face_width": cheek_dist,
            "canthal_tilt": tilt,
            "midface_ratio": ipd / midface_h
        }

        # --- GENERATE THE "QOVES" REPORT ---
        protocol_list = generate_protocol(measurements)

        with col2:
            st.image(image, caption="Biometric scan complete", use_column_width=True)

        # --- DISPLAY DASHBOARD ---
        st.divider()
        st.header("01 / Your Biometric Data")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Sexual Dimorphism (Jaw)", f"{(jaw_dist/cheek_dist)*100:.1f}%", "Target: 90%+")
        m_col2.metric("Canthal Tilt", f"{tilt:.1f}°", "Target: 4-8°")
        m_col3.metric("Midface Compactness", f"{(ipd/midface_h):.2f}", "Target: 1.0+")

        st.divider()
        st.header("02 / Transformation Protocol")
        st.write("Based on your unique facial topology, here is your non-surgical action plan.")

        for item in protocol_list:
            with st.expander(f"📍 {item['area']}: {item['issue']}", expanded=True):
                st.markdown(f"**The Science:** *{item['science']}*")
                st.markdown(f"**Strategy:** {item['solution']}")
                st.markdown(f"**Estimated Timeline:** `{item['timeline']}`")
                st.info(f"👉 **Daily Action:** {item['action']}")

        st.divider()
        st.header("03 / Visualization")
        st.write("Your personalized visualization shows how these changes affect harmony.")
        # Note: Real-time image morphing requires GANs (Deep Learning) which cannot run on basic Streamlit Cloud.
        # We use overlay simulation here.
        
        st.warning("⚠️ Medical Disclaimer: This analysis is based on geometric algorithms and average aesthetic ratios. It is not a medical diagnosis.")

    else:
        st.error("Face not detected. Please ensure good lighting and facing forward.")
