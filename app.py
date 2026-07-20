import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib
import time

# --- Page Config ---
st.set_page_config(page_title="Disease Prediction System", page_icon="🏥", layout="wide")

# --- Initialize Menu State Tracking ---
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = True
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Diabetes"

# --- Custom Global UI CSS Styling ---
# 1. Hides standard header decorations, clears entry prompts.
# 2. CRITICAL: Deletes Streamlit's native '<<' arrow button so frontend states don't break.
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Dynamic Sidebar Visibility Override ---
if not st.session_state.show_sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

# --- Persistent Navigation Control Button Layout ---
# Allocating a specific layout weight keeps your toggle text sitting cleanly on a single line
col_toggle, col_spacer = st.columns([1, 4])
with col_toggle:
    if st.session_state.show_sidebar:
        if st.button("⬅️ Hide Sidebar", use_container_width=True):
            st.session_state.show_sidebar = False
            st.rerun()
    else:
        if st.button("➡️ Show Sidebar", use_container_width=True):
            st.session_state.show_sidebar = True
            st.rerun()

# --- Load Models and Scalers ---
@st.cache_resource
def load_assets():
    diabetes_model = joblib.load('models/diabetes_model.sav')
    heart_model = joblib.load('models/heart_disease_model.sav')
    heart_scaler = joblib.load('models/heart_disease_scaler.sav')
    parkinsons_model = joblib.load('models/parkinsons_model.sav')
    parkinsons_scaler = joblib.load('models/parkinsons_scaler.sav')
    return diabetes_model, heart_model, heart_scaler, parkinsons_model, parkinsons_scaler

diabetes_model, heart_model, heart_scaler, parkinsons_model, parkinsons_scaler = load_assets()

# --- Sidebar Navigation Framework ---
if st.session_state.show_sidebar:
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🏥 Med-Predict</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        menu_options = ["Diabetes", "Heart Disease", "Parkinson's"]
        current_index = menu_options.index(st.session_state.selected_page)
        
        page_selection = option_menu(
            menu_title=None,
            options=menu_options,
            icons=["activity", "heart-pulse", "person-lines-fill"],
            default_index=current_index,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "gray", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#ff4b4b", "color": "white", "icon-color": "white"},
            }
        )
        st.session_state.selected_page = page_selection

page = st.session_state.selected_page

# ==========================================
# 1. DIABETES PREDICTION
# ==========================================
if page == "Diabetes":
    st.title("📈 Diabetes Risk Assessment")
    st.markdown("Enter patient metrics below to evaluate the risk of diabetes onset.")
    
    with st.expander("ℹ️ Click here for input instructions and normal ranges"):
        st.write("* **Glucose Level:** Normal fasting blood sugar is below 99 mg/dL.\n* **Blood Pressure:** Normal diastolic blood pressure is generally under 80 mmHg.\n* **BMI:** A normal Body Mass Index falls between 18.5 and 24.9.")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=300, step=1, value=120)
        bp = st.number_input("Blood Pressure (Diastolic)", min_value=20, max_value=140, step=1, value=70)
    with col2:
        skin = st.number_input("Skin Thickness (mm)", min_value=5, max_value=100, step=1, value=15)
        insulin = st.number_input("Insulin Level (IU/mL)", min_value=14, max_value=900, step=1, value=125)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, step=0.1, value=25.0)
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01, value=0.5)
        age = st.number_input("Age", min_value=21, max_value=120, step=1, value=33)

    st.markdown("---")
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        if st.button("Generate Diagnostic Report", use_container_width=True):
            with st.spinner('Analyzing metrics...'):
                time.sleep(1)
                user_input = np.array([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1, -1)
                prediction = diabetes_model.predict(user_input)
                
            if prediction[0] == 1:
                st.error("⚠️ **High Risk Detected:** The model indicates a high probability of Diabetes.")
            else:
                st.success("✅ **Low Risk Detected:** The patient's metrics suggest a low probability of Diabetes.")

# ==========================================
# 2. HEART DISEASE PREDICTION
# ==========================================
elif page == "Heart Disease":
    st.title("🫀 Heart Disease Risk Assessment")
    st.markdown("Please enter the patient's vitals below to evaluate the risk of cardiovascular disease.")
    
    with st.expander("ℹ️ Click here for input instructions and normal ranges"):
        st.write("* **Resting Blood Pressure:** Normal is typically around 120/80 mmHg.\n* **Serum Cholestoral:** Desirable levels are under 200 mg/dl.\n* **Maximum Heart Rate:** Varies by age, generally 220 minus the patient's age.")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="0:Asymptomatic, 1: Typical Angina, 2: Atypical Angina, 3: Non-anginal,")
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, step=1, value=120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, step=1, value=200)
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, step=1, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col3:
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", options=[1, 2, 3], help="1: Normal, 2: Fixed Defect, 3: Reversible Defect")

    st.markdown("---")
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        if st.button("Generate Diagnostic Report", use_container_width=True):
            with st.spinner('Analyzing vitals against cardiovascular models...'):
                time.sleep(1)
                user_input = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
                scaled_input = heart_scaler.transform(user_input)
                prediction = heart_model.predict(scaled_input)
                
            if prediction[0] == 1:
                st.error("⚠️ **High Risk Detected:** The model indicates a high probability of heart disease.")
            else:
                st.success("✅ **Less Risk:** The patient's vitals are currently within healthy parameters.")

# ==========================================
# 3. PARKINSON'S PREDICTION
# ==========================================
elif page == "Parkinson's":
    st.title("👤 Parkinson's Disease Assessment")
    st.markdown("Enter biomedical voice measurements below to evaluate the likelihood of Parkinson's Disease.")
    
    with st.expander("ℹ️ Click here for input instructions"):
        st.write("* These fields require high-precision acoustic measurements extracted from voice recordings.\n* Ensure data is entered exactly as formatted in the laboratory results.")
    st.markdown("---")
    
    feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
                     'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
                     'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
                     'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    
    inputs = []
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(feature, value=0.00000, format="%.5f")
            inputs.append(val)

    st.markdown("---")
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        if st.button("Generate Diagnostic Report", use_container_width=True):
            with st.spinner('Analyzing vocal biomarkers...'):
                time.sleep(1)
                user_input = np.array(inputs).reshape(1, -1)
                scaled_input = parkinsons_scaler.transform(user_input)
                prediction = parkinsons_model.predict(scaled_input)
                
            if prediction[0] == 1:
                st.error("⚠️ **High Risk Detected:** Acoustic analysis indicates a high probability of Parkinson's.")
            else:
                st.success("✅ **Low Risk Detected:** Acoustic analysis falls within normal parameters.")
