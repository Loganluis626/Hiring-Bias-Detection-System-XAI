import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime import lime_tabular

# Import the Free Google Gemini API Library
import google.generativeai as genai

# Set up clean corporate configurations
st.set_page_config(
    page_title="Enterprise AI Recruitment Governance Portal", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize navigation state if it doesn't exist yet
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Function to switch pages
def navigate_to(page_name):
    st.session_state.page = page_name

# ==========================================
# 1. ENTERPRISE LANDING PAGE
# ==========================================
if st.session_state.page == 'landing':
    # Professional dark header banner
    st.markdown("""
        <div style="background-color:#0F172A; padding:40px 30px; border-radius:8px; margin-bottom:30px; text-align:left; border-left: 5px solid #3B82F6;">
            <h1 style="color:#F8FAFC; margin:0; font-family:sans-serif; font-size:2.2rem; font-weight:600;">Automated Recruitment Governance System</h1>
            <p style="color:#94A3B8; font-size:1.1rem; margin-top:8px; font-family:sans-serif;">
                Auditing automated screening pipelines for algorithmic fairness, proxy biases, and structural variance.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Operational Framework Pillars")
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        with st.container(border=True):
            st.markdown("#### **Local Interpretability**")
            st.write("Traditional resume evaluation pipelines treat matching weights as a black box. This framework applies linear perturbations via LIME to isolate localized decisions.")
    
    with col2:
        with st.container(border=True):
            st.markdown("#### **Proxy Variable Isolation**")
            st.write("Historical organizational skews inherently propagate systemic disparities. The audit environment flags criteria that act as indirect demographic filters.")
    
    with col3:
        with st.container(border=True):
            st.markdown("#### **Risk Stress-Testing**")
            st.write("Compliance authorities can evaluate synthetic and boundary-case applicant profiles in real-time to assess operational consistency.")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Clean, wide action button
    if st.button("Initialize Compliance Dashboard System", type="primary", use_container_width=True):
        navigate_to('auditor')

# ==========================================
# 2. MAIN CORE AUDITOR INTERFACE
# ==========================================
elif st.session_state.page == 'auditor':
    # Header navigation row
    nav_col, title_col = st.columns([1, 4])
    with nav_col:
        st.button("Back to Portal", on_click=lambda: navigate_to('landing'), use_container_width=True)
    
    st.markdown("## Algorithmic Review: Candidate Risk Auditor")
    st.markdown("Evaluate predictive outcomes, isolate local feature attribution weights, and review compliance logs.")
    st.markdown("---")

    @st.cache_resource
    def load_assets():
        model = joblib.load("hiring_rf_model.pkl")
        X_train = pd.read_csv("X_train.csv")
        return model, X_train

    try:
        model, X_train = load_assets()
    except Exception as e:
        st.error("Missing Assets: Ensure 'hiring_rf_model.pkl' and 'X_train.csv' reside in the execution root directory.")
        st.stop()

    # Sidebar layout for candidate features - cleaned up formatting
    st.sidebar.markdown("### Evaluation Attributes")
    
    gender_labels = {0: "Man", 1: "Woman", 2: "Non-Binary"}
    gender_input = st.sidebar.selectbox("Demographic Identity", options=list(gender_labels.keys()), format_func=lambda x: gender_labels[x])
    
    age_labels = {0: "<35 Years Old", 1: ">35 Years Old"}
    age_input = st.sidebar.selectbox("Age Classification", options=list(age_labels.keys()), format_func=lambda x: age_labels[x])
    
    ed_labels = {0: "No Higher Education", 1: "Other", 2: "Undergraduate Degree", 3: "Master's Degree", 4: "PhD"}
    ed_input = st.sidebar.selectbox("Educational Milestones", options=list(ed_labels.keys()), format_func=lambda x: ed_labels[x])
    
    branch_labels = {1: "Technical Developer (Dev)", 0: "Non-Technical (NotDev)"}
    branch_input = st.sidebar.selectbox("Core Track Placement", options=list(branch_labels.keys()), format_func=lambda x: branch_labels[x])

    st.sidebar.markdown("---")
    comp_skills = st.sidebar.slider("Verified Computer Skills Sum", min_value=0, max_value=50, value=12)
    years_code = st.sidebar.slider("Gross Programming Experience (Years)", min_value=0, max_value=60, value=8)
    years_pro = st.sidebar.slider("Enterprise Development Tenure (Years)", min_value=0, max_value=55, value=4)
    mental_input = st.sidebar.selectbox("Mental Health Status Registry", options=[0, 1], format_func=lambda x: "Disclosed" if x==1 else "None Logged")
    access_input = st.sidebar.selectbox("Accommodation Requirements", options=[0, 1], format_func=lambda x: "Requested" if x==1 else "None Logged")

    # Structure user data input
    input_data = pd.DataFrame([{
        'Age': age_input, 'EdLevel': ed_input, 'Gender': gender_input,
        'YearsCode': years_code, 'YearsCodePro': years_pro, 'ComputerSkills': comp_skills,
        'Accessibility': access_input, 'MentalHealth': mental_input, 'MainBranch': branch_input
    }])[X_train.columns]

    prob = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]

    # Content Layout Splits
    col_metrics, col_viz = st.columns([1, 2], gap="large")

    with col_metrics:
        st.markdown("### Classification Output")
        
        # Wrapped metric inside a clean structured card
        with st.container(border=True):
            if prediction == 1:
                st.markdown("<span style='color:#10B981; font-weight:bold; font-size:1.1rem;'>SYSTEM DETERMINATION: APPROVE</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#EF4444; font-weight:bold; font-size:1.1rem;'>SYSTEM DETERMINATION: CLASSIFY REJECT</span>", unsafe_allow_html=True)
                
            st.metric(label="Calculated Hiring Probability", value=f"{prob[1]*100:.1f}%")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Generative AI Compliance Summary section moved to left column for better horizontal design balance
        st.markdown("### Executive Compliance Overview")
        with st.container(border=True):
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                api_key = "NOT_FOUND"
            
            if api_key == "NOT_FOUND" or api_key == "PASTE_YOUR_GEMINI_KEY_HERE":
                st.info("System Notification: Add your GEMINI_API_KEY credentials to the local secrets index to populate automated logs.")
            else:
                # Compute LIME arrays prior to generation
                lime_explainer = lime_tabular.LimeTabularExplainer(
                    training_data=np.array(X_train),
                    feature_names=X_train.columns,
                    class_names=['Rejected', 'Hired'],
                    mode='classification',
                    random_state=42
                )
                exp = lime_explainer.explain_instance(
                    data_row=input_data.iloc[0].values,
                    predict_fn=model.predict_proba,
                    num_features=5
                )
                lime_features_text = ", ".join([str(item[0]) for item in exp.as_list()])

                with st.spinner("Processing local weights..."):
                    try:
                        genai.configure(api_key=api_key)
                        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
                        
                        prompt = f"""
                        You are an expert AI Fairness and Governance auditor evaluating an automated HR screening system.
                        The model gave a recommendation outcome of {'HIRE' if prediction == 1 else 'REJECT'} with a hiring probability of {prob[1]*100:.1f}%.
                        The top 5 feature weights extracted from the local LIME graph are: {lime_features_text}.
                        
                        Provide a concise, 3-sentence executive summary for an HR manager:
                        1. Explain exactly why the model made this choice based on the LIME metrics.
                        2. Explicitly comment on whether this choice displays potential 'proxy bias' (e.g., if experience metrics or demographic factors are unfairly penalizing the candidate).
                        3. Give a clear recommendation on whether a human manager should trust or override this decision.
                        """
                        
                        response = gemini_model.generate_content(prompt)
                        st.markdown(f"<div style='font-size:0.95rem; line-height:1.5; color:#334155;'>{response.text}</div>", unsafe_allow_html=True)
                        
                    except Exception as api_error:
                        st.error(f"Inference Log Fault: {api_error}")

    with col_viz:
        st.markdown("### Local Attribution Diagnostics (LIME)")
        
        with st.container(border=True):
            # Regenerate plot cleanly if not processed in the left column
            if 'lime_explainer' not in locals():
                lime_explainer = lime_tabular.LimeTabularExplainer(
                    training_data=np.array(X_train),
                    feature_names=X_train.columns,
                    class_names=['Rejected', 'Hired'],
                    mode='classification',
                    random_state=42
                )
                exp = lime_explainer.explain_instance(
                    data_row=input_data.iloc[0].values,
                    predict_fn=model.predict_proba,
                    num_features=5
                )
            
            # Cleanly style the matplotlib figure canvas to fit a corporate dashboard aesthetic
            fig = exp.as_pyplot_figure()
            plt.subplots_adjust(left=0.35, right=0.95, top=0.85, bottom=0.15)
            fig.patch.set_facecolor('#F8FAFC') # Soft grey border canvas match
            ax = plt.gca()
            ax.set_facecolor('#F8FAFC')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.title("Localized Feature Impact Metrics\n(Positive Shifting Variance vs. Negative Margin Penalties)", fontsize=9, fontweight='bold', pad=15)
            
            st.pyplot(fig)