import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime import lime_tabular
import google.generativeai as genai

# Page setup
st.set_page_config(
    page_title="Recruitment Bias Auditor", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# 1. LOAD MODEL AND TRAINING DATA CODES

@st.cache_resource
def load_saved_assets():
    model = joblib.load("hiring_rf_model.pkl")
    X_train = pd.read_csv("X_train.csv")
    return model, X_train

try:
    hiring_model, training_data = load_saved_assets()
except Exception:
    st.error("Error: Could not load 'hiring_rf_model.pkl' or 'X_train.csv'. Please check your folder.")
    st.stop()



# 2. SIDEBAR INPUT FORM (RATE-LIMIT PROTECTION)

st.sidebar.header("Candidate Evaluation Profiles")

# Using a form prevents Gemini from being spammed with requests while moving sliders
with st.sidebar.form(key="candidate_form"):
    
    # Categorical dropdown selections
    gender_map = {0: "Man", 1: "Woman", 2: "Non-Binary"}
    gender = st.selectbox("Gender/Demographic Identity", options=[0, 1, 2], format_func=lambda x: gender_map[x])
    
    age_map = {0: "<35 Years Old", 1: ">35 Years Old"}
    age = st.selectbox("Age Bracket", options=[0, 1], format_func=lambda x: age_map[x])
    
    education_map = {0: "No Higher Education", 1: "Other", 2: "Undergraduate Degree", 3: "Master's Degree", 4: "PhD"}
    education = st.selectbox("Education Level", options=[0, 1, 2, 3, 4], format_func=lambda x: education_map[x])
    
    track_map = {1: "Technical Developer", 0: "Non-Technical"}
    main_branch = st.selectbox("Professional Track", options=[1, 0], format_func=lambda x: track_map[x])
    
    st.markdown("---")
    
    # Numerical slider values
    computer_skills = st.slider("Verified Computer Skills Sum", min_value=0, max_value=50, value=12)
    years_code = st.slider("Total Coding Experience (Years)", min_value=0, max_value=60, value=8)
    years_pro = st.slider("Professional Coding Experience (Years)", min_value=0, max_value=55, value=4)
    
    # Binary inclusion flags
    mental_health = st.selectbox("Mental Health Disclosure", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    accessibility = st.selectbox("Accessibility Accommodation Request", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Submission button to trigger the calculations
    submit_audit = st.form_submit_button(label="Execute Compliance Audit", type="primary", use_container_width=True)



# 3. DATA PROCESSING & MODEL PREDICTIONS

# Reconstruct input features into a structured DataFrame matching the training data schema
applicant_profile = pd.DataFrame([{
    'Age': age, 
    'EdLevel': education, 
    'Gender': gender,
    'YearsCode': years_code, 
    'YearsCodePro': years_pro, 
    'ComputerSkills': computer_skills,
    'Accessibility': accessibility, 
    'MentalHealth': mental_health, 
    'MainBranch': main_branch
}])[training_data.columns]

# Generate raw target predictions and matching probability distributions
prediction_probabilities = hiring_model.predict_proba(applicant_profile)[0]
final_prediction = hiring_model.predict(applicant_profile)[0]



# 4. DASHBOARD INTERFACE LAYOUT

st.title("Responsible AI Recruitment: Hiring Bias Auditor")
st.markdown("Evaluate predictive outcomes, isolate local feature attribution weights, and review compliance logs.")
st.markdown("---")

# Split screen layout into two primary columns
left_column, right_column = st.columns([1, 2], gap="large")

# --- LEFT COLUMN: METRICS & GENERATIVE AI ---
with left_column:
    st.subheader("Model Decision Output")
    
    with st.container(border=True):
        if final_prediction == 1:
            st.success("SYSTEM DETERMINATION: **HIRE**")
        else:
            st.error("SYSTEM DETERMINATION: **REJECT**")
            
        st.metric(label="Calculated Hiring Probability", value=f"{prediction_probabilities[1]*100:.1f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Executive Governance Log")
    with st.container(border=True):
        if not submit_audit:
            st.info("ℹ️ Tweak candidate attributes in the sidebar panel and click 'Execute Compliance Audit' to generate natural language analysis.")
        else:
            # Safely fetch key configuration parameters
            api_key = st.secrets.get("GEMINI_API_KEY", "NOT_FOUND")
            
            if api_key == "NOT_FOUND" or api_key == "PASTE_YOUR_GEMINI_KEY_HERE":
                st.warning("Configuration Notice: Please insert an active GEMINI_API_KEY into your secrets index to load text logging details.")
            else:
                # 1. Compute LIME explanations ahead of time
                explainer = lime_tabular.LimeTabularExplainer(
                    training_data=np.array(training_data),
                    feature_names=training_data.columns,
                    class_names=['Rejected', 'Hired'],
                    mode='classification',
                    random_state=42
                )
                local_explanation = explainer.explain_instance(
                    data_row=applicant_profile.iloc[0].values,
                    predict_fn=hiring_model.predict_proba,
                    num_features=5
                )
                features_summary_text = ", ".join([str(item[0]) for item in local_explanation.as_list()])

                # 2. Feed parameters to Gemini model endpoint
                with st.spinner("Analyzing decision pathways..."):
                    try:
                        genai.configure(api_key=api_key)
                        ai_model = genai.GenerativeModel("models/gemini-2.5-flash")
                        
                        prompt_template = f"""
                        You are an expert AI Fairness and Governance auditor evaluating an automated HR screening system.
                        The model gave a recommendation outcome of {'HIRE' if final_prediction == 1 else 'REJECT'} with a hiring probability of {prediction_probabilities[1]*100:.1f}%.
                        The top 5 feature weights extracted from the local LIME graph are: {features_summary_text}.
                        
                        Provide a concise, 3-sentence executive summary for an HR manager:
                        1. Explain exactly why the model made this choice based on the LIME metrics.
                        2. Explicitly comment on whether this choice displays potential 'proxy bias' (e.g., if experience metrics or demographic factors are unfairly penalizing the candidate).
                        3. Give a clear recommendation on whether a human manager should trust or override this decision.
                        """
                        
                        ai_response = ai_model.generate_content(prompt_template)
                        st.write(ai_response.text)
                        
                    except Exception as error_log:
                        st.error(f"Inference Log Fault: {error_log}")

# --- RIGHT COLUMN: LIME GRAPH DICTIONARY VISUALIZATION ---
with right_column:
    st.subheader("Local Attribution Diagnostics (LIME)")
    
    with st.container(border=True):
        # Build explainer logic locally if not already handled inside the generative AI loop
        if 'local_explanation' not in locals():
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.array(training_data),
                feature_names=training_data.columns,
                class_names=['Rejected', 'Hired'],
                mode='classification',
                random_state=42
            )
            local_explanation = explainer.explain_instance(
                data_row=applicant_profile.iloc[0].values,
                predict_fn=hiring_model.predict_proba,
                num_features=5
            )
        
        # Draw and display the clean local weights graph canvas
        matplotlib_figure = local_explanation.as_pyplot_figure()
        plt.subplots_adjust(left=0.35, right=0.95, top=0.85, bottom=0.15)
        
        # Clean background canvas properties to look integrated
        matplotlib_figure.patch.set_facecolor('#F8FAFC')
        plot_axes = plt.gca()
        plot_axes.set_facecolor('#F8FAFC')
        plot_axes.spines['top'].set_visible(False)
        plot_axes.spines['right'].set_visible(False)
        plt.title("Localized Feature Impact Metrics\n(Positive Shifting Variance vs. Negative Margin Penalties)", fontsize=9, fontweight='bold', pad=15)
        
        st.pyplot(matplotlib_figure)