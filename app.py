# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import joblib
# import pandas as pd
#
# # Load the trained model and scaler
# model = tf.keras.models.load_model("churn_model.h5")
# scaler = joblib.load("scaler.pkl")
#
# st.set_page_config(page_title="ChurnGuardAI - Customer Churn Predictor", layout="centered")
# st.title("ChurnGuardAI 💡")
# st.subheader("Predict if a customer will churn based on input features")
#
# # Input fields
# age = st.slider("Age", 18, 100, 30)
# tenure = st.slider("Tenure (in months)", 0, 72, 12)
# monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
# total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
# support_calls = st.slider("Support Calls", 0, 10, 2)
#
# gender = st.selectbox("Gender", ["Male", "Female"])
# has_internet_service = st.radio("Has Internet Service?", ["Yes", "No"])
# streaming_services = st.radio("Uses Streaming Services?", ["Yes", "No"])
#
# contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
# payment_method = st.selectbox("Payment Method", [
#     "Electronic check",
#     "Mailed check",
#     "Bank transfer (automatic)",
#     "Credit card (automatic)"
# ])
#
# # Map binary features
# gender_bin = 1 if gender == "Male" else 0
# internet_bin = 1 if has_internet_service == "Yes" else 0
# streaming_bin = 1 if streaming_services == "Yes" else 0
#
# # One-hot encoding manually (excluding one payment column to match training)
# contract_map = {
#     "Month-to-month": [0, 0],
#     "One year": [1, 0],
#     "Two year": [0, 1],
# }
# payment_map = {
#     "Electronic check": [1, 0],
#     "Mailed check": [0, 1],
#     # These will both be [0, 0] as the dropped base
#     "Bank transfer (automatic)": [0, 0],
#     "Credit card (automatic)": [0, 0],
# }
#
# contract_encoded = contract_map[contract_type]
# payment_encoded = payment_map[payment_method]
#
# # Combine all inputs (total 12 features now)
# input_data = [
#     age, tenure, monthly_charges, total_charges, support_calls,
#     gender_bin, internet_bin, streaming_bin
# ] + contract_encoded + payment_encoded
#
# # Convert to DataFrame and scale numeric features
# input_df = pd.DataFrame([input_data], columns=[
#     "age", "tenure_months", "monthly_charges", "total_charges", "support_calls",
#     "gender", "has_internet_service", "streaming_services",
#     "contract_type_One year", "contract_type_Two year",
#     "payment_method_Electronic check", "payment_method_Mailed check"
# ])
#
# numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
# input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
#
# # Predict
# if st.button("Predict Churn"):
#     pred_prob = model.predict(input_df)[0][0]
#     churned = pred_prob > 0.5
#     st.markdown("---")
#     st.metric(label="Churn Probability", value=f"{pred_prob:.2%}")
#     if churned:
#         st.error("⚠️ This customer is likely to churn.")
#     else:
#         st.success("✅ This customer is likely to stay.")
#
# st.markdown("""
# ---
# ### 🔧 How to Use
# 1. Place `churn_model.h5` and `scaler.pkl` in the same folder as this script.
# 2. Run the app with `streamlit run app.py` in terminal or PyCharm.
# 3. Enter customer details and click **Predict Churn**.
# """)

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import joblib
# import pandas as pd
#
# # Load model and scaler
# model = tf.keras.models.load_model("churn_model.h5")
# scaler = joblib.load("scaler.pkl")
#
# # Page settings
# st.set_page_config(
#     page_title="ChurnGuardAI - Enterprise Churn Predictor",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
#
# # --- Custom CSS for Pro Design ---
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
#
#     html, body, [class*="css"] {
#         font-family: 'Inter', sans-serif;
#         background-color: #f4f6f9;
#         color: #212529;
#     }
#
#     h1, h2, h3, h4 {
#         font-weight: 600;
#         text-align: center;
#     }
#
#     .stApp {
#         padding-top: 2rem;
#     }
#
#     .section {
#         background-color: #ffffff;
#         padding: 2rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 24px rgba(0, 0, 0, 0.03);
#         margin-bottom: 2rem;
#     }
#
#     .stButton>button {
#         background-color: #2d2f33;
#         color: white;
#         font-weight: 600;
#         border-radius: 8px;
#         padding: 0.75rem 1.5rem;
#         transition: all 0.3s ease;
#         border: none;
#     }
#
#     .stButton>button:hover {
#         background-color: #4b4e57;
#     }
#
#     .result-box {
#         text-align: center;
#         font-size: 1.2rem;
#         padding: 1rem;
#         background-color: #ffffff;
#         border-radius: 10px;
#         margin-top: 1.5rem;
#         box-shadow: 0 3px 10px rgba(0,0,0,0.05);
#     }
#
#     .footer {
#         text-align: center;
#         font-size: 0.9rem;
#         color: #6c757d;
#         margin-top: 3rem;
#     }
#     </style>
# """, unsafe_allow_html=True)
#
# # --- Header ---
# st.title("📊 ChurnGuardAI")
# st.subheader("Enterprise-Grade Customer Churn Predictor")
#
# st.markdown("""<p style='text-align: center; font-size: 1.1rem; color: #5c636a;'>
# Make data-backed decisions to retain valuable customers.
# </p>""", unsafe_allow_html=True)
#
# # --- Input Section ---
# with st.container():
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         age = st.slider("Customer Age", 18, 100, 30)
#         tenure = st.slider("Tenure (months)", 0, 72, 12)
#         support_calls = st.slider("Support Calls", 0, 10, 2)
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         has_internet_service = st.radio("Has Internet Service?", ["Yes", "No"])
#
#     with col2:
#         monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
#         total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
#         streaming_services = st.radio("Uses Streaming Services?", ["Yes", "No"])
#         contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
#         payment_method = st.selectbox("Payment Method", [
#             "Electronic check", "Mailed check",
#             "Bank transfer (automatic)", "Credit card (automatic)"
#         ])
#
#     st.markdown("</div>", unsafe_allow_html=True)
#
# # --- Feature Engineering ---
# gender_bin = 1 if gender == "Male" else 0
# internet_bin = 1 if has_internet_service == "Yes" else 0
# streaming_bin = 1 if streaming_services == "Yes" else 0
#
# contract_map = {
#     "Month-to-month": [0, 0],
#     "One year": [1, 0],
#     "Two year": [0, 1],
# }
# payment_map = {
#     "Electronic check": [1, 0],
#     "Mailed check": [0, 1],
#     "Bank transfer (automatic)": [0, 0],
#     "Credit card (automatic)": [0, 0],
# }
#
# contract_encoded = contract_map[contract_type]
# payment_encoded = payment_map[payment_method]
#
# input_data = [
#     age, tenure, monthly_charges, total_charges, support_calls,
#     gender_bin, internet_bin, streaming_bin
# ] + contract_encoded + payment_encoded
#
# input_df = pd.DataFrame([input_data], columns=[
#     "age", "tenure_months", "monthly_charges", "total_charges", "support_calls",
#     "gender", "has_internet_service", "streaming_services",
#     "contract_type_One year", "contract_type_Two year",
#     "payment_method_Electronic check", "payment_method_Mailed check"
# ])
#
# numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
# input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
#
# # --- Prediction ---
# if st.button("🔍 Predict Now"):
#     pred_prob = model.predict(input_df)[0][0]
#     churned = pred_prob > 0.5
#
#     st.markdown("<div class='result-box'>", unsafe_allow_html=True)
#     st.markdown(f"<h4>Churn Probability: {pred_prob:.2%}</h4>", unsafe_allow_html=True)
#
#     if churned:
#         st.error("🚨 Likely to Churn")
#     else:
#         st.success("✅ Likely to Stay")
#     st.markdown("</div>", unsafe_allow_html=True)
#
# # --- Footer ---
# st.markdown("""
# <div class="footer">
# 📁 Place <code>churn_model.h5</code> and <code>scaler.pkl</code> in the same directory.<br>
# ▶️ Launch with <code>streamlit run app.py</code><br><br>
# © 2025 ChurnGuardAI. All rights reserved.
# </div>
# """, unsafe_allow_html=True)

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import joblib
# import pandas as pd
#
# # Load model and scaler
# model = tf.keras.models.load_model("churn_model.h5")
# scaler = joblib.load("scaler.pkl")
#
# # Page configuration
# st.set_page_config(
#     page_title="ChurnGuardAI - Customer Churn Predictor",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
#
# # --- Custom CSS Styling ---
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
#
#     html, body, [class*="css"] {
#         font-family: 'Inter', sans-serif;
#         background-color: #f4f6f9;
#         color: #212529;
#     }
#
#     h1 {
#         font-weight: 800;
#         font-size: 3rem;
#         text-align: center;
#         color: red;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
#     }
#
#     .blue-subheader {
#         font-size: 1.4rem;
#         font-weight: 700;
#         color: #1a73e8;
#         text-align: center;
#     }
#
#     .section {
#         background-color: #ffffff;
#         padding: 2rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 24px rgba(0, 0, 0, 0.03);
#         margin-bottom: 2rem;
#     }
#
#     .stButton>button {
#         background-color: #2d2f33;
#         color: white;
#         font-weight: 600;
#         border-radius: 8px;
#         padding: 0.75rem 1.5rem;
#         transition: all 0.3s ease;
#         border: none;
#     }
#
#     .stButton>button:hover {
#         background-color: #4b4e57;
#     }
#
#     .result-box {
#         text-align: center;
#         font-size: 1.2rem;
#         padding: 1rem;
#         background-color: #ffffff;
#         border-radius: 10px;
#         margin-top: 1.5rem;
#         box-shadow: 0 3px 10px rgba(0,0,0,0.05);
#     }
#
#     .footer {
#         text-align: center;
#         font-size: 0.9rem;
#         color: #6c757d;
#         margin-top: 3rem;
#     }
#     </style>
# """, unsafe_allow_html=True)
#
# # --- Header ---
# st.markdown("<h1>ChurnGuardAI</h1>", unsafe_allow_html=True)
# st.markdown("<p class='blue-subheader'>Enterprise-Grade Customer Churn Predictor</p>", unsafe_allow_html=True)
#
# st.markdown("""<p style='text-align: center; font-size: 1.1rem; color: #5c636a;'>
# Make data-backed decisions to retain valuable customers.
# </p>""", unsafe_allow_html=True)
#
# # --- Input Section ---
# with st.container():
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         age = st.slider("Customer Age", 18, 100, 30)
#         tenure = st.slider("Tenure (months)", 0, 72, 12)
#         support_calls = st.slider("Support Calls", 0, 10, 2)
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         has_internet_service = st.radio("Has Internet Service?", ["Yes", "No"])
#
#     with col2:
#         monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
#         total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
#         streaming_services = st.radio("Uses Streaming Services?", ["Yes", "No"])
#         contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
#         payment_method = st.selectbox("Payment Method", [
#             "Electronic check", "Mailed check",
#             "Bank transfer (automatic)", "Credit card (automatic)"
#         ])
#
#     st.markdown("</div>", unsafe_allow_html=True)
#
# # --- Feature Engineering ---
# gender_bin = 1 if gender == "Male" else 0
# internet_bin = 1 if has_internet_service == "Yes" else 0
# streaming_bin = 1 if streaming_services == "Yes" else 0
#
# contract_map = {
#     "Month-to-month": [0, 0],
#     "One year": [1, 0],
#     "Two year": [0, 1],
# }
# payment_map = {
#     "Electronic check": [1, 0],
#     "Mailed check": [0, 1],
#     "Bank transfer (automatic)": [0, 0],
#     "Credit card (automatic)": [0, 0],
# }
#
# contract_encoded = contract_map[contract_type]
# payment_encoded = payment_map[payment_method]
#
# input_data = [
#     age, tenure, monthly_charges, total_charges, support_calls,
#     gender_bin, internet_bin, streaming_bin
# ] + contract_encoded + payment_encoded
#
# input_df = pd.DataFrame([input_data], columns=[
#     "age", "tenure_months", "monthly_charges", "total_charges", "support_calls",
#     "gender", "has_internet_service", "streaming_services",
#     "contract_type_One year", "contract_type_Two year",
#     "payment_method_Electronic check", "payment_method_Mailed check"
# ])
#
# numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
# input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
#
# # --- Prediction ---
# if st.button("Predict Now"):
#     pred_prob = model.predict(input_df)[0][0]
#     churned = pred_prob > 0.5
#
#     st.markdown("<div class='result-box'>", unsafe_allow_html=True)
#     st.markdown(f"<h4>Churn Probability: {pred_prob:.2%}</h4>", unsafe_allow_html=True)
#
#     if churned:
#         st.error("Likely to Churn")
#     else:
#         st.success("Likely to Stay")
#     st.markdown("</div>", unsafe_allow_html=True)
#
# # --- Footer ---
# st.markdown("""
# <div class="footer">
# Place <code>churn_model.h5</code> and <code>scaler.pkl</code> in the same directory.<br>
# Launch with <code>streamlit run app.py</code><br><br>
# © 2025 ChurnGuardAI. All rights reserved.
# </div>
# """, unsafe_allow_html=True)

#
# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import joblib
# import pandas as pd
#
# # Load model and scaler
# model = tf.keras.models.load_model("churn_model.h5")
# scaler = joblib.load("scaler.pkl")
#
# # Page configuration
# st.set_page_config(
#     page_title="ChurnGuardAI - Customer Churn Predictor",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
#
# # --- Custom CSS Styling with Clean White Background & Modern Header ---
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
#
#     html, body, [class*="css"] {
#         font-family: 'Poppins', sans-serif;
#         background-color: #ffffff;
#         color: #212529;
#         margin: 0;
#         padding: 0;
#     }
#
#     .title-container {
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         flex-direction: column;
#         margin-bottom: 30px;
#         padding: 30px 0;
#     }
#
#     .title {
#         font-size: 60px;
#         font-weight: 800;
#         text-align: center;
#         background: linear-gradient(90deg, #00c6ff, #0072ff);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-transform: uppercase;
#         letter-spacing: 3px;
#         margin-bottom: 10px;
#     }
#
#     .subheader {
#         font-size: 24px;
#         text-align: center;
#         color: #444444;
#         font-weight: 500;
#     }
#
#     .blue-label label {
#         color: #1a73e8 !important;
#         font-weight: 700 !important;
#     }
#
#     .section {
#         background-color: #f8f9fa;
#         padding: 2rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 24px rgba(0, 0, 0, 0.03);
#         margin-bottom: 2rem;
#     }
#
#     .stButton>button {
#         background-color: #2d2f33;
#         color: white;
#         font-weight: 600;
#         border-radius: 8px;
#         padding: 0.75rem 1.5rem;
#         transition: all 0.3s ease;
#         border: none;
#     }
#
#     .stButton>button:hover {
#         background-color: #4b4e57;
#     }
#
#     .result-box {
#         text-align: center;
#         font-size: 1.2rem;
#         padding: 1rem;
#         background-color: #ffffff;
#         border-radius: 10px;
#         margin-top: 1.5rem;
#         box-shadow: 0 3px 10px rgba(0,0,0,0.05);
#     }
#
#     .footer {
#         text-align: center;
#         font-size: 0.9rem;
#         color: #6c757d;
#         margin-top: 3rem;
#     }
#     </style>
# """, unsafe_allow_html=True)
#
# # --- Header Section ---
# st.markdown("""
# <div class='title-container'>
#     <h1 class='title'>ChurnGuardAI</h1>
#     <h3 class='subheader'>Predict Customer Churn with Ease</h3>
# </div>
# """, unsafe_allow_html=True)
#
# # --- Input Section ---
# with st.container():
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         age = st.slider("Customer Age", 18, 100, 30)
#         tenure = st.slider("Tenure (months)", 0, 72, 12)
#         support_calls = st.slider("Support Calls", 0, 10, 2)
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         has_internet_service = st.radio("Has Internet Service?", ["Yes", "No"])
#
#     with col2:
#         monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
#         total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
#         streaming_services = st.radio("Uses Streaming Services?", ["Yes", "No"])
#         contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
#         payment_method = st.selectbox("Payment Method", [
#             "Electronic check", "Mailed check",
#             "Bank transfer (automatic)", "Credit card (automatic)"
#         ])
#
#     st.markdown("</div>", unsafe_allow_html=True)
#
# # --- Feature Engineering ---
# gender_bin = 1 if gender == "Male" else 0
# internet_bin = 1 if has_internet_service == "Yes" else 0
# streaming_bin = 1 if streaming_services == "Yes" else 0
#
# contract_map = {
#     "Month-to-month": [0, 0],
#     "One year": [1, 0],
#     "Two year": [0, 1],
# }
# payment_map = {
#     "Electronic check": [1, 0],
#     "Mailed check": [0, 1],
#     "Bank transfer (automatic)": [0, 0],
#     "Credit card (automatic)": [0, 0],
# }
#
# contract_encoded = contract_map[contract_type]
# payment_encoded = payment_map[payment_method]
#
# input_data = [
#     age, tenure, monthly_charges, total_charges, support_calls,
#     gender_bin, internet_bin, streaming_bin
# ] + contract_encoded + payment_encoded
#
# input_df = pd.DataFrame([input_data], columns=[
#     "age", "tenure_months", "monthly_charges", "total_charges", "support_calls",
#     "gender", "has_internet_service", "streaming_services",
#     "contract_type_One year", "contract_type_Two year",
#     "payment_method_Electronic check", "payment_method_Mailed check"
# ])
#
# numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
# input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
#
# # --- Prediction ---
# if st.button("Predict Now"):
#     pred_prob = model.predict(input_df)[0][0]
#     churned = pred_prob > 0.5
#
#     st.markdown("<div class='result-box'>", unsafe_allow_html=True)
#     st.markdown(f"<h4>Churn Probability: {pred_prob:.2%}</h4>", unsafe_allow_html=True)
#
#     if churned:
#         st.error("Likely to Churn")
#     else:
#         st.success("Likely to Stay")
#     st.markdown("</div>", unsafe_allow_html=True)

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import joblib
# import pandas as pd
#
# # Load model and scaler
# model = tf.keras.models.load_model("churn_model.h5")
# scaler = joblib.load("scaler.pkl")
#
# # Page configuration
# st.set_page_config(
#     page_title="ChurnGuardAI - Customer Churn Predictor",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
#
# # --- Custom CSS Styling with Clean White Background & Modern Header ---
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
#
#     html, body, [class*="css"] {
#         font-family: 'Poppins', sans-serif;
#         background-color: #ffffff;
#         color: #212529;
#         margin: 0;
#         padding: 0;
#     }
#
#     .title-container {
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         flex-direction: column;
#         margin-bottom: 30px;
#         padding: 30px 0;
#     }
#
#     .title {
#         font-size: 60px;
#         font-weight: 800;
#         text-align: center;
#         background: linear-gradient(90deg, #00c6ff, #0072ff);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-transform: uppercase;
#         letter-spacing: 3px;
#         margin-bottom: 10px;
#     }
#
#     .subheader {
#         font-size: 24px;
#         text-align: center;
#         color: #444444;
#         font-weight: 500;
#     }
#
#     .blue-label label {
#         color: #1a73e8 !important;
#         font-weight: 700 !important;
#     }
#
#     .section {
#         background-color: #f8f9fa;
#         padding: 2rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 24px rgba(0, 0, 0, 0.03);
#         margin-bottom: 2rem;
#     }
#
#     .stButton>button {
#         background-color: #2d2f33;
#         color: white;
#         font-weight: 600;
#         border-radius: 8px;
#         padding: 0.75rem 1.5rem;
#         transition: all 0.3s ease;
#         border: none;
#     }
#
#     .stButton>button:hover {
#         background-color: #4b4e57;
#     }
#
#     .result-box {
#         text-align: center;
#         font-size: 1.2rem;
#         padding: 1rem;
#         background-color: #ffffff;
#         border-radius: 10px;
#         margin-top: 1.5rem;
#         box-shadow: 0 3px 10px rgba(0,0,0,0.05);
#     }
#     </style>
# """, unsafe_allow_html=True)
#
# # --- Header Section ---
# st.markdown("""
# <div class='title-container'>
#     <h1 class='title'>ChurnGuardAI</h1>
#     <h3 class='subheader'>Predict Customer Churn with Ease</h3>
# </div>
# """, unsafe_allow_html=True)
#
# # --- Input Section ---
# with st.container():
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         age = st.slider("Customer Age", 18, 100, 30)
#         tenure = st.slider("Tenure (months)", 0, 72, 12)
#         support_calls = st.slider("Support Calls", 0, 10, 2)
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         has_internet_service = st.radio("Has Internet Service?", ["Yes", "No"])
#
#     with col2:
#         monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
#         total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
#         streaming_services = st.radio("Uses Streaming Services?", ["Yes", "No"])
#         contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
#         payment_method = st.selectbox("Payment Method", [
#             "Electronic check", "Mailed check",
#             "Bank transfer (automatic)", "Credit card (automatic)"
#         ])
#
#     st.markdown("</div>", unsafe_allow_html=True)
#
# # --- Feature Engineering ---
# gender_bin = 1 if gender == "Male" else 0
# internet_bin = 1 if has_internet_service == "Yes" else 0
# streaming_bin = 1 if streaming_services == "Yes" else 0
#
# contract_map = {
#     "Month-to-month": [0, 0],
#     "One year": [1, 0],
#     "Two year": [0, 1],
# }
# payment_map = {
#     "Electronic check": [1, 0],
#     "Mailed check": [0, 1],
#     "Bank transfer (automatic)": [0, 0],
#     "Credit card (automatic)": [0, 0],
# }
#
# contract_encoded = contract_map[contract_type]
# payment_encoded = payment_map[payment_method]
#
# input_data = [
#     age, tenure, monthly_charges, total_charges, support_calls,
#     gender_bin, internet_bin, streaming_bin
# ] + contract_encoded + payment_encoded
#
# input_df = pd.DataFrame([input_data], columns=[
#     "age", "tenure_months", "monthly_charges", "total_charges", "support_calls",
#     "gender", "has_internet_service", "streaming_services",
#     "contract_type_One year", "contract_type_Two year",
#     "payment_method_Electronic check", "payment_method_Mailed check"
# ])
#
# numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
# input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
#
# # --- Prediction ---
# if st.button("Predict Now"):
#     pred_prob = model.predict(input_df)[0][0]
#     churned = pred_prob > 0.5
#
#     st.markdown("<div class='result-box'>", unsafe_allow_html=True)
#     st.markdown(f"<h4>Churn Probability: {pred_prob:.2%}</h4>", unsafe_allow_html=True)
#
#     if churned:
#         st.markdown(
#             "<span style='color: red; font-weight: bold; font-size: 1.2rem;'>🚨 Likely to Churn</span>",
#             unsafe_allow_html=True
#         )
#     else:
#         st.markdown(
#             "<span style='color: green; font-weight: bold; font-size: 1.2rem;'>✅ Likely to Stay</span>",
#             unsafe_allow_html=True
#         )
#     st.markdown("</div>", unsafe_allow_html=True)

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import joblib
# import pandas as pd
#
# # Load model and scaler
# model = tf.keras.models.load_model("churn_model.h5")
# scaler = joblib.load("scaler.pkl")
#
# # Page configuration
# st.set_page_config(
#     page_title="ChurnGuardAI - Customer Churn Predictor",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
#
# # --- Custom CSS Styling with Clean White Background & Modern Header ---
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
#
#     html, body, [class*="css"] {
#         font-family: 'Poppins', sans-serif;
#         background-color: #ffffff;
#         color: #212529;
#         margin: 0;
#         padding: 0;
#     }
#
#     .title-container {
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         flex-direction: column;
#         margin-bottom: 30px;
#         padding: 30px 0;
#     }
#
#     .title {
#         font-size: 60px;
#         font-weight: 800;
#         text-align: center;
#         background: linear-gradient(90deg, #00c6ff, #0072ff);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-transform: uppercase;
#         letter-spacing: 3px;
#         margin-bottom: 10px;
#     }
#
#     .subheader {
#         font-size: 24px;
#         text-align: center;
#         color: #444444;
#         font-weight: 500;
#     }
#
#     .blue-label label {
#         color: #1a73e8 !important;
#         font-weight: 700 !important;
#     }
#
#     .section {
#         background-color: #f8f9fa;
#         padding: 2rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 24px rgba(0, 0, 0, 0.03);
#         margin-bottom: 2rem;
#     }
#
#     .stButton>button {
#         background-color: #2d2f33;
#         color: white;
#         font-weight: 600;
#         border-radius: 8px;
#         padding: 0.75rem 1.5rem;
#         transition: all 0.3s ease;
#         border: none;
#     }
#
#     .stButton>button:hover {
#         background-color: #4b4e57;
#     }
#
#     .result-box {
#         text-align: center;
#         font-size: 1.2rem;
#         margin-top: 1.5rem;
#     }
#     </style>
# """, unsafe_allow_html=True)
#
# # --- Header Section ---
# st.markdown("""
# <div class='title-container'>
#     <h1 class='title'>ChurnGuardAI</h1>
#     <h3 class='subheader'>Predict Customer Churn with Ease</h3>
# </div>
# """, unsafe_allow_html=True)
#
# # --- Input Section ---
# with st.container():
#     st.markdown("<div class='section'>", unsafe_allow_html=True)
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         age = st.slider("Customer Age", 18, 100, 30)
#         tenure = st.slider("Tenure (months)", 0, 72, 12)
#         support_calls = st.slider("Support Calls", 0, 10, 2)
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         has_internet_service = st.radio("Has Internet Service?", ["Yes", "No"])
#
#     with col2:
#         monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
#         total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
#         streaming_services = st.radio("Uses Streaming Services?", ["Yes", "No"])
#         contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
#         payment_method = st.selectbox("Payment Method", [
#             "Electronic check", "Mailed check",
#             "Bank transfer (automatic)", "Credit card (automatic)"
#         ])
#
#     st.markdown("</div>", unsafe_allow_html=True)
#
# # --- Feature Engineering ---
# gender_bin = 1 if gender == "Male" else 0
# internet_bin = 1 if has_internet_service == "Yes" else 0
# streaming_bin = 1 if streaming_services == "Yes" else 0
#
# contract_map = {
#     "Month-to-month": [0, 0],
#     "One year": [1, 0],
#     "Two year": [0, 1],
# }
# payment_map = {
#     "Electronic check": [1, 0],
#     "Mailed check": [0, 1],
#     "Bank transfer (automatic)": [0, 0],
#     "Credit card (automatic)": [0, 0],
# }
#
# contract_encoded = contract_map[contract_type]
# payment_encoded = payment_map[payment_method]
#
# input_data = [
#     age, tenure, monthly_charges, total_charges, support_calls,
#     gender_bin, internet_bin, streaming_bin
# ] + contract_encoded + payment_encoded
#
# input_df = pd.DataFrame([input_data], columns=[
#     "age", "tenure_months", "monthly_charges", "total_charges", "support_calls",
#     "gender", "has_internet_service", "streaming_services",
#     "contract_type_One year", "contract_type_Two year",
#     "payment_method_Electronic check", "payment_method_Mailed check"
# ])
#
# numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
# input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
#
# # --- Prediction ---
# if st.button("Predict Now"):
#     pred_prob = model.predict(input_df)[0][0]
#     churned = pred_prob > 0.5
#
#     st.markdown("<div class='result-box'>", unsafe_allow_html=True)
#     st.markdown(f"<h4>Churn Probability: {pred_prob:.2%}</h4>", unsafe_allow_html=True)
#
#     if churned:
#         st.markdown(
#             "<span style='color: red; font-weight: bold; font-size: 1.3rem;'>🚨 Likely to Churn</span>",
#             unsafe_allow_html=True
#         )
#     else:
#         st.markdown(
#             "<span style='color: green; font-weight: bold; font-size: 1.3rem;'>✅ Likely to Stay</span>",
#             unsafe_allow_html=True
#         )
#     st.markdown("</div>", unsafe_allow_html=True)
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# Load model and scaler
model = tf.keras.models.load_model("churn_model.h5")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="ChurnguardAi - Customer Churn Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Styling with Clean White Background & Modern Header ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #ffffff;
        color: #212529;
        margin: 0;
        padding: 0;
    }

    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        margin-bottom: 30px;
        padding: 30px 0;
    }

    .title {
        font-size: 60px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 3px;
        margin-bottom: 10px;
    }

    .subheader {
        font-size: 24px;
        text-align: center;
        color: #444444;
        font-weight: 500;
    }

    .blue-label label {
        color: #1a73e8 !important;
        font-weight: 700 !important;
    }

    .section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.03);
        margin-bottom: 2rem;
    }

    .stButton>button {
        background-color: #2d2f33;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton>button:hover {
        background-color: #4b4e57;
    }

    .result-box {
        text-align: center;
        font-size: 1.2rem;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
<div class='title-container'>
    <h1 class='title'>ChurnguardAi</h1>
    <h3 class='subheader'>Predict Customer Churn with Ease</h3>
</div>
""", unsafe_allow_html=True)

# --- Input Section ---
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Customer Age", 18, 100, 30)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        support_calls = st.slider("Support Calls", 0, 10, 2)
        gender = st.selectbox("Gender", ["Male", "Female"])
        has_internet_service = st.radio("Has Internet Service?", ["Yes", "No"])

    with col2:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
        streaming_services = st.radio("Uses Streaming Services?", ["Yes", "No"])
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    st.markdown("</div>", unsafe_allow_html=True)

# --- Feature Engineering ---
gender_bin = 1 if gender == "Male" else 0
internet_bin = 1 if has_internet_service == "Yes" else 0
streaming_bin = 1 if streaming_services == "Yes" else 0

contract_map = {
    "Month-to-month": [0, 0],
    "One year": [1, 0],
    "Two year": [0, 1],
}
payment_map = {
    "Electronic check": [1, 0],
    "Mailed check": [0, 1],
    "Bank transfer (automatic)": [0, 0],
    "Credit card (automatic)": [0, 0],
}

contract_encoded = contract_map[contract_type]
payment_encoded = payment_map[payment_method]

input_data = [
    age, tenure, monthly_charges, total_charges, support_calls,
    gender_bin, internet_bin, streaming_bin
] + contract_encoded + payment_encoded

input_df = pd.DataFrame([input_data], columns=[
    "age", "tenure_months", "monthly_charges", "total_charges", "support_calls",
    "gender", "has_internet_service", "streaming_services",
    "contract_type_One year", "contract_type_Two year",
    "payment_method_Electronic check", "payment_method_Mailed check"
])

numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges", "support_calls"]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# --- Prediction ---
if st.button("Predict Now"):
    pred_prob = model.predict(input_df)[0][0]
    churned = pred_prob > 0.5

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(f"<h4>Churn Probability: {pred_prob:.2%}</h4>", unsafe_allow_html=True)

    if churned:
        st.markdown(
            "<span style='color: red; font-weight: bold; font-size: 1.3rem;'>🚨 Likely to Churn</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span style='color: green; font-weight: bold; font-size: 1.3rem;'>✅ Likely to Stay</span>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
