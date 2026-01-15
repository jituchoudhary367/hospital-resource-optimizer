import streamlit as st
import pandas as pd
import mlflow.pyfunc

MODEL_NAME = "HospitalLOSModel"
MODEL_URI = f"models:/{MODEL_NAME}@production"

@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)

st.set_page_config(page_title="Hospital LOS Predictor", layout="centered")

st.title("🏥 Hospital Length of Stay Predictor")
st.write("Estimate how long a patient may stay in hospital based on admission details.")

model = load_model()

# ------------------ INPUTS ------------------

race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"])
gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])

age = st.selectbox("Age Range", [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
])

admission_type_id = st.selectbox("Admission Type ID", list(range(1, 9)))
discharge_disposition_id = st.selectbox("Discharge Disposition ID", list(range(1, 30)))
admission_source_id = st.selectbox("Admission Source ID", list(range(1, 26)))

num_lab_procedures = st.slider("Number of Lab Procedures", 0, 150, 40)
num_procedures = st.slider("Number of Procedures", 0, 10, 1)
num_medications = st.slider("Number of Medications", 0, 80, 10)

number_outpatient = st.slider("Outpatient Visits", 0, 20, 0)
number_emergency = st.slider("Emergency Visits", 0, 20, 0)
number_inpatient = st.slider("Inpatient Visits", 0, 20, 0)

number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

max_glu_serum = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
A1Cresult = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
insulin = st.selectbox("Insulin", ["No", "Up", "Down", "Steady"])
change = st.selectbox("Medication Change", ["No", "Ch"])
diabetesMed = st.selectbox("Diabetes Medication", ["Yes", "No"])

# ------------------ PREDICT ------------------

if st.button("Predict Length of Stay"):
    input_df = pd.DataFrame([{
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetesMed,
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"🕒 Estimated Length of Stay: **{prediction:.1f} days**")
