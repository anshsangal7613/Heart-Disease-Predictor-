import joblib
import pandas as pd
import streamlit as st

# Load artifacts
model = joblib.load('logistic_Regression_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title('Heart Disease Prediction')

age = st.slider('Age', 18, 100, 40)
sex = st.selectbox('SEX', ['MALE', 'FEMALE'])
chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
cholesterol = st.number_input('Cholesterol (mg/dL)', 100, 600, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', ['NO', 'YES'])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_hr = st.slider('Max Heart Rate', 60, 220, 150)
exercise_angina = st.selectbox('Exercise-Induced Angina', ['YES', 'NO'])
oldpeak = st.slider('Oldpeak (ST Depression)', 0.0, 6.0, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

if st.button('Predict'):
  raw_input = {
      'Age': age,
      'RestingBP': resting_bp,
      'Cholesterol': cholesterol,
      'FastingBS': fasting_bs,
      'MaxHR': max_hr,
      'Oldpeak': oldpeak,
      'Sex_M': 1 if sex == 'M' else 0,
      'ChestPainType_ATA': 1 if chest_pain == 'ATA' else 0,
      'ChestPainType_NAP': 1 if chest_pain == 'NAP' else 0,
      'ChestPainType_TA': 1 if chest_pain == 'TA' else 0,
      'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
      'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
      'ExerciseAngina_Y': 1 if exercise_angina == 'Y' else 0,
      'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
      'ST_Slope_Up': 1 if st_slope == 'Up' else 0,
  }

  input_df = pd.DataFrame([raw_input]).reindex(
      columns=expected_columns, fill_value=0
  )
  scaled_input = scaler.transform(input_df)

  prediction = model.predict(scaled_input)[0]

  if prediction == 1:
    st.error('⚠️ HIGH RISK OF HEART DISEASE!!')
  else:
    st.success('🟢 LOW RISK OF HEART DISEASE!! BUT STILL EAT HEALTHY😊')