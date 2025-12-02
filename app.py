import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd
import numpy as np

#3 loading the tained model

model = tf.keras.models.load_model('model.h5')

## loading pickle file
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')
## user input

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌏 Geography',onehot_encoder_geo.categories_[0],index=0)
    age = st.slider('🎂 Age',18,92)
    tenure = st.slider('📅 Tenure',0,10)

with col2:
    gender = st.selectbox('🧑‍🦰 Gender',label_encoder_gender.classes_, index=0)
    num_of_products = st.slider('🛍️ Number of Products ', 1,4)
    is_active_member = 1 if st.selectbox('✅ Is active member', ['No', 'Yes']) == 'Yes' else 0

with st.expander("💰 Financial Information ", expanded=True):
    credit_score = st.number_input("📊 Credit Score",min_value=300, max_value=900,step=1, value=None)
    balance = st.number_input('🏦 Balance',value=None)
    estimated_salary = st.number_input('💵 Estimated Salary',value=None)
    has_cr_card= st.selectbox('💳 Has Credit Card',[0,1])

if credit_score is not None and balance is not None and estimated_salary is not None:
    # prepare input data
    input_data = pd.DataFrame({
        'CreditScore':[credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure' : [tenure],
        'Balance':[balance],
        'NumOfProducts' : [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    ## one hot encode 'geography
    geo_encoded = onehot_encoder_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    ## combine one hot with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df],axis=1)

    ## scale the input
    input_data_scaled = scaler.transform(input_data)

    ## predict churn
    prediction = model.predict(input_data_scaled)
    prediction_probab = prediction[0][0]

    #
    st.subheader("📊  Prediction Result")

    col1 , col2 = st.columns(2)
    with col1:
        st.metric(label="Churn Probability", value=f"{prediction_probab: .2%}")

    with col2:
        if prediction_probab > 0.5:
            st.error("⚠️ High risk of churn!")
        else:
            st.success("✅ Customer is likely to stay.")
else:
    st.info("Please fill in all financial information fields to get a prediction.")

   
