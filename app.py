import streamlit as streamlit
import numpy as nump
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#loading the trained model
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender= pickle.load(file)
with open('one_hot_encoder_geography.pkl','rb') as file:
    one_hot_encoder_geography= pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scalar = pickle.load(file)

##streamlit App
streamlit.title('Customer chrun prediction')

# User Input
geography = streamlit.selectbox('Geography', one_hot_encoder_geography.categories_[0])
gender = streamlit.selectbox('Gender', label_encoder_gender.classes_)
age= streamlit.slider('Age', 18, 92)
balance=streamlit.number_input('Balance')
credit_score=streamlit.number_input('Credit Score')
estimated_salary= streamlit.number_input('estimated salary')
tenure=streamlit.slider('Tenure',0,10)
num_of_product=streamlit.slider('Number of Product', 1, 4)
has_cr_card=streamlit.selectbox('Has Credit Card', [0,1])
is_active_member=streamlit.selectbox('Is Active Member', [0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':  [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_product],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded=one_hot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded,columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
#input_df
#Scalaer the input data
input_data_scaled=scalar.transform(input_data)

#Predict Churn
prediction=model.predict(input_data_scaled)
prediction_porba=prediction[0][0]
streamlit.write(f'Churn Probability {prediction_porba:.2f}')
if prediction_porba>0.5:
    print("Customer like to churn")
else:
    print("customer not likely to")