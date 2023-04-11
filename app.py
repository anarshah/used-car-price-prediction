import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model and encoders
model = joblib.load('car_price_prediction_model.joblib')
brand_encoder = joblib.load('brand_encoder.joblib')
model_encoder = joblib.load('model_encoder.joblib')
condition_encoder = joblib.load('condition_encoder.joblib')
fuel_encoder = joblib.load('fuel_encoder.joblib')

def get_input():
    # Load the car data from the CSV file
    df_car = pd.read_csv('olx_car_data_csv.csv', encoding='ISO-8859-1')
    df_car = df_car.dropna()

    brand_options = df_car['Brand'].unique().tolist()
    brand = st.selectbox('Brand', brand_options)

    model_options = df_car[df_car['Brand'] == brand]['Model'].dropna().unique().tolist()
    model = st.selectbox('Model', model_options)

    condition_options = df_car['Condition'].unique().tolist()
    condition = st.selectbox('Condition', condition_options)

    fuel_options = df_car['Fuel'].unique().tolist()
    fuel = st.selectbox('Fuel', fuel_options)

    km_driven = st.slider('KMs Driven', min_value=1, max_value=1000000, step=1000)
    year = st.slider('Year', min_value=1980, max_value=2023, step=1)

    inputs = {
        'Brand': brand,
        'Model': model,
        'Condition': condition,
        'Fuel': fuel,
        'KMs Driven': km_driven,
        'Year': year
    }
    return inputs

def preprocess_inputs(inputs):
    inputs['Brand'] = brand_encoder.transform([inputs['Brand']])[0]
    inputs['Model'] = model_encoder.transform([inputs['Model']])[0]
    inputs['Condition'] = condition_encoder.transform([inputs['Condition']])[0]
    inputs['Fuel'] = fuel_encoder.transform([inputs['Fuel']])[0]

    return [list(inputs.values())]

def app():
    st.set_page_config(page_title='Car Price Prediction', page_icon=':car:', layout='wide')
    st.title('Car Price Prediction')
    st.write('This app predicts the price of a car based on user inputs')

    inputs = get_input()
    X = preprocess_inputs(inputs)
    price = model.predict(X)[0]

    st.subheader('Predicted Price')
    st.write('PKR', int(price))

if __name__ == '__main__':
    app()
