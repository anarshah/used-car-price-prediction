# Import libraries
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('car_price_prediction_model.joblib')

# Define a function to get user inputs
def get_input():
    # Load the car data from the CSV file
    df_car = pd.read_csv('olx_car_data_csv.csv', encoding='ISO-8859-1')
    
    brand_options = df_car['Brand'].unique().tolist()
    brand = st.selectbox('Brand', brand_options)
    
    model_options = df_car[df_car['Brand'] == brand]['Model'].unique().tolist()
    model = st.selectbox('Model', model_options)
    
    condition_options = df_car[df_car['Condition']].unique().tolist()
    condition = st.selectbox('Condition', condition_options)
    
    fuel_options = df_car[df_car['Fuel']].unique().tolist()
    fuel = st.selectbox('Fuel', fuel_options)
    
    km_driven = st.slider('KMs Driven', min_value=1, max_value=1000000, step=1000)
    year = st.slider('Year', min_value=1980, max_value=2023, step=1)
    
    # Return a dictionary of inputs
    inputs = {
        'Brand': brand,
        'Model': model,
        'Condition': condition,
        'Fuel': fuel,
        'KMs Driven': km_driven,
        'Year': year
    }
    return inputs

# Define a function to preprocess the inputs
def preprocess_inputs(inputs):
    # Load the car data from the CSV file
    df_car = pd.read_csv('olx_car_data_csv.csv', encoding='ISO-8859-1')
    
    # Convert categorical variables into numeric form
    inputs['Brand'] = df_car[df_car['Brand'] == inputs['Brand']].index[0]
    inputs['Model'] = df_car[df_car['Model'] == inputs['Model']].index[0]
    inputs['Condition'] = df_car[df_car['Condition'] == inputs['Condition'].index[0]
    inputs['Fuel'] = df_car[df_car['Fuel'] == inputs['Fuel'].index[0]
    
    # Return a 2D array of preprocessed inputs
    return [list(inputs.values())]

# Define the Streamlit app
def app():
    st.set_page_config(page_title='Car Price Prediction', page_icon=':car:', layout='wide')
    st.title('Car Price Prediction')
    st.write('This app predicts the price of a car based on user inputs')
    
    # Get user inputs
    inputs = get_input()
    
    # Preprocess the inputs
    X = preprocess_inputs(inputs)
    
    # Make a prediction using the trained model
    price = model.predict(X)[0]
    
    # Display the predicted price to the user
    st.subheader('Predicted Price')
    st.write('PKR', int(price))

# Run the app
if __name__ == '__main__':
    app()
