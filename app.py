# Import libraries
import streamlit as st
import joblib

# Load the trained model
model = joblib.load('car_price_prediction_model.joblib')

# Define a function to get user inputs
def get_input():
    brand = st.selectbox('Brand', ['Toyota', 'Suzuki', 'Honda', 'Daihatsu', 'Mitsubishi'])
    condition = st.selectbox('Condition', ['Used', 'New'])
    fuel = st.selectbox('Fuel', ['Petrol', 'Diesel', 'CNG'])
    km_driven = st.slider('KMs Driven', min_value=1, max_value=1000000, step=1000)
    model = st.text_input('Model')
    year = st.slider('Year', min_value=1980, max_value=2023, step=1)
    
    # Return a dictionary of inputs
    inputs = {
        'Brand': brand,
        'Condition': condition,
        'Fuel': fuel,
        'KMs Driven': km_driven,
        'Model': model,
        'Year': year
    }
    return inputs

# Define a function to preprocess the inputs
# Define a function to preprocess the inputs
def preprocess_inputs(inputs):
    # Convert categorical variables into numeric form
    inputs['Brand'] = ['Toyota', 'Suzuki', 'Honda', 'Daihatsu', 'Mitsubishi'].index(inputs['Brand'])
    inputs['Condition'] = ['Used', 'New'].index(inputs['Condition'])
    inputs['Fuel'] = ['Petrol', 'Diesel', 'CNG'].index(inputs['Fuel'])
    inputs['Model'] = -1 if inputs['Model'] == '' else inputs['Model']
    
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
