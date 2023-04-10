# Car Price Prediction
This is a machine learning project that predicts the price of a used car based on various features such as the brand, condition, fuel type, kilometers driven, model, and year. The project uses a dataset of used cars from OLX Pakistan and trains a machine learning model using scikit-learn to make price predictions.

# Dataset
The dataset used in this project is available in the olx_car_data_csv.csv file. It contains information about various used cars such as the brand, model, condition, fuel type, kilometers driven, year, price, and location.

# Model
The machine learning model used in this project is a Gradient Boosting Regression model. It was trained using scikit-learn and achieved a Mean Absolute Error (MAE) of 250545.58 on the test set.

The trained model is saved to the car_price_prediction_model.joblib file and can be loaded using the joblib.load() function from the joblib library.

# Dependencies
To run the car price prediction model and app, you will need the following dependencies:

Python 3.7 or higher
scikit-learn
pandas
numpy
streamlit
joblib
You can install these dependencies using pip with the following command:

pip install -r requirements.txt

# Usage
To use the car price prediction model, you can run the car_price_prediction.py script. This script loads the trained model, preprocesses user inputs, and makes a price prediction based on the user inputs.

To use the Streamlit app, you can run the streamlit_app.py script. This app provides a user interface for entering car features and displays the predicted price based on the entered features.

# Credits
This project was created by Anar Shah. The OLX Pakistan dataset used in this project was scraped by Karim Ali and is available on (https://www.kaggle.com/karimali/used-cars-data-pakistan).

# License
This project is licensed under the MIT License.
