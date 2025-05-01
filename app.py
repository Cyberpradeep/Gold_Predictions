from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
import joblib
import os
from data_cleaning import clean_gold_data  # Ensure this module is available

app = Flask(__name__)

# Load and clean the dataset
try:
    df = clean_gold_data('data/gold_prices.csv')
    if df is None:
        raise ValueError("Data cleaning failed.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Load the trained model
model_path = 'gold_price_model.pkl'
try:
    if not os.path.exists('gold_price.pkl'):
        print("Model not found. Please train the model.")
        exit(1)

    model = joblib.load('gold_price.pkl')

except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define the route for the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        date_str = request.form.get('date')
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            features = pd.DataFrame({
                'Day': [date_obj.day],
                'Month': [date_obj.month],
                'Year': [date_obj.year]
            })
            prediction = model.predict(features)[0]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

# Entry point for running the app
if __name__ == '__main__':
    app.run(debug=True)
