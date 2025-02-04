from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pymysql
import logging
import config

# Load the trained model and scaler
model = pickle.load(open('rf_model_sc.pkl', 'rb'))
scaler = pickle.load(open('std_scaler.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Aditya.6404',
    'database': 'classification_db_scaled'
}

# Define input validation function
def validate_input_data(input_data):
    expected_features = [
        'Depreciation and amortization', 'EBITDA', 'Inventory', 'Net Income',
        'Total Receivables', 'Market value', 'Total assets', 'Total Current Liabilities',
        'Total Long-term Debt', 'Total Revenue'
    ]
    
    for feature in expected_features:
        if feature not in input_data:
            raise ValueError(f"Missing required feature: {feature}")
    
    for key, value in input_data.items():
        try:
            float_value = float(value)
            if float_value < -100 or float_value > 1000000:
                raise ValueError(f"Invalid input: {key} must be between -100 and 1,000,000.")
        except ValueError:
            raise ValueError(f"Invalid input: {key} must be a numeric value.")

# Define preprocessing function
def preprocess_features(data):
    original_features = {key: float(data.get(key, 0)) for key in [
        'Depreciation and amortization', 'EBITDA', 'Inventory', 'Net Income',
        'Total Receivables', 'Market value', 'Total assets', 'Total Current Liabilities',
        'Total Long-term Debt', 'Total Revenue'
    ]}

    processed_features = [
        original_features['Depreciation and amortization'],
        original_features['EBITDA'],
        original_features['Inventory'],
        original_features['Net Income'],
        original_features['Total Receivables'],
        original_features['Market value'],
        original_features['Total assets'],
        original_features['EBITDA'] / original_features['Total Revenue'] if original_features['Total Revenue'] != 0 else 0,
        original_features['Net Income'] / original_features['Total Revenue'] if original_features['Total Revenue'] != 0 else 0,
        (original_features['Total Current Liabilities'] + original_features['Total Long-term Debt']) / original_features['Total assets'] if original_features['Total assets'] != 0 else 0,
        original_features['Net Income'] / original_features['Total assets'] if original_features['Total assets'] != 0 else 0,
        original_features['Total Revenue'] / original_features['Total assets'] if original_features['Total assets'] != 0 else 0,
        (original_features['Total assets'] - original_features['Total Current Liabilities']) / original_features['Total Revenue'] if original_features['Total Revenue'] != 0 else 0,
        original_features['Total Long-term Debt'] / (original_features['Total assets'] - original_features['Total Current Liabilities']) if (original_features['Total assets'] - original_features['Total Current Liabilities']) != 0 else 0,
        original_features['Net Income'] / original_features['Total assets'] if original_features['Total assets'] != 0 else 0,
        original_features['Market value'] / original_features['Net Income'] if original_features['Net Income'] != 0 else 0,
    ]

    scaled_features = scaler.transform([processed_features])
    return scaled_features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        logger.info(f"Received input data: {input_data}")

        validate_input_data(input_data)
        processed_features = preprocess_features(input_data)
        logger.info(f"Processed features: {processed_features}")

        proba = model.predict_proba(processed_features)[0]
        logger.info(f"Prediction probabilities: {proba}")

        threshold = 0.5
        prediction = 1 if proba[1] > threshold else 0
        logger.info(f"Final prediction: {prediction}")

        save_prediction_to_db(input_data, prediction)

        return jsonify({'prediction': int(prediction), 'probabilities': {'class_0': float(proba[0]), 'class_1': float(proba[1])}})

    except ValueError as ve:
        logger.error(f"Validation Error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Save prediction to the database
def save_prediction_to_db(input_data, prediction):
    try:
        connection = pymysql.connect(**DB_CONFIG)
        cursor = connection.cursor()

        create_table_query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            `Depreciation and amortization` FLOAT,
            EBITDA FLOAT,
            Inventory FLOAT,
            `Net Income` FLOAT,
            `Total Receivables` FLOAT,
            `Market value` FLOAT,
            `Total assets` FLOAT,
            `Total Current Liabilities` FLOAT,
            `Total Long-term Debt` FLOAT,
            `Total Revenue` FLOAT,
            prediction FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)

        insert_query = """
        INSERT INTO predictions (
            `Depreciation and amortization`, EBITDA, Inventory, `Net Income`,
            `Total Receivables`, `Market value`, `Total assets`, `Total Current Liabilities`,
            `Total Long-term Debt`, `Total Revenue`, prediction
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, (
            input_data['Depreciation and amortization'], input_data['EBITDA'], input_data['Inventory'],
            input_data['Net Income'], input_data['Total Receivables'], input_data['Market value'],
            input_data['Total assets'], input_data['Total Current Liabilities'], input_data['Total Long-term Debt'],
            input_data['Total Revenue'], prediction
        ))

        connection.commit()
        connection.close()
    except pymysql.MySQLError as e:
        logger.error(f"Database Error: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port= config.PORT_NUMBER,debug=False)
