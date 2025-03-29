from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        area = int(request.form.get('area'))
        bedrooms = int(request.form.get('bhk'))  # Using 'bhk' from your HTML form
        
        # Create DataFrame with correct column names
        input_data = pd.DataFrame([[area, bedrooms]], columns=['area', 'bedrooms'])
        
        # Make prediction
        prediction = model.predict(input_data)
        output = int(prediction[0])
        
        # Return JSON response
        return jsonify({
            'status': 'success',
            'prediction': output,
            'prediction_text': f'Predicted Price: ₹{output:,}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
