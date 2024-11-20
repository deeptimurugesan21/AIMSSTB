from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load('fan_status_model.pkl')

app = Flask(__name__)

# Store the latest sensor data and prediction
latest_data = {
    'temperature': None,
    'gas': None,
    'smoke': None,
    'noise': None,
    'vibration': None,
    'fan_status': None
}

@app.route('/predict', methods=['POST'])
def predict():
    global latest_data
    data = request.json
    temperature = data['temperature']
    gas = data['gas']
    smoke = data['smoke']
    noise = data['noise']
    vibration = data['vibration']
    
    # Create a feature array for prediction
    features = np.array([[temperature, gas, smoke, noise, vibration]])
    prediction = model.predict(features)[0]
    
    # Update the latest data and prediction
    latest_data = {
        'temperature': temperature,
        'gas': gas,
        'smoke': smoke,
        'noise': noise,
        'vibration': vibration,
        'fan_status': int(prediction)
    }

    # Respond with the prediction
    return jsonify({'fan_status': int(prediction)})

# New endpoint to get the latest sensor data for updating the graphs
@app.route('/get_sensor_data', methods=['GET'])
def get_sensor_data():
    return jsonify(latest_data)

@app.route('/')
def index():
    return render_template('index.html', data=latest_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
