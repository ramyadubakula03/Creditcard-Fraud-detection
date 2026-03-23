from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/fraud_model.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract values
        amount = float(data.get('amount', 0))
        time = float(data.get('time', 0))
        v_features = [float(data.get(f'V{i}', 0)) for i in range(1, 29)]

        # Scale amount and time
        amount_scaled = scaler.transform([[amount]])[0][0]
        time_scaled_val = (time - 94813.86) / 47488.14  # manual z-score approx

        # Build feature array in correct order
        # Features: V1-V28, Amount_scaled, Time_scaled
        features = v_features + [amount_scaled, time_scaled_val]
        features_array = np.array(features).reshape(1, -1)

        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]

        fraud_prob = round(float(probability[1]) * 100, 2)
        legit_prob = round(float(probability[0]) * 100, 2)

        return jsonify({
            'prediction': int(prediction),
            'label': '🚨 FRAUDULENT' if prediction == 1 else '✅ LEGITIMATE',
            'fraud_probability': fraud_prob,
            'legit_probability': legit_prob,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
