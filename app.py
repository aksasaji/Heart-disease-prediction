from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('heart_disease_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])
        
        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])
        prediction = model.predict(input_data)
        
        result = "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
