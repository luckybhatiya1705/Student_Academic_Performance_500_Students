from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the SVM model
MODEL_PATH = 'svm_model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# Embedded HTML & CSS for the UI (with animations added)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Model Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem;
        }
        
        /* Animation Keyframes */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }

        .container {
            background-color: var(--card-bg);
            max-width: 600px;
            width: 100%;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            /* Apply Fade In Animation */
            animation: fadeInUp 0.6s ease-out forwards;
        }
        
        h2 {
            text-align: center;
            margin-bottom: 0.5rem;
            color: var(--text-main);
        }
        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 2rem;
            font-size: 0.9rem;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }
        .form-group { display: flex; flex-direction: column; }
        .form-group.full-width { grid-column: span 2; }
        label {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: var(--text-main);
            transition: color 0.3s ease;
        }
        input {
            padding: 0.75rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.15);
            transform: translateY(-2px);
        }
        input:focus + label {
            color: var(--primary);
        }
        button {
            grid-column: span 2;
            padding: 0.85rem;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 1rem;
        }
        button:hover { 
            background-color: var(--primary-hover); 
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        button:active {
            transform: translateY(0);
        }
        .result-box {
            grid-column: span 2;
            margin-top: 1.5rem;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            background-color: #e0e7ff;
            color: var(--primary);
            border: 1px solid #c7d2fe;
            /* Apply pulse animation to result */
            animation: pulse 0.5s ease-in-out;
        }
        .error-box {
            background-color: #fee2e2;
            color: #b91c1c;
            border-color: #fecaca;
        }
        @media (max-width: 500px) {
            .form-grid { grid-template-columns: 1fr; }
            .form-group.full-width, button, .result-box { grid-column: span 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>SVM Prediction Model</h2>
        <p class="subtitle">Enter student data features to generate a prediction</p>
        
        <form method="POST" class="form-grid">
            <div class="form-group">
                <label for="gender">Gender (Numeric)</label>
                <input type="number" step="any" name="gender" required placeholder="e.g. 0 or 1">
            </div>
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" step="any" name="age" required placeholder="e.g. 20">
            </div>
            <div class="form-group">
                <label for="study_hours_per_week">Study Hours / Week</label>
                <input type="number" step="any" name="study_hours_per_week" required placeholder="e.g. 15">
            </div>
            <div class="form-group">
                <label for="attendance_rate">Attendance Rate</label>
                <input type="number" step="any" name="attendance_rate" required placeholder="e.g. 85.5">
            </div>
            <div class="form-group">
                <label for="parent_education">Parent Education (Numeric)</label>
                <input type="number" step="any" name="parent_education" required placeholder="e.g. 2">
            </div>
            <div class="form-group">
                <label for="internet_access">Internet Access (0 or 1)</label>
                <input type="number" step="any" name="internet_access" required placeholder="0 = No, 1 = Yes">
            </div>
            <div class="form-group">
                <label for="extracurricular">Extracurriculars (0 or 1)</label>
                <input type="number" step="any" name="extracurricular" required placeholder="0 = No, 1 = Yes">
            </div>
            <div class="form-group">
                <label for="previous_score">Previous Score</label>
                <input type="number" step="any" name="previous_score" required placeholder="e.g. 75">
            </div>
            <div class="form-group full-width">
                <label for="final_score">Final Score</label>
                <input type="number" step="any" name="final_score" required placeholder="e.g. 82.5">
            </div>
            
            <button type="submit">Run Prediction</button>

            {% if prediction %}
                {% if 'Error' in prediction %}
                    <div class="result-box error-box">{{ prediction }}</div>
                {% else %}
                    <div class="result-box">Prediction Result: {{ prediction }}</div>
                {% endif %}
            {% endif %}
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if model is None:
            prediction = "Error: Model file 'svm_model.pkl' not found on the server."
        else:
            try:
                # Capture features in the exact order found in the provided .pkl dump
                features = [
                    float(request.form['gender']),
                    float(request.form['age']),
                    float(request.form['study_hours_per_week']),
                    float(request.form['attendance_rate']),
                    float(request.form['parent_education']),
                    float(request.form['internet_access']),
                    float(request.form['extracurricular']),
                    float(request.form['previous_score']),
                    float(request.form['final_score'])
                ]
                
                # Reshape for a single sample prediction
                input_data = np.array(features).reshape(1, -1)
                
                # Make prediction
                pred = model.predict(input_data)
                prediction = str(pred[0])
                
            except ValueError:
                prediction = "Error: Please ensure all inputs are valid numbers."
            except Exception as e:
                prediction = f"Error during prediction: {str(e)}"
                
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    # Render requires binding to 0.0.0.0 and dynamically assigning the port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
