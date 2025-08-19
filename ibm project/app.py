from flask import Flask, render_template_string, request
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

app = Flask(__name__)

# Load artifacts
try:
    model = joblib.load("stress_model.joblib")
    encoders = joblib.load("label_encoders.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_names = joblib.load("feature_names.joblib")
    print("Model artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: One or more model files not found. Please run 'train.py' first.")
    exit()

# Suggestions
suggestions = {
    0: "Low stress. Keep maintaining your well-being.",
    1: "Moderate stress. Take regular breaks and talk to someone.",
    2: "High stress. Seek help from a counselor or support network."
}

# Form HTML (modified for text input for all fields)
form_html = """<!DOCTYPE html><html><head><title>Mental Health Form</title>
<style>body {font-family: Arial; background: #eef2f7; padding: 30px;}
form {background: white; padding: 20px; border-radius: 10px; max-width: 600px; margin: auto;}
input[type=text], select {width: 100%; padding: 8px; margin: 5px 0; border-radius: 5px; border: 1px solid #ccc;}
input[type=submit] {padding: 10px; background: green; color: white; border: none; border-radius: 5px; cursor: pointer;}
</style></head><body>
<h2 align=center>🧠 Student Mental Health Assessment</h2>
<form method="POST" action="/predict">
{% for field in fields %}
<label>{{ field.replace('_', ' ') }}</label>
<input type="text" name="{{ field }}" required><br>
{% endfor %}
<input type="submit" value="Predict">
</form></body></html>"""

# Result HTML
result_html = """<!DOCTYPE html><html><head><title>Result</title>
<style>body {font-family: Arial; padding: 30px; background: #fdfaf5;}
.box {background: white; padding: 20px; border-radius: 10px; width: 600px; margin: auto; text-align: center;}
.btn {display: inline-block; margin-top: 20px; padding: 10px 20px; background: blue; color: white; border-radius: 5px; text-decoration: none;}
</style></head><body>
<div class="box">
<h2>📄 Prediction Result</h2>
<p><strong>Stress Level:</strong> {{ prediction }}</p>
<p><strong>Suggestion:</strong> {{ suggestion }}</p>
<img src="/static/prediction_chart.png" width="500">
<br><a class="btn" href="/">🔙 Try Again</a>
</div></body></html>"""

# Graph function
def draw_graph(pred):
    colors = ['#00cc66', '#ffcc00', '#cc0000']
    plt.figure(figsize=(6, 1.5))
    sns.set_style("white")
    for i in range(3):
        plt.barh(0, 1, left=i, color=colors[i], edgecolor='black')
    plt.axvline(pred + 0.5, color='black', linestyle='--', lw=2)
    plt.text(pred + 0.5, 0.3, f'← You (Level {pred})', fontsize=10, ha='left', color='black')
    plt.yticks([]); plt.xticks([0, 1, 2]); plt.xlim(0, 3)
    plt.title("Stress Level Indicator")
    plt.tight_layout(); os.makedirs("static", exist_ok=True)
    plt.savefig("static/prediction_chart.png"); plt.close()

@app.route('/')
def home():
    return render_template_string(form_html, fields=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = []
        for field in feature_names:
            val = request.form[field]
            if field in encoders:
                # Handle case-insensitivity for 'Gender'
                val = val.title()
                val = encoders[field].transform([val])[0]
            else:
                val = float(val)
            values.append(val)

        values_scaled = scaler.transform([values])
        pred = model.predict(values_scaled)[0]
        draw_graph(pred)

        return render_template_string(result_html, prediction=pred, suggestion=suggestions[pred])
    except Exception as e:
        return f"<p><strong>Error:</strong> {e}</p><a href='/'>🔙 Back</a>"

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)