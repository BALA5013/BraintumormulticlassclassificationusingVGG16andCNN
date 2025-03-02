from flask import Flask, request, render_template_string, jsonify, url_for
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import joblib  # For Random Forest

# Initialize Flask app
app = Flask(__name__)

# Configure static folder for assets
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Paths to models
MODELS = {
    'CNN': os.path.join('C:/Desktop/brain_tumor', 'tumor_classification_model.h5'),
    'Random Forest': os.path.join('C:/Desktop/brain_tumor', 'random_forest_model.pkl'),
    'VGG16': os.path.join('C:/Desktop/brain_tumor', 'tumorVGG16.h5')
}

# Class labels
CLASSES = {0: 'No Tumor', 1: 'Pituitary Tumor', 2: 'Meningioma Tumor', 3: 'Glioma Tumor'}

# Algorithm metrics
METRICS = [
    {'Algorithm': 'CNN', 'Accuracy': '88.29%', 'Precision': '88.32%', 'Recall': '88.29%', 'F1 Score': '88.25%'},
    {'Algorithm': 'Random Forest', 'Accuracy': '88.13%', 'Precision': '88.48%', 'Recall': '88.13%', 'F1 Score': '88.07%'},
    {'Algorithm': 'VGG16', 'Accuracy': '90.66%', 'Precision': '90.97%', 'Recall': '90.66%', 'F1 Score': '90.70%'}
]

# HTML and CSS for the app (updated with a dropdown)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Tumor Classification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 20px;
            background-image: url('{{ url_for('static', filename='background.jpg') }}');
            background-size: cover;
            background-position: center;
            color: #000000;
        }
        h1 {
            color: #00008B;
        }
        form {
            margin: 20px auto;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            max-width: 400px;
        }
        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result, .error, .metrics, .images-container {
            margin: 20px auto;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            max-width: 800px;
        }
        table {
            margin: 0 auto;
            width: 100%;
            border-collapse: collapse;
            text-align: center;
        }
        table, th, td {
            border: 1px solid #000;
        }
        th, td {
            padding: 10px;
        }
        th {
            background-color: #007bff;
            color: white;
        }
        .images-container {
    display: flex;
    flex-wrap: wrap; /* Allows images to wrap to the next row if necessary */
    justify-content: center;
    gap: 20px; /* Space between images */
    margin-top: 20px;
}

.images-container div {
    text-align: center;
    flex: 1 1 30%; /* Each image container takes up to 30% of the row */
}

.images-container img {
    width: 100%; /* Makes images responsive */
    max-width: 300px; /* Limits the size for consistency */
    height: auto; /* Keeps the aspect ratio intact */
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.images-container p {
    font-size: 16px;
    color: #333;
    margin-top: 10px;
}        
    </style>
</head>
<body>
    <h1>Brain Tumor Classification</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <label for="model">Select Model:</label><br>
        <select name="model" required>
            <option value="CNN">CNN</option>
            <option value="Random Forest">Random Forest</option>
            <option value="VGG16">VGG16</option>
        </select><br><br>
        <label for="file">Upload an Image:</label><br>
        <input type="file" name="file" accept="image/*" required><br><br>
        <button type="submit">Predict</button>
    </form>
    {% if prediction %}
    <div class="result">
        <h2>Prediction: {{ prediction }}</h2>
        <p>Confidence: {{ confidence }}%</p>
    </div>
    {% elif error %}
    <div class="error">
        <h2>Error: {{ error }}</h2>
    </div>
    {% endif %}
<div class="metrics">
    <h2>Algorithm Metrics</h2>
    <table>
        <tr>
            <th>Algorithm</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
        </tr>
        {% for metric in metrics %}
        <tr>
            <td>{{ metric.Algorithm }}</td>
            <td>{{ metric.Accuracy }}</td>
            <td>{{ metric.Precision }}</td>
            <td>{{ metric.Recall }}</td>
            <td>{{ metric['F1 Score'] }}</td>
        </tr>
        {% endfor %}
    </table>
</div>


<!-- Display confusion matrices -->
<div class="images-container">
    <div>
        <img src="{{ url_for('static', filename='confusionmatrix_RF.png') }}" alt="Random Forest Confusion Matrix">
        <p>Random Forest Confusion Matrix</p>
    </div>
    <div>
        <img src="{{ url_for('static', filename='confusionmatrix_CNN.png') }}" alt="CNN Confusion Matrix">
        <p>CNN Confusion Matrix</p>
    </div>
    <div>
        <img src="{{ url_for('static', filename='VGG16CONFUSIONMATRIX.png') }}" alt="VGG16 Confusion Matrix">
        <p>VGG16 Confusion Matrix</p>
    </div>
</div>

<!-- Display accuracy plots -->
<div class="images-container">
    <div>
        <img src="{{ url_for('static', filename='CNNoutputplot.png') }}" alt="CNN Accuracy Chart">
        <p>CNN Accuracy Chart</p>
    </div>
    <div>
        <img src="{{ url_for('static', filename='VGG16output.png') }}" alt="VGG16 Accuracy Chart">
        <p>VGG16 Accuracy Chart</p>
    </div>
</div>

</body>
</html>
"""

# Define the home route
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, metrics=METRICS)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return render_template_string(HTML_TEMPLATE, error="Invalid form submission", metrics=METRICS)

    file = request.files['file']
    selected_model = request.form['model']
    if file.filename == '':
        return render_template_string(HTML_TEMPLATE, error="No file selected", metrics=METRICS)

    try:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(file_path)

        # Predict based on the selected model
        if selected_model == 'CNN':
            model = load_model(MODELS['CNN'])

            # Preprocess image for CNN
            img = cv2.imread(file_path, 0)  # Read as grayscale
            img = cv2.resize(img, (200, 200))  # Resize to 200x200
            img = img.reshape(1, 200, 200, 1)  # Add batch and channel dimensions
            img = img.astype('float32') / 255.0  # Normalize pixel values

            predictions = model.predict(img)

        elif selected_model == 'Random Forest':
            model = joblib.load(MODELS['Random Forest'])

            # Preprocess image for Random Forest
            img = cv2.imread(file_path, 0)  # Read as grayscale
            img = cv2.resize(img, (200, 200))  # Resize to 200x200
            img_flat = img.flatten().reshape(1, -1)  # Flatten the image for Random Forest

            predictions = model.predict_proba(img_flat)

        elif selected_model == 'VGG16':
            model = load_model(MODELS['VGG16'])

            # Preprocess image for VGG16
            img_rgb = cv2.cvtColor(cv2.resize(cv2.imread(file_path), (224, 224)), cv2.COLOR_BGR2RGB)  # Convert to RGB
            img_rgb = img_rgb.astype('float32') / 255.0  # Normalize pixel values
            img_rgb = img_rgb.reshape(1, 224, 224, 3)  # Add batch dimension

            predictions = model.predict(img_rgb)

        # Determine the predicted class and confidence
        class_index = np.argmax(predictions)
        class_label = CLASSES[class_index]
        confidence = np.max(predictions) * 100

        # Remove the uploaded file after processing
        os.remove(file_path)

        return render_template_string(HTML_TEMPLATE, prediction=class_label, confidence=f"{confidence:.2f}", metrics=METRICS)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e), metrics=METRICS)

# Run the app
if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
