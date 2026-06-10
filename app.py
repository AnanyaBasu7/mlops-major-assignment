from flask import Flask, request, render_template_string
import joblib
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

MODEL_PATH = "savedmodel.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("savedmodel.pth not found. Please run train.py first.")

model = joblib.load(MODEL_PATH)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Olivetti Face Classifier</title>
</head>
<body>
    <h1>Olivetti Face Classifier</h1>
    <p>Upload a face image. The model will predict the class.</p>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br><br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction is not none %}
        <h2>Predicted Class: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""


def preprocess_image(image_file):
    image = Image.open(image_file).convert("L")
    image = image.resize((64, 64))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.reshape(1, -1)
    return image_array


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            processed_image = preprocess_image(file)
            prediction = model.predict(processed_image)[0]

    return render_template_string(HTML_PAGE, prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)