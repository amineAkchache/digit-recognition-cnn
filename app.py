from flask import Flask, render_template, request
import numpy as np
import cv2
import base64
import tensorflow as tf   # <-- ADD THIS

app = Flask(__name__)

# Load model correctly
model = tf.keras.models.load_model("mnist_cnn_model.h5", compile=False)

def preprocess_image(img):
    img = cv2.resize(img, (28, 28))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    img_data = request.form["image"]
    img_data = img_data.split(",")[1]
    img_bytes = base64.b64decode(img_data)

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    processed = preprocess_image(img)

    prediction = model.predict(processed)[0]

    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    return {
        "digit": digit,
        "confidence": round(confidence, 2)
    }

if __name__ == "__main__":
    app.run(debug=True)