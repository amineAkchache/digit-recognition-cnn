import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load trained CNN model
model = load_model("mnist_cnn_model.h5")

# Create window
window = tk.Tk()
window.title("Draw a Digit (0-9)")

canvas_width = 300
canvas_height = 300

canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="black")
canvas.pack()

# PIL image for drawing
image = Image.new("L", (canvas_width, canvas_height), "black")
draw = ImageDraw.Draw(image)

def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
    draw.ellipse([x1, y1, x2, y2], fill="white")

canvas.bind("<B1-Motion>", paint)

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill="black")

def predict_digit():
    # Convert PIL image to numpy
    img = np.array(image)

    # Resize to manageable size first
    img = cv2.resize(img, (280, 280))

    # Apply Gaussian Blur (smoothing)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold (binary image)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Find bounding box of digit
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)

    digit = img[y:y+h, x:x+w]

    # Resize to 20x20 (MNIST standard inner size)
    digit = cv2.resize(digit, (20, 20))

    # Create blank 28x28 image
    new_img = np.zeros((28, 28), dtype=np.uint8)

    # Center digit inside 28x28
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    new_img[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    # Normalize
    new_img = new_img / 255.0

    # Reshape for CNN
    new_img = new_img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(new_img)
    digit_pred = np.argmax(prediction)

    result_label.config(text=f"Prediction: {digit_pred}")