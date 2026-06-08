# 🔢 Digit Recognition CNN

A web-based handwritten digit recognition app powered by a Convolutional Neural Network (CNN) trained on the MNIST dataset. Draw a digit in your browser and get an instant prediction with confidence score.

---

## 📸 Demo

> Draw a digit on the canvas → click **Predict** → see the result!

*(Add a screenshot or GIF here)*

---

## 🚀 Features

- Interactive drawing canvas in the browser
- Real-time digit prediction (0–9)
- Confidence score displayed per prediction
- CNN model trained on MNIST (99%+ accuracy)
- Lightweight Flask backend

---

## 🛠️ Tech Stack

| Layer     | Technology                  |
|-----------|-----------------------------|
| Frontend  | HTML, CSS, JavaScript       |
| Backend   | Python, Flask               |
| ML Model  | TensorFlow / Keras (CNN)    |
| Image Proc| OpenCV, NumPy               |
| Model File| mnist_cnn_model.h5          |

---

## 📁 Project Structure

```
digit-recognition-cnn/
├── templates/
│   └── index.html          # Frontend drawing canvas
├── app.py                  # Flask server + prediction endpoint
├── digit_gui.py            # (Optional) standalone GUI version
├── main.py                 # Model training script
├── mnist_cnn_model.h5      # Pre-trained CNN model
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## ⚙️ How It Works

1. User draws a digit on an HTML5 canvas
2. Canvas is exported as a base64-encoded image via JavaScript
3. Flask backend receives the image via POST request to `/predict`
4. Image is decoded, resized to 28×28, normalized, and fed into the CNN
5. Model returns the predicted digit + confidence score
6. Result is displayed on the web page

---

## 🧠 Model Architecture

The CNN is trained on the MNIST dataset and follows this pipeline:

- Input: 28×28 grayscale image
- Convolutional layers with ReLU activation
- MaxPooling layers
- Fully connected (Dense) layers
- Output: Softmax over 10 classes (digits 0–9)

---

## 🔧 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/amineAkchache/digit-recognition-cnn.git
cd digit-recognition-cnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in your browser
```
http://127.0.0.1:5000
```

---

## 📦 Requirements

Key packages (see `requirements.txt` for full list):

- Flask
- TensorFlow
- OpenCV-python
- NumPy

---

## 📊 Model Performance

| Metric    | Value     |
|-----------|-----------|
| Dataset   | MNIST     |
| Training  | ~60,000 samples |
| Testing   | ~10,000 samples |
| Accuracy  | ~99%      |

---

## 🙋 Author

**Amine Akchache**  

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
