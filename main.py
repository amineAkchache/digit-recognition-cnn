import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report


# ==============================
# 1️⃣ Load Dataset
# ==============================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Original Training shape:", X_train.shape)
print("Original Test shape:", X_test.shape)


# ==============================
# 2️⃣ Normalize
# ==============================
X_train = X_train / 255.0
X_test = X_test / 255.0


# ==============================
# 3️⃣ Reshape for CNN
# ==============================
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("After reshape:", X_train.shape)


# ==============================
# 4️⃣ Build CNN Model
# ==============================
model = Sequential()

# Convolution layer 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))

# Pooling layer
model.add(MaxPooling2D((2,2)))

# Convolution layer 2
model.add(Conv2D(64, (3,3), activation='relu'))

# Pooling layer
model.add(MaxPooling2D((2,2)))

# Flatten before Dense layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(10, activation='softmax'))

model.summary()


# ==============================
# 5️⃣ Compile
# ==============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ==============================
# 6️⃣ Train
# ==============================
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)


# ==============================
# 7️⃣ Evaluate
# ==============================
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)


# ==============================
# 8️⃣ Predictions
# ==============================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


# ==============================
# 9️⃣ Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ==============================
# 🔟 Classification Report
# ==============================


# ==============================
# 🔟 Show Wrong Predictions
# ==============================
wrong = np.where(y_pred != y_test)[0]

plt.figure(figsize=(10,6))
for i in range(6):
    index = wrong[i]
    plt.subplot(2,3,i+1)
    plt.imshow(X_test[index].reshape(28,28), cmap='gray')
    plt.title(f"Pred: {y_pred[index]}, True: {y_test[index]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

print(classification_report(y_test, y_pred))
model.save("mnist_cnn_model.h5")
print("Model saved successfully.")