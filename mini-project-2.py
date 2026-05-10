# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# =========================
# Setup
# =========================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("Setup complete")


# =========================
# Load MNIST CSV files
# =========================

train_path = r"C:\Users\GAMER\Downloads\mnist\mnist_train.csv"
test_path = r"C:\Users\GAMER\Downloads\mnist\mnist_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# First column = label
y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values / 255.0

y_test = test_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values / 255.0

# Convert labels to one-hot encoding
y_train_oh = to_categorical(y_train, num_classes=10)
y_test_oh = to_categorical(y_test, num_classes=10)

print("Train shape:", X_train.shape, y_train_oh.shape)
print("Test shape:", X_test.shape, y_test_oh.shape)


# =========================
# Model Settings
# =========================

hidden_layers = 3
neurons_per_layer = 128
hidden_activation = "relu"

epochs = 10
learning_rate = 0.001


# =========================
# Build Neural Network
# =========================

model = Sequential()

# First hidden layer
model.add(Dense(
    neurons_per_layer,
    activation=hidden_activation,
    input_shape=(X_train.shape[1],)
))

# Add remaining hidden layers
for _ in range(hidden_layers - 1):
    model.add(Dense(
        neurons_per_layer,
        activation=hidden_activation
    ))

# Output layer: 10 digits/classes
model.add(Dense(10, activation="softmax"))


# =========================
# Compile Model
# =========================

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

optimizer = tf.keras.optimizers.ٍSGD(learning_rate=0.01)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# Train Model
# =========================

history = model.fit(
    X_train,
    y_train_oh,
    epochs=epochs,
    validation_data=(X_test, y_test_oh),
    verbose=1
)


# =========================
# Evaluate Model
# =========================

test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)

print("Final train accuracy:", round(float(history.history["accuracy"][-1]), 3))
print("Final validation accuracy:", round(float(history.history["val_accuracy"][-1]), 3))
print("Test accuracy:", round(float(test_acc), 3))


# =========================
# Predictions
# =========================

preds = model.predict(X_test[:5], verbose=0)
pred_labels = np.argmax(preds, axis=1)

print("Predictions head:", pred_labels.tolist())
print("True labels head:", y_test[:5].tolist())
print("Prob row sums:", np.round(preds.sum(axis=1), 3).tolist())

assert preds.shape == (5, 10)
assert np.allclose(preds.sum(axis=1), 1, atol=1e-5)

print("Interpretation: softmax outputs probabilities over 10 classes.")


# =========================
# Plot Training History
# =========================

plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()
