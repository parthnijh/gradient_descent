
---

# 🧠 Logistic Regression with Keras & Manual Gradient Descent

This project demonstrates **binary classification (insurance prediction)** using both:

1. **Keras implementation** (with `Sequential` and `Dense` layers)
2. **Manual implementation** of gradient descent for logistic regression

It’s meant to show what happens “under the hood” when training a neural network.

---

## 📂 Contents

* [Data Preparation](#-data-preparation)
* [Keras Model](#-keras-model)
* [Manual Gradient Descent](#-manual-gradient-descent)
* [Gradient Descent vs Optimizer](#-gradient-descent-vs-optimizer)
* [Dense Layer](#-dense-layer)
* [Sequential Model](#-sequential-model)
* [Model Choices](#-model-choices)

---

## 🔹 Data Preparation

We start with insurance data that has:

* **Features**:

  * `age` (scaled by dividing by 100)
  * `affordability`
* **Target**:

  * `bought_insurance` (0 = No, 1 = Yes)

We split into train/test using `train_test_split`.

---

## 🔹 Keras Model

The Keras model is defined as:

```python
model = keras.Sequential([
    keras.layers.Dense(
        1,
        input_shape=(2,),       # 2 input features
        activation="sigmoid",   # logistic regression
        kernel_initializer="ones",
        bias_initializer="zeros"
    )
])
```

* **Dense(1, sigmoid)** = logistic regression
* Loss = **binary crossentropy**
* Optimizer = **Adam**
* Trained with `.fit(...)` for 5000 epochs

---

## 🔹 Manual Gradient Descent

We also implement training manually:

```python
def grad_dec(age, affordability, y_true, epochs):
    w1 = w2 = 1
    bias = 0
    rate = 0.5   # learning rate
    n = len(age)

    for i in range(epochs):
        # Forward pass
        weightedsum = w1*age + w2*affordability + bias
        ypred = sigmoid_numpy(weightedsum)

        # Loss (binary crossentropy)
        loss = log_loss(y_true, ypred)

        # Backpropagation (gradients)
        bias_d = np.mean(ypred - y_true)
        w1d = (1/n) * np.dot(age.T, (ypred - y_true))
        w2d = (1/n) * np.dot(affordability.T, (ypred - y_true))

        # Gradient descent update
        bias = bias - rate * bias_d
        w1 = w1 - rate * w1d
        w2 = w2 - rate * w2d

    return w1, w2, bias
```

This reproduces what Keras does internally:

1. Forward pass
2. Compute loss
3. Compute gradients (backpropagation)
4. Update weights with gradient descent

---

## 🔹 Gradient Descent vs Optimizer

* **Manual GD** → you explicitly define the learning rate (`rate=0.5`) and apply weight updates.
* **Keras Optimizers** (`Adam`, `SGD`, etc.) → handle gradient descent automatically. You just pick the optimizer and optionally its learning rate:

```python
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy"
)
```

---

## 🔹 Dense Layer

`keras.layers.Dense(units, activation=...)` = fully connected layer

$$
y = f(Wx + b)
$$

* `units`: number of neurons (outputs)
* `activation`: function (`relu`, `sigmoid`, `softmax`, etc.)
* `W`: weight matrix, `b`: bias vector

Example:

```python
Dense(3, input_shape=(2,), activation="relu")
```

* Input = 2 features
* Output = 3 neurons
* Weight matrix = (2 × 3), Bias = (3,)

---

## 🔹 Sequential Model

`keras.Sequential([...])` = stack of layers in order.

Example:

```python
model = keras.Sequential([
    Dense(10, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

* Input → 10 hidden units (ReLU) → 1 output (Sigmoid)
* Used for problems where data flows in a straight pipeline (no branching).

---



---

## ✅ Summary

* Keras `Dense(1, sigmoid)` is logistic regression.
* Gradient descent is always used under the hood in Keras (via `optimizers` and `tf.GradientTape`).
* We showed how to implement gradient descent manually, and saw the backpropagation step explicitly.
* Learning rate controls step size: in manual code we define it (`rate=0.5`), in Keras the optimizer manages it.
* `Dense` = fully connected layer, `Sequential` = stack of layers.

This project helps you **understand what’s happening behind the scenes** when training models in TensorFlow/Keras.

---
