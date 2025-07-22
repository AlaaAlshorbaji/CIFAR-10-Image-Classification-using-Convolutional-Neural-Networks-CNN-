# 🖼️ CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

This project demonstrates how to classify images from the CIFAR-10 dataset using a deep Convolutional Neural Network (CNN) built with TensorFlow and Keras.

The CIFAR-10 dataset is a well-known benchmark in computer vision, consisting of small 32x32 color images across 10 classes. The objective of this notebook is to build, train, and evaluate a CNN model that can accurately classify these images.

## 🎯 Objective

- Build a CNN model to classify 10 different image categories
- Preprocess and normalize image data
- Train and evaluate the model on CIFAR-10
- Visualize performance and predictions

## 📦 Dataset: CIFAR-10

- 60,000 32x32 color images in 10 classes:
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images + 10,000 testing images
- Available directly via: `tf.keras.datasets.cifar10`
## 📁 File Structure

The project is fully implemented in a single Jupyter notebook:

📄 CIFAR‑10_Image_Classification_with_CNN.ipynb

This notebook is structured into the following sections:

---

### 1. 📚 Importing Libraries

Essential libraries used:
- `tensorflow` – to build and train the CNN
- `matplotlib` – for data visualization
- `numpy` – for numerical operations

---

### 2. 📥 Loading the Dataset

- Loads the CIFAR-10 dataset via `tf.keras.datasets.cifar10`.
- Splits the data into training and test sets:
  - `x_train`, `y_train`
  - `x_test`, `y_test`

---

### 3. 🧼 Data Preprocessing

- Normalizes image pixel values to the `[0, 1]` range.
- Optionally converts labels to one-hot encoding (if used in categorical crossentropy).

---

### 4. 🔎 Data Visualization

- Displays sample images from the training set.
- Helps visualize the 10 different image categories.

---

### 5. 🧠 Building the CNN Model

The model is constructed using `tf.keras.Sequential` and includes:
- Convolutional layers (`Conv2D`) with ReLU activation
- Pooling layers (`MaxPooling2D`)
- Dropout layer to prevent overfitting
- Fully connected (`Dense`) layers
- Final output layer with softmax activation for 10 classes

---

### 6. 🏋️ Model Compilation & Training

- Compiled with:
  - Optimizer: `adam`
  - Loss: `sparse_categorical_crossentropy`
  - Metrics: `accuracy`
- Trained over multiple epochs using `.fit()`
- Uses validation split to monitor performance

---

### 7. 📈 Model Evaluation

- Evaluates performance on test data using `.evaluate()`
- Displays test loss and accuracy

---

### 8. 🔮 Predictions & Visualization

- Predicts labels for test images using `.predict()`
- Compares predicted vs. true labels
- Visualizes results, optionally highlighting correct/incorrect classifications

👨‍💻 Author
Alaa Shorbaji
Artificial Intelligence Instructor
Computer Vision & Deep Learning Specialist

📜 License
This project is licensed under the MIT License.

You are free to:

✅ Use and share the code for personal, academic, or commercial purposes.

✅ Modify, distribute, and build upon the code with proper credit.

You must:

❗ Provide appropriate attribution to the original author.

❗ Include this license notice in any copies or substantial portions of the project.

Disclaimer: This notebook uses open-source tools and datasets provided by TensorFlow. The author does not claim ownership of CIFAR-10 or any third-party resources used within the notebook.
