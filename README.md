# 🎬 Movie Recommendation System using Deep Learning

This project builds a personalized movie recommendation system leveraging Deep Learning techniques on the MovieLens 100k dataset. The system's core functionality involves predicting user ratings for unseen movies and subsequently recommending top N movies based on a user's past preferences and the trained deep neural network model.

---

## 📌 Project Objective

The main objectives of this development are to:
* Analyze and understand complex user-movie interaction patterns.
* Accurately predict movie ratings for items that a user has not yet interacted with.
* Generate personalized recommendations, suggesting the top N most relevant unseen movies to any user through a well-trained Deep Neural Network (DNN) model.

---

## 📂 Dataset

The project utilizes the **MovieLens 100k** dataset, a widely used benchmark for recommendation systems.
🔗 **Download Link:** [http://files.grouplens.org/datasets/movielens/ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)

The dataset contains:
* `u.data`: Comprises 100,000 explicit ratings (1-5 stars) provided by 943 unique users on 1682 different movies.
* `u.item`: Provides essential movie metadata, including movie titles and associated genres.
* `u.info`: Offers a concise summary of the dataset's scale, detailing the number of users, movies, and ratings.

---

## 🧪 Technologies Used

This project is implemented primarily in Python and utilizes several key libraries and tools:
* **Python 3.x**: The primary programming language.
* **Pandas & NumPy**: Essential libraries for efficient data manipulation, analysis, and numerical operations.
* **Matplotlib**: Used for creating static, interactive, and animated visualizations in Python.
* **scikit-learn**: Provides various utilities for machine learning tasks, such as data splitting (train-test split).
* **TensorFlow / Keras**: The core deep learning framework used to build, train, and evaluate the neural network model.
* **Google Colab**: The preferred cloud-based environment for development and execution, offering free GPU/TPU access.

---

## 🏗️ Model Architecture

The Deep Neural Network (DNN) designed for this recommendation system comprises:
* **Two Embedding Layers**: Separate embedding layers for User IDs and Movie IDs are used to learn dense, low-dimensional representations (embeddings) for each user and movie.
* **Dense Layers**: Multiple fully connected layers with **ReLU (Rectified Linear Unit) activation functions** are stacked to capture complex non-linear relationships within the data.
* **Final Softmax Layer**: The output layer consists of 9 classes (representing normalized ratings or rating bins). The Softmax activation function provides a probability distribution over these classes, indicating the likelihood of each rating.
* **Optimizer**: **Adam optimizer** is employed for efficient gradient-based optimization during training.
* **Loss Function**: **SparseCategoricalCrossentropy** is used as the loss function, suitable for multi-class classification where the target labels are integers.

---

## 🚀 How to Run This Project

You can execute and experiment with this Movie Recommendation System project quickly and easily using Google Colab.

### 🔗 Step 1: Open the Colab Notebook

The entire project code and execution environment are pre-configured in a Google Colab notebook.
👉 [**Click here to open the notebook in Google Colab**](https://colab.research.google.com/drive/1O-d7VsEqSoT4bAs4J2i0PI-4NU0PXZkG)

Once opened in Colab, you can run all the cells sequentially to execute the project from data loading to model training and recommendation generation.

### 🧰 Step 2: Install Required Libraries (Optional for Colab, Essential for Local)

While most necessary libraries are pre-installed in Google Colab, it's good practice to include installation commands, especially if you plan to run the project in a local Python environment.

pip install numpy pandas matplotlib scikit-learn tensorflow




---

## 🔍 Workflow Overview



```mermaid

graph TD

  A[Load Dataset] --> B[Merge Ratings & Titles]

  B --> C[Average multiple ratings]

  C --> D[Encode users & movies]

  D --> E[Train–Test Split]

  E --> F[Build Neural Network]

  F --> G[Train Model]

  G --> H[Predict ratings for unseen movies]

  H --> I[Sort and Recommend Top-N Movies]
