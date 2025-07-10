# 🎬 Movie Recommendation System using Deep Learning

This project builds a personalized movie recommendation system using Deep Learning on the MovieLens 100k dataset. It predicts ratings and recommends unseen movies for each user based on their past preferences.

---

## 📌 Project Objective

To develop a system that can:
- Analyze user-movie interactions.
- Predict movie ratings for unseen items.
- Recommend top N movies to any user using a trained DNN model.

---

## 📂 Dataset

**MovieLens 100k**  
🔗 [Download Here](http://files.grouplens.org/datasets/movielens/ml-100k.zip)

Contains:
- `u.data`: 100,000 ratings from 943 users on 1682 movies.
- `u.item`: Movie details including title and genres.
- `u.info`: Summary of users, movies, and ratings.

---

## 🧪 Technologies Used

- Python 3.x
- Pandas & NumPy
- Matplotlib
- scikit-learn
- TensorFlow / Keras
- Google Colab

---

## 🏗️ Model Architecture

- 2 Embedding layers (User and Movie)
- Dense layers with ReLU activations
- Final Softmax layer with 9 output classes (for normalized ratings)
- Optimizer: Adam
- Loss: SparseCategoricalCrossentropy

---

## 🚀 How to Run This Project

You can run this Movie Recommendation System in just a few simple steps using Google Colab.

### 🔗 Step 1: Open the Colab Notebook

👉 [Click here to open the notebook in Google Colab](https://colab.research.google.com/drive/1O-d7VsEqSoT4bAs4J2i0PI-4NU0PXZkG)

### 🧰 Step 2: Install Required Libraries (if local)

```bash
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
