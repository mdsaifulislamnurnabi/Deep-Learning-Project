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

**MovieLens 100k** \
📎 Download: [http://files.grouplens.org/datasets/movielens/ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)

Contains:
- `u.data`: 100,000 ratings from 943 users on 1682 movies.
- `u.item`: Movie details including title and genres.
- `u.info`: Summary of users, movies, and ratings.

---

## 🧪 Technologies Used

- Python 3.10+
- Pandas & NumPy
- Matplotlib
- scikit-learn (LabelEncoder, train_test_split)
- TensorFlow / Keras

---

## 🏗️ Model Architecture

- 2 **Embedding layers** (User and Movie)
- Dense layers with ReLU activations
- Final **Softmax layer** with 9 output categories (for normalized ratings)
- Optimizer: Adam
- Loss: SparseCategoricalCrossentropy

---

## 🔍 Workflow Overview

```mermaid
graph TD
A[Load Dataset] --> B[Merge Ratings & Titles]
B --> C[Average multiple ratings]
C --> D[Encode users & movies]
D --> E[Train-Test Split]
E --> F[Build Neural Network]
F --> G[Train Model]
G --> H[Predict ratings for unseen movies]
H --> I[Sort and Recommend Top-N Movies]


🧪 Model Training
Epochs: 70

Batch Size: 128

Dropout: 5% to prevent overfitting

Accuracy & Loss monitored on validation set

history = model.fit(
    x = X_train_array, y = y_train,
    validation_data=(X_test_array, y_test),
    epochs=70, batch_size=128, shuffle=True
)


🔍 Prediction Function

def recommender_system(user_id, model, n_movies):
    encoded_user = user_enc.transform([user_id])
    seen = refined_dataset[refined_dataset['user id'] == user_id]['movie']
    unseen = [i for i in range(n_movies_total) if i not in seen]
    model_input = [np.asarray([encoded_user[0]] * len(unseen)), np.asarray(unseen)]
    predictions = model.predict(model_input)
    predicted_ratings = np.max(predictions, axis=1)
    sorted_indices = np.argsort(predicted_ratings)[::-1]
    return item_enc.inverse_transform(sorted_indices[:n_movies])
📈 Sample Output
markdown

Top 10 Movie recommendations for user 777:
1. Contact (1997)
2. Return of the Jedi (1983)
3. L.A. Confidential (1997)
...
🧠 Future Improvements
Use Transformer models (e.g., BERT, Attention layers)

Incorporate timestamp-based temporal analysis

Evaluate using RMSE or MSE for continuous ratings

Add user demographics or genres as extra features

🙋‍♂️ Author
Mohammad Nurnabi
CSE, North East University Bangladesh
ID: 0562210005101011

📜 License
This project is for academic and research purposes. Not for commercial use.

yaml
Copy
Edit

---

🔹 **Tips:**
- Save the above content in a file named `README.md`
- If using GitHub, simply place it in the root folder of your repository
- Add screenshots, model architecture diagram, or colab notebook links as needed
