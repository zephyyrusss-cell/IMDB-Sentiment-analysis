# IMDB-Sentiment-analysis

Hey guys 👋 this is Sukanth, and this is my IMDB movie review sentiment prediction model.  
I used **Logistic Regression** and **Linear SVM** to classify reviews as positive or negative.  
---

## Steps I followed

### 1. Preprocessing
- The dataset contains raw IMDB reviews with a sentiment label (positive / negative).
- Before using it for training, I cleaned the reviews:
  - Converted all text to lowercase (so “Movie” and “movie” are treated the same)
  - Removed numbers, punctuation, and special characters using regex
  - Kept only alphabetic words
- This makes the input consistent and reduces noise, so the model focuses only on meaningful words.

### 2. Feature Extraction (TF-IDF)
- After cleaning, the text is converted into numerical vectors (ML models can't read raw text).
- I used **TF-IDF (Term Frequency – Inverse Document Frequency)** to measure word importance:
  - **Term Frequency** → how often a word appears in a review
  - **Inverse Document Frequency** → reduces the weight of very common words
- TF-IDF settings I used:
  - `max_features=5000` → keep only the top 5000 most important words/phrases
  - `stop_words='english'` → remove very common words like “the”, “is”, “and”
  - `ngram_range=(1,2)` → use both single words (“good”) and pairs (“not good”)
- Result: Each review is transformed into a vector of length 5000, where each number represents the importance of a word/phrase.

### 3. Train/Test Split
- Split the dataset into:
  - 80% for training → used to teach the model
  - 20% for testing → used to evaluate generalization
- Used `stratify=y` to maintain the same proportion of positive/negative reviews in both sets
- Ensures balanced data and avoids bias toward one class.

### 4. Models
I trained **two models** on the TF-IDF features:

#### a) Logistic Regression
- Treats the task as a probability problem
- Each word/phrase is assigned a weight → positive weight pushes toward positive sentiment, negative toward negative
- Outputs probabilities (e.g., “this review has a 90% chance of being positive”)

#### b) Linear SVM (Support Vector Machine)
- Finds the best boundary (hyperplane) that separates positive from negative reviews
- Only the most important reviews (**support vectors**) determine this boundary
- Robust for high-dimensional text data

### 5. Evaluation
- Compared the models using multiple metrics:
  - **Accuracy** → how many reviews were classified correctly overall
  - **Precision** → of predicted positives, how many were actually positive
  - **Recall** → of all actual positives, how many did the model find
  - **F1-score** → balance between precision and recall
- Visualized results:
  - **Confusion Matrix** → true positives, true negatives, false positives, false negatives
  - **Learning Curves** → how training and validation accuracy change as more data is added

---

## Results
- **Logistic Regression** → ~89% accuracy
- **Linear SVM** → ~88% accuracy

Both performed closely, which is expected because:
- Both are linear models
- TF-IDF produces high-dimensional sparse vectors (linear models handle these well)
