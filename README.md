# 📧 Email Spam Classifier

An end-to-end **Email Spam Classification system** built using **Machine Learning and NLP**. The project trains and evaluates multiple ML models using both **Count Vectorizer** and **TF-IDF** representations, and exports the best-performing model as a deployable pickle file for integration into web application.

---

## 📂 Project Structure

```
EMAIL_SPAM_CLASSIFIER_PROJECT/
│
├── preprocessing.py             
├── streamlit_app.py            
├── vectorizer.pkl               
├── model.pkl                    
├── requirements.txt             
└── README.md
```

---

## 🧠 Dataset & Training Details

* **Task:** Binary Classification (Spam / Ham)

* **Text Vectorization Techniques Used:**

  * Count Vectorizer
  * TF-IDF Vectorizer

* **Models Evaluated:**

  * Support Vector Classifier (SVC)
  * K-Nearest Neighbors (KNN)
  * Multinomial Naive Bayes (NB)
  * Decision Tree (DT)
  * Logistic Regression (LR)
  * Random Forest (RF)
  * AdaBoost
  * Bagging Classifier (BgC)
  * Extra Trees Classifier (ETC)
  * Gradient Boosted Decision Trees (GBDT)
  * XGBoost (XGB)

* The dataset was split into **training and validation sets**, and all models were evaluated primarily on **accuracy and precision**, with special emphasis on **high precision for spam detection**.
---

## 📊 Model Performance Snapshot

The models were trained and evaluated using **two different text vectorization techniques**:

### 🔹 Count Vectorizer (CV)

| Model                  | Accuracy | Precision |
|------------------------|----------|-----------|
| Random Forest          | 97.19%   | 100.00%   |
| K-Nearest Neighbors    | 90.71%   | 100.00%   |
| Extra Trees Classifier | 97.67%   | 99.14%    |
| XGBoost                | 97.38%   | 97.43%    |
| Linear Regression      | 97.09%   | 97.36%    |

### 🔹 TF-IDF Vectorizer

| Model                  | Accuracy | Precision  |
|------------------------|----------|------------|
| Naive Bayes            | 97.09%   | 100.00%    |
| K-Nearest Neighbors    | 90.52%   | 100.00%    |
| Random Forest          | 97.38%   | 98.26%     |
| Support Vector         | 97.58%   | 97.47%     |
| Extra Trees Classifier | 97.48%   | 97.45%     |

---

📌 **Key Observations**

* **TF-IDF consistently improved precision** across most classifiers, which is critical for spam detection
* **Naive Bayes achieved stable and high precision under TF-IDF**
* Tree-based ensemble models performed well but are heavier for deployment
* Since the dataset is **class-imbalanced**, accuracy alone is misleading — precision is the decisive metric

---

## ✅ Final Model Selection

* **Vectorizer:** TF-IDF
* **Classifier:** Multinomial Naive Bayes


📌 **Why Multinomial Naive Bayes?**

* Highest precision for spam detection (minimizes false positives)
* Extremely fast training and inference
* Naturally suited for sparse, high-dimensional text features

This configuration provides an optimal balance between **performance, simplicity, and deployment readiness**.

---

### Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## 🛠 Tech Stack

* Python
* NLTK
* Scikit-learn
* XGBoost
* TF-IDF & Count Vectorizer
* WordCloud
* Pickle (model serialization)

---

## 📌 Key Highlights

* Compared multiple ML classifiers for SMS spam detection
* Used two different text vectorization techniques
* Applied full NLP preprocessing pipeline: lowercasing, tokenization, stopword removal, and Porter stemming
* Selected model based on precision-focused evaluation due to class imbalance

---

## 📄 License

This project is open-source and available for learning and experimentation.
