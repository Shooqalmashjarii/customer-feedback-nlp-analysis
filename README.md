# 💬 NLP Sentiment Analysis — Feedforward Neural Network

A complete NLP pipeline that classifies customer feedback as **positive or negative** using TF-IDF vectorisation and a feedforward neural network — built from scratch in Python.

---

## 📌 Overview

This project implements an end-to-end text classification system, from raw text preprocessing through to neural network training and evaluation. The focus is on understanding how text can be transformed into numerical representations that a neural network can learn from.

---

## 🧠 Pipeline

```
Raw Customer Feedback (CSV)
        ↓
Text Preprocessing (lowercase, punctuation removal, stopword removal)
        ↓
TF-IDF Vectorisation  ←── builds sparse feature matrix
        ↓
Train/Test Split (80/20)
        ↓
Feedforward Neural Network (20 epochs)
        ↓
Accuracy Evaluation + VADER Inspection
```

---

## 🏗️ Neural Network Architecture

| Layer | Details |
|---|---|
| Input | TF-IDF feature vector (auto-sized) |
| Hidden | Dense(16), ReLU activation |
| Output | Dense(1), Sigmoid activation |
| Loss | Binary Crossentropy |
| Optimizer | Adam |
| Epochs | 20, Batch size 4 |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** `pandas`, `numpy`, `nltk`, `scikit-learn`, `tensorflow`
- **Environment:** Jupyter Notebook

---

## 📁 Project Structure

```
nlp-sentiment-analysis/
├── task3.ipynb            # Full pipeline notebook
├── feedback_dataset.csv   # Labelled customer feedback dataset
└── README.md
```

---

## 🚀 Getting Started

```bash
pip install pandas numpy nltk scikit-learn tensorflow
jupyter notebook task3.ipynb
```

Dataset requires two columns: `cleaned` (preprocessed text) and `label` (0 or 1).

---

## 💡 Key Learnings

- How TF-IDF captures word importance across a corpus
- Designing and training a neural network for binary text classification
- Iterative evaluation — tracking training vs. validation accuracy to detect overfitting

---

## 👩‍💻 Author

**Shooq Al Mashjari** — Computer Engineering Student, Khalifa University
[LinkedIn](http://www.linkedin.com/in/shooq-al-mashjari-521b85353)
