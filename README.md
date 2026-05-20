# 🎬 Movie Rating Categorization
### A Data Mining Project using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=flat&logo=kaggle)
![XGBoost](https://img.shields.io/badge/XGBoost-66.30%25_Accuracy-brightgreen?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 Project Overview

This project applies **Data Mining and Machine Learning** techniques to automatically categorize movie ratings into three classes — **Low**, **Medium**, and **High** — using the MovieLens dataset.

Rather than manually reviewing movies, the goal is to train a machine learning model that understands **human rating behaviour at scale** — the same core idea behind recommendation systems used by Netflix, Amazon Prime, and Hotstar.

---

## 🎯 Objective

> **Can we predict whether a user will rate a movie as Low, Medium, or High — before they watch it?**

- **Low** → Rating ≤ 2.5 (bad movie experience)
- **Medium** → Rating 3.0 – 3.5 (average movie experience)
- **High** → Rating ≥ 4.0 (great movie experience)

---

## 📂 Project Structure

```
movie-rating-categorization/
│
├── notebook/
│   └── movie_rating_categorization.ipynb   ← Full Kaggle notebook
│
├── charts/
│   ├── chart1_rating_distribution.png
│   ├── chart2_top_movies.png
│   ├── chart3_genre_analysis.png
│   ├── chart4_genre_avg_rating.png
│   ├── chart5_model_comparison.png
│   ├── chart6_confusion_matrix.png
│   ├── chart7_feature_importance.png
│   └── final_summary_chart.png
│
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [MovieLens Latest Small — Kaggle](https://www.kaggle.com/datasets/sriharshabsprasad/movielens-dataset-100k-ratings) |
| Total Ratings | 100,836 |
| Unique Movies | 9,724 |
| Unique Users | 610 |
| Rating Scale | 0.5 to 5.0 (half-star increments) |
| Files Used | `ratings.csv`, `movies.csv`, `tags.csv` |

---

## 🔄 Project Pipeline

```
Data Collection → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation
```

### Step 1 — Data Collection & Loading
- Loaded MovieLens dataset from Kaggle
- Merged ratings, movies, and tags into one unified dataframe
- Confirmed zero missing values across all columns

### Step 2 — Exploratory Data Analysis (EDA)
- Visualised rating distribution across all 10 possible values
- Identified top 10 most-rated movies (Forrest Gump leads with 329 ratings)
- Analysed rating counts and average ratings across 19 genres
- Key finding: **4.0 is the most common rating** — users tend to rate positively

### Step 3 — Preprocessing & Feature Engineering
- Created rating category labels (Low / Medium / High)
- Extracted **release year** from movie title strings
- Encoded 19 genres into binary columns (1 = belongs to genre, 0 = does not)
- Engineered **movie-level stats**: average rating, rating count, rating std
- Engineered **user-level stats**: average rating, rating count, rating std
- Final feature matrix: **26 features**
- Train/Test split: **80% training (80,668 rows) / 20% testing (20,168 rows)**

### Step 4 — Model Training & Comparison
- Trained 4 different ML models and compared their performance
- Used stratified split to maintain class balance

### Step 5 — Evaluation & Insights
- Evaluated using Accuracy, Precision, Recall, F1-Score
- Generated confusion matrix for the best model
- Extracted feature importances to understand what drives predictions

---

## 🤖 Model Results

| Model | Accuracy | Training Time | Notes |
|---|---|---|---|
| Naive Bayes | 52.20% | 0.0 sec | Too simple for this problem |
| Logistic Regression | 63.94% | 0.6 sec | Good linear baseline |
| Random Forest | 65.61% | 4.3 sec | Strong ensemble model |
| **XGBoost** | **66.30%** | **3.4 sec** | **🏆 Best Model** |

### Detailed XGBoost Report

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Low | 0.68 | 0.50 | 0.57 | 3,815 |
| Medium | 0.58 | 0.52 | 0.54 | 6,637 |
| High | 0.70 | 0.83 | 0.76 | 9,716 |
| **Weighted Avg** | **0.66** | **0.66** | **0.65** | **20,168** |

---

## 🔍 Key Findings

### 1. Rating Behaviour
- **4.0 is the most common rating** — people rate generously
- **48.2%** of all ratings are High — positive bias exists in movie ratings
- Only **18.9%** of ratings are Low — users rarely rate movies poorly

### 2. Feature Importance (XGBoost)
| Rank | Feature | Importance | Meaning |
|---|---|---|---|
| 1 | `movie_avg_rating` | 27.7% | How the movie is rated overall |
| 2 | `user_avg_rating` | 15.3% | How generous/strict the user is |
| 3 | `movie_rating_std` | 6.9% | How consistent movie ratings are |
| 4 | `user_rating_std` | 5.2% | How varied the user's ratings are |
| 5 | `user_rating_count` | 2.9% | How active the user is |

> 💡 **Insight:** A user's personal rating habits matter **more** than what genre a movie belongs to. Who is rating is more predictive than what they are rating.

### 3. Genre Insights
- **Drama** is the most-rated genre with 41,928 ratings
- **Film-Noir** has the **highest average rating (3.92)** despite fewest ratings — niche but loved
- **Horror** has the **lowest average rating (3.26)** — most-watched but least-enjoyed
- **Comedy** is second lowest (3.38) despite being second most-watched genre

### 4. Model Behaviour
- **High ratings are easiest to predict** (82.9% recall) — clear signal in data
- **Medium ratings are hardest** — they sit at the boundary between Low and High
- This is a common and expected pattern in ordinal classification problems

---

## 📈 Visualisations

| Chart | Description |
|---|---|
| Rating Distribution | Bar chart of all 10 rating values + category pie chart |
| Top 10 Movies | Most-rated movies in the dataset |
| Ratings by Genre | Which genres get rated the most |
| Avg Rating by Genre | Which genres are rated the highest on average |
| Model Comparison | Accuracy comparison across all 4 models |
| Confusion Matrix | XGBoost prediction breakdown |
| Feature Importance | Top 15 predictive features |
| Final Summary | Combined dashboard of all key results |

---

## 🌍 Real-World Applications

This project demonstrates the core logic behind:

- **Recommendation Systems** — Predict if a user will love a movie before they watch it
- **Content Acquisition** — Help platforms decide which movies to buy/license
- **User Personalisation** — Tailor UI and messaging based on predicted rating behaviour
- **Rating Moderation** — Flag anomalous ratings that don't match predicted patterns

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | ML models, preprocessing, evaluation |
| XGBoost | Best-performing gradient boosting model |
| Matplotlib | Charts and visualisations |
| Seaborn | Statistical visualisations |
| Kaggle Notebooks | Cloud-based development environment |

---

## ⚙️ How to Reproduce

### Option 1 — Run on Kaggle (Recommended)
1. Open the [Kaggle Notebook](https://www.kaggle.com/code/yukeshs05/movie-rating-categorization)
2. Click **Copy & Edit**
3. Click **Run All**

### Option 2 — Run Locally
```bash
# Clone the repository
git clone https://github.com/YukeshS05/movie-rating-categorization.git
cd movie-rating-categorization

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebook/movie_rating_categorization.ipynb
```

### requirements.txt
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
jupyter
```

---

## 📉 Current Limitations & Future Scope

| Limitation | Planned Improvement |
|---|---|
| 66% accuracy — not production-ready | Add text review sentiment analysis (NLP) |
| No movie review text used | Apply TF-IDF / BERT on user reviews |
| Only predicts category, not exact rating | Switch to regression for exact score |
| Dataset is from 1990s–2000s era | Use a more recent, modern dataset |
| No real-time prediction interface | Deploy as a Streamlit web app |

---

## 👨‍💻 Author

**Yukesh S**
- 🌐 Kaggle: [kaggle.com/yukeshs05](https://www.kaggle.com/yukeshs05)
- 💼 LinkedIn: [www.linkedin.com/in/yukesh-s]
- 🐙 GitHub: [github.com/YukeshS05](https://github.com/YukeshS05)

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- [GroupLens Research](https://grouplens.org/) for the MovieLens dataset
- [Sriharsha B S Prasad](https://www.kaggle.com/sriharshabsprasad) for hosting the dataset on Kaggle
- [Kaggle](https://www.kaggle.com/) for the free cloud notebook environment

---

> *"The goal is not to review movies — it is to teach machines to understand human rating behaviour at scale."*
