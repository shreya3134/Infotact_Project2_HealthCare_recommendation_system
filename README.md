
# 🩺 Healthcare Recommendation System

This project is a **machine learning-based healthcare assistant** that predicts diseases based on user symptoms and recommends appropriate treatments, including medicines, exercises, recovery period, and blood pressure categorization.

---

## 🎯 Project Goals

- Predict the most likely **disease** based on 6 user-selected symptoms.
- Recommend **treatments** (medicines and exercises) based on disease, age, vitals (heart rate, oxygen level), blood pressure, and past medical history.
- Classify **blood pressure** levels (Low, Normal, High).
- Estimate **recovery days** using regression.

---

## 🛠 Technologies Used

- **Python, Flask** – Backend web framework
- **HTML/CSS + Bootstrap** – Frontend templates
- **scikit-learn** – ML models: RandomForestClassifier, MultiOutputClassifier, RandomForestRegressor
- **Pandas, NumPy** – Data manipulation
- **Pickle/Joblib** – Model and encoder serialization
- **TfidfVectorizer** – For processing text features

---

## 🧠 Machine Learning Models

- `pred.pkl`: Predicts disease from encoded symptoms  
- `clf_pipeline.pkl`: Classifies recommended medicines and exercises  
- `reg_pipeline.pkl`: Predicts number of recovery days  
- `disease_encoder.pkl`: Encodes predicted disease  
- `symptom_encoders.pkl`: Encodes each symptom column  

---

## 📁 Dataset

- `detailed_exercise_disease_dataset.csv`  
  Includes columns:
  - Disease, Age, Heart_Beats, Oxygen_Level, Past_History
  - BP_Level (e.g., High, Low, Normal)
  - Recovery_Days, Medicine_1 to Medicine_4
  - Exercise 1 to Exercise 3

---

## 🌐 Web Application Structure

### Routes:

- `/signup` – User registration  
- `/login` – Secure login page  
- `/home` – Select symptoms to predict disease  
- `/predict` – Disease prediction (AJAX/JSON POST)  
- `/treatment` – Input vital signs and get recommendations  
- `/result` – View predicted treatment plan  
- `/diseases` – List of supported diseases  

---

## ✅ How to Run

### Install dependencies
```bash
pip install flask scikit-learn pandas numpy
