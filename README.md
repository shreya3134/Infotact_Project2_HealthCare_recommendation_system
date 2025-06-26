# Infotact_Project2_HealthCare_recommendation_system

# Task Intelligence Suite

This repository contains two integrated machine learning projects aimed at optimizing task management:
1. **Task Prediction & Classification** using Deep Learning and Random Forests.
2. **Task Scheduling & On-Time Completion Prediction** using ensemble models and ranking logic.

---

## 🔍 1: Task Prediction & Classification

### 📌 Objective
To predict task names and recommend intelligent treatment (e.g., task rank or scheduling priority) based on features such as:
- Priority
- Status
- Risk Level
- Estimated and Actual Effort

### 🧠 Techniques Used
- **TensorFlow/Keras**: For deep learning-based task name prediction.
- **RandomForestClassifier**: For predicting task label and priority.
- **Label Encoding**: For converting categorical features to numerical format.
- **Confusion Matrix & Evaluation Metrics**: To assess prediction performance.

### 📁 Dataset
- Input CSV (`Data3.csv`) includes fields like:
  - Task Metadata: `Priority`, `Status`, `Risk Level`
  - Effort: `Estimated Effort (Hours)`, `Actual Effort (Hours)`
  - Label: `Task Name`

### 🏁 Output
- Predicted task name
- Evaluation plots (confusion matrix, classification report)
- Model saved as `.h5` and encoders as `.pkl`

---

## 🧩 2: Task Scheduling & On-Time Completion Prediction

### 📌 Objective
To determine:
- Whether a task will be completed on time
- Rank or order tasks based on importance for scheduling

### 🧠 Techniques Used
- **RandomForestClassifier / HistGradientBoostingClassifier**
- **SimpleImputer** for handling missing values
- **Feature Importance Plots**
- **Confusion Matrix, Accuracy, R² Score**

### 📁 Dataset
- Input CSV (`Book.csv` or `task_data.csv`) includes:
  - Categorical: `Category`, `Sub-Category`, `Priority`, `Status`, `Risk Level`
  - Time Features: `Estimated Effort (Hours)`, `Due Date`, `Last Updated`
  - Target Labels: `On_Time_Completion`, `Task_Rank`

### 🏁 Output
- Whether task is likely to be completed on time (Yes/No)
- Predicted task rank (for scheduling)
- Feature importance visualization
- Saved model and encoders (`model.pkl`, `preprocessor.pkl`)

---

## 🚀 How to Run

### 📦 Dependencies
```bash
pip install pandas scikit-learn matplotlib seaborn tensorflow joblib
