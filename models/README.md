# 🤖 AI Candidate Success Predictor

An **AI-powered employee attrition prediction system** built using **Machine Learning and Streamlit**.
This application analyzes employee attributes and predicts whether an employee is **likely to stay in the company or leave**.

The system provides **interactive dashboards, model performance analysis, and live prediction capabilities** for HR analytics.

---

# 📊 Project Overview

Employee attrition is a major challenge for organizations. This project uses **machine learning algorithms** to analyze employee data and predict potential attrition risks.

The system helps HR teams:

* Identify employees at **high risk of leaving**
* Analyze **employee behavior patterns**
* Improve **employee retention strategies**

---

# 🚀 Features

✔ Interactive **Streamlit dashboard**
✔ **Multiple Machine Learning models**
✔ **Live employee attrition prediction**
✔ Model performance **evaluation metrics**
✔ **Cross validation analysis**
✔ **Confusion matrix visualization**
✔ **Probability based prediction system**
✔ Dataset analytics and visualizations

---

# 🧠 Machine Learning Models Used

The project evaluates multiple ML models to compare performance:

### 1️⃣ Logistic Regression

Best suited for binary classification problems.

### 2️⃣ K-Nearest Neighbors (KNN)

Classifies employees based on similarity with neighboring data points.

### 3️⃣ Linear Regression

Used as an experimental model and converted into classification using a threshold.

---

# 📈 Model Evaluation Metrics

The system evaluates model performance using:

* Accuracy Score
* F1 Score
* Mean Squared Error (MSE)
* Confusion Matrix
* Classification Report
* Cross Validation

These metrics help determine the **most reliable model for attrition prediction**.

---

# 🖥 Application Dashboard

The Streamlit application contains several modules:

### 🏠 Home

Project introduction and overview.

### 📂 Dataset

Displays dataset statistics and visualizations.

### 🔮 Prediction

Shows model predictions and classification results.

### ⚡ Live Prediction

Allows users to enter employee information and predict attrition risk.

### 📊 Model Accuracy

Displays evaluation metrics and performance graphs.

### 📞 Contact

Project information and developer details.

---

# 📊 Example Prediction Output

The system predicts:

```
0 → Employee Stay
1 → Employee Leave
```

Example output:

| Prediction | Meaning        |
| ---------- | -------------- |
| 0          | Employee Stay  |
| 1          | Employee Leave |

---

# 🛠 Technology Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**

---

# 📁 Project Structure

```
ai_candidate_success_predictor
│
├── app.py
├── resume_dataset.csv
├── req.txt
├── README.md
│
├── models
│   └── train_model.py
│
├── utils
│   └── metrics.py
│
└── assets
    ├── home.jpg
    ├── prediction.jpg
    ├── accuracy.jpg
    └── contact.jpg
```

---

# ⚙ Installation

Clone the repository:

```
git clone https://github.com/22AD040/ai-candidate-success-predictor.git
```

Move into project folder:

```
cd ai-candidate-success-predictor
```

Install dependencies:

```
pip install -r req.txt
```

---

# ▶ Running the Application

Start the Streamlit app:

```
streamlit run app.py
```

The application will open in your browser:

```
http://localhost:8501
```

---

# 📌 Use Cases

This system can help:

* HR analytics teams
* Workforce management
* Employee retention analysis
* Organizational behavior research

---

# 👩‍💻 Developer

**Ratchita B**

AI & Machine Learning Enthusiast
Passionate about building intelligent systems and data-driven solutions.

---

# 📜 License

This project is created for **educational and research purposes**.

---

⭐ If you find this project useful, feel free to **star the repository**.
