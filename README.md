# 🎓 Student Academic Performance Predictor

[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-success?logo=render)](https://student-academic-performance-500.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](#)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM%20%2F%20Scikit--Learn-orange?logo=scikit-learn)](#)

## 📌 Project Overview
This repository contains the source code and deployment configuration for the **Student Academic Performance Predictor**. It is a machine learning-powered web application designed to forecast a student's academic success based on their behavioral and educational metrics. 

By analyzing patterns in the 500-student dataset, this application serves as an educational data mining tool to identify key factors influencing student grades and to provide early performance predictions.

> **Live Application:** [View the deployed model here](https://student-academic-performance-500.onrender.com)

---

## 📊 Dataset Information
The machine learning model is trained on a comprehensive dataset of 500 students, capturing various demographic, behavioral, and academic features. Common predictors analyzed in this project include:
*   Study hours and classroom attendance
*   Previous academic scores and assessments
*   Extracurricular activities and daily routines

---

## ⚙️ Tech Stack
| Component | Technology Used |
| :--- | :--- |
| **Language** | Python 3 |
| **Web Framework**| Flask |
| **Machine Learning**| Scikit-Learn (SVM), Pandas, NumPy |
| **Frontend** | HTML5, CSS3 |
| **Deployment** | Render (PaaS) |

---

## 🚀 Local Installation & Setup
Follow these steps to run the application on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows use: venv\Scripts\activate
    source venv/bin/activate  
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```

5.  **Access the web app:**
    Open your browser and navigate to `http://localhost:5000` (or the specific port indicated in your terminal).

---

## 💡 Usage
1.  Navigate to the web interface via the live link or local host.
2.  Enter the required student metrics (e.g., comma-separated feature values) into the respective input fields.
3.  Click the **Predict** button.
4.  The application will process your inputs through the trained machine learning model and instantly display the predicted academic performance.

---

## 👨‍💻 Author
*   **Lucky Manish Bhatiya**
