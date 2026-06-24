````markdown
# 🧠 Parkinson's Disease Prediction System

This is a **Machine Learning-based web application** built using **Streamlit** that predicts whether a person is likely to have **Parkinson's Disease** based on voice measurement features.

The application uses a trained Machine Learning model and provides both **single-patient prediction** and **bulk CSV prediction** capabilities through an interactive web interface.

This project is part of my **Data Science & Machine Learning learning journey**, where I applied concepts such as **data preprocessing, feature engineering, model training, model serialization, and deployment**.

---

## 📊 Features

✔️ **User-Friendly Interface** – Enter patient voice measurements through an intuitive dashboard.

✔️ **Machine Learning Prediction** – Predicts the likelihood of Parkinson's Disease using a trained model.

✔️ **Bulk CSV Prediction** – Upload CSV files and generate predictions for multiple records at once.

✔️ **Automatic Preprocessing** – Applies scaling automatically if a saved scaler is available.

✔️ **Download Results** – Export prediction results as a CSV file.

✔️ **Interactive Dashboard** – Built with Streamlit for real-time predictions.

✔️ **Cloud Deployment Ready** – Easily deployable on Streamlit Community Cloud.

---

## 📂 Repository Structure

```text
📦 ParkinsonDiseasePrediction
│── app.py
│── parkinsons_model.pkl
│── scaler.pkl
│── parkinsons.csv
│── requirements.txt
│── README.md
````

---

## ⚙️ Installation & Usage

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/abdulwasay8905/parkinsons-disease-prediction.git
cd parkinsons-disease-prediction
```

### 🔹 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 3: Run the Application

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

## 📈 Input Features

The model uses the following 22 biomedical voice measurement features:

* MDVP:Fo(Hz)
* MDVP:Fhi(Hz)
* MDVP:Flo(Hz)
* MDVP:Jitter(%)
* MDVP:Jitter(Abs)
* MDVP:RAP
* MDVP:PPQ
* Jitter:DDP
* MDVP:Shimmer
* MDVP:Shimmer(dB)
* Shimmer:APQ3
* Shimmer:APQ5
* MDVP:APQ
* Shimmer:DDA
* NHR
* HNR
* RPDE
* DFA
* spread1
* spread2
* D2
* PPE

---

## 🎯 Prediction Output

```text
0 = Healthy
1 = Parkinson's Disease
```

---

## 📂 Bulk CSV Prediction

Upload a CSV file containing all 22 required feature columns.

The application will:

* Process all records automatically
* Generate predictions
* Preserve additional columns such as Patient ID
* Allow downloading results as CSV

---

## 🌍 Deployment on Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select the repository.
5. Set `app.py` as the entry file.
6. Deploy and share the public URL.

---

## 🧑‍💻 Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Streamlit
* Joblib / Pickle

---

## 📸 Demo

Add screenshots of your Streamlit application here.

```markdown
![App Screenshot](screenshot.png)
```

---

## 👨‍💻 Author

**Abdul Wasay**

### 📬 Connect with Me

* GitHub: [https://github.com/abdulwasay8905](https://github.com/abdulwasay8905)
* LinkedIn: [https://www.linkedin.com/in/abdul-wasay-2a602329b](https://www.linkedin.com/in/abdul-wasay-2a602329b)

---

✨ This project is part of my **Machine Learning and Data Science portfolio**. It demonstrates the complete workflow of building, training, deploying, and sharing a healthcare-focused predictive analytics application.

```
```
