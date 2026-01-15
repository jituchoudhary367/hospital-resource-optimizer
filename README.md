# 🏥 Hospital Resource Optimization — MLOps Project

An end-to-end Machine Learning and MLOps system for predicting and optimizing hospital resource utilization using structured operational and clinical data.  
This project demonstrates the complete ML lifecycle: data processing, feature engineering, model training, experiment tracking, and deployment-ready inference.

---

## 🚀 Project Highlights

- End-to-end ML pipeline (data → features → model → evaluation)
- Modular and reproducible training pipeline
- Streamlit-based inference application
- Clean project structure following MLOps best practices

---

## 🧠 Problem Statement

Hospitals must efficiently manage limited resources such as ICU beds, ventilators, and medical staff.  
Inaccurate forecasting leads to overcrowding, delayed treatment, and inefficient scheduling.

This system predicts future resource demand from historical data to support data-driven capacity planning and operational decisions.

---

## 🏗️ System Architecture

### Training Flow

Data → Feature Engineering → Model Training → Evaluation 

### Inference Flow

User Input → Feature Processing → Trained Model → Prediction Output

Training and inference pipelines are separated to support scalable deployment.

---

## 📁 Project Structure

```bash
hospital-mlops/
│
├── app/ # Streamlit inference application
│ └── app.py
│
├── src/
│ ├── components/ # Data ingestion, transformation, training modules
│ ├── features/ # Feature engineering logic
│ ├── pipeline/ # Training pipeline orchestration
│ ├── monitoring/ # Drift & performance monitoring hooks
│
├── data/ # Raw and processed datasets
├── notebooks/ # EDA and experimentation
├── requirements.txt
├── .gitignore
└── README.md
```


---

## 🛠️ Tech Stack

- Python 3.10+
- Scikit-learn
- Pandas, NumPy
- MLflow (experiment tracking & model registry)
- Streamlit (inference UI)
- Joblib (local pipeline serialization)

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/jituchoudhary367/hospital-mlops.git
cd hospital-mlops
```

2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

4. Install Dependencies
```bash
pip install -r requirements.txt
```

🧪 Run Training Pipeline

This step loads data, builds features, trains the model, and logs experiments to MLflow.
```bash
python src/pipeline/train_pipeline.py
```

🖥️ Run Inference Application

After training:
```bash
streamlit run app/app.py
```

Enter hospital parameters to receive real-time predictions.



## 👨‍💻 Author

**Jitendra Choudhary**  
Computer Science Engineering Student  
Focused on Machine Learning, AI Engineering, SAP ABAP & UI  

🔗 GitHub: [jituchoudhary367](https://github.com/jituchoudhary367)
