
# 🧬 BCS Drug Classification Pipeline

## 📘 Overview

This project implements an end-to-end machine learning pipeline for automated classification of drug compounds into the Biopharmaceutics Classification System (BCS).

The system combines physicochemical drug properties and molecular structure information derived from SMILES strings to predict BCS classes, supporting early-stage drug development and formulation research.

The application follows a multi-service architecture, integrating:

* Python-based machine learning inference
* Explainable AI (SHAP)
* Node.js backend API
* React-based frontend interface

The entire pipeline was successfully executed locally.

---

## 🎯 Objectives

* Automate BCS classification of drug compounds
* Utilize molecular descriptors and SMILES-based features
* Compare classical ML models for multiclass prediction
* Provide model interpretability using Explainable AI
* Build a deployable full-stack ML application

---

## ⚙️ System Architecture

```
Frontend (React)
        ↓
Node.js Backend (API Gateway)
        ↓
Python ML Service (FastAPI)
        ↓
ML Models + SHAP Explanation
```

---

## 🔬 Key Features

### 1. Molecular & Physicochemical Processing

* Input parameters:

  * Solubility (logS)
  * AlogP
  * Permeability Index
  * SMILES string
* Molecular parsing using RDKit
* Feature preprocessing and normalization
* Pre-trained feature pipelines loaded from artifacts

---

### 2. Machine Learning Models

* LightGBM classifier
* Random Forest classifier
* CatBoost classifier

Models were trained and saved using pickle for efficient inference.

Evaluation techniques included:

* Accuracy
* Precision, Recall, F1-score
* Confusion matrix analysis
* Multiclass performance comparison

---

### 3. Explainable AI

* Integrated SHAP (SHapley Additive exPlanations)
* Provides feature-level contribution explanations
* Improves transparency of BCS predictions

---

### 4. Backend Services

#### Python ML Service

* Built using FastAPI
* Loads trained models at runtime
* Performs prediction and explanation
* Runs on local API server

#### Node.js Backend

* Acts as middleware between frontend and ML service
* Handles request routing and timeout management
* Environment-based configuration

---

### 5. Frontend Interface

* Developed using React (Vite)
* User-friendly input interface
* Sends drug properties and SMILES for prediction
* Displays predicted BCS class and results

---

## 🧪 Technologies Used

* Python 3.10
* FastAPI
* Uvicorn
* Scikit-learn
* LightGBM
* CatBoost
* RDKit
* SHAP
* Node.js
* React (Vite)

---

## ▶️ How to Run Locally

### Prerequisites

* Python 3.10.x
* Node.js (LTS)
* Git

---

### Step 1: Clone repository

```bash
git clone https://github.com/Krupa994/BCS_PIPELINE.git
cd BCS_PIPELINE
```

---

### Step 2: Start Python ML Service

```bash
py -3.10 -m venv venv
venv\Scripts\activate
cd ml_services
pip install fastapi uvicorn shap rdkit-pypi catboost lightgbm scikit-learn pandas numpy
uvicorn app:app --reload
```

ML service runs at:

```
http://127.0.0.1:8000
```

Swagger API:

```
http://127.0.0.1:8000/docs
```

---

### Step 3: Start Node Backend

Open a new terminal:

```bash
cd backend
npm install
npm start
```

Backend runs at:

```
http://localhost:5000
```

---

### Step 4: Start Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

## 📊 Output

* Predicted BCS Class (I–IV)
* Model inference results
* Explainability insights (SHAP-based)
* Interactive UI output

---

## 🎓 Academic Significance

This project demonstrates:

* Practical application of machine learning in pharmaceutical sciences
* Integration of molecular informatics with ML
* Explainable AI for healthcare decision systems
* Full-stack ML system design

---

## 👩‍💻 Author

Krupa S  

This project was developed as part of a final-year capstone project.
B.Tech Computer Science and Engineering  
PES University

---

## 📌 Notes

* Model inference time may vary due to RDKit molecular processing and SHAP explanation.
* Timeout values are configurable in the Node backend.
* Project executed and tested locally on Windows environment.

---
