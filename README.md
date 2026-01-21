# BCS_PIPELINE
BCS Classification of Drugs - Capstone batch -80

📘 BCS Classification Pipeline – README
🔬 Overview

This repository contains a complete machine learning pipeline designed to classify drug compounds into the Biopharmaceutics Classification System (BCS) using both physicochemical properties and molecular representations derived from SMILES strings. The project integrates rule-based classification, supervised learning models, and advanced graph-based molecular featurization to support early-stage drug development and formulation research.

The system processes user-provided drug attributes and SMILES, extracts molecular features, predicts solubility/permeability behavior, and assigns a BCS class with explainability. A web interface enables interactive predictions.

## 🚀 Features
 1. Data Processing & Featurization

    Extraction of Solubility (logS), AlogP, and Permeability Index from dataset.

    Molecular graph construction via DeepChem MolGraphConvFeaturizer.

    Conversion to PyTorch Geometric Data objects for GNN models.

    Handling missing values and normalization pipelines.

    Optional generation of:

    3D conformers using RDKit

2. Machine Learning Models

    Multiple ML algorithms were trained, tuned, and evaluated, including:

    LightGBM

    CatBoost

    Random Forest

    Graph Convolution Networks (GCN)

    GINE Conv (Graph Isomorphism Network with Edge Features)

    The best performance was achieved using a combination of classical models and graph features.

    Performance metrics evaluated include:

    Accuracy

    Precision, Recall, F1

    Cohen’s Kappa

    MCC

    Classification reports

    Confusion matrices

3. Web Application (Frontend + Backend)

    Built using React + Flask/Node backend (based on your code context)

    Accepts:

    Solubility

    AlogP

    Permeability Index

    SMILES

    Runs model inference

    Displays BCS class + explanation

    Includes:

    Error handling

    Model interpretation support

    Deployment-ready structure

📊 Results Summary

    The final trained models achieved strong predictive performance.

    LightGBM delivered highest classical accuracy with R² = 0.851.

    GCN captured structural SMILES patterns with R² = 0.71.

    CatBoost achieved robust multiclass performance with balanced precision/recall.

    Combined framework provides reliable BCS prediction for real-world use cases.

## 👨‍💻 Developed By

HARI KIRAN K - PES2UG22CS212
HARI SHANKAR - PESUGEECS213
KRUPA S      - PES2UG22CS272
RHUSHYA K C  - PES2UG22CS440
Dept. of Computer Science 