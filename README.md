# Hiring-Bias-Detection-System-XAI

Explainable AI project for detecting bias in automated hiring systems using SHAP and LIME.

## Overview

This project investigates bias in AI-powered hiring systems using Explainable Artificial Intelligence (XAI) techniques such as SHAP and LIME.

The system predicts whether a candidate should be hired or rejected based on historical recruitment data while analyzing fairness, transparency, and proxy discrimination.

---

## Project Links

### Documentation

📄 Project Report: [View Documentation](./reports/hiring_bias_detection_system.docx)

### YouTube Demo

🎥 Demo Video: [Watch Project Demo](https://youtu.be/WbjrpYtUa4U)

### Live Application

🚀 Streamlit App: [Open Application](PASTE_YOUR_STREAMLIT_LINK_HERE)

---

## Objectives

* Build a hiring prediction model
* Detect gender bias in recruitment decisions
* Explain predictions using SHAP and LIME
* Evaluate fairness metrics
* Deploy a simple Streamlit application

---

## Features

* Predict candidate hiring outcomes (Hire / Reject)
* Random Forest classification model
* SHAP global explainability analysis
* LIME local explainability analysis
* Bias and fairness detection
* Interactive Streamlit web application
* AI-based transparency support

---

## Technologies Used

* Python
* Scikit-learn
* SHAP
* LIME
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Joblib

---

## Project Structure

```text
Hiring-Bias-Detection-System-XAI/
│
├── app/
│   ├── app.py
│   ├── hiring_rf_model.pkl
│   └── X_train.csv
│
├── data/
│   └── raw/
│       └── hiring_dataset.csv
│
├── images/
│   ├── shap_global_summary.png
│   └── lime_local_explanation.png
│
├── notebooks/
│   └── HIRING_BIAS_DETECTION_SYSTEM.ipynb
│
├── reports/
│   ├── final_report.docx
│   └── presentation.pptx
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How To Run The Project

Clone the repository:

```bash
git clone PASTE_REPOSITORY_LINK_HERE
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

---

## Explainability Methods

### SHAP

Used for global explanations by identifying how features affect model behavior across the entire dataset.

### LIME

Used for local explanations by showing why individual hiring decisions were made.

---

## Project Focus

This project focuses on fairness, transparency, and responsible AI by examining whether automated hiring systems introduce hidden bias or proxy discrimination.

---

## Group Members

* RITA JONATHAN - VUG/CSC/24/13208
* NEGEDU EMMANUEL OJODOMO - VUG/CSC/23/10588
* KWAPYONG MATTHEW MANTOE - VUG/CSC/23/8829
* AMAEFULE VICTOR CHUKWUEBUKA - VUG/CSC/23/10215
* ABDULJALAL ISHABA -VUG/CSC/23/9145
* JUDE CHIDERA JUDE - VUG/CSC/24/13106
* ADAMA JARREN  OJOCHIDE - VUG/CSC/23/9961
* RAPHAEL CHINEDU OKECHUKWU - VUG/CSC/23/9222
* OSHIBOTE ADEBUSAYO ESTHER - VUG/CSC/23/10043
* WANI PAUL WISDOM - VUG/CSC/23/9826
* OKORONKWO CHIDOZIE PETER  - VUG/CSC/23/10614
* SOLOMON VICTOR ENEIRAMO - VUG/CSC/23/8942
* OLUEHI DELIGHT - VUG/CSC/23/9564
* EZE FAVOUR - VUG/CSC/23/9414
* OKECHUKWU-PRINCE JOSEPH CHUKWUEME - VUG/CSC/23/9339
* AHMED MUTAWAKEEL - VUG/CSC/23/10262
* IJEBUWA-NWOKOH KAMSI KENECHI - VUG/CSC/23/9427
* ISIORO OGHENEKOME - VUG/CSC/24/13067
* ELISHA MUSA NAMAN - VUG/CSC/24/13279
* ANTHONY VICTOR ENOMHEN - VUG/CSC/23/8819
* NWACHUKWU UGOCHUKWU COLLIN - VUG/CSC/23/10710
* LUCKY COVENANT - VUG/CSC/23/9672
* EMEFIELE DIVINEFAVOUR - VUG/CSC/23/8985
* ANONO DEBORAH - VUG/CSC/23/9870
* ESSIET ENOBONG VICTOR - VUG/CSC/23/10556
* OBIOKALA ODIAKAOSE PATRICK - VUG/CSC/23/8937
