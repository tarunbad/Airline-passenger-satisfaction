
# Airline Passenger Satisfaction Analysis

**Course**: CSE 4/587 Spring 2025
**Project**: Phases 1–4 of the Data Science Pipeline

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
4. [Usage](#usage)

   * [Phase 1 – Data Collection & Cleaning](#phase-1-–-data-collection--cleaning)
   * [Phase 2 – Modeling & Evaluation](#phase-2-–-modeling--evaluation)
   * [Phase 3 – Distributed Processing with PySpark](#phase-3-–-distributed-processing-with-pyspark)
   * [Phase 4 – Data Product & Streamlit App](#phase-4-–-data-product--streamlit-app)
5. [Results & Key Findings](#results--key-findings)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Overview

This repository contains the end-to-end project for predicting airline passenger satisfaction, broken down into four phases aligned with standard data science workflows:

1. **Phase 1 (Data Collection & Cleaning)** – Exploratory Data Analysis (EDA), cleaning, and feature engineering in Jupyter Notebooks.
2. **Phase 2 (Modeling & Evaluation)** – Building and tuning multiple classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost, etc.), comparing performance, and saving pipelines.
3. **Phase 3 (Distributed Processing with PySpark)** – Scaling preprocessing and model training on large datasets using Apache Spark ML.
4. **Phase 4 (Data Product & Streamlit App)** – Wrapping up models in an interactive Streamlit application for end‐user predictions and visual insights.

## Repository Structure

```
├── data/
│   └── sample_airline_data.csv      # Sample CSV for quick testing
├── models/
│   └── phase2/                      # Saved trained pipelines and scaler
│       ├── logistic_pipeline.pkl
│       ├── decision_tree_pipeline.pkl
│       └── ...
├── src/
│   ├── phase1/                      # Notebooks & scripts for EDA & cleaning
│   ├── phase2/                      # Training scripts & notebooks
│   ├── phase3/                      # PySpark notebooks & scripts
│   └── phase4/                      # Streamlit app (`app.py`)
├── notebooks/                       # Consolidated Jupyter notebooks
│   ├── Phase1_EDA.ipynb
│   └── Phase2_Modeling.ipynb
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Getting Started

### Prerequisites

* Python 3.8+
* Java 8+ (required for PySpark)
* Git

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/airline-passenger-satisfaction.git
cd airline-passenger-satisfaction

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Phase 1 – Data Collection & Cleaning

```bash
# Launch Jupyter and explore data cleaning steps:
jupyter notebook notebooks/Phase1_EDA.ipynb
```

### Phase 2 – Modeling & Evaluation

```bash
# Run the training notebook:
jupyter notebook notebooks/Phase2_Modeling.ipynb

# Or execute training script to save pipelines:
python src/phase2/train_models.py
```

### Phase 3 – Distributed Processing with PySpark

```bash
# Start a PySpark shell or execute notebook:
jupyter notebook src/phase3/Phase3_Spark.ipynb
# Or run as script:
spark-submit src/phase3/spark_phase3.py
```

### Phase 4 – Data Product & Streamlit App

```bash
# From the project root:
streamlit run src/phase4/app.py
```

* Open `http://localhost:8501` to access the interactive app.
* Use the Data Input tab to make single or batch predictions.

---

## Results & Key Findings

* **Top Predictor**: Inflight Entertainment showed the highest correlation with satisfaction (\~0.72).
* **Best Models**: Tree‐based ensembles (Random Forest, XGBoost, Stacking) achieved ≥95% accuracy.
* **Recommendations**: Focus on entertainment, seat comfort, Wi-Fi, and cleanliness to boost satisfaction.

---

## Contributing

Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
