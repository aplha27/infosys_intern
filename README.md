# Automated Resume Screening and Interview Assistance

## Overview
This project leverages machine learning and natural language processing to streamline the hiring process. It includes modules for generating interview questions, exploratory data analysis (EDA), prediction of candidate selection, and resume screening. The system processes candidate resumes, transcripts, and job descriptions to derive features, perform similarity analysis, and build a predictive model to classify candidates as "select" or "reject."

The project utilizes data preprocessing, feature engineering, and machine learning techniques to make hiring decisions more efficient and interpretable.

---

## Features

### General Features:
1. **Automated Interview Question Generation**  
   - Uses the Together API and a job description to create unique, insightful interview questions tailored to specific roles.

2. **Exploratory Data Analysis (EDA)**  
   - Analyzes resume and interview data for trends and insights.  
   - Visualizes correlations, distributions, and decision-making patterns.

3. **Prediction Model**  
   - Utilizes an XGBoost model to predict candidate selection.  
   - Features include text similarity, keyword analysis, and decision encoding.

4. **Resume Screening**  
   - Employs TF-IDF and cosine similarity to match resumes with job descriptions.  
   - Categorizes candidates into match categories (e.g., Excellent Match, Strong Match).

5. **Email Integration**  
   - Sends the results via email for easy access and sharing.

### Resume and Transcript Processing Features:
- **Text Feature Extraction**: Analyze resumes and transcripts for word count, character count, sentence count, keyword density, and more.
- **Keyword Analysis**: Count technical, positive, and negative keywords in resumes and transcripts.
- **TF-IDF Similarity**: Compute cosine similarity scores between resumes, transcripts, and job descriptions.
- **Predictive Model**: Use an XGBoost classifier to predict hiring decisions based on engineered features.
- **SHAP Analysis**: Perform explainable AI analysis to understand feature importance.
- **Visualization**: Generate SHAP plots, partial dependence plots, and interaction effects for interpretability.

---

## Files

1. **data_generation.py**  
   - **Purpose**: Generates interview questions using the Together API.  
   - **Input**: Excel file containing candidate details and job descriptions.  
   - **Output**: Excel file with generated interview questions.  
   - **Key Dependencies**: `pandas`, `os`, `together`.

2. **EDA.py**  
   - **Purpose**: Performs exploratory data analysis on combined data.  
   - **Analysis Highlights**:  
     - Word counts, transcript lengths, and correlation matrices.  
     - Count plots and box plots for decision-making trends.  
   - **Key Dependencies**: `pandas`, `matplotlib`, `seaborn`.

3. **Prediction.py**  
   - **Purpose**: Predicts candidate selection using an XGBoost model.  
   - **Key Features**:  
     - Keyword overlap, TF-IDF-based text similarity.  
     - Feature engineering for resumes and transcripts.  
     - Sends email notifications with results.  
   - **Key Dependencies**: `pandas`, `pickle`, `smtplib`, `sklearn`.

4. **Resume_screener.py**  
   - **Purpose**: Screens resumes against job descriptions.  
   - **Features**:  
     - Calculates similarity scores using TF-IDF and cosine similarity.  
     - Categorizes candidates based on match quality.  
     - Visualizes match distributions and decision trends.  
   - **Key Dependencies**: `pandas`, `seaborn`, `matplotlib`.

5. **Training.py**  
   - **Purpose**: Main script for data processing, feature engineering, model training, and visualization.  
   - **Key Outputs**: SHAP plots, partial dependence plots, and interaction effects.  
   - **Dependencies**: `pandas`, `numpy`, `xgboost`, `shap`, `matplotlib`.

6. **cleaned_data.xlsx**  
   - Input data file containing resumes, transcripts, and job descriptions.

7. **tfidf_vectorizer.pkl**  
   - Saved TF-IDF vectorizer for reuse.

8. **xgb_model.pkl**  
   - Saved trained XGBoost model.

---

## Installation

1. Clone this repository.  
2. Install dependencies:  
   ```bash
   pip install pandas numpy matplotlib seaborn sklearn xgboost together-python shap
   ```
3. Ensure all required Excel files are in the appropriate directory.

---

## Usage

### Generate Interview Questions
1. Place candidate data in an Excel file (`prediction_data.xlsx`).
2. Run `data_generation.py`:  
   ```bash
   python data_generation.py
   ```

### Perform EDA
1. Place combined data in `combined_data.xlsx`.
2. Run `EDA.py`:  
   ```bash
   python EDA.py
   ```

### Predict Candidate Selection
1. Place prediction data in `prediction_data.xlsx`.
2. Ensure the XGBoost model (`xgb_model.pkl`) is available.
3. Run `Prediction.py`:  
   ```bash
   python Prediction.py
   ```

### Screen Resumes
1. Place cleaned data in `cleaned_data.xlsx`.
2. Run `Resume_screener.py`:  
   ```bash
   python Resume_screener.py
   ```

### Train Model
1. Use `Training.py` for data preprocessing, feature engineering, and model training.  
   ```bash
   python Training.py
   ```

---

## Outputs
- **Generated Questions**: Saved as Excel files.
- **EDA Visualizations**: Interactive plots displayed in real-time.
- **Prediction Results**: Output as Excel files and emailed.
- **SHAP Analysis**: Interpretability plots for predictive models.

---

## Notes
- Replace sensitive credentials (e.g., API keys, email passwords) with environment variables or secure methods.
- Ensure data formatting matches the expected structure for seamless operation.

---

## Contact
For questions or feedback, please reach out to the project maintainer.

