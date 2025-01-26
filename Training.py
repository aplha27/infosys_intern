import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Load data
df = pd.read_excel('cleaned_data.xlsx')

# Feature extraction functions (as provided)
def count_words(text):
    return len(text.split()) if pd.notnull(text) else 0

def count_characters(text):
    return len(text) if pd.notnull(text) else 0

def avg_word_length(text):
    words = text.split() if pd.notnull(text) else []
    return sum(len(word) for word in words) / len(words) if words else 0

def count_sentences(text):
    return len(re.split(r'[.!?]', text)) - 1 if pd.notnull(text) else 0

def count_uppercase_ratio(text):
    return sum(1 for char in text if char.isupper()) / len(text) if pd.notnull(text) and len(text) > 0 else 0

def keyword_count(text, keywords):
    words = text.split() if pd.notnull(text) else []
    return sum(1 for word in words if word.lower() in keywords)

def unique_word_ratio(text):
    words = text.split() if pd.notnull(text) else []
    return len(set(words)) / len(words) if words else 0

def check_role_in_resume(resume, role):
    return 1 if pd.notnull(resume) and role.lower() in resume.lower() else 0

def keyword_overlap(text1, text2):
    if pd.notnull(text1) and pd.notnull(text2):
        words1 = set(text1.split())
        words2 = set(text2.split())
        return len(words1 & words2)
    return 0

# Predefined keyword dictionaries
technical_keywords = {'python', 'java', 'sql', 'machine learning', 'cloud', 'design', 'analysis', 'management'}
positive_keywords = {'excellent', 'success', 'outstanding', 'achievement', 'skilled'}
negative_keywords = {'poor', 'inadequate', 'lacking', 'failure', 'weak'}

# Feature engineering (as before)
df['resume_word_count'] = df['Cleaned_Resume'].apply(count_words)
df['resume_char_count'] = df['Cleaned_Resume'].apply(count_characters)
df['resume_avg_word_length'] = df['Cleaned_Resume'].apply(avg_word_length)
df['resume_sentence_count'] = df['Cleaned_Resume'].apply(count_sentences)
df['resume_uppercase_ratio'] = df['Cleaned_Resume'].apply(count_uppercase_ratio)
df['resume_technical_keyword_count'] = df['Cleaned_Resume'].apply(lambda x: keyword_count(x, technical_keywords))
df['resume_positive_keyword_count'] = df['Cleaned_Resume'].apply(lambda x: keyword_count(x, positive_keywords))
df['resume_negative_keyword_count'] = df['Cleaned_Resume'].apply(lambda x: keyword_count(x, negative_keywords))
df['resume_unique_word_ratio'] = df['Cleaned_Resume'].apply(unique_word_ratio)

df['transcript_word_count'] = df['Cleaned_Transcript'].apply(count_words)
df['transcript_char_count'] = df['Cleaned_Transcript'].apply(count_characters)
df['transcript_avg_word_length'] = df['Cleaned_Transcript'].apply(avg_word_length)
df['transcript_sentence_count'] = df['Cleaned_Transcript'].apply(count_sentences)
df['transcript_uppercase_ratio'] = df['Cleaned_Transcript'].apply(count_uppercase_ratio)
df['transcript_positive_keyword_count'] = df['Cleaned_Transcript'].apply(lambda x: keyword_count(x, positive_keywords))
df['transcript_negative_keyword_count'] = df['Cleaned_Transcript'].apply(lambda x: keyword_count(x, negative_keywords))
df['transcript_unique_word_ratio'] = df['Cleaned_Transcript'].apply(unique_word_ratio)

df['job_role_in_resume'] = df.apply(lambda row: check_role_in_resume(row['Cleaned_Resume'], row['Role']), axis=1)

df['resume_job_keyword_overlap'] = df.apply(lambda row: keyword_overlap(row['Cleaned_Resume'], row['Cleaned_Job_Description']), axis=1)
df['transcript_job_keyword_overlap'] = df.apply(lambda row: keyword_overlap(row['Cleaned_Transcript'], row['Cleaned_Job_Description']), axis=1)

# Role popularity (frequency encoding)
role_counts = df['Role'].value_counts()
df['role_popularity'] = df['Role'].map(role_counts)

# Decision reason encoding
df['decision_reason_encoded'] = df['Reason for decision'].astype('category').cat.codes

# TF-IDF and cosine similarity
tfidf_vectorizer = TfidfVectorizer()

# Concatenating all text columns for TF-IDF
df['All_Text'] = df['Cleaned_Resume'] + ' ' + df['Cleaned_Transcript'] + ' ' + df['Cleaned_Job_Description']

# Fitting and transforming the text columns
tfidf_matrix = tfidf_vectorizer.fit_transform(df['All_Text'])

# Save the TF-IDF vectorizer to a pickle file
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Cosine similarities
df['resume_job_similarity'] = [
    cosine_similarity(tfidf_matrix[i, :].reshape(1, -1), tfidf_matrix[j, :].reshape(1, -1))[0, 0]
    for i, j in zip(df.index, df.index)
]

df['transcript_job_similarity'] = [
    cosine_similarity(tfidf_matrix[i, :].reshape(1, -1), tfidf_matrix[k, :].reshape(1, -1))[0, 0]
    for i, k in zip(df.index, df.index)
]

df['transcript_resume_similarity'] = [
    cosine_similarity(tfidf_matrix[j, :].reshape(1, -1), tfidf_matrix[k, :].reshape(1, -1))[0, 0]
    for j, k in zip(df.index, df.index)
]

# Display sample of engineered features and similarities
engineered_features = [
    'resume_word_count', 'resume_char_count', 'resume_avg_word_length', 'resume_sentence_count',
    'resume_uppercase_ratio', 'resume_technical_keyword_count', 'resume_positive_keyword_count',
    'resume_negative_keyword_count', 'resume_unique_word_ratio', 'transcript_word_count',
    'transcript_char_count', 'transcript_avg_word_length', 'transcript_sentence_count',
    'transcript_uppercase_ratio', 'transcript_positive_keyword_count', 'transcript_negative_keyword_count',
    'transcript_unique_word_ratio', 'job_role_in_resume', 'resume_job_keyword_overlap',
    'transcript_job_keyword_overlap', 'role_popularity', 'decision_reason_encoded',
    'resume_job_similarity', 'transcript_job_similarity', 'transcript_resume_similarity'
]

print(df[engineered_features].head())

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Features and target variable
features = [
    'transcript_positive_keyword_count',
    'resume_positive_keyword_count',
    'transcript_avg_word_length',
    'decision_reason_encoded',
    'transcript_char_count',
    'transcript_job_keyword_overlap',
    'resume_negative_keyword_count',
    'resume_job_keyword_overlap',
    'resume_char_count',
    'transcript_unique_word_ratio'
]


target = 'decision'  # Replace with the correct target column if different

X = df[features]
y = df[target]

y = y.map({'reject': 0, 'select': 1})
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
    }

# Update the XGBClassifier initialization with the best parameters
xgb = XGBClassifier(
    n_estimators=300,          # Best parameter
    learning_rate=0.05,        # Best parameter
    max_depth=5,               # Best parameter
    subsample=1.0,             # Best parameter
    colsample_bytree=0.6,      # Best parameter
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss'
)

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model summary
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("\nBest Cross-Validation Score:", grid_search.best_score_)

# Save the model to a pickle file
joblib.dump(best_model, 'xgb_model.pkl')

# Predict on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nTest Set Accuracy:", accuracy)
print("Test Set ROC-AUC Score:", roc_auc)



import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_train)
shap.plots.beeswarm(shap_values)
base_value = explainer.expected_value
print("Base Value:", base_value)
shap_values_test = explainer(X_test)
import numpy as np
import shap

# Get prediction probabilities for test set
test_probas = best_model.predict_proba(X_test)[:, 1]

# Find indices for low, medium, and high predictions
low_idx = np.argmin(test_probas)
high_idx = np.argmax(test_probas)
med_idx = np.argmin(np.abs(test_probas - np.median(test_probas)))

# Create a directory for SHAP analysis if it doesn't exist
os.makedirs('shaply_analysis', exist_ok=True)

# SHAP Beeswarm plot
shap.plots.beeswarm(shap_values)
plt.title("SHAP Beeswarm Plot")
plt.tight_layout()
plt.savefig('shaply_analysis/shap_beeswarm_plot.png')  # Save the beeswarm plot
plt.show()

# Low prediction waterfall plot
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_test[low_idx])
plt.title(f'Low Prediction (Probability: {test_probas[low_idx]:.3f})')
plt.tight_layout()
plt.savefig('shaply_analysis/shap_low_prediction_waterfall.png')  # Save the low prediction waterfall plot
plt.show()

# Medium prediction waterfall plot
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_test[med_idx])
plt.title(f'Medium Prediction (Probability: {test_probas[med_idx]:.3f})')
plt.tight_layout()
plt.savefig('shaply_analysis/shap_medium_prediction_waterfall.png')  # Save the medium prediction waterfall plot
plt.show()

# High prediction waterfall plot
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_test[high_idx])
plt.title(f'High Prediction (Probability: {test_probas[high_idx]:.3f})')
plt.tight_layout()
plt.savefig('shaply_analysis/shap_high_prediction_waterfall.png')  # Save the high prediction waterfall plot
plt.show()

# Calculate mean absolute SHAP values for feature importance ranking
feature_importance = np.abs(shap_values_test.values).mean(0)
feature_importance_dict = dict(zip(features, feature_importance))

# Sort features by importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
top_3_features = [feat[0] for feat in sorted_features[:3]]

print("Top 3 most important features:")
for feat, importance in sorted_features[:3]:
    print(f"{feat}: {importance:.4f}")

# Create dependence plots for top 3 features
for feature in top_3_features:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_values_test.values,
        X_test,
        show=False
    )
    plt.title(f'SHAP Dependence Plot for {feature}')
    plt.tight_layout()
    plt.savefig(f'shaply_analysis/shap_dependence_plot_{feature}.png')  # Save each dependence plot
    plt.show()

# Create a combined plot for all three features
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Partial Dependence Plots for Top 3 Features', fontsize=16)

display = shap.PartialDependenceDisplay.from_estimator(
    best_model,
    X_train,
    top_3_features,
    kind="average",  # Shows only PDP without ICE plots for clearer comparison
    centered=True,
    ax=axes,
    random_state=42
)

plt.tight_layout()
plt.savefig('shaply_analysis/shap_combined_partial_dependence_plot.png')  # Save the combined plot
plt.show()

# Print feature importance scores
print("\nFeature Importance Scores:")
for feature in top_3_features:
    print(f"{feature}: {feature_importance_dict[feature]:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Get top 2 features from SHAP analysis
feature_importance = np.abs(shap_values_test.values).mean(0)
feature_importance_dict = dict(zip(features, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
top_2_features = [feat[0] for feat in sorted_features[:2]]

print(f"Creating 2D PDP interaction for features: {top_2_features[0]} and {top_2_features[1]}")

# Calculate partial dependence
pdp_interact_values = shap.PartialDependenceDisplay.from_estimator(
    best_model,
    X_train,
    features=top_2_features,
    kind='average',
    grid_resolution=20
)

# Extract values and feature grid points
pdp_values = pdp_interact_values.average[0]
feature1_grid = pdp_interact_values['values'][0]  # Use dictionary-style access
feature2_grid = pdp_interact_values['values'][1]

# Create meshgrid for contour plot
X1, X2 = np.meshgrid(feature1_grid, feature2_grid)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create contour plot
contours = ax.contour(X1, X2, pdp_values.T, levels=10, colors='black', alpha=0.6)
plt.clabel(contours, inline=True, fontsize=8)

# Create color mesh
im = ax.contourf(X1, X2, pdp_values.T, levels=20, cmap='RdYlBu')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Predicted Probability', rotation=270, labelpad=15)

# Set labels and title
ax.set_xlabel(top_2_features[0])
ax.set_ylabel(top_2_features[1])
plt.title(f'2D Partial Dependence Plot: Interaction Effects\n{top_2_features[0]} vs {top_2_features[1]}')

# Add scatter plot of actual data points
ax.scatter(
    X_train[top_2_features[0]],
    X_train[top_2_features[1]],
    c='black',
    alpha=0.1,
    s=10
)

plt.tight_layout()
plt.show()

# Calculate and print interaction statistics
print("\nFeature Statistics:")
for feature in top_2_features:
    print(f"\n{feature}:")
    print(f"Mean: {X_train[feature].mean():.4f}")
    print(f"Std: {X_train[feature].std():.4f}")
    print(f"Range: [{X_train[feature].min():.4f}, {X_train[feature].max():.4f}]")
    print(f"SHAP importance: {feature_importance_dict[feature]:.4f}")

# Calculate correlation and print
correlation = X_train[top_2_features[0]].corr(X_train[top_2_features[1]])
print(f"\nFeature Correlation: {correlation:.4f}")

# Calculate the range of PDP values
pdp_range = np.max(pdp_values) - np.min(pdp_values)
print(f"\nPDP Effect Range: {pdp_range:.4f}")
print(f"PDP Min Effect: {np.min(pdp_values):.4f}")
print(f"PDP Max Effect: {np.max(pdp_values):.4f}")

