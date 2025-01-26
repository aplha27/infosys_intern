import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data from Excel file
try:
    combined_df = pd.read_excel('cleaned_data.xlsx')
except FileNotFoundError:
    print("Error: The file 'combined_data.xlsx' was not found.")
    combined_df = pd.DataFrame()  # Create an empty DataFrame to avoid further errors

# Check if the required columns exist
if 'Job Description' in combined_df.columns and 'Resume' in combined_df.columns:
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the job descriptions and resumes
    tfidf_job_desc = vectorizer.fit_transform(combined_df['Job Description'])
    tfidf_resumes = vectorizer.transform(combined_df['Resume'])

    # Calculate the cosine similarity between job descriptions and resumes
    combined_df['resume_job_similarity'] = cosine_similarity(tfidf_resumes, tfidf_job_desc).max(axis=1)

    # Defining bins for similarity scores based on TF-IDF vectorizer similarity
    bins = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    labels = ['0-0.1', '0.1-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']

    # Create a new column with binned values for TF-IDF similarity scores
    combined_df['similarity_bin'] = pd.cut(combined_df['resume_job_similarity'], bins=bins, labels=labels, include_lowest=True)

    # Grouping data for analysis
    grouped_data = combined_df.groupby(['similarity_bin', 'decision', 'Role']).size().unstack(fill_value=0)

    # Print the grouped data
    print(grouped_data)

    # Optional: Visualize the grouped data
    grouped_data.plot(kind='bar', stacked=True)
    plt.title('Grouped Data by TF-IDF Similarity Bin, Decision, and Role')
    plt.xlabel('TF-IDF Similarity Bin')
    plt.ylabel('Count')
    plt.show()
else:
    print("Error: Required columns 'job_description' or 'resume' are missing from the DataFrame.")
