import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def screen_resume(job_description, resume_text):
    # Save the input resume and job description to an Excel file
    file_path = "uploaded_resumes.xlsx"

    # Create or append to the file
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        new_entry = pd.DataFrame({'Job Description': [job_description], 'Resume': [resume_text]})
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = pd.DataFrame({'Job Description': [job_description], 'Resume': [resume_text]})

    # Save updated file
    df.to_excel(file_path, index=False)

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the job descriptions and resumes
    tfidf_job_desc = vectorizer.fit_transform(df['Job Description'])
    tfidf_resumes = vectorizer.transform(df['Resume'])

    # Calculate the cosine similarity
    similarity_score = cosine_similarity(tfidf_resumes[-1:], tfidf_job_desc[-1:])[0][0]

    # Determine match category
    match_category = get_match_category(similarity_score)

    return {
        'similarity_score': round(similarity_score * 100, 2),
        'match_category': match_category
    }

def get_match_category(similarity_score):
    if similarity_score >= 0.8: return 'Excellent Match'
    elif similarity_score >= 0.6: return 'Strong Match'
    elif similarity_score >= 0.4: return 'Moderate Match'
    elif similarity_score >= 0.2: return 'Weak Match'
    else: return 'Poor Match'
