import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

combined_df = pd.read_excel('combined_data.xlsx')
combined_df = combined_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
combined_df['length_of_transcript'] = combined_df['Transcript'].apply(lambda x: len(x))

# Define a function to map the decision values
def map_decision(decision):
    if decision in ['reject', 'rejected']:
        return 'reject'
    elif decision in ['select', 'selected']:
        return 'select'
    else:
        return decision  # Return other values as they are


# Apply the mapping function to the 'decision' column
combined_df['decision'] = combined_df['decision'].apply(map_decision)

combined_df[['length_of_transcript','decision']].groupby('decision').mean()
combined_df['num_words_in_transcript'] = combined_df['Transcript'].apply(lambda x: len(str(x).split()))
combined_df[['num_words_in_transcript','decision','Role']].groupby(['Role','decision']).agg({'mean','median','std'})
sns.distplot(combined_df[combined_df['decision']=='reject']['num_words_in_transcript'])
combined_df[['num_words_in_transcript','decision']].groupby(['decision']).mean()
sns.distplot(combined_df[combined_df['decision']=='select']['num_words_in_transcript'])
# Calculate the correlation matrix
correlation_matrix = combined_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Box plots for numerical features grouped by decision
numerical_cols = combined_df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='decision', y=col, data=combined_df)
    plt.title(f'Box Plot of {col} by Decision')
    plt.show()

# Explore relationships between categorical features and decision
categorical_cols = combined_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col, hue='decision', data=combined_df)
    plt.title(f'Count Plot of {col} by Decision')
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
    plt.show()

# Calculate the correlation matrix, only including numerical features
correlation_matrix = combined_df.select_dtypes(include=np.number).corr()
# Select only numerical columns before calculating correlations

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Box plots for numerical features grouped by decision
numerical_cols = combined_df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='decision', y=col, data=combined_df)
    plt.title(f'Box Plot of {col} by Decision')
    plt.show()
