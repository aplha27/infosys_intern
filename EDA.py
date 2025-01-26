import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the Excel file
data_path = 'combined_data.xlsx'
df = pd.read_excel(data_path)

# Function to clean and convert text to lowercase
def clean_text(text):
    if isinstance(text, str):
        return ' '.join(text.lower().split())  # Convert to lowercase and remove extra spaces/newlines
    return text

# Apply the cleaning function to all text-based columns
df = df.applymap(clean_text)

# Save the cleaned data back to a new file
output_path = 'cleaned_combined_data.xlsx'
df.to_excel(output_path, index=False)

# Basic Data Analysis
print("Basic Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include='all'))

print("\nMissing Values:")
print(df.isnull().sum())

# Create a directory for saving plots
output_dir = 'EDA analysis'
os.makedirs(output_dir, exist_ok=True)

# Advanced Analysis
# Distribution of Numerical Columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'))  # Save the plot
    plt.close()  # Close the figure to free memory

# Count Plots for Categorical Columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, y=col, order=df[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.savefig(os.path.join(output_dir, f'count_plot_{col}.png'))  # Save the plot
    plt.close()  # Close the figure to free memory

# Pairplot for numerical columns
if len(numerical_cols) > 1:
    pairplot = sns.pairplot(df[numerical_cols])
    pairplot.savefig(os.path.join(output_dir, 'pairplot.png'))  # Save the pairplot
    plt.close()  # Close the figure to free memory

# Grouping and Aggregation
print("\nGrouping and Aggregation Example:")
grouped_data = df.groupby(categorical_cols[0])[numerical_cols].mean()
print(grouped_data)

# Save the cleaned data
print(f"Cleaned data saved to {output_path}")

