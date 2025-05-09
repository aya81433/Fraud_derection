import pandas as pd

# Load the dataset
file_path = "/Users/macbook/Desktop/missclassification/synthetic_fraud_dataset_1M_optimized.csv"
df = pd.read_csv(file_path)

import pandas as pd
import re

# Sample: read your data into a DataFrame (adjust the path and method as needed)
# df = pd.read_csv('your_file.csv')  # or however you load your data

# Show column names to debug if needed
print("Available columns:", df.columns)

# Function to clean and preprocess product names
def preprocess_text(text):
    if isinstance(text, str):  # Ensure input is a string
        # Convert to lowercase (keep accents intact)
        text = text.lower()
        # Only allow alphabetic characters (including accented ones) and spaces
        text = re.sub(r'[^a-z\séèêëàâôùîïäöü]', '', text)  # This keeps accented characters and spaces
        return text
    return ""

# Adjust this based on the actual column name (example: 'product_name')
# Replace 'product_name' with the correct column name if different
#if 'inputs' not in df.columns:
    #df['inputs'] = df['product_description'] # or any other relevant existing column

# Apply the preprocessing function to the 'inputs' column
#df['inputs'] = df['inputs'].apply(preprocess_text)

# Optional: print a preview to confirm
#print(df[['inputs']].head())


# Save the cleaned dataset
df.to_csv("/Users/macbook/Desktop/missclassification/cleaned_fraud_dataset_ready.csv", index=False)

print("Dataset cleaned, filtered by product_code, and saved successfully.")