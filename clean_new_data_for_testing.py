import pandas as pd
import re
import csv
from datetime import datetime

import pandas as pd
import csv

# Load product code database with robust parsing
try:
    product_code_df = pd.read_csv("products_code.csv", on_bad_lines="warn", quoting=csv.QUOTE_ALL)
    
    # Clean up column names by stripping any extra semicolons and spaces
    product_code_df.columns = [col.strip().rstrip(';') for col in product_code_df.columns]

    # Validate required columns
    if not all(col in product_code_df.columns for col in ["Code", "Désignation_Complète"]):
        raise ValueError(f"❌ Error: products_code.csv must contain 'Code' and 'Désignation_Complète' columns. Found: {product_code_df.columns.tolist()}")
except FileNotFoundError:
    print("❌ Error: products_code.csv not found")
    exit(1)
except pd.errors.ParserError as e:
    print(f"❌ Error parsing products_code.csv: {e}")
    print("Try opening the file in a text editor to check for unescaped commas or inconsistent fields.")
    exit(1)

valid_hs_codes = set(product_code_df["Code"].astype(str).str.strip())

# Fallback descriptions for known HS codes
fallback_descriptions = {
    "2828": "Hypochlorites",
    # Add more if needed
}

# Load new_data_for_testing.csv with robust parsing
try:
    df = pd.read_csv("new_data_for_testing.csv", on_bad_lines="warn", quoting=csv.QUOTE_ALL)
except FileNotFoundError:
    print("❌ Error: new_data_for_testing.csv not found")
    exit(1)
except pd.errors.ParserError as e:
    print(f"❌ Error parsing new_data_for_testing.csv: {e}")
    exit(1)

# Function to clean declared_code (extract HS code)
def clean_declared_code(code):
    if pd.isna(code):
        return None
    code = str(code).strip()
    # Extract the first numeric-like HS code (e.g., "290349,Autres" -> "290349")
    match = re.match(r"(\d+)", code)
    if match:
        return match.group(1)
    return code

# Function to clean product_description
def clean_product_description(desc, declared_code):
    # Handle missing or empty description
    if pd.isna(desc) or not desc.strip():
        # Use fallback description if available
        declared_code_clean = clean_declared_code(declared_code)
        if declared_code_clean in fallback_descriptions:
            return fallback_descriptions[declared_code_clean]
        return None
    
    # Clean and split description
    desc = str(desc).strip().strip('"')
    
    # If the description contains complex hierarchies (e.g., "['29', '2903', '290349']")
    if "," in desc:
        # Extract the part before the first comma (the product name)
        parts = re.split(r",\['[^\]]*'\]", desc)
        clean_desc = parts[0].strip().rstrip(',')
    else:
        clean_desc = desc
    
    # Clean description further
    clean_desc = re.sub(r";+$", "", clean_desc)
    
    # If description is empty or generic, try declared_code description
    if not clean_desc or clean_desc.lower() in ["autres", "other"]:
        declared_code_clean = clean_declared_code(declared_code)
        if "," in str(declared_code):
            parts = declared_code.split(",", 1)
            if len(parts) > 1:
                clean_desc = parts[1].strip().replace('"', '')
    
    return clean_desc if clean_desc else None

# Clean the DataFrame
cleaned_data = []
for idx, row in df.iterrows():
    declared_code = clean_declared_code(row["declared_code"])
    product_desc = clean_product_description(row["product_description"], row["declared_code"])
    
    # Validate HS code
    if declared_code and declared_code not in valid_hs_codes:
        print(f"⚠️ Warning: HS code {declared_code} at row {idx+2} not found in products_code.csv")
    
    # Skip rows with missing product_description or declared_code
    if product_desc and declared_code:
        cleaned_data.append({
            "product_description": product_desc,
            "declared_code": declared_code,
            "is_fraud": row["is_fraud"]
        })
    else:
        print(f"⚠️ Warning: Skipping row {idx+2} due to missing product_description or declared_code")

# Create cleaned DataFrame
cleaned_df = pd.DataFrame(cleaned_data)

# Save cleaned file
output_file = "new_data_for_testing_cleaned.csv"
cleaned_df.to_csv(output_file, index=False)
print(f"✅ Cleaned data saved to {output_file} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Sample cleaned rows:")
print(cleaned_df.head())

# Validate column presence
required_cols = ["product_description", "declared_code", "is_fraud"]
if not all(col in cleaned_df.columns for col in required_cols):
    print(f"❌ Error: Cleaned data missing required columns. Found: {cleaned_df.columns.tolist()}")
else:
    print("✅ All required columns present in cleaned data")