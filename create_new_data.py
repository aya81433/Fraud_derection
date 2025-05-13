import pandas as pd
import random
from datetime import datetime

# Load product code database safely
try:
    product_code_df = pd.read_csv(
        "products_code.csv",
        usecols=[0, 1, 2, 3],
        on_bad_lines='skip'  # Skip malformed lines
    )
    if not all(col in product_code_df.columns for col in ["Code", "Désignation_Complète;;;;;;;;;;;;;;;;"]):
        raise ValueError("❌ Error: products_code.csv must contain 'Code' and 'Désignation_Complète;;;;;;;;;;;;;;;;' columns")
except FileNotFoundError:
    print("❌ Error: products_code.csv not found")
    exit(1)
except Exception as e:
    print(f"❌ Error reading CSV: {e}")
    exit(1)

# Sample HS codes and descriptions
codes = product_code_df["Code"].tolist()
descriptions = product_code_df["Désignation_Complète;;;;;;;;;;;;;;;;"].tolist()

# Generate test data
test_data = []
for _ in range(10):  # Create 10 test rows
    correct_idx = random.randint(0, len(codes) - 1)
    correct_code = codes[correct_idx]
    correct_desc = descriptions[correct_idx]

    # Randomly decide if this row is fraudulent
    is_fraud = random.choice([0, 1])
    if is_fraud:
        # For fraudulent rows, pick a random incorrect code
        incorrect_codes = [c for c in codes if c != correct_code]
        declared_code = random.choice(incorrect_codes)
    else:
        # For non-fraudulent rows, use the correct code
        declared_code = correct_code

    test_data.append({
        "product_description": correct_desc,
        "declared_code": declared_code,
        "is_fraud": is_fraud
    })

# Create DataFrame and save
test_df = pd.DataFrame(test_data)
test_df.to_csv("new_data_for_testing.csv", index=False)
print(f"✅ Generated new_data_for_testing.csv with {len(test_df)} rows at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Sample rows:")
print(test_df.head())