
import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from datetime import datetime
import os
import csv
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("❌ Error: HUGGINGFACE_API_KEY not found in environment variables")

# Load product code database with robust parsing
try:
    product_code_df = pd.read_csv("products_code.csv", on_bad_lines="warn", quoting=csv.QUOTE_ALL, encoding='utf-8')
    # Rename problematic column if present
    if 'Désignation_Complète;;;;;;;;;;;;;;;;' in product_code_df.columns:
        product_code_df = product_code_df.rename(columns={'Désignation_Complète;;;;;;;;;;;;;;;;': 'Désignation_Complète'})
    required_cols = ["Code", "Désignation_Complète"]
    if not all(col in product_code_df.columns for col in required_cols):
        raise ValueError(f"❌ Error: products_code.csv must contain 'Code' and 'Désignation_Complète' columns. Found: {product_code_df.columns.tolist()}")
except FileNotFoundError:
    raise FileNotFoundError("❌ Error: products_code.csv not found")
except pd.errors.ParserError as e:
    print(f"❌ Error parsing products_code.csv: {e}")
    print("Try opening the file in a text editor to check for unescaped commas or inconsistent fields.")
    exit(1)
except UnicodeDecodeError as e:
    print(f"❌ Error: Encoding issue in products_code.csv: {e}")
    print("Ensure the file is UTF-8 encoded or specify the correct encoding.")
    exit(1)

# Clean Désignation_Complète and Code
product_code_df["Désignation_Complète"] = product_code_df["Désignation_Complète"].fillna("Unknown").astype(str)
product_code_df["Code"] = product_code_df["Code"].astype(str)

# Validate Code column (numeric, 2-10 digits, or clean malformed entries)
def is_valid_hs_code(code):
    # Extract numeric part (e.g., "0902200090, dans..." -> "0902200090")
    match = re.match(r'^\d{2,10}(?=\D|$)', code)
    return match is not None

# Clean Code by extracting numeric part
product_code_df["Code"] = product_code_df["Code"].apply(
    lambda x: re.match(r'^\d{2,10}(?=\D|$)', x).group() if re.match(r'^\d{2,10}(?=\D|$)', x) else "INVALID"
)

# Log and filter invalid codes
invalid_codes = product_code_df[product_code_df["Code"] == "INVALID"]
if not invalid_codes.empty:
    print(f"⚠️ Warning: Found {len(invalid_codes)} invalid HS codes in products_code.csv:")
    print(invalid_codes[["Code", "Désignation_Complète"]].head(10))  # Show first 10 for brevity
    product_code_df = product_code_df[product_code_df["Code"] != "INVALID"]

# Log rows with empty Désignation_Complète
invalid_rows = product_code_df[product_code_df["Désignation_Complète"].str.strip() == ""]
if not invalid_rows.empty:
    print(f"⚠️ Warning: Found {len(invalid_rows)} rows with empty Désignation_Complète:")
    print(invalid_rows[["Code", "Désignation_Complète"]].head(10))

# Check if product_code_df is empty
if product_code_df.empty:
    print("❌ Error: No valid rows in products_code.csv after filtering. Check data and fix invalid HS codes.")
    exit(1)

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"❌ Error initializing embeddings: {e}")
    exit(1)

# Prepare texts and metadata for FAISS
texts = product_code_df["Désignation_Complète"].tolist()
product_codes = product_code_df["Code"].tolist()

# Create FAISS index
try:
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=[{"product_code": code} for code in product_codes]
    )
    vector_store.save_local("faiss_index")
    print("✅ FAISS vector store created and saved as faiss_index")
except Exception as e:
    print(f"❌ Error creating FAISS index: {e}")
    exit(1)

# Initialize Mistral LLM
try:
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    exit(1)

# Define prompt template for fraud detection
prompt_template = PromptTemplate(
    input_variables=["product_description", "retrieved_code", "declared_code"],
    template="""
    Compare the retrieved Harmonized System (HS) code with the declared HS code for the given product description.

    Product Description: {product_description}
    Retrieved HS Code (from database): {retrieved_code}
    Declared HS Code (from input): {declared_code}

    If the retrieved and declared HS codes match exactly, classify as Legitimate. Otherwise, classify as Fraudulent.

    Output only the fraud status in the following format:
    Fraud Status: [Legitimate/Fraudulent]
    """
)

# Create LLM chain using RunnableSequence
fraud_detection_chain = RunnableSequence(prompt_template | llm)

# Load new data for testing with robust parsing
try:
    new_df = pd.read_csv("new_data_for_testing.csv", on_bad_lines="warn", quoting=csv.QUOTE_ALL, encoding='utf-8')
except FileNotFoundError:
    print("❌ Error: new_data_for_testing.csv not found. Run create_new_data.py first.")
    exit(1)
except pd.errors.ParserError as e:
    print(f"❌ Error parsing new_data_for_testing.csv: {e}")
    exit(1)
except UnicodeDecodeError as e:
    print(f"❌ Error: Encoding issue in new_data_for_testing.csv: {e}")
    print("Ensure the file is UTF8 encoded or specify the correct encoding.")
    exit(1)

# Validate required columns
required_cols = ["product_description", "declared_code", "is_fraud"]
if not all(col in new_df.columns for col in required_cols):
    print(f"❌ Error: New data missing required columns. Found: {new_df.columns.tolist()}")
    exit(1)

# Validate HS codes in new_df against product_code_df
valid_hs_codes = set(product_code_df["Code"])
new_df["declared_code"] = new_df["declared_code"].astype(str)
invalid_hs_codes = new_df[~new_df["declared_code"].isin(valid_hs_codes)]
if not invalid_hs_codes.empty:
    print(f"⚠️ Warning: Found {len(invalid_hs_codes)} invalid HS codes in new_data_for_testing.csv:")
    print(invalid_hs_codes[["product_description", "declared_code"]])
    new_df.loc[~new_df["declared_code"].isin(valid_hs_codes), "declared_code"] = "UNKNOWN"

# Function to process a single row for fraud detection
def detect_fraud(row):
    product_description = row["product_description"]
    declared_code = str(row["declared_code"]).strip()
    
    try:
        # Retrieve closest product code using RAG
        docs = vector_store.similarity_search(product_description, k=1)
        retrieved_code = docs[0].metadata["product_code"]
        
        # Run LLM chain for fraud detection
        result = fraud_detection_chain.invoke({
            "product_description": product_description,
            "retrieved_code": retrieved_code,
            "declared_code": declared_code
        })
        
        # Parse LLM output
        lines = result.strip().split("\n")
        fraud_status = lines[0].replace("Fraud Status: ", "").strip() if len(lines) > 0 else "Unknown"
        
        return {
            "product_description": product_description,
            "declared_code": declared_code,
            "retrieved_code": retrieved_code,
            "fraud_status": fraud_status,
            "is_fraud": row["is_fraud"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"⚠️ Warning: Failed to process row with product_description '{product_description}': {e}")
        return {
            "product_description": product_description,
            "declared_code": declared_code,
            "retrieved_code": "Unknown",
            "fraud_status": "Unknown",
            "is_fraud": row["is_fraud"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Process all rows and collect results
results = []
for idx, row in new_df.iterrows():
    result = detect_fraud(row)
    results.append(result)

# Convert results to DataFrame
if not results:
    print("❌ Error: No results processed. Check input data or logs.")
    exit(1)
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv("rag_fraud_detection_results.csv", index=False)
print("✅ Fraud detection results saved to rag_fraud_detection_results.csv")

# Summary of fraud detections
print("\nFraud Detection Summary:")
print(results_df["fraud_status"].value_counts())
print("\nSample Results (Top 5):")
print(results_df[["product_description", "declared_code", "retrieved_code", "fraud_status", "is_fraud"]].head())
