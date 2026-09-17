"""
Final Data Cleaning Script for Resume Dataset
"""

import pandas as pd
import re
from pathlib import Path

RAW_PATH = Path("data/raw/cleaned_dataset (1).csv")
PROCESSED_DIR = Path("data/cleaned")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)

def clean_multiline_block(text):
    """
    Cleans complex multi-line text fields (skills, experience, education).
    Preserves structure with "|" separator, helps later parsing
    """
    if pd.isna(text):
        return text

    text = str(text)

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = re.sub(r'^\s*\d+\s*[-–]\s*', '', line)
        line = re.sub(r'\s+', ' ', line)
        line = line.strip(' ,')
        if line:
            cleaned_lines.append(line)

    # join all cleaned lines with "|" separator
    return " | ".join(cleaned_lines)

def clean_simple(text):
    """
    Clean simple single-line text fields (name, job_title, job_description), 
    which should be single-line.
    """
    if pd.isna(text):
        return text

    text = str(text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# optimize data structure
df["name"] = df["name"].str.strip().str.title()
# Use .apply() to apply function to each row (single value at a time)
# This converts Series to individual values for clean_simple() to process
df["job_title"] = df["job_title"].apply(clean_simple)
df["job_description"] = df["job_description"].apply(clean_simple)

df["skills"] = df["skills"].apply(clean_multiline_block)
df["experience"] = df["experience"].apply(clean_multiline_block)
df["education"] = df["education"].apply(clean_multiline_block)


# separate protected attributes & keep them in a separate copy for fairness evaluation later
protected_cols = ["Candidate ID", "nationality", "age_group", "gender"]
protected = df[protected_cols].copy()

# candidates dataframe WITHOUT protected attributes: USED FOR MODEL TRAINING
candidates = df.drop(columns=["nationality", "age_group", "gender"])


# data the model will use for training
candidates.to_csv(PROCESSED_DIR / "cleaned_candidates.csv", index=False)

# protected attributes for evaluation ONLY
protected.to_csv(PROCESSED_DIR / "protected_attributes.csv", index=False)

print(f"✓ Data cleaning complete!")
print(f"✓ Cleaned candidates saved to: {PROCESSED_DIR / 'cleaned_candidates.csv'}")
print(f"✓ Protected attributes saved to: {PROCESSED_DIR / 'protected_attributes.csv'}")