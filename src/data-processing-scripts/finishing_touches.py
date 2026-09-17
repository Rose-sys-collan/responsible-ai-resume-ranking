import pandas as pd
import pandera.pandas as pa
import os
import re
from pandera.pandas import Column, Check


# Define protected attributes (sensitive personal information)
PROTECTED_ATTRIBUTES = ['name', 'nationality', 'gender', 'age_group']

def separate_protected_attributes(df, protected_attrs=None, output_dir='../cleaned'):
    """
    Separate protected attributes from the main dataset and store them in a separate file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
    protected_attrs : list, optional
        List of column names to treat as protected attributes.
        If None, uses PROTECTED_ATTRIBUTES
    output_dir : str
        Directory where the protected attributes file will be saved
    
    Returns:
    --------
    df_clean : pd.DataFrame
        Dataframe with protected attributes removed
    """
    if protected_attrs is None:
        protected_attrs = PROTECTED_ATTRIBUTES
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate protected attributes that exist in the dataframe
    protected_cols = [col for col in protected_attrs if col in df.columns]
    
    if not protected_cols:
        print(f"Warning: No protected attributes found in dataframe")
        return df
    
    # Create separate dataframe with protected attributes and the ID column if it exists
    id_col = 'Candidate ID' if 'Candidate ID' in df.columns else None
    if id_col:
        protected_df = df[[id_col] + protected_cols]
    else:
        protected_df = df[protected_cols]
    
    # Save protected attributes to a separate file
    protected_file = os.path.join(output_dir, 'protected_attributes.csv')
    protected_df.to_csv(protected_file, index=False)
    print(f"Protected attributes saved to: {protected_file}")
    
    # Remove protected attributes from main dataframe
    df_clean = df.drop(columns=protected_cols)
    
    return df_clean

# Load dataset
df = pd.read_csv("cleaned_dataset (1).csv")
df = df.drop_duplicates().reset_index(drop=True)

df_clean = separate_protected_attributes(df)


def no_invalid_chars(series):
    """Check that strings don't contain invalid characters like �"""
    # Pattern matches: replacement char (�), null bytes, and other control chars
    invalid_pattern = r'[\ufffd\x00\ufffe\uffff\x00-\x08\x0b-\x0c\x0e-\x1f]'
    return ~series.astype(str).str.contains(invalid_pattern, regex=True, na=False)

schema = pa.DataFrameSchema({
    "ID": Column(int, Check.in_range(0, 100000000), nullable=False),
    "job_title": Column(str, [
        Check.str_length(min_value=1),
        Check(no_invalid_chars, error="Contains invalid characters")
    ], nullable=False),
    "job_description": Column(str, [
        Check.str_length(min_value=1),
        Check(no_invalid_chars, error="Contains invalid characters")
    ], nullable=False),
    "skills": Column(str, [
        Check.str_length(min_value=1),
        Check(no_invalid_chars, error="Contains invalid characters")
    ], nullable=False),
    "experience": Column(str, [
        Check.str_length(min_value=1),
        Check(no_invalid_chars, error="Contains invalid characters")
    ], nullable=False),
    "education": Column(str, [
        Check.str_length(min_value=1),
        Check(no_invalid_chars, error="Contains invalid characters")
    ], nullable=False),
})

# Drop any rows that fail validation
try:
    validated_df = schema.validate(df_clean, lazy=True)
    print("✓ All rows passed validation!")
    clean_df = validated_df
except pa.errors.SchemaErrors as e:
    print(f"Validation errors found: {len(e.schema_errors)} errors")
    
    # Get the failure cases and extract indices
    failed_indices = set()
    for error in e.schema_errors:
        try:
            # Different pandera versions structure this differently
            fc = error.failure_cases
            if fc is not None and len(fc) > 0:
                # Try multiple ways to get the index
                if 'index' in fc.columns:
                    failed_indices.update(fc['index'].dropna().astype(int).tolist())
                elif fc.index is not None:
                    failed_indices.update(fc.index.tolist())
        except Exception as inner_e:
            print(f"Could not extract indices from error: {inner_e}")
            continue
    
    if failed_indices:
        print(f"Dropping {len(failed_indices)} failed rows")
        clean_df = df_clean.drop(index=list(failed_indices)).reset_index(drop=True)
    else:
        print("Keeping all rows (couldn't identify specific failures)")
        clean_df = df_clean

print(f"\nFinal dataframe shape: {clean_df.shape}")
print(clean_df.head(10))
clean_df.to_csv('../cleaned/final_cleaned_dataset.csv', index=False)


