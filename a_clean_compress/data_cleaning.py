"""
Data cleaning module responsible for standardizing and cleaning pandas DataFrames.
Uses datacleaner library for core cleaning operations and implements additional
domain-specific cleaning as needed.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
from datacleaner import autoclean


class DataCleaningError(Exception):
    """Custom exception for data cleaning related errors."""
    pass


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the input DataFrame meets minimum requirements for cleaning.
    
    Args:
        df: Input DataFrame to validate
        
    Raises:
        DataCleaningError: If DataFrame is empty or has invalid structure
    """
    if df is None or df.empty:
        raise DataCleaningError("Input DataFrame is empty")
    if df.columns.empty:
        raise DataCleaningError("DataFrame has no columns")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    df.columns = (df.columns
                 .str.lower()
                 .str.replace(r'[\s\-]+', '_', regex=True)
                 .str.replace(r'[^a-z0-9_]', '', regex=True))
    return df


def handle_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to datetime format.
    
    Args:
        df: Input DataFrame
        date_columns: List of column names to convert to datetime
        
    Returns:
        DataFrame with standardized date columns
        
    Raises:
        DataCleaningError: If date conversion fails
    """
    df = df.copy()
    for col in date_columns:
        if col not in df.columns:
            continue
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            raise DataCleaningError(f"Failed to convert column {col} to datetime: {str(e)}")
    return df


def remove_outliers(
    df: pd.DataFrame,
    numeric_columns: List[str],
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers from specified numeric columns using z-score method.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names to check for outliers
        threshold: Z-score threshold for outlier detection (default: 3.0)
        
    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    for col in numeric_columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        z_scores = abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < threshold]
    return df


def clean_data(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    custom_cleaning_rules: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Clean and standardize input DataFrame using datacleaner and custom rules.
    
    Args:
        df: Input DataFrame to clean
        date_columns: Optional list of columns to convert to datetime
        numeric_columns: Optional list of numeric columns to check for outliers
        custom_cleaning_rules: Optional dictionary of custom cleaning rules
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        DataCleaningError: If cleaning operations fail
        
    Example:
        >>> df = pd.DataFrame({
        ...     'DATE': ['2023-01-01', '2023-01-02'],
        ...     'VALUE': [100, 200],
        ...     'CATEGORY': ['A', 'B']
        ... })
        >>> cleaned_df = clean_data(
        ...     df,
        ...     date_columns=['DATE'],
        ...     numeric_columns=['VALUE']
        ... )
    """
    try:
        # Validate input
        validate_dataframe(df)
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Standardize column names
        cleaned_df = standardize_column_names(cleaned_df)
        
        # Apply datacleaner's autoclean
        cleaned_df = autoclean(cleaned_df)
        
        # Handle date columns if specified
        if date_columns:
            cleaned_df = handle_date_columns(cleaned_df, date_columns)
        
        # Remove outliers from numeric columns if specified
        if numeric_columns:
            cleaned_df = remove_outliers(cleaned_df, numeric_columns)
        
        # Apply custom cleaning rules if provided
        if custom_cleaning_rules:
            for column, rule in custom_cleaning_rules.items():
                if column in cleaned_df.columns and callable(rule):
                    cleaned_df[column] = cleaned_df[column].apply(rule)
        
        # Final validation of cleaned data
        validate_dataframe(cleaned_df)
        
        return cleaned_df
    
    except Exception as e:
        raise DataCleaningError(f"Data cleaning failed: {str(e)}")


def get_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a report comparing original and cleaned DataFrames.
    
    Args:
        original_df: Original DataFrame before cleaning
        cleaned_df: Cleaned DataFrame after processing
        
    Returns:
        Dictionary containing cleaning statistics and changes
    """
    return {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'null_values_before': original_df.isnull().sum().to_dict(),
        'null_values_after': cleaned_df.isnull().sum().to_dict(),
        'columns_modified': list(set(original_df.columns) ^ set(cleaned_df.columns))
    } 