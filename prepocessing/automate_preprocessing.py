"""
Automated Preprocessing Script for Telco Customer Churn Dataset
Author: Nama Siswa
Purpose: Kriteria 2 - MSML Submission
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import argparse


def load_data(filepath):
    """
    Load dataset from CSV file

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame: Loaded dataset
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def convert_total_charges(df):
    """
    Convert TotalCharges from object to numeric

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame with converted TotalCharges
    """
    print("\nConverting TotalCharges to numeric...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    nan_count = df['TotalCharges'].isnull().sum()
    print(f"NaN values after conversion: {nan_count}")
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame with handled missing values
    """
    print("\nHandling missing values...")

    # Fill missing TotalCharges with median
    if df['TotalCharges'].isnull().sum() > 0:
        median_value = df['TotalCharges'].median()
        df['TotalCharges'].fillna(median_value, inplace=True)
        print(f"Filled missing TotalCharges with median: {median_value:.2f}")

    # Verify no missing values remain
    total_missing = df.isnull().sum().sum()
    print(f"Total missing values after handling: {total_missing}")

    return df


def drop_unnecessary_columns(df):
    """
    Drop columns not needed for modeling

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame without unnecessary columns
    """
    print("\nDropping customerID column...")
    df = df.drop('customerID', axis=1)
    print(f"Shape after dropping: {df.shape}")
    return df


def encode_categorical_features(df):
    """
    Encode categorical features using Label Encoding

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame with encoded categorical features
        dict: Dictionary of label encoders
    """
    print("\nEncoding categorical features...")

    # Identify categorical columns (exclude target)
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Remove target from encoding list
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')

    # Apply Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"  Encoded {col}: {len(le.classes_)} classes")

    return df, label_encoders


def encode_target(df):
    """
    Encode target variable (Churn)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame with encoded target
        LabelEncoder: Fitted label encoder for target
    """
    print("\nEncoding target variable (Churn)...")

    le_target = LabelEncoder()
    df['Churn'] = le_target.fit_transform(df['Churn'])

    print("Mapping:")
    for i, label in enumerate(le_target.classes_):
        print(f"  {label} -> {i}")

    return df, le_target


def scale_numerical_features(df):
    """
    Scale numerical features using StandardScaler

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: DataFrame with scaled numerical features
        StandardScaler: Fitted scaler
    """
    print("\nScaling numerical features...")

    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    print(f"Scaled features: {numerical_features}")

    return df, scaler


def save_preprocessed_data(df, output_path):
    """
    Save preprocessed data to CSV file

    Args:
        df: Preprocessed DataFrame
        output_path: Path to save the CSV file
    """
    print(f"\nSaving preprocessed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully. Shape: {df.shape}")


def save_artifacts(label_encoders, le_target, scaler, output_dir):
    """
    Save preprocessing artifacts (encoders and scaler)

    Args:
        label_encoders: Dictionary of label encoders
        le_target: Target label encoder
        scaler: StandardScaler object
        output_dir: Directory to save artifacts
    """
    print("\nSaving preprocessing artifacts...")

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Save label encoders
    with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"  Saved label_encoders.pkl")

    # Save target encoder
    with open(os.path.join(output_dir, 'target_encoder.pkl'), 'wb') as f:
        pickle.dump(le_target, f)
    print(f"  Saved target_encoder.pkl")

    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler.pkl")


def preprocess_pipeline(input_filepath, output_filepath, artifacts_dir):
    """
    Complete preprocessing pipeline

    Args:
        input_filepath: Path to input CSV file
        output_filepath: Path to save preprocessed CSV file
        artifacts_dir: Directory to save preprocessing artifacts
    """
    print("="*60)
    print("AUTOMATED PREPROCESSING PIPELINE")
    print("="*60)

    # Step 1: Load data
    df = load_data(input_filepath)

    # Step 2: Convert TotalCharges
    df = convert_total_charges(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # Step 5: Encode categorical features
    df, label_encoders = encode_categorical_features(df)

    # Step 6: Encode target
    df, le_target = encode_target(df)

    # Step 7: Scale numerical features
    df, scaler = scale_numerical_features(df)

    # Step 8: Save preprocessed data
    save_preprocessed_data(df, output_filepath)

    # Step 9: Save artifacts
    save_artifacts(label_encoders, le_target, scaler, artifacts_dir)

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Preprocessed data: {output_filepath}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Telco Customer Churn dataset')
    parser.add_argument('--input', type=str, default='WA_Fn-UseC_-Telco-Customer-Churn.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default='preprocessed_data.csv',
                        help='Output CSV file path')
    parser.add_argument('--artifacts-dir', type=str, default='preprocessing/artifacts',
                        help='Directory to save preprocessing artifacts')

    args = parser.parse_args()

    # Run preprocessing pipeline
    preprocess_pipeline(
        input_filepath=args.input,
        output_filepath=args.output,
        artifacts_dir=args.artifacts_dir
    )
