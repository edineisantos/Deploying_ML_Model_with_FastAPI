"""
Census Data Cleaner

This script cleans census data by removing extra spaces and duplicates, then overwrites
the original file with the cleaned data. It is designed to follow PEP8 standards and can
be run from the command line.

Dataset: Census data in CSV format.

Author: Edinei Santos
Date: 2023-12-05
"""

import pandas as pd
import os

def clean_data(file_path):
    """
    Clean a CSV file by stripping extra spaces from column names and string values,
    then overwrite the original file with the cleaned data.
    
    Parameters:
    file_path (str): The path to the CSV file to be cleaned.

    Returns:
    None: The function directly modifies the file at file_path.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Strip spaces from the column names
    df.columns = df.columns.str.strip()

    # Strip spaces from object type columns (which are usually strings)
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()

    # Drop duplicates
    df = df.drop_duplicates()

    # Overwrite the original file with the cleaned data
    df.to_csv(file_path, index=False)
    print(f"File '{file_path}' has been cleaned and overwritten.")

def main():
    # The file name of the CSV to be cleaned
    file_name = 'census.csv'
    data_folder = 'data'
    file_path = os.path.join(os.getcwd(), data_folder, file_name)

    # Clean the data and overwrite the original file
    clean_data(file_path)

    # Print completion message
    print("Data cleaning process completed.")

if __name__ == "__main__":
    main()
