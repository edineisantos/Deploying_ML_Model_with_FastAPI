"""
Census Data Cleaner

This script cleans census data by removing extra spaces and duplicates, then saves
the cleaned data as 'cleaned_census.csv' in the same directory. It is designed 
to follow PEP8 standards and can be run from the command line.

Dataset: Census data in CSV format.

Author: Edinei Santos
Date: 2023-12-05
"""

import os
import pandas as pd

def clean_data(file_path):
    """
    Clean a CSV file by stripping extra spaces from column names and string values,
    then save the cleaned data as 'cleaned_census.csv' in the same directory.
    
    Parameters:
    file_path (str): The path to the CSV file to be cleaned.

    Returns:
    str: The path to the cleaned file.
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

    # Save the cleaned data as 'cleaned_census.csv'
    cleaned_file_path = os.path.join(os.path.dirname(file_path), 'cleaned_census.csv')
    df.to_csv(cleaned_file_path, index=False)

    return cleaned_file_path

def main():
    """
    Main function to clean the census data and save it as 'cleaned_census.csv'.

    This function defines the file path for the original census data, invokes the cleaning
    process, and saves the cleaned data to a new file. It also prints a message upon
    successful completion.
    """
    # The file name of the CSV to be cleaned
    file_name = 'census.csv'
    data_folder = 'data'
    file_path = os.path.join(os.getcwd(), data_folder, file_name)

    # Clean the data and save as 'cleaned_census.csv'
    cleaned_file_path = clean_data(file_path)

    # Print completion message
    print(f"Data cleaning process completed. Cleaned file saved as: {cleaned_file_path}")

if __name__ == "__main__":
    main()
