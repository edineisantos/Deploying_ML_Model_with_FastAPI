import os
import pandas as pd
import pytest

from starter.clean_data import clean_data


def test_file_not_found():
    """
    Test if FileNotFoundError is raised for a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        clean_data("non_existent_file.csv")


def test_clean_data_output():
    """
    Test clean_data function for stripping spaces and saving correctly.
    """
    # Creating a test CSV file
    test_data = pd.DataFrame({
        ' Column 1 ': ['data1 ', ' data2'],
        'Column 2': ['data3', 'data4 ']
    })
    test_file_name = 'test.csv'
    test_data.to_csv(test_file_name, index=False)

    # Running the clean_data function
    cleaned_file_path = clean_data(test_file_name)

    # Checking if cleaned file exists
    assert os.path.exists(cleaned_file_path)

    # Reading the cleaned data
    cleaned_data = pd.read_csv(cleaned_file_path)

    # Verifying column name stripping
    assert all(col.strip() == col for col in cleaned_data.columns)

    # Verifying data stripping
    assert all(cell.strip() == cell for cell in cleaned_data.iloc[:, 0])

    # Cleaning up (removing the created test files)
    os.remove(test_file_name)
    os.remove(cleaned_file_path)
