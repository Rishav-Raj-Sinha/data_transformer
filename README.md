
# Streamlit Data Preprocessing Application

## Overview

This Streamlit application provides an interactive interface for data preprocessing tasks including:
- Importing data
- Handling missing values
- Detecting and removing outliers
- Feature selection/removal
- Encoding categorical variables

The application is designed to assist users in cleaning and preparing datasets for analysis or machine learning.

## Features

1. **Data Import**: Upload CSV files and view dataset dimensions and content.
2. **Handling Null Values**: Various methods to handle missing values including:
   - Dropping rows or columns
   - Replacing missing values with mean/median for numeric columns and the most frequent value for categorical columns
3. **Handling Outliers**: Methods to detect and remove outliers using:
   - Interquartile Range (IQR)
   - Z-Score
4. **Feature Selection/Removal**: Option to select and remove irrelevant columns.
5. **Encoding**: Convert categorical values using:
   - Label Encoding
   - One-Hot Encoding

## Installation

### Prerequisites

Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### Required Packages

You will need the following Python packages:
- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

You can install these packages using pip:
```bash
pip install streamlit scikit-learn pandas numpy matplotlib
```

## Usage

1. **Run the Application**:

   Navigate to the directory where your `app.py` file is located and run:
   ```bash
   streamlit run app.py
   ```

2. **Upload Data**:

   Go to the "Data Import" tab and use the file uploader to upload your CSV file.

3. **Handle Missing Values**:

   In the "Handling Null Values" tab, choose a method to handle missing values and view the transformed dataset.

4. **Detect and Remove Outliers**:

   In the "Handling Outliers" tab, select an outlier detection method (IQR or Z-Score) to identify and optionally remove outliers.

5. **Feature Selection/Removal**:

   In the "Feature Selection/Removal" tab, choose columns to remove from the dataset and view the updated dataset.

6. **Encode Categorical Variables**:

   In the "Encoding" tab, choose an encoding method (Label Encoding or One-Hot Encoding) and see the transformed dataset.

## Code Explanation

### Handling Missing Values

- **Drop Rows**: Removes rows with any missing values.
- **Drop Columns**: Removes columns with any missing values.
- **Replace Numeric Values with Mean**: Replaces missing numeric values with the column mean and non-numeric values with the most frequent value.
- **Replace Numeric Values with Median**: Replaces missing numeric values with the column median and non-numeric values with the most frequent value.

### Handling Outliers

- **IQR Method**:
  - Computes quartiles (Q1, Q3) and IQR.
  - Identifies outliers as values below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR`.
  - Removes outliers based on these bounds.

- **Z-Score Method**:
  - Computes the mean and standard deviation of each numeric column.
  - Calculates Z-Scores for each value.
  - Identifies outliers as values with Z-Scores beyond a specified threshold (e.g., ±3).
  - Removes outliers based on these Z-Scores.

### Encoding

- **Label Encoding**: Converts categorical values to integers.
- **One-Hot Encoding**: Converts categorical values into binary columns, with each column representing a unique category.

## Example

Here’s a quick example of how to use the application:

1. Upload your dataset.
2. Handle any missing values by choosing an appropriate method.
3. Detect and handle outliers using either the IQR or Z-Score method.
4. Select and remove any irrelevant features.
5. Encode categorical variables as needed.

## Contributing

Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Key Points

- The README.md provides a comprehensive overview of the Streamlit application, including installation instructions, usage details, and explanations of the main features.
- It includes code snippets and examples to help users understand how to interact with the application.
- It outlines the methods used for handling missing values, detecting and removing outliers, and encoding categorical variables.