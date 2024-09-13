import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Function to detect outliers using IQR
def detect_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outliers
        outliers[col] = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
        
    return outliers

# Function to detect outliers using Z-score
def detect_outliers_zscore(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_cols:
        col_mean = np.mean(df[col])
        col_std = np.std(df[col])

        # Z-score formula
        z_scores = (df[col] - col_mean) / col_std

        # Find outliers
        outliers[col] = df[col][np.abs(z_scores) > threshold]

    return outliers


# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Import", "Handling Null Values", "Handling Outliers","Feature Selection/Removal","Encoding"])

# Data Import Tab
with tab1:
    with st.container(border=True):
        uploaded_df = st.file_uploader("Upload your data in CSV format")
    with st.container(border=True):
        if uploaded_df is not None:
            st.session_state.df = pd.read_csv(uploaded_df)
            st.write(f"The dimension of your dataset is: {st.session_state.df.shape}")
            st.dataframe(st.session_state.df)
        else:
            st.error("Data field empty")

# Handling Null Values Tab
if st.session_state.df is not None:
    with tab2:
        col1, col2, col3 = st.columns([1, 0.5, 1])

        with col1:
            with st.container(border=True,height=700):
                st.header("Null Value Analysis")
                st.dataframe(st.session_state.df.isnull().sum(), width=650, height=600)

        with col2:
            with st.container(border=True,height=700):

                st.header("Handle Missing Values")
                choice = st.radio("Choose a method:", [
                    "Drop rows",
                    "Drop columns",
                    "Replace numeric values with mean and non-numeric with most frequently occurring value",
                    "Replace numeric values with median and non-numeric with most frequently occurring value"
                ])

        with col3:
            with st.container(border=True,height=700):
                st.header("Transformed Dataset")
                if choice == "Drop rows":
                    df_cleaned = st.session_state.df.dropna()
                    st.session_state.df = df_cleaned
                    st.write(f"New dimensions are {df_cleaned.shape}")
                    st.dataframe(df_cleaned, width=650)
                elif choice == "Drop columns":
                    df_cleaned = st.session_state.df.dropna(axis=1)
                    st.session_state.df = df_cleaned
                    st.write(f"New dimensions are {df_cleaned.shape}")
                    st.dataframe(df_cleaned, width=650)
                elif choice == "Replace numeric values with mean and non-numeric with most frequently occurring value":
                    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                    non_numeric_cols = st.session_state.df.select_dtypes(exclude=[np.number]).columns

                    numeric_imputer = SimpleImputer(strategy='mean')
                    st.session_state.df[numeric_cols] = numeric_imputer.fit_transform(st.session_state.df[numeric_cols])

                    categorical_imputer = SimpleImputer(strategy='most_frequent')
                    st.session_state.df[non_numeric_cols] = categorical_imputer.fit_transform(st.session_state.df[non_numeric_cols])

                    st.dataframe(st.session_state.df, width=650)
                    mean_values = numeric_imputer.statistics_
                    mean_values_dict = dict(zip(numeric_cols, mean_values))
                    st.dataframe(pd.DataFrame.from_dict(mean_values_dict, orient='index', columns=['Mean']), width=650)

                elif choice == "Replace numeric values with median and non-numeric with most frequently occurring value":
                    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
                    non_numeric_cols = st.session_state.df.select_dtypes(exclude=[np.number]).columns

                    numeric_imputer = SimpleImputer(strategy='median')
                    st.session_state.df[numeric_cols] = numeric_imputer.fit_transform(st.session_state.df[numeric_cols])

                    categorical_imputer = SimpleImputer(strategy='most_frequent')
                    st.session_state.df[non_numeric_cols] = categorical_imputer.fit_transform(st.session_state.df[non_numeric_cols])

                    st.dataframe(st.session_state.df, width=650)
                    median_values = numeric_imputer.statistics_
                    median_values_dict = dict(zip(numeric_cols, median_values))
                    st.dataframe(pd.DataFrame.from_dict(median_values_dict, orient='index', columns=['Median']), width=650)

# Handling Outliers Tab
if st.session_state.df is not None:

#if 'df' in st.session_state:

    with tab3:
        col11, col22, col33 = st.columns([0.5, 1, 1])

        with col11:
            with st.container(border=True,height=700):
                df_cleaned = st.session_state['df']  # Use the cleaned data from the session state
                st.header("Outlier Analysis")

                # Select method for outlier detection
                outlier_method = st.radio("Select outlier detection method:", ["IQR", "Z-Score"])
        with col22:
            with st.container(border=True,height=700):
                # Handling outliers
                if outlier_method == "IQR":
                    # Outlier detection using IQR
                    outliers_iqr = detect_outliers_iqr(df_cleaned)
                    st.header("Outlier Detection using IQR:")
                    st.dataframe(outliers_iqr,height=600)
                    
                
                elif outlier_method == "Z-Score":
                    # Outlier detection using Z-Score
                    outliers_zscore = detect_outliers_zscore(df_cleaned)
                    st.header("Outlier Detection using Z-Score:")
                    st.dataframe(outliers_zscore,height=600)


        with col33:
            with st.container(border=True,height=700):
                # Option to remove outliers
                    if outlier_method == "IQR":
                        for col in outliers_iqr:
                            Q1 = df_cleaned[col].quantile(0.25)
                            Q3 = df_cleaned[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                        st.header("Outliers removed using IQR.")
                        st.dataframe(df_cleaned,height=600)

                    # Option to remove outliers
                    if outlier_method == "Z-Score":
                        for col in outliers_zscore:
                            col_mean = np.mean(df_cleaned[col])
                            col_std = np.std(df_cleaned[col])
                            z_scores = (df_cleaned[col] - col_mean) / col_std
                            df_cleaned = df_cleaned[np.abs(z_scores) <= 3]  # Remove rows with Z-scores > threshold (3)
                        st.header("Outliers removed using Z-Score.")
                        st.dataframe(df_cleaned,height=600)
                    
                    st.session_state['df'] = df_cleaned

# feature selection/removal
if st.session_state.df is not None:

    with tab4:

        col111, col222 = st.columns([0.2,0.8])

        with col111:
            with st.container(height=700):
                st.header("Choose columns that are not needed")
                dff = st.session_state.df
                options = list(dff.columns)
                irrelevant_column = st.multiselect("choose one or more columns for removal",options)
                # def delete_column():
                #     dff.drop(columns=irrelevant_column, inplace=True)  # Define a function to drop the column
                #     st.session_state.df = dff
                #     st.session_state.irrelevant_column = []

                # # Use the on_click to call the function when the button is pressed
                # st.button("Delete", on_click=delete_column)
                if st.button("Delete Columns"):
                # Step 3: Modify the dataframe in session_state
                    dffnew = dff.drop(columns=irrelevant_column)
                    st.session_state.df = dffnew  # Update the dataframe in session_state

            
        with col222:
            with st.container(height=700):
                st.header("Transformed dataset")
                st.dataframe(st.session_state.df,height=600)

if st.session_state.df is not None:

    with tab5:
        col51, col52 = st.columns([0.2,0.8])

        with col51:
            with st.container(height=700):
                st.header("Choose encoding method")
                encoding_method = st.radio("Select encoding method :", ["Label encoding", "One-hot encoding"])
                encodingdf = st.session_state.df
                if encoding_method == "Label encoding":
                    label_encoder = LabelEncoder()
                    for col in encodingdf.columns:
                        if encodingdf[col].dtype == 'object':  # Check if the column is non-numeric
                            encodingdf[col] = label_encoder.fit_transform(encodingdf[col])
                    st.session_state.df = encodingdf

                if encoding_method == "One-hot encoding":
                    non_numeric_columns = encodingdf.select_dtypes(include=['object']).columns.tolist()
                    df_encoded = pd.get_dummies(encodingdf, columns=non_numeric_columns, drop_first=True)

                    # encodingdf = pd.get_dummies(encodingdf, drop_first=True)  # drop_first=True avoids dummy variable trap
                    st.session_state.df = df_encoded
        with col52:
        
            with st.container(height=700):
                st.header("Transformed dataset")
                st.dataframe(st.session_state.df,height=600)

