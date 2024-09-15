import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#page configs

st.set_page_config(page_title="Data Transformer",layout="wide",initial_sidebar_state="collapsed")
hide_decoration_bar_style = '''<style>header {visibility: hidden;}
</style><style> .main {overflow: hidden} </style><style>footer{visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

# Function to detect outliers using IQR
def detect_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_count = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outliers
        outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_count[col] = len(outliers)
    return outliers_count

def remove_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    mask = np.ones(len(df), dtype=bool)
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Update mask to filter out outliers
        mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
    
    # Remove outliers from DataFrame
    df_cleaned = df[mask]
    
    return df_cleaned


# Function to detect outliers using Z-score
def detect_outliers_zscore(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_count = {}
    
    for col in numeric_cols:
        col_mean = np.mean(df[col])
        col_std = np.std(df[col])

        # Z-score formula
        z_scores = (df[col] - col_mean) / col_std

        # Find outliers
        outliers = df[col][np.abs(z_scores) > threshold]
        
        # Store count of outliers for each column
        outliers_count[col] = len(outliers)
        
    return outliers_count

def remove_outliers_zscore(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    mask = np.ones(len(df), dtype=bool)
    
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()

        # Z-score formula
        z_scores = (df[col] - col_mean) / col_std

        # Update mask to filter out outliers
        mask &= np.abs(z_scores) <= threshold
    
    # Remove outliers from DataFrame
    df_cleaned = df[mask]
    
    return df_cleaned

#df = pd.DataFrame()

col1 , col2 = st.columns(2)

with col1:
    col11, col22 = st.columns(2)
    with col11:
        
        with st.container(height=263):
            st.subheader("Upload Data")
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None :
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Upload Data")
                df = pd.DataFrame() 
    
        with st.container(height=180):
            if df is not None:

                st.subheader("Sort Data")
                col = st.selectbox("Select column to sort", df.columns,index=None)

        with st.container(height=75):
            if df is not None:
                # st.subheader("Remove Duplicate Values")
                # choice = st.checkbox("Remove duplicates")
                activated = st.toggle("Delete Duplicate Values")
        with st.container(height=208):
            if df is not None:
                st.subheader("Remove Null Values")
                choice = st.selectbox("Choose a method:", [
                                "Drop rows",
                                "Drop columns",
                                "Replace numeric with mean and non-numeric with most frequently occurring value",
                                "Replace numeric values with median and non-numeric with most frequently occurring value"
                            ],index=None)


    
    with col22:
        if df is not None:

            with st.container(height=185):
                st.subheader("Remove Outliers")
                outlier_method = st.radio("Select outlier detection method:", ["IQR", "Z-Score"],index=None)

            with st.container(height=175):
                st.subheader("Remove Features")
                options = list(df.columns)
                irrelevant_column = st.multiselect("choose one or more columns for removal",options)

            with st.container(height=180):
                st.subheader("Choose Encoding Method")
                encoding_method = st.radio("Select encoding method :", ["Label encoding", "One-hot encoding"],index=None)

            with st.container(height=186):
                st.subheader("Choose Scaling Method")
                scaling_method = st.radio("Select scaling method :", ["StandardScaler", "MinMaxScaler"],index=None)

with col2:
    if df is not None:
        col21, col222 = st.columns([0.84,0.16])

        with col21:
            placeholder = st.empty()
            with placeholder.container(height=773):
                st.subheader("Uploaded Data")
                st.dataframe(df,height=450,width=820,hide_index=True)

                dfrows, dfcols = df.shape
                st.write(f"Dimensions : {str(dfrows)} X {str(dfcols)}")      

                dfnull = df.isnull().sum().sum()
                st.write(f"Total number of null values : {dfnull}")

                # outliers detection usign IQR
                st.write(f"Total number of outliers (IQR) : {detect_outliers_iqr(df)}")
                        
                # Outlier detection using Z-Score
                st.write(f"Total number of outliers (Zscore) : {detect_outliers_zscore(df)}")


        with col222:
                container =  st.container(height=621)

        with col222:
            with st.container(height=136):
                downloader = True
                if st.button("Transform"):
                    #drop null
                    if choice == "Drop rows":
                        df_cleaned = df.dropna()
                        df = df_cleaned
                        # st.write(f"New dimensions are {df_cleaned.shape}")
                        # st.dataframe(df_cleaned, width=650)
                    elif choice == "Drop columns":
                        df_cleaned = df.dropna(axis=1)
                        df = df_cleaned
                        # st.write(f"New dimensions are {df_cleaned.shape}")
                        # st.dataframe(df_cleaned, width=650)
                    elif choice == "Replace numeric values with mean and non-numeric with most frequently occurring value":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

                        numeric_imputer = SimpleImputer(strategy='mean')
                        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

                        categorical_imputer = SimpleImputer(strategy='most_frequent')
                        df[non_numeric_cols] = categorical_imputer.fit_transform(df[non_numeric_cols])

                        # st.dataframe(df, width=650)
                        # mean_values = numeric_imputer.statistics_
                        # mean_values_dict = dict(zip(numeric_cols, mean_values))
                        # st.dataframe(pd.DataFrame.from_dict(mean_values_dict, orient='index', columns=['Mean']), width=650)

                    elif choice == "Replace numeric values with median and non-numeric with most frequently occurring value":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

                        numeric_imputer = SimpleImputer(strategy='median')
                        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

                        categorical_imputer = SimpleImputer(strategy='most_frequent')
                        df[non_numeric_cols] = categorical_imputer.fit_transform(df[non_numeric_cols])

                        # st.dataframe(df, width=650)
                        # median_values = numeric_imputer.statistics_
                        # median_values_dict = dict(zip(numeric_cols, median_values))
                        # st.dataframe(pd.DataFrame.from_dict(median_values_dict, orient='index', columns=['Median']), width=650)
                    
                    # outlier detection
                    if outlier_method == "IQR":
                        # Outlier detection using IQR
                        outlier_removed = remove_outliers_iqr(df)
                        df = outlier_removed
                        # st.write("outlier removal done")

                
                    elif outlier_method == "Z-Score":
                        # Outlier detection using Z-Score
                        outliers_removed= remove_outliers_zscore(df)
                        df = outliers_removed

                    #remove unwanted feature
                    if irrelevant_column is not None :
                        remove_unwanted_df = df.drop(columns=irrelevant_column)
                        df = remove_unwanted_df

                    #encoding
                    if encoding_method == "Label encoding":
                        # st.write("encoding done")

                        label_encoder = LabelEncoder()
                        for col in df.columns:
                            if df[col].dtype == 'object':  # Check if the column is non-numeric
                                df[col] = label_encoder.fit_transform(df[col])

                    elif encoding_method == "One-hot encoding":
                        non_numeric_columns = df.select_dtypes(include=['object']).columns.tolist()
                        df_encoded = pd.get_dummies(df, columns=non_numeric_columns, drop_first=True)
                        df = df_encoded


                    # data scaling
                    if scaling_method is not None:
                        # 1. Standardization (Z-Score Scaling)
                        scaler_standard = StandardScaler()
                        df_standard_scaled = scaler_standard.fit_transform(df)

                        # Convert back to DataFrame for better readability
                        df_standard_scaled = pd.DataFrame(df_standard_scaled, columns=df.columns)
                        df = df_standard_scaled

                        # 2. Min-Max Scaling (Scaling between 0 and 1)
                        scaler_minmax = MinMaxScaler()
                        df_minmax_scaled = scaler_minmax.fit_transform(df)

                        # Convert back to DataFrame for better readability
                        df_minmax_scaled = pd.DataFrame(df_minmax_scaled, columns=df.columns)

                    # sorting data

                    if col is not None:
                        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                            df = df.sort_values(by=col).reset_index(drop=True)
                            # st.write("sorted")
                        else:
                            st.warning("Selected column cannot be sorted")

                    with placeholder.container(height=773):
                        st.subheader("Transformed Data")
                        st.dataframe(df,height=450,width=820,hide_index=True)

                        dfrowsnew, dfcolsnew = df.shape
                        st.write(f"Dimensions : {str(dfrowsnew)} X {str(dfcolsnew)}")      

                        dfnullnew = df.isnull().sum().sum()
                        st.write(f"Total number of null values : {dfnullnew}")

                        # outliers detection usign IQR
                        st.write(f"Total number of outliers (IQR) : {detect_outliers_iqr(df)}")
                                
                        # Outlier detection using Z-Score
                        st.write(f"Total number of outliers (Zscore) : {detect_outliers_zscore(df)}")

                    st.toast('Your data has been transformed!')
                    downloader = False
                    
                csv = df.to_csv(index=False).encode('utf-8')

                if st.download_button(
                    label="Download",
                    data=csv,
                    file_name='my_data.csv',
                    mime='text/csv', 
                    disabled=downloader):
                    st.toast('Your data has been downloaded!')




#container =  st.container(height=186)

# colrows, colcolumns, colnull, colnumeric, noncolnumeric = container.columns(5)

num_rows, num_cols = df.shape
row_delta = num_rows - dfrows
container.metric("Number of rows",num_rows,delta=row_delta)
col_delta = num_cols - dfcols
container.metric("Number of columns",num_cols,col_delta)

null_cells = df.isnull().sum().sum()
null_delta = int(null_cells - dfnull)
container.metric("Null values",null_cells,null_delta)
# container.metric("Number of rows",num_rows,row_delta)
# container.metric("Number of rows",num_rows,row_delta)