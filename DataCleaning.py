import pandas as pd


# ============================================================
# Loading
# ============================================================

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['Order Date', 'Ship Date'])
        return df
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


# ============================================================
# Inspection / EDA
# ============================================================

def get_info(df):
    print("Dataset Information:")
    df.info(memory_usage='deep')

def get_shape(df):
    print("Dataset Shape:")
    print(df.shape)

def get_head(df):
    print(df.head())

def get_summary_statistics(df):
    print("Summary Statistics:")
    print(df.describe(include='all'))

def check_data_types(df):
    print("Data Types:")
    print(df.dtypes)

def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing Values:")
    print(missing_values[missing_values > 0])

def check_duplicates(df):
    duplicates = df.duplicated().sum()
    print(f"Duplicate records found: {duplicates}")

def check_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]

    print(f"Outliers in {column_name}:")
    print(outliers[[column_name]])

def check_inconsistent_data(df, column_names):
    for column_name in column_names:
        inconsistent_values = df[column_name].unique()
        print(f"Inconsistent values in {column_name}:")
        print(inconsistent_values)


# ============================================================
# Cleaning
# ============================================================

def handle_missing_values(df):
    df.dropna(subset=['Row ID'], inplace=True)
    df['Postal Code'] = df['Postal Code'].fillna('05401')
    return df

def convert_data_types(df):
    df['Row ID'] = df['Row ID'].astype('int64')
    df['Postal Code'] = df['Postal Code'].astype('str')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Ship Mode'] = df['Ship Mode'].astype('category')
    df['Segment'] = df['Segment'].astype('category')
    df['Country/Region'] = df['Country/Region'].astype('category')
    df['City'] = df['City'].astype('category')
    df['State'] = df['State'].astype('category')
    df['Region'] = df['Region'].astype('category')
    df['Category'] = df['Category'].astype('category')
    df['Sub-Category'] = df['Sub-Category'].astype('category')
    return df

def remove_duplicates(df):
    duplicates = df.duplicated().sum()
    print(f"Duplicate records found: {duplicates}")
    df.drop_duplicates(inplace=True)
    return df

def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return df

def inconsistent_data(df, column_names):
    for column_name in column_names:
        df[column_name] = df[column_name].str.strip().str.title()
    return df


# ============================================================
# Pipeline
# ============================================================

def clean_data(df):
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers(df, 'Sales')
    df = inconsistent_data(
        df,
        ['Ship Mode', 'Segment', 'Country/Region', 'City', 'State', 'Region', 'Category', 'Sub-Category']
    )
    df = convert_data_types(df)
    return df