import pandas as pd


def load_data(file_path):
    try:
        df = pd.read_csv(file_path,parse_dates=['Order Date', 'Ship Date'])

        return df

    except FileNotFoundError:
        print("File not found.")
        return None

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def inspect_data(df):
    print("Dataset Information:")
    df.info(memory_usage='deep')

    print("\nFirst 5 Rows:")
    print(df.head())


def handle_missing_values(df):

    df.dropna(subset=['Row ID'],inplace=True)

    df['Postal Code'] = (df['Postal Code'].fillna('05401'))

    return df


def convert_data_types(df):

    df['Row ID'] = df['Row ID'].astype('int64')

    df['Postal Code'] = df['Postal Code'].astype('str')

    df['Order Date'] = pd.to_datetime(df['Order Date'])

    df['Ship Date'] = pd.to_datetime(df['Ship Date'])

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
        df[column_name] = (df[column_name].str.strip().str.title())

    return df


def clean_data(df):

    df = handle_missing_values(df)

    df = convert_data_types(df)

    df = remove_duplicates(df)

    df = remove_outliers(df, 'Sales')

    df = inconsistent_data(df,['Ship Mode','Segment','Country/Region','City','State','Region','Category','Sub-Category'])

    return df