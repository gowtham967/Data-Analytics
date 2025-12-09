import pandas as pd
from sklearn.preprocessing import LabelEncoder
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, errors='ignore')
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    df = df.fillna(df.median())
    return df
