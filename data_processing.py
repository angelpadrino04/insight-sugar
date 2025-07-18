import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():
    try:
        df = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        print("Error: Archivo 'diabetes.csv' no encontrado.")
        print("Descarga el dataset de Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        exit()

    print("\nAnálisis inicial de datos:")
    print(df.describe())

    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df[col].replace(0, np.nan, inplace=True)

    # Manejar valores faltantes (en este caso, NaN por el reemplazo anterior)
    if df.isnull().sum().sum() > 0:
        print(f"\nValores faltantes encontrados:\n{df.isnull().sum()}")
        # Imputar con la mediana por grupo (diabéticos/no diabéticos)
        for col in zero_cols:
            df[col] = df.groupby('Outcome')[col].transform(
                lambda x: x.fillna(x.median()))

    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
