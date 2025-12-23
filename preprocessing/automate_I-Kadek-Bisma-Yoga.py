# automate_Nama.py
import pandas as pd
import os
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def run_automation():
    # Load
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Preprocessing
    scaler = StandardScaler()
    df[iris.feature_names] = scaler.fit_transform(df[iris.feature_names])

    # Save ke folder preprocessing
    os.makedirs('preprocessing', exist_ok=True)
    df.to_csv('preprocessing/iris_preprocessed.csv', index=False)
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    run_automation()