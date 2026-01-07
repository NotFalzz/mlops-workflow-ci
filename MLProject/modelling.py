import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import argparse

# Setup Argument agar bisa dipanggil oleh MLproject
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="credit_risk_preprocessing.csv")
parser.add_argument("--target_column", type=str, default="status")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Start MLflow Run
with mlflow.start_run():
    # 1. Load Data
    # Pastikan file csv ada di folder yang sama atau sesuaikan path-nya
    df = pd.read_csv(args.data_path)
    
    
    # 2. Preprocessing Sederhana (Contoh: Encode target)
    # Sesuaikan 'status' dengan nama kolom target kamu di CSV
    le = LabelEncoder()
    df[args.target_column] = le.fit_transform(df[args.target_column])
    
    X = df.drop(args.target_column, axis=1)
    y = df[args.target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)

    # 3. Training Model
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train, y_train)

    # 4. Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # 5. LOGGING (WAJIB ADA)
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # PENTING: Simpan model agar valid (bukan simulasi)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Model saved. Accuracy: {acc}")
