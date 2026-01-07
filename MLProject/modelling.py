import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="credit_risk_preprocessing.csv")
parser.add_argument("--target_column", type=str, default="loan_status")  # ✅ FIX INI!
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# 1. Load Data dengan path yang benar
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, args.data_path)

print(f"📂 Loading data from: {file_path}")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Dataset tidak ditemukan di: {file_path}")

df = pd.read_csv(file_path)
print(f"✅ Dataset loaded. Shape: {df.shape}")
print(f"📊 Columns: {df.columns.tolist()}")

# Label encoding untuk target
le = LabelEncoder()
df[args.target_column] = le.fit_transform(df[args.target_column])

X = df.drop(args.target_column, axis=1)
y = df[args.target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=args.random_state
)

print(f"✅ Data split - Train: {len(X_train)}, Test: {len(X_test)}")

# 2. Train Model
print("🚀 Training model...")
model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    random_state=args.random_state
)
model.fit(X_train, y_train)

# 3. Evaluasi
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(f"✅ F1-Score: {f1:.4f}")

# 4. Logging ke MLflow (run sudah aktif otomatis dari MLproject)
mlflow.log_param("n_estimators", args.n_estimators)
mlflow.log_param("random_state", args.random_state)
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)

mlflow.sklearn.log_model(model, "model")

print(f"\n🎉 Training selesai! Model berhasil di-log ke MLflow.")
