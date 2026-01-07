import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "model")
