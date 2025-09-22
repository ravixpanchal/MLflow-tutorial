import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# (Optional) Set tracking to local file-based storage
mlflow.set_tracking_uri("file:///C:/Study Materials/5th Sem/MLOps/Lab/Lab-3/MLflow-tutorial/mlruns")

# Enable autologging (automatically logs model, params, metrics, confusion matrix, etc.)
mlflow.sklearn.autolog()

# Load dataset
wine = load_wine()
x = wine.data
y = wine.target

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Define parameters for RF model
max_depth = 10
n_estimators = 6

mlflow.set_experiment("mlflow123")
# experiment_id=555295523475723985
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(x_train, y_train)
    
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix visualization (manual logging of plot)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

