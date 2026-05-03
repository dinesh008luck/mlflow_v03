import mlflow
import mlflow.sklearn
import dagshub
import os

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier  # if needed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Init DagsHub
dagshub.init(repo_owner='dinesh008luck', repo_name='mlflow_v02', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/dinesh008luck/mlflow_v02.mlflow")

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Params
max_depth = 1

mlflow.set_experiment('iris-decision-tree')

with mlflow.start_run():

    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log params & metrics
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_metric('accuracy', accuracy)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # Log script safely
    if "__file__" in globals():
        mlflow.log_artifact(__file__)

    # Log model
    mlflow.sklearn.log_model(model, "decision_tree_model")

    mlflow.set_tag('author','Dinesh')
    mlflow.set_tag('model','decision_tree')

    print('accuracy:', accuracy)