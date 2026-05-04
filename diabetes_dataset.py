from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import dagshub
import os

import mlflow
mlflow.autolog()
# from sklearn.datasets import load_diabetes


# Init DagsHub
dagshub.init(repo_owner='dinesh008luck', repo_name='mlflow_v03', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/dinesh008luck/mlflow_v03.mlflow")

df = pd.read_csv("https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv")

# splitting the data into features and target


X= df.drop('Outcome',axis=1)
y= df['Outcome']

# splitting into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# creating a Random Forest Classifier
rf= RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
#Applying Grid Search CV

grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=- 1,verbose=2)

mlflow.set_experiment('diabetes_data_set')

with mlflow.start_run():

    grid_search.fit(X_train,y_train)

    #Displaying the best parameters and best scores

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    #params

    # mlflow.log_params(best_params)

    #metrics

    # mlflow.log_metric('accuracy',best_score)

    #data

    train_df = X_train.copy()
    train_df['Outcome'] = y_train.values

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training")
    
    test_df = X_test.copy()
    test_df['Outcome'] = y_test.values

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"validation")

    #source code
    
    if "__file__" in globals():
        mlflow.log_artifact(__file__)

    #model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest_model")

    #tags
    mlflow.set_tag("author","Dinesh")

    print(best_params)
    print(best_score)