import sys
import pathlib
import sys
import joblib
import mlflow

import pandas as pd
import numpy as np
from hyperopt import hp
from sklearn.model_selection import train_test_split
from hyperopt.pyll.base import scope
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from xgboost import XGBRegressor




curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
sys.path.append(home_dir.as_posix())
from src.features import transformer


def find_best_model_with_params(X_train, y_train, X_test, y_test):

    hyperparameters = {
        "RandomForestRegressor": {
            "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
            "max_depth": hp.choice("max_depth", [6, 8, 10]),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
        },
        "XGBRegressor": {
            "n_estimators": hp.choice("n_estimators", np.random.randint(950, 1000, size=10)),
            "max_depth": hp.choice("max_depth", np.random.randint(3, 10, size=5)),
            "learning_rate": hp.choice("learning_rate", np.random.uniform(0.01, 0.1, size=5)),
            "subsample": hp.choice("subsample", np.random.uniform(0.8, 1.0, size=5)),
            "colsample_bytree": hp.choice("colsample_bytree", np.random.uniform(0.8, 1.0, size=5)),
            "gamma": hp.choice("gamma", np.random.uniform(0.0, 0.1, size=5)),
            "reg_alpha": hp.choice("reg_alpha", np.random.uniform(0.0, 0.1, size=5)),
            "reg_lambda": hp.choice("reg_lambda", np.random.randint(0.5, 1, size=5)),
            "min_child_weight": hp.choice("min_child_weight", np.random.randint(10, 20, size=5)),
        },
    }



    def evaluate_model(hyperopt_params):
        params = hyperopt_params
        if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
        if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) 
        if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_rmse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric('RMSE', model_rmse)  # record actual metric with mlflow run
        loss = model_rmse  
        return {'loss': loss, 'status': STATUS_OK}

    space = hyperparameters['XGBRegressor']
    with mlflow.start_run(run_name='XGBRegressor'):
        argmin = fmin(
            fn=evaluate_model,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=Trials(),
            verbose=True
            )
    run_ids = []
    with mlflow.start_run(run_name='XGB Final Model') as run:
        run_id = run.info.run_id
        run_name = run.data.tags['mlflow.runName']
        run_ids += [(run_name, run_id)]
        
        # configure params
        params = space_eval(space, argmin)
        if 'max_depth' in params: params['max_depth']=int(params['max_depth'])       
        if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight'])
        if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])  
        mlflow.log_params(params)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'discount')  # persist model with mlflow for registering
    return model


def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + "/discount.joblib")


def main():
    input_file = '/data/processed'
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + "/models"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    TARGET = "Discount"

    train_features = pd.read_csv(data_path + "/data_discount.csv")
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    ct = transformer.column_transformer_1()
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    joblib.dump(ct, output_path + '/column_transformer_discount.pkl')

    trained_model = find_best_model_with_params(X_train, y_train, X_test, y_test)
    save_model(trained_model, output_path)
    # push this model to S3 and also copy in the root folder for Dockerfile to pick


if __name__ == "__main__":
    main()
