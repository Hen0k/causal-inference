import sys
sys.path.append("../")
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             log_loss,
                             precision_score,
                             recall_score)

import mlflow
import time

from scripts.cleaning import CleanDataFrame

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


class TrainingPipeline(Pipeline):
    '''
    Class -> TrainingPipeline, ParentClass -> Sklearn-Pipeline
    Extends from Scikit-Learn Pipeline class. Has additional functionality to track 
    model metrics and log model artifacts with mlflow
    params:
    steps: list of tuple (similar to Scikit-Learn Pipeline class)
    '''

    def __init__(self, steps):
        super().__init__(steps)

    def fit(self, X_train, y_train):
        self.__pipeline = super().fit(X_train, y_train)
        return self.__pipeline

    def make_model_name(self, experiment_name, run_name):
        clock_time = time.ctime().replace(' ', '-')
        return experiment_name + '_' + run_name + '_' + clock_time

    def log_model(self,
                  experiment_name,
                  run_name,
                  run_params,
                  run_metrics,
                  run_columns,
                  cm_fig,
                  model_name,
                  save_pipeline=False
                  ):
        mlflow.set_tracking_uri('http://localhost:5000')
        # mlflow.set_tracking_uri('../mlflow_outputs/mlruns')
        mlflow.set_experiment(experiment_name)
        model = self.__pipeline.get_params()["model"]

        # Commented out because of this: https://lifesaver.codes/answer/runid-not-found-when-executing-mlflow-run-with-remote-tracking-server-608

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("columns", run_columns)
            mlflow.log_params(run_params)
            print("Run params saved")

            mlflow.log_metrics(run_metrics)
            print("Run metrics saved")

            mlflow.log_figure(cm_fig, "confusion_matrix.png")

            cm_fig.savefig("./reports/confusion_matrix.png")
            print("figures saved")

        if save_pipeline:
            mlflow.sklearn.log_model(
                sk_model=self.__pipeline,
                artifact_path='models',
                registered_model_name=model_name)
        print(f'Run - {run_name} is logged to Experiment - {experiment_name}')


def plot_cm(y_test, y_pred, model_name):
    fig = plt.figure(figsize=(8, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{model_name} predictions confusion metrix", fontsize=25)
    return fig


def label_encoder(df: pd.DataFrame, cat_columns: list) -> pd.DataFrame:
    lb = LabelEncoder()
    for col in cat_columns:
        df[col] = lb.fit_transform(df[col].astype(str))

    return df


def get_pipeline(model, x):
    # cat_cols = CleanDataFrame.get_categorical_columns(x)
    num_cols = CleanDataFrame.get_numerical_columns(x)

    # categorical_transformer = Pipeline(steps=[
    #     ("cat_encoder", FunctionTransformer(
    #         label_encoder, kw_args={"cat_columns": cat_cols})),
    # ])
    numerical_transformer = Pipeline(steps=[
        ('scale', StandardScaler()),
        # ('norm', Normalizer()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            # ('cat', categorical_transformer, cat_cols)
        ])
    train_pipeline = TrainingPipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return train_pipeline


def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ac = accuracy_score(y_true, y_pred)
    return {
        'mse': round(mse, 2),
        'r2': round(r2, 2),
        'mae': round(mae, 2),
        'ac': round(ac, 2),
    }


def run_train_pipeline(model,
                       x_train: pd.DataFrame,
                       y_train: pd.Series,
                       x_test: pd.DataFrame,
                       y_test: pd.Series,
                       experiment_name: str,
                       run_name: str
                       ):

    train_pipeline = get_pipeline(model, x_train)
    model_name = train_pipeline.make_model_name(experiment_name, run_name)
    run_params = model.get_params()

    train_pipeline.fit(x_train, y_train)
    y_pred = train_pipeline.predict(x_test)
    metrics = get_metrics(y_test, y_pred)
    cm_fig = plot_cm(y_test, y_pred, model_name)
    train_pipeline.log_model(experiment_name=experiment_name,
                             run_name=run_name,
                             run_params=run_params,
                             run_metrics=metrics,
                             cm_fig=cm_fig,
                             run_columns=str(x_test.columns),
                             model_name=model_name,
                             save_pipeline=False)
    return {
        "pipeline": train_pipeline,
        "metrics": metrics,
        "comfusion matrix": cm_fig
    }
