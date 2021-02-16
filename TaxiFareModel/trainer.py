# imports
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer
import numpy as np


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        pipe_time = Pipeline([
        ('features', TimeFeaturesEncoder('pickup_datetime')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        X_distance = ['pickup_longitude', 'pickup_latitude',
           'dropoff_longitude', 'dropoff_latitude']

        X_time = ['pickup_datetime']

        pipe_distance = Pipeline([
            ('distance_transformer', DistanceTransformer()),
            ('standardize', StandardScaler())
            ])

        pipe_preproc = ColumnTransformer(
             [("pipe_distance", pipe_distance, X_distance),
              ("pipe_time", pipe_time, X_time)
             ])

        # Add the model of your choice to the pipeline

        final_pipe = Pipeline([
            ('pipelines_aggregated', pipe_preproc),
            ('model', LinearRegression())

        ])

        # display the pipeline with model


        return final_pipe



    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        final_pipe_trained = pipe.fit(X_train,y_train)
        return final_pipe_trained

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = final_pipe_trained.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test)**2).mean())
        print(rmse)
        return rmse


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X = df[['pickup_datetime', 'pickup_longitude',
           'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
           'passenger_count']]
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
    trainer = Trainer(X,y)
    pipe = trainer.set_pipeline()
    final_pipe_trained = trainer.run()
    rmse = trainer.evaluate(X_test,y_test)
    # print('TODO')
