from TFM_TrainAtScalePipeline.pipeline import TaxiFarePipeline
from TFM_TrainAtScalePipeline.data_ import get_data, clean_data, holdout
from TFM_TrainAtScalePipeline.utils import compute_rmse
from TFM_TrainAtScalePipeline.mlflow_class import MLFlowBase
from google.cloud import storage
import joblib


class Trainer(MLFlowBase):


    def __init__(self, X, y, experiment_name, MLFLOW_URI):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        super().__init__(experiment_name, MLFLOW_URI)
        self.pipeline = None
        self.X = X
        self.y = y



    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''

        tf_pipeline = TaxiFarePipeline()
        self.pipeline = tf_pipeline.create_pipeline()


    def run(self):
        """set and train the pipeline"""

        self.X_train, self.X_test, self.y_train, self.y_test = holdout(
            self.X, self.y)

        self.pipeline.fit(self.X_train, self.y_train)


    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""

        self.y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(self.y_pred, self.y_test)
        return rmse



def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(
        f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}"
    )

if __name__ == "__main__":
    # get data
    data = clean_data(get_data())
    # test_data = clean_data(get_test_data())
    # clean data
    # set X and y
    X = data[0]
    y = data[1]
    EXPERIMENT_NAME = "[ES] [Madrid] [Nicofragon] TaxiFare_Pipeline 1"
    MLFLOW_URI = 'https://mlflow.lewagon.co/'
    STORAGE_LOCATION = 'models/TFM_TrainAtScalePipeline/model.joblib'
    BUCKET_NAME = 'wagon-data-739-frate'
    # hold out
    # train
    # evaluate
    train = Trainer(X, y, EXPERIMENT_NAME, MLFLOW_URI)
    train.set_pipeline()
    train.run()
    train.mlflow_create_run()
    train.mlflow_log_param('model','Linear_regression')
    result = train.evaluate()
    train.mlflow_log_metric('rmse', result)
    save_model(result)
    print(result)
