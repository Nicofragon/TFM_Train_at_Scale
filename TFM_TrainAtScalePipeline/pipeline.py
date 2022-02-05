from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from TFM_TrainAtScalePipeline.encoders import TimeFeaturesEncoder, DistanceTransformer


class TaxiFarePipeline():

    def __init__(self):
        pass

    def create_pipeline(self):

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])
        return pipe
