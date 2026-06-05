# TFM — Train at Scale (Taxi Fare ML pipeline on GCP)

An end-to-end, cloud-trained regression pipeline that predicts NYC taxi fares.
The focus is the **engineering**: a packaged, reproducible training pipeline that
reads data from Google Cloud Storage, tracks runs in MLflow, and ships the
trained model back to GCS.

## Approach
- **Problem:** predict `fare_amount` from pickup/dropoff coordinates, datetime
  and passenger count (NYC Taxi Fare dataset).
- **Pipeline (scikit-learn):** custom transformers wired into a `ColumnTransformer`:
  - `DistanceTransformer` — haversine distance between pickup/dropoff GPS points.
  - `TimeFeaturesEncoder` — extracts day-of-week / hour / month / year (tz-aware),
    then one-hot encodes them.
  - `StandardScaler` on distance + `LinearRegression` as the estimator.
- **Cloud / scale:** training data pulled from a **GCS bucket** (`gcsfs`); model
  serialized with `joblib` and uploaded back to GCS; experiments logged to
  **MLflow**; `Makefile` targets for `create_bucket`, `upload_data` and
  `gcp_submit_training` (GCP AI Platform). Installable Python package with a
  `…-run` entrypoint and a GitHub Actions CI skeleton.
- **Validation:** train/test hold-out, metric = RMSE (`compute_rmse`).

## Results
- Trains and evaluates on GCP, logging **RMSE** to MLflow per run.
- *(This repo stores the pipeline and the trained `model.joblib`, not a final
  RMSE figure — the metric is logged to MLflow at train time.)*

## Stack
Python · scikit-learn · pandas · NumPy · joblib · MLflow · Google Cloud Storage
(`gcsfs`, `google-cloud-storage`) · GCP AI Platform · Make · GitHub Actions

## Run
```bash
pip install -r requirements.txt
make clean install test     # install package and run tests
make run_locally            # train the pipeline locally
```

> _(2021–22, Le Wagon "Train at Scale" bootcamp project — my production work since
> lives in private client repos. Trained on Le Wagon's shared GCP/MLflow infra.)_
