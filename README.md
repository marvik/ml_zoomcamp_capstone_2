# Predicting Airbnb Listing Prices in New York City

## Summary

1.  Problem Description
2.  Data
3.  Project Structure
4.  Dependency and Environment Management
5.  Exploratory Data Analysis (EDA)
6.  Data Preparation
7.  Model Training and Tuning
8.  Model Evaluation
9.  Creating Python Scripts from the Notebook
10. Local Model Deployment with Docker
11. Cloud Model Deployment with Google Cloud Run

## 1. Problem Description

This project focuses on predicting the price of Airbnb listings in New York City. The goal is to build a regression model that can accurately estimate the nightly price (log-transformed) of a listing based on its features, such as location, room type, and other relevant factors. This is a valuable task for both hosts (to optimize their pricing strategy) and guests (to find the best deals).

## 2. Data

The dataset used for this project is the **New York City Airbnb Open Data** from Kaggle:

[https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

The dataset contains information about Airbnb listings in NYC, including:

*   **Listing Information:**  latitude, longitude, room type, minimum nights, number of reviews, reviews per month, calculated host listings count, availability 365, neighborhood group.

## 3. Project Structure

The project consists of the following files and directories:

*   `/`:
    *   `capstone_2_airbnb_price_prediction.ipynb`: Jupyter Notebook containing the EDA, data preparation, model training, and evaluation code.
    *   `train.py`: Python script for training the final model.
    *   `predict.py`: Python script for creating the prediction service with Flask.
    *   `predict_sample.py`: Python script for testing the locally deployed model using Docker.
    *   `predict_sample_google_cloud.py`: Python script for testing the deployed model on Google Cloud Run.
    *   `optimized_model.h5`: Saved Keras model file.
    *   `preprocessor_final.pkl`: Saved preprocessor (scaler and one-hot encoder) using joblib.
    *   `Dockerfile`: Defines the Docker image for the prediction service.
    *   `cloudbuild.yaml`: Configuration file for building the Docker image on Google Cloud Build.
    *   `Pipfile` and `Pipfile.lock`: Define the project dependencies managed by `pipenv`.
  

## 4. Dependency and Environment Management

This project uses `pipenv` for dependency and environment management. To set up the project environment, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/marvik/ml_zoomcamp_capstone_2.git
    cd ml_zoomcamp_capstone_2
    ```



2.  **Install `pipenv`:**

    ```bash
    pip install pipenv
    ```

3.  **Create a virtual environment with Python 3.12:**

    ```bash
    pipenv --python 3.12
    ```

4.  **Install project dependencies:**

    ```bash
    pipenv install
    ```

5.  **Activate the environment:**

    ```bash
    pipenv shell
    ```

## 5. Exploratory Data Analysis (EDA)

The Exploratory Data Analysis (EDA) section of the notebook provides insights into the dataset. Key findings include:

*   **Data Overview:** The dataset has 48,895 rows and 16 columns before cleaning and preprocessing.
*   **Missing Values:** The `name` and `host_name` columns have a small number of missing values, while `last_review` and `reviews_per_month` have a significant number of missing entries.
*   **Price Distribution:** The `price` distribution is highly skewed, with most listings concentrated in the lower price range and a long tail of higher-priced listings. Log transformation is used to normalize the distribution.
*   **Geospatial Distribution:** The majority of listings are clustered around Manhattan, with varying densities across different boroughs.
*   **Correlation Analysis:** Numerical features like `latitude`, `longitude` show varying degrees of correlation with price.
*   **Mutual Information:** Categorical features like `room_type`, `neighbourhood_group`, and their interaction `room_type_neighbourhood` show significant mutual information with price, suggesting their importance in predicting prices.

## 6. Data Preparation

In the Data Preparation section of the notebook:

*   Unnecessary columns (`id`, `name`, `host_id`, `host_name`, `last_review`) are dropped.
*   `price` outliers are removed based on quantiles (1% and 99%).
*   Missing values in `reviews_per_month` are filled with 0.
*   The target variable `price` is log-transformed to `log_price` to improve model performance.
*   New features are engineered:
    *   `room_type_neighbourhood`: Interaction between `room_type` and `neighbourhood_group`.

## 7. Model Training and Tuning

A Keras Sequential model (neural network) is used for this regression task.

*   **Model Building:** The model consists of an input layer,  hidden layers with tunable units and dropout rates, and an output layer with a single neuron for predicting the log-transformed price.
*   **Hyperparameter Tuning:** `RandomSearch` from `keras_tuner` is employed to find the best hyperparameter combination, including the number of units in each layer, dropout rates, and learning rate.
*   **Training:** The model is trained using the Adam optimizer, with early stopping and learning rate reduction on a plateau as callbacks to prevent overfitting and improve training efficiency.

**Simplified Model Architecture:**

The final model architecture was simplified based on experimentation and to avoid overfitting.

## 8. Model Evaluation

The model's performance is evaluated using the following metrics:

*   **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual prices.
*   **Root Mean Squared Error (RMSE):** Square root of the average squared difference between predicted and actual prices.
*   **R2 Score:** Coefficient of determination, representing the proportion of variance in the target variable explained by the model.

The final model achieved the following results on the test set:

*   **MAE (log scale):** 0.3026
*   **RMSE (log scale):** 0.4003
*   **R² Score:** 0.6050

These metrics are also calculated and displayed after converting the predictions back to the original USD scale using `np.expm1()`:

*   **MAE (USD):** $45.15
*   **RMSE (USD):** $80.67

## 9. Creating Python Scripts from the Notebook

The code from the notebook `capstone_2_airbnb_price_prediction.ipynb` was used to create three Python scripts for training and deploying the model:

*   **`train.py`:** Trains the final model on the full training data and saves the trained model and preprocessor to files (`optimized_model.h5` and `preprocessor.pkl`).
*   **`predict.py`:** Loads the trained model and preprocessor, and creates a Flask web service to serve predictions via an API endpoint (`/predict`).
*   **`predict_sample.py`:** Sends a sample request to the locally deployed prediction service (for testing with Docker).
*   **`predict_sample_google_cloud.py`:** Sends a sample request to the deployed prediction service on Cloud Run.

## 10. Local Model Deployment with Docker

The prediction service can be deployed locally using Docker:

1.  **Build the Docker image:**

    ```bash
    docker build -t airbnb-price-prediction .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -it -p 9696:9696 airbnb-price-prediction
    ```

3.  **Test the service:**
    You can test the service locally using `predict_sample.py`:

    ```bash
    python predict_sample.py
    ```

## 11. Cloud Model Deployment with Google Cloud Run

The prediction service is deployed to Google Cloud Run using a Docker image built and stored in Artifact Registry.

**Deployment Steps:**

1.  **Enable Google Cloud Services:** Enable the Artifact Registry API, Cloud Build API, and Cloud Run Admin API in your Google Cloud project.
2.  **Create an Artifact Registry Repository:** Create a Docker repository in Artifact Registry (e.g., `airbnb-repo`).
3.  **Build and Push the Docker Image:** Use Cloud Build and the provided `cloudbuild.yaml` to build and push the image to Artifact Registry:

    ```bash
    gcloud builds submit --config cloudbuild.yaml .
    ```

4.  **Deploy to Cloud Run:** Use the `gcloud` command to deploy the service:

    ```bash
    gcloud run deploy airbnb-price-predictor \
     --image=us-central1-docker.pkg.dev/ml-zoomcamp-capstone-1/airbnb-repo/airbnb-price-prediction:latest \
     --port=9696 \
     --region=us-central1 \
     --allow-unauthenticated \
     --memory=2Gi
    ```

    *   Replace `airbnb-repo` with the actual name of your Artifact Registry repository.
    *   Replace `airbnb-price-prediction` with your chosen image name.
    *   Replace `latest` with the appropriate tag if you are not using the `latest` tag.

**Service URL:**

After successful deployment, Cloud Run will provide a service URL. The deployed service will be accessible at a URL similar to:  https://airbnb-price-predictor-xxxxxxxxxx-uc.a.run.app/predict

**Testing the Deployed Service:** Use the `predict_sample_google_cloud.py` script to test the deployed service. Make sure to update the `url` variable in the script with your service URL. ```bash python predict_sample_google_cloud.py

**Service URL:**

The deployed service is accessible at the following URL, use `predict_sample_google_cloud.py` in repository:

https://airbnb-price-predictor-558797510368.us-central1.run.app/predict

You can find screenshots of working service in repository(Cloud_run_1.png, Cloud_run_2.png)

**Testing the Deployed Service:**

1.  Run the script:

    ```bash
    python predict_sample_google_cloud.py
    ```

**Using the Notebook in Google Colab:**

The notebook `capstone_2_airbnb_price_prediction_keras_ipynb_.ipynb` was developed and trained in Google Colab. To run the notebook in Colab:

1.  **Mount Google Drive:** You'll need to mount your Google Drive to access the dataset and save the model. The notebook contains the following code snippet to do this:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

    This will prompt you to authorize Colab to access your Google Drive.

2.  **Adjust File Paths:** Update the file paths in the notebook to point to the correct locations within your Google Drive. For example, change the dataset loading path to something like:

    ```python
    file_path = '/content/drive/MyDrive/Colab Notebooks/capstone_2_airbnb/AB_NYC_2019.csv'
    ```

    Make sure to adjust other paths (e.g., where you save the model) accordingly.

3.  **Run the Cells:** Execute the cells in the notebook sequentially.
