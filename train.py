import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch
import joblib

# Data Preparation 
def prepare_data(df):
    df = df.copy()
    df = df.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1)
    df = df[df['price'].between(df['price'].quantile(0.01), df['price'].quantile(0.99))]
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['log_price'] = np.log1p(df['price'])
    df['room_neighborhood'] = df['room_type'] + "_" + df['neighbourhood_group']
    return df.drop('price', axis=1)

# Load and prepare data
file_path = 'AB_NYC_2019.csv'  
df = pd.read_csv(file_path)
processed_df = prepare_data(df)

# Split data
train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42)

# Feature Configuration
numerical_features = ['latitude', 'longitude', 'minimum_nights',
                      'number_of_reviews', 'reviews_per_month',
                      'calculated_host_listings_count', 'availability_365']
categorical_features = ['neighbourhood_group', 'room_type', 'room_neighborhood']

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Prepare data
X_train = preprocessor.fit_transform(train_df)
X_test = preprocessor.transform(test_df)
y_train = train_df['log_price'].values
y_test = test_df['log_price'].values

# Model Building
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', 64, 256, step=64), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5)))
    model.add(Dense(units=hp.Int('units_2', 32, 128, step=32), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 1e-3])), loss='mse', metrics=['mae'])
    return model

# Hyperparameter Tuning
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='airbnb_tuning',
    project_name='price_prediction'
)

tuner.search(X_train, y_train, epochs=20, validation_split=0.2, batch_size=256, verbose=1)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]

# Final Training
history = best_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ],
    verbose=1
)

# Save model and preprocessor
MODEL_FILE = 'optimized_model.h5'
PREPROCESSOR_FILE = 'preprocessor.pkl'
best_model.save(MODEL_FILE)
joblib.dump(preprocessor, PREPROCESSOR_FILE)
print(f"Model saved to {MODEL_FILE}, preprocessor saved to {PREPROCESSOR_FILE}")