import os 

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pickle


# Make the model directory if it doesn't already exist
os.makedirs('model',exist_ok=True)

# Load the cleaned data to a pandas dataframe
df = pd.read_csv('data/cleaned_properties.csv')

# Make the training and test splits

X = df.drop('price',axis=1)
y = df.price

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42) 


# Define columns for preprocessor
num_cols = X.select_dtypes('number').columns
cat_cols = X.select_dtypes('object').columns

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore',sparse=False), cat_cols)
], remainder='passthrough')


# Create the model with optimal parameters, and combine model and preprocessor into a pipeline
rf = RandomForestRegressor(
        n_estimators= 530,
 max_depth= 29,
 min_samples_split= 3,
 min_samples_leaf= 1,
 max_features = 'auto',
        random_state=42,
        n_jobs=-1
    )

pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', rf)
    ])

# Train the pipeline
pipeline.fit(X_train,y_train)

# Save model at desired directory
with open('model/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)