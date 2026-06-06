import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRFRegressor

# Load datasets
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Combine datasets
data = pd.concat([exercise, calories['Calories']], axis=1)

# Features and target
X = data.drop(columns=['User_ID', 'Calories'])
y = data['Calories']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Preprocessing
numerical_features = [
    'Age', 'Height', 'Weight',
    'Duration', 'Heart_Rate', 'Body_Temp'
]

categorical_features = ['Gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OrdinalEncoder(), categorical_features)
    ]
)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRFRegressor())
])

# Train
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Metrics
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Cross Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_val_score(
    pipeline,
    X,
    y,
    cv=kfold,
    scoring='r2'
)

print("CV Mean R2:", cv_results.mean())

# Save model
with open("calorie_predictor.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully!")