import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("dataset/sales_data.csv")

print("Dataset Loaded Successfully!")
print("\nFirst 5 Records:")
print(df.head())


# ==============================
# FEATURES AND TARGET
# ==============================

X = df[
    [
        "Product",
        "Category",
        "Price",
        "Discount",
        "Current_Stock",
        "Lead_Time",
        "Season"
    ]
]

y = df["Demand"]


# ==============================
# CATEGORICAL AND NUMERICAL COLUMNS
# ==============================

categorical_features = [
    "Product",
    "Category",
    "Season"
]

numerical_features = [
    "Price",
    "Discount",
    "Current_Stock",
    "Lead_Time"
]


# ==============================
# PREPROCESSING
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# ==============================
# CREATE MODEL
# ==============================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==============================
# CREATE PIPELINE
# ==============================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# TRAIN MODEL
# ==============================

pipeline.fit(X_train, y_train)


# ==============================
# MODEL EVALUATION
# ==============================

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("\nModel trained successfully!")
print("Mean Absolute Error:", mae)


# ==============================
# SAVE MODEL
# ==============================

with open("model.pkl", "wb") as file:
    pickle.dump(pipeline, file)


print("\nNew model.pkl saved successfully!")
print("Unknown products will now be handled safely!")