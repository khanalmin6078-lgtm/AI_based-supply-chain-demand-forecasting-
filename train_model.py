import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("dataset/sales_data.csv")

print("Dataset Loaded Successfully!")
print("\nFirst 5 Records:")
print(df.head())


# =========================================================
# DATE FEATURES
# =========================================================

df["Date"] = pd.to_datetime(df["Date"])

df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)


# =========================================================
# INPUT FEATURES
# =========================================================

X = df[
    [
        "Product",
        "Category",
        "Price",
        "Discount",
        "Current_Stock",
        "Lead_Time",
        "Season",
        "Day",
        "DayOfWeek",
        "Month",
        "WeekOfYear"
    ]
]

y = df["Demand"]


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

categorical_features = [
    "Product",
    "Category",
    "Season"
]


# =========================================================
# NUMERICAL FEATURES
# =========================================================

numerical_features = [
    "Price",
    "Discount",
    "Current_Stock",
    "Lead_Time",
    "Day",
    "DayOfWeek",
    "Month",
    "WeekOfYear"
]


# =========================================================
# PREPROCESSING
# =========================================================

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


# =========================================================
# MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)


# =========================================================
# PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# TRAIN MODEL
# =========================================================

pipeline.fit(X_train, y_train)


# =========================================================
# TEST MODEL
# =========================================================

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)


print("\nModel trained successfully!")
print("Mean Absolute Error:", round(mae, 2))


# =========================================================
# SAVE MODEL
# =========================================================

with open("model.pkl", "wb") as file:
    pickle.dump(pipeline, file)


print("\nNew model.pkl saved successfully!")
print("Date-based future prediction is ready!")