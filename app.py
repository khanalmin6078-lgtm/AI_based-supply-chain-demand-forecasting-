from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os
import json
from datetime import datetime, date


app = Flask(__name__)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_FILE = "model.pkl"

HISTORICAL_DATA = "dataset/sales_data.csv"

UPLOADED_DATA = "dataset/uploaded_sales_data.csv"

PREDICTION_FILE = "prediction_result.json"


# =========================================================
# LOAD MODEL
# =========================================================

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully!")


# =========================================================
# GET ACTIVE DATA
# =========================================================

def get_active_data():

    if os.path.exists(UPLOADED_DATA):

        try:

            df = pd.read_csv(UPLOADED_DATA)

            if not df.empty:
                return df

        except Exception:
            pass

    return pd.read_csv(HISTORICAL_DATA)


# =========================================================
# SAVE LAST PREDICTION
# =========================================================

def save_prediction(result):

    with open(
        PREDICTION_FILE,
        "w"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )


# =========================================================
# LOAD LAST PREDICTION
# =========================================================

def load_prediction():

    if not os.path.exists(PREDICTION_FILE):
        return None

    try:

        with open(
            PREDICTION_FILE,
            "r"
        ) as file:

            return json.load(file)

    except Exception:

        return None


# =========================================================
# GET SEASON FROM DATE
# =========================================================

def get_season_from_date(prediction_date):

    month = prediction_date.month

    # March - May
    if month in [3, 4, 5]:

        return "Summer"

    # June - September
    elif month in [6, 7, 8, 9]:

        return "Monsoon"

    # October - February
    else:

        return "Winter"


# =========================================================
# STOCK-OUT RISK
# =========================================================

def calculate_risk(stock, demand):

    if stock < demand:

        return "High Risk"

    elif stock <= demand * 1.2:

        return "Medium Risk"

    else:

        return "Low Risk"


# =========================================================
# REORDER QUANTITY
# =========================================================

def calculate_reorder(stock, demand):

    recommended_stock = demand * 1.20

    reorder_quantity = (
        recommended_stock - stock
    )

    if reorder_quantity < 0:

        reorder_quantity = 0

    return round(reorder_quantity)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# FUTURE DEMAND PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    df = get_active_data()

    # Product list
    products = sorted(
        df["Product"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Latest details for each product
    product_details = {}

    for product_name in products:
        product_rows = df[
            df["Product"].astype(str) == product_name
        ]

        if not product_rows.empty:
            row = product_rows.iloc[-1]
            product_details[product_name] = {
                "category": str(row["Category"]),
                "price": float(row["Price"]),
                "discount": float(row["Discount"]),
                "current_stock": float(row["Current_Stock"]),
                "lead_time": float(row["Lead_Time"])
            }

    prediction = None
    recommendation = None
    risk = None
    reorder_quantity = 0

    selected_product = ""
    prediction_date = ""
    category = ""
    price = ""
    discount = ""
    current_stock = ""
    lead_time = ""
    season = ""
    error = None

    # =====================================================
    # POST
    # =====================================================

    if request.method == "POST":

        selected_product = request.form.get(
            "product", ""
        ).strip()

        prediction_date = request.form.get(
            "prediction_date", ""
        ).strip()

        # User-entered values. If empty, CSV defaults are used.
        manual_category = request.form.get(
            "category", ""
        ).strip()

        manual_price = request.form.get(
            "price", ""
        ).strip()

        manual_discount = request.form.get(
            "discount", ""
        ).strip()

        manual_current_stock = request.form.get(
            "current_stock", ""
        ).strip()

        manual_lead_time = request.form.get(
            "lead_time", ""
        ).strip()

        # =================================================
        # PRODUCT VALIDATION
        # =================================================

        if not selected_product:
            error = "Please select a product."

        elif selected_product not in product_details:
            error = "Selected product was not found."

        # =================================================
        # DATE VALIDATION
        # =================================================

        elif not prediction_date:
            error = "Please select a prediction date."

        else:

            try:
                selected_date = datetime.strptime(
                    prediction_date,
                    "%Y-%m-%d"
                ).date()

                today = date.today()

                if selected_date < today:
                    error = (
                        "Please select today or a future "
                        "date for demand prediction."
                    )

            except ValueError:
                error = (
                    "Invalid date. Please select a valid date."
                )

        # =================================================
        # GET DEFAULT PRODUCT DETAILS
        # =================================================

        if error is None:

            product_rows = df[
                df["Product"].astype(str) == selected_product
            ]

            row = product_rows.iloc[-1]

            default_category = row["Category"]
            default_price = row["Price"]
            default_discount = row["Discount"]
            default_current_stock = row["Current_Stock"]
            default_lead_time = row["Lead_Time"]

            # Auto-fill values unless the user edited them.
            category = (
                manual_category
                if manual_category
                else str(default_category)
            )

            try:
                price = (
                    float(manual_price)
                    if manual_price
                    else float(default_price)
                )

                discount = (
                    float(manual_discount)
                    if manual_discount
                    else float(default_discount)
                )

                current_stock = (
                    float(manual_current_stock)
                    if manual_current_stock
                    else float(default_current_stock)
                )

                lead_time = (
                    float(manual_lead_time)
                    if manual_lead_time
                    else float(default_lead_time)
                )

            except ValueError:
                error = (
                    "Please enter valid numeric values for "
                    "Price, Discount, Current Stock and Lead Time."
                )

            # =================================================
            # VALIDATION OF EDITABLE VALUES
            # =================================================

            if error is None:

                if not category:
                    error = "Category cannot be empty."

                elif price < 0:
                    error = "Price cannot be negative."

                elif discount < 0 or discount > 100:
                    error = "Discount must be between 0 and 100%."

                elif current_stock < 0:
                    error = "Current Stock cannot be negative."

                elif lead_time < 0:
                    error = "Lead Time cannot be negative."

        # =================================================
        # PREDICTION
        # =================================================

        if error is None:

            season = get_season_from_date(selected_date)

            try:

                day = selected_date.day
                day_of_week = selected_date.weekday()
                month = selected_date.month
                week_of_year = selected_date.isocalendar().week

                input_data = pd.DataFrame([
                    {
                        "Product": selected_product,
                        "Category": category,
                        "Price": float(price),
                        "Discount": float(discount),
                        "Current_Stock": float(current_stock),
                        "Lead_Time": float(lead_time),
                        "Season": season,
                        "Day": int(day),
                        "DayOfWeek": int(day_of_week),
                        "Month": int(month),
                        "WeekOfYear": int(week_of_year)
                    }
                ])

                prediction = round(
                    float(model.predict(input_data)[0])
                )

                if prediction < 0:
                    prediction = 0

                risk = calculate_risk(
                    float(current_stock),
                    prediction
                )

                reorder_quantity = calculate_reorder(
                    float(current_stock),
                    prediction
                )

                if reorder_quantity > 0:
                    recommendation = (
                        f"Recommended reorder quantity: "
                        f"{reorder_quantity} units."
                    )
                else:
                    recommendation = (
                        "No additional stock is required."
                    )

                prediction_result = {
                    "product": selected_product,
                    "date": prediction_date,
                    "category": str(category),
                    "price": float(price),
                    "discount": float(discount),
                    "current_stock": float(current_stock),
                    "lead_time": float(lead_time),
                    "season": season,
                    "predicted_demand": int(prediction),
                    "risk": risk,
                    "reorder_quantity": int(reorder_quantity)
                }

                save_prediction(prediction_result)

            except Exception as e:
                error = (
                    "Prediction could not be completed. "
                    f"Error: {e}"
                )

    return render_template(
        "predict.html",
        products=products,
        product_details=product_details,
        prediction=prediction,
        recommendation=recommendation,
        risk=risk,
        reorder_quantity=reorder_quantity,
        selected_product=selected_product,
        prediction_date=prediction_date,
        category=category,
        price=price,
        discount=discount,
        current_stock=current_stock,
        lead_time=lead_time,
        season=season,
        error=error
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route(
    "/dashboard",
    methods=["GET"]
)
def dashboard():

    df = get_active_data()


    # =====================================================
    # PRODUCT SUMMARY
    # =====================================================

    product_summary = (

        df

        .groupby(
            "Product",
            as_index=False
        )

        .agg({

            "Category": "first",

            "Current_Stock": "mean",

            "Demand": "mean"

        })

    )


    product_summary[
        "Current_Stock"
    ] = (

        product_summary[
            "Current_Stock"
        ]

        .round(0)

    )


    product_summary[
        "Demand"
    ] = (

        product_summary[
            "Demand"
        ]

        .round(0)

    )


    # =====================================================
    # RISK COUNTERS
    # =====================================================

    high_risk_count = 0

    medium_risk_count = 0

    low_risk_count = 0


    product_data = []


    # =====================================================
    # ALL PRODUCT ANALYSIS
    # =====================================================

    for _, row in product_summary.iterrows():


        product = row["Product"]


        stock = float(
            row["Current_Stock"]
        )


        demand = float(
            row["Demand"]
        )


        risk = calculate_risk(

            stock,

            demand

        )


        reorder_quantity = calculate_reorder(

            stock,

            demand

        )


        if risk == "High Risk":

            high_risk_count += 1

        elif risk == "Medium Risk":

            medium_risk_count += 1

        else:

            low_risk_count += 1


        product_data.append({

            "Product":
                product,

            "Category":
                row["Category"],

            "Current_Stock":
                round(stock),

            "Demand":
                round(demand),

            "Risk":
                risk,

            "Reorder":
                reorder_quantity

        })


    # =====================================================
    # GRAPH DATA
    # =====================================================

    products = (

        product_summary[
            "Product"
        ]

        .tolist()

    )


    stocks = (

        product_summary[
            "Current_Stock"
        ]

        .astype(float)

        .round(0)

        .tolist()

    )


    demands = (

        product_summary[
            "Demand"
        ]

        .astype(float)

        .round(0)

        .tolist()

    )


    total_products = len(
        products
    )


    # =====================================================
    
    # =====================================================
    # TOP / LOW PERFORMING PRODUCTS
    # =====================================================

    # Performance is based on average Demand.
    # Higher average demand = Top Performing.
    # Lower average demand = Low Performing.
    performance_summary = (
        product_summary[["Product", "Demand"]]
        .copy()
        .sort_values("Demand", ascending=False)
    )

    top_products = []
    for _, row in performance_summary.head(3).iterrows():
        top_products.append({
            "Product": row["Product"],
            "Demand": round(float(row["Demand"]))
        })

    low_products = []
    for _, row in performance_summary.tail(3).sort_values(
        "Demand", ascending=True
    ).iterrows():
        low_products.append({
            "Product": row["Product"],
            "Demand": round(float(row["Demand"]))
        })

    # =====================================================
    # DISCOUNT IMPACT ANALYSIS
    # =====================================================

    discount_summary = (
        df.groupby("Discount", as_index=False)
        .agg({"Demand": "mean"})
        .sort_values("Discount")
    )

    discount_summary["Discount"] = discount_summary["Discount"].round(2)
    discount_summary["Demand"] = discount_summary["Demand"].round(0)

    discount_values = (
        discount_summary["Discount"]
        .astype(float)
        .tolist()
    )

    discount_demands = (
        discount_summary["Demand"]
        .astype(float)
        .tolist()
    )

    # =====================================================
    # SEASONAL DEMAND ANALYSIS
    # =====================================================

    season_summary = (

        df

        .groupby(
            "Season",
            as_index=False
        )

        .agg({

            "Demand": "mean"

        })

    )


    season_summary[
        "Demand"
    ] = (

        season_summary[
            "Demand"
        ]

        .round(0)

    )


    season_order = [

        "Winter",

        "Summer",

        "Monsoon"

    ]


    season_summary[
        "Season"
    ] = pd.Categorical(

        season_summary[
            "Season"
        ],

        categories=season_order,

        ordered=True

    )


    season_summary = (
        season_summary
        .sort_values("Season")
    )


    seasons = (

        season_summary[
            "Season"
        ]

        .astype(str)

        .tolist()

    )


    season_demands = (

        season_summary[
            "Demand"
        ]

        .astype(float)

        .tolist()

    )


    # =====================================================
    # HIGHEST SEASON
    # =====================================================

    if len(season_demands) > 0:

        highest_index = (

            season_demands.index(

                max(
                    season_demands
                )

            )

        )


        highest_demand_season = (

            seasons[
                highest_index
            ]

        )


        highest_season_demand = (

            season_demands[
                highest_index
            ]

        )

    else:

        highest_demand_season = "N/A"

        highest_season_demand = 0


    # =====================================================
    # LAST PREDICTION
    # =====================================================

    last_prediction = (
        load_prediction()
    )


    # =====================================================
    # SELECTED PRODUCT TABLE
    # =====================================================

    selected_product_data = []


    if last_prediction:


        selected_product_name = (
            last_prediction["product"]
        )


        selected_stock = float(
            last_prediction[
                "current_stock"
            ]
        )


        selected_predicted_demand = float(
            last_prediction[
                "predicted_demand"
            ]
        )


        selected_risk = (
            last_prediction["risk"]
        )


        selected_reorder = int(
            last_prediction[
                "reorder_quantity"
            ]
        )


        selected_product_data.append({

            "Product":
                selected_product_name,

            "Category":
                last_prediction[
                    "category"
                ],

            "Current_Stock":
                round(
                    selected_stock
                ),

            "Demand":
                round(
                    selected_predicted_demand
                ),

            "Risk":
                selected_risk,

            "Reorder":
                selected_reorder

        })


    # =====================================================
    # DASHBOARD
    # =====================================================

    return render_template(

        "dashboard.html",

        total_products=
            total_products,

        high_risk=
            high_risk_count,

        medium_risk=
            medium_risk_count,

        low_risk=
            low_risk_count,

        products=
            products,

        stocks=
            stocks,

        demands=
            demands,

        product_data=
            selected_product_data,

        all_product_data=
            product_data,

        top_products=
            top_products,

        low_products=
            low_products,

        discount_values=
            discount_values,

        discount_demands=
            discount_demands,

        seasons=
            seasons,

        season_demands=
            season_demands,

        highest_demand_season=
            highest_demand_season,

        highest_season_demand=
            highest_season_demand,

        last_prediction=
            last_prediction

    )


# =========================================================
# UPLOAD DATA
# =========================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():


    if request.method == "GET":

        return render_template(

            "upload.html",

            error=None,

            success=None

        )


    file = request.files.get(
        "file"
    )


    if file is None or file.filename == "":

        return render_template(

            "upload.html",

            error="Please select a CSV file.",

            success=None

        )


    if not file.filename.lower().endswith(
        ".csv"
    ):

        return render_template(

            "upload.html",

            error=(
                "Invalid file type. "
                "Please upload a CSV file."
            ),

            success=None

        )


    try:

        df = pd.read_csv(file)

    except Exception as e:

        return render_template(

            "upload.html",

            error=(
                "Unable to read CSV file: "
                f"{e}"
            ),

            success=None

        )


    if df.empty:

        return render_template(

            "upload.html",

            error="The uploaded CSV is empty.",

            success=None

        )


    required_columns = [

        "Product",

        "Category",

        "Price",

        "Discount",

        "Current_Stock",

        "Lead_Time",

        "Season",

        "Demand"

    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        return render_template(

            "upload.html",

            error=(

                "Missing columns: "

                + ", ".join(
                    missing_columns
                )

            ),

            success=None

        )


    blank_columns = []


    for column in required_columns:

        if df[column].isna().any():

            blank_columns.append(
                column
            )

        elif (

            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .any()

        ):

            blank_columns.append(
                column
            )


    if blank_columns:

        return render_template(

            "upload.html",

            error=(

                "Blank values found in: "

                + ", ".join(
                    blank_columns
                )

            ),

            success=None

        )


    numeric_columns = [

        "Price",

        "Discount",

        "Current_Stock",

        "Lead_Time",

        "Demand"

    ]


    invalid_numeric = []


    for column in numeric_columns:

        converted = pd.to_numeric(

            df[column],

            errors="coerce"

        )


        if converted.isna().any():

            invalid_numeric.append(
                column
            )


    if invalid_numeric:

        return render_template(

            "upload.html",

            error=(

                "Invalid numeric values in: "

                + ", ".join(
                    invalid_numeric
                )

            ),

            success=None

        )


    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column]
        )


    if (
        df["Discount"] > 100
    ).any():

        return render_template(

            "upload.html",

            error=(
                "Discount cannot be greater than 100%."
            ),

            success=None

        )


    if (
        df[numeric_columns] < 0
    ).any().any():

        return render_template(

            "upload.html",

            error=(
                "Negative values are not allowed."
            ),

            success=None

        )


    os.makedirs(
        "dataset",
        exist_ok=True
    )


    df.to_csv(

        UPLOADED_DATA,

        index=False

    )


    print(
        "Latest CSV uploaded successfully!"
    )


    return redirect(
        url_for("predict")
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )