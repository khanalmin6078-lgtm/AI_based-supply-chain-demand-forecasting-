from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os
from datetime import datetime, date


app = Flask(__name__)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_FILE = "model.pkl"

HISTORICAL_DATA = "dataset/sales_data.csv"

UPLOADED_DATA = "dataset/uploaded_sales_data.csv"


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
# GET SEASON FROM DATE
# =========================================================

def get_season_from_date(prediction_date):

    month = prediction_date.month

    # March, April, May
    if month in [3, 4, 5]:

        return "Summer"

    # June, July, August, September
    elif month in [6, 7, 8, 9]:

        return "Monsoon"

    # October to February
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

    # 20% safety stock

    recommended_stock = demand * 1.20

    reorder_quantity = recommended_stock - stock

    if reorder_quantity < 0:

        reorder_quantity = 0

    return round(reorder_quantity)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# FUTURE DEMAND PREDICTION
# =========================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    df = get_active_data()


    # -----------------------------------------------------
    # PRODUCT LIST
    # -----------------------------------------------------

    products = sorted(
        df["Product"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    # -----------------------------------------------------
    # DEFAULT VALUES
    # -----------------------------------------------------

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
    # POST REQUEST
    # =====================================================

    if request.method == "POST":


        # -------------------------------------------------
        # PRODUCT
        # -------------------------------------------------

        selected_product = request.form.get(
            "product",
            ""
        ).strip()


        # -------------------------------------------------
        # PREDICTION DATE
        # -------------------------------------------------

        prediction_date = request.form.get(
            "prediction_date",
            ""
        ).strip()


        # -------------------------------------------------
        # VALIDATE PRODUCT
        # -------------------------------------------------

        if not selected_product:

            error = "Please select a product."


        # -------------------------------------------------
        # VALIDATE DATE
        # -------------------------------------------------

        elif not prediction_date:

            error = "Please select a prediction date."


        else:

            try:

                selected_date = datetime.strptime(
                    prediction_date,
                    "%Y-%m-%d"
                ).date()


                # Today's date
                today = date.today()


                if selected_date < today:

                    error = (
                        "Please select today or a future "
                        "date for demand prediction."
                    )


            except ValueError:

                error = (
                    "Invalid date format. "
                    "Please select a valid date."
                )


        # =================================================
        # PRODUCT DATA
        # =================================================

        if error is None:


            product_rows = df[
                df["Product"].astype(str)
                == selected_product
            ]


            if product_rows.empty:

                error = (
                    "Selected product was not found "
                    "in the data."
                )


            else:

                # Get latest available record
                row = product_rows.iloc[-1]


                category = row["Category"]

                price = row["Price"]

                discount = row["Discount"]

                current_stock = row["Current_Stock"]

                lead_time = row["Lead_Time"]


                # -------------------------------------------------
                # SEASON FROM FUTURE DATE
                # -------------------------------------------------

                season = get_season_from_date(
                    selected_date
                )


                try:


                    # =================================================
                    # DATE FEATURES
                    # =================================================

                    day = selected_date.day

                    day_of_week = selected_date.weekday()

                    month = selected_date.month

                    week_of_year = (
                        selected_date.isocalendar().week
                    )


                    # =================================================
                    # MODEL INPUT
                    # =================================================

                    input_data = pd.DataFrame([

                        {

                            "Product": selected_product,

                            "Category": category,

                            "Price": float(price),

                            "Discount": float(discount),

                            "Current_Stock": float(
                                current_stock
                            ),

                            "Lead_Time": float(
                                lead_time
                            ),

                            "Season": season,

                            "Day": int(day),

                            "DayOfWeek": int(
                                day_of_week
                            ),

                            "Month": int(month),

                            "WeekOfYear": int(
                                week_of_year
                            )

                        }

                    ])


                    # =================================================
                    # ML PREDICTION
                    # =================================================

                    prediction = round(

                        float(

                            model.predict(
                                input_data
                            )[0]

                        )

                    )


                    # Prevent negative demand

                    if prediction < 0:

                        prediction = 0


                    # =================================================
                    # STOCK RISK
                    # =================================================

                    risk = calculate_risk(

                        float(current_stock),

                        prediction

                    )


                    # =================================================
                    # REORDER
                    # =================================================

                    reorder_quantity = calculate_reorder(

                        float(current_stock),

                        prediction

                    )


                    # =================================================
                    # RECOMMENDATION
                    # =================================================

                    if reorder_quantity > 0:

                        recommendation = (

                            f"Recommended reorder quantity: "
                            f"{reorder_quantity} units."

                        )

                    else:

                        recommendation = (

                            "No additional stock is required."

                        )


                except Exception as e:

                    error = (

                        "Prediction could not be completed. "
                        f"Error: {e}"

                    )


    # =====================================================
    # PREDICTION PAGE
    # =====================================================

    return render_template(

        "predict.html",

        products=products,

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

@app.route("/dashboard")
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


    product_summary["Current_Stock"] = (

        product_summary["Current_Stock"]

        .round(0)

    )


    product_summary["Demand"] = (

        product_summary["Demand"]

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
    # PRODUCT ANALYSIS
    # =====================================================

    for _, row in product_summary.iterrows():


        product = row["Product"]


        stock = float(
            row["Current_Stock"]
        )


        demand = float(
            row["Demand"]
        )


        # Risk

        risk = calculate_risk(

            stock,

            demand

        )


        # Reorder

        reorder_quantity = calculate_reorder(

            stock,

            demand

        )


        # Count risks

        if risk == "High Risk":

            high_risk_count += 1


        elif risk == "Medium Risk":

            medium_risk_count += 1


        else:

            low_risk_count += 1


        # Product data

        product_data.append({

            "Product": product,

            "Category": row["Category"],

            "Current_Stock": round(stock),

            "Demand": round(demand),

            "Risk": risk,

            "Reorder": reorder_quantity

        })


    # =====================================================
    # PRODUCT CHART DATA
    # =====================================================

    products = (

        product_summary["Product"]

        .tolist()

    )


    stocks = (

        product_summary["Current_Stock"]

        .astype(float)

        .round(0)

        .tolist()

    )


    demands = (

        product_summary["Demand"]

        .astype(float)

        .round(0)

        .tolist()

    )


    total_products = len(products)


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


    season_summary["Demand"] = (

        season_summary["Demand"]

        .round(0)

    )


    # Keep the seasons in a fixed order

    season_order = [

        "Winter",

        "Summer",

        "Monsoon"

    ]


    season_summary["Season"] = pd.Categorical(

        season_summary["Season"],

        categories=season_order,

        ordered=True

    )


    season_summary = season_summary.sort_values(

        "Season"

    )


    # Season names

    seasons = (

        season_summary["Season"]

        .astype(str)

        .tolist()

    )


    # Season demand

    season_demands = (

        season_summary["Demand"]

        .astype(float)

        .tolist()

    )


    # =====================================================
    # HIGHEST DEMAND SEASON
    # =====================================================

    if len(season_demands) > 0:


        highest_demand_index = (

            season_demands.index(

                max(season_demands)

            )

        )


        highest_demand_season = (

            seasons[highest_demand_index]

        )


        highest_season_demand = (

            season_demands[highest_demand_index]

        )


    else:


        highest_demand_season = "N/A"

        highest_season_demand = 0


    # =====================================================
    # DASHBOARD PAGE
    # =====================================================

    return render_template(

        "dashboard.html",

        total_products=total_products,

        high_risk=high_risk_count,

        medium_risk=medium_risk_count,

        low_risk=low_risk_count,

        products=products,

        stocks=stocks,

        demands=demands,

        product_data=product_data,

        seasons=seasons,

        season_demands=season_demands,

        highest_demand_season=highest_demand_season,

        highest_season_demand=highest_season_demand

    )


# =========================================================
# UPLOAD DATA
# =========================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():


    # =====================================================
    # GET
    # =====================================================

    if request.method == "GET":

        return render_template(

            "upload.html",

            error=None,

            success=None

        )


    # =====================================================
    # GET FILE
    # =====================================================

    file = request.files.get("file")


    if file is None or file.filename == "":

        return render_template(

            "upload.html",

            error="Please select a CSV file.",

            success=None

        )


    # =====================================================
    # CHECK FILE TYPE
    # =====================================================

    if not file.filename.lower().endswith(".csv"):

        return render_template(

            "upload.html",

            error=(
                "Invalid file type. "
                "Please upload a CSV file."
            ),

            success=None

        )


    # =====================================================
    # READ CSV
    # =====================================================

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


    # =====================================================
    # EMPTY CHECK
    # =====================================================

    if df.empty:

        return render_template(

            "upload.html",

            error="The uploaded CSV is empty.",

            success=None

        )


    # =====================================================
    # REQUIRED COLUMNS
    # =====================================================

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

                "Invalid CSV format. "
                "Missing columns: "

                + ", ".join(
                    missing_columns
                )

            ),

            success=None

        )


    # =====================================================
    # BLANK CHECK
    # =====================================================

    blank_columns = []


    for column in required_columns:


        if df[column].isna().any():

            blank_columns.append(column)


        elif (

            df[column]

            .astype(str)

            .str.strip()

            .eq("")

            .any()

        ):

            blank_columns.append(column)


    if blank_columns:

        return render_template(

            "upload.html",

            error=(

                "Blank or missing values found in: "

                + ", ".join(
                    blank_columns
                )

            ),

            success=None

        )


    # =====================================================
    # NUMERIC CHECK
    # =====================================================

    numeric_columns = [

        "Price",

        "Discount",

        "Current_Stock",

        "Lead_Time",

        "Demand"

    ]


    invalid_numeric_columns = []


    for column in numeric_columns:


        converted = pd.to_numeric(

            df[column],

            errors="coerce"

        )


        if converted.isna().any():

            invalid_numeric_columns.append(
                column
            )


    if invalid_numeric_columns:

        return render_template(

            "upload.html",

            error=(

                "These columns must contain "
                "numeric values: "

                + ", ".join(
                    invalid_numeric_columns
                )

            ),

            success=None

        )


    # =====================================================
    # NEGATIVE CHECK
    # =====================================================

    negative_columns = []


    for column in numeric_columns:


        values = pd.to_numeric(

            df[column],

            errors="coerce"

        )


        if (values < 0).any():

            negative_columns.append(
                column
            )


    if negative_columns:

        return render_template(

            "upload.html",

            error=(

                "Negative values are not allowed in: "

                + ", ".join(
                    negative_columns
                )

            ),

            success=None

        )


    # =====================================================
    # DISCOUNT CHECK
    # =====================================================

    discount_values = pd.to_numeric(

        df["Discount"],

        errors="coerce"

    )


    if (discount_values > 100).any():

        return render_template(

            "upload.html",

            error=(
                "Discount cannot be greater than 100%."
            ),

            success=None

        )


    # =====================================================
    # CONVERT NUMERIC COLUMNS
    # =====================================================

    for column in numeric_columns:

        df[column] = pd.to_numeric(

            df[column]

        )


    # =====================================================
    # SAVE UPLOADED DATA
    # =====================================================

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
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )