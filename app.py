from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os

app = Flask(__name__)

# -------------------------------------------------
# FILE PATHS
# -------------------------------------------------

MODEL_FILE = "model.pkl"
HISTORICAL_DATA = "dataset/sales_data.csv"
UPLOADED_DATA = "dataset/uploaded_sales_data.csv"


# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully!")


# -------------------------------------------------
# GET ACTIVE DATA
# -------------------------------------------------

def get_active_data():

    if os.path.exists(UPLOADED_DATA):
        try:
            df = pd.read_csv(UPLOADED_DATA)

            if not df.empty:
                return df

        except Exception:
            pass

    return pd.read_csv(HISTORICAL_DATA)


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():

    df = get_active_data()

    # Product list from latest uploaded data
    products = sorted(
        df["Product"].dropna().astype(str).unique().tolist()
    )

    prediction = None
    recommendation = None
    selected_product = ""

    # Values to automatically fill
    category = ""
    price = ""
    discount = ""
    current_stock = ""
    lead_time = ""
    season = ""

    if request.method == "POST":

        selected_product = request.form.get("product", "").strip()

        # -------------------------------------------------
        # FIND SELECTED PRODUCT IN LATEST DATA
        # -------------------------------------------------

        product_rows = df[
            df["Product"].astype(str) == selected_product
        ]

        if not product_rows.empty:

            # Use the latest uploaded snapshot row
            row = product_rows.iloc[-1]

            category = row["Category"]
            price = row["Price"]
            discount = row["Discount"]
            current_stock = row["Current_Stock"]
            lead_time = row["Lead_Time"]
            season = row["Season"]

            # -------------------------------------------------
            # CREATE INPUT FOR ML MODEL
            # -------------------------------------------------

            input_data = pd.DataFrame([
                {
                    "Product": selected_product,
                    "Category": category,
                    "Price": float(price),
                    "Discount": float(discount),
                    "Current_Stock": float(current_stock),
                    "Lead_Time": float(lead_time),
                    "Season": season
                }
            ])

            # -------------------------------------------------
            # PREDICT DEMAND
            # -------------------------------------------------

            prediction = round(
                float(
                    model.predict(input_data)[0]
                )
            )

            # Prevent negative demand
            if prediction < 0:
                prediction = 0

            # -------------------------------------------------
            # INVENTORY RECOMMENDATION
            # -------------------------------------------------

            if float(current_stock) < prediction:

                shortage = round(
                    prediction - float(current_stock)
                )

                recommendation = (
                    f"Low Stock - Order approximately "
                    f"{shortage} more units."
                )

            elif float(current_stock) <= prediction * 1.2:

                recommendation = (
                    "Monitor Stock - Inventory is close "
                    "to predicted demand."
                )

            else:

                recommendation = (
                    "Stock Safe - Current inventory is "
                    "sufficient."
                )

    return render_template(
        "predict.html",
        products=products,
        prediction=prediction,
        recommendation=recommendation,
        selected_product=selected_product,
        category=category,
        price=price,
        discount=discount,
        current_stock=current_stock,
        lead_time=lead_time,
        season=season
    )


# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------

@app.route("/dashboard")
def dashboard():

    df = get_active_data()

    # ---------------------------------------------
    # PRODUCT SUMMARY
    # ---------------------------------------------

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

    # Round values
    product_summary["Current_Stock"] = (
        product_summary["Current_Stock"].round(0)
    )

    product_summary["Demand"] = (
        product_summary["Demand"].round(0)
    )

    # ---------------------------------------------
    # PRODUCT STATUS
    # ---------------------------------------------

    product_data = []

    low_stock_count = 0
    monitor_count = 0
    safe_count = 0

    for _, row in product_summary.iterrows():

        stock = float(row["Current_Stock"])
        demand = float(row["Demand"])

        if stock < demand:

            status = "Low Stock"
            low_stock_count += 1

        elif stock <= demand * 1.2:

            status = "Monitor"
            monitor_count += 1

        else:

            status = "Stock Safe"
            safe_count += 1

        product_data.append({
            "Product": row["Product"],
            "Category": row["Category"],
            "Current_Stock": round(stock),
            "Demand": round(demand),
            "Status": status
        })

    # ---------------------------------------------
    # CHART DATA
    # ---------------------------------------------

    products = product_summary["Product"].tolist()

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

    return render_template(
        "dashboard.html",
        total_products=total_products,
        low_stock=low_stock_count,
        monitor=monitor_count,
        safe_stock=safe_count,
        products=products,
        stocks=stocks,
        demands=demands,
        product_data=product_data
    )


# -------------------------------------------------
# UPLOAD LATEST DATA
# -------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        file = request.files.get("file")

        if file is None or file.filename == "":
            return "Please select a CSV file."

        # Check extension
        if not file.filename.lower().endswith(".csv"):
            return "Only CSV files are allowed."

        try:

            df = pd.read_csv(file)

        except Exception as e:

            return f"Unable to read CSV file: {e}"

        # ---------------------------------------------
        # REQUIRED COLUMNS
        # ---------------------------------------------

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

            return (
                "Missing columns: "
                + ", ".join(missing_columns)
            )

        # ---------------------------------------------
        # CHECK EMPTY DATA
        # ---------------------------------------------

        if df.empty:

            return "Uploaded CSV is empty."

        # ---------------------------------------------
        # CREATE DATASET FOLDER
        # ---------------------------------------------

        os.makedirs(
            "dataset",
            exist_ok=True
        )

        # ---------------------------------------------
        # SAVE LATEST DATA
        # ---------------------------------------------

        df.to_csv(
            UPLOADED_DATA,
            index=False
        )

        print(
            "Latest data uploaded successfully!"
        )

        return redirect(
            url_for("predict")
        )

    return render_template("upload.html")


# -------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )