from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os

app = Flask(__name__)

# ==============================
# FILE PATHS
# ==============================

DATASET_FOLDER = "dataset"

DEFAULT_DATASET = os.path.join(
    DATASET_FOLDER,
    "sales_data.csv"
)

UPLOADED_DATASET = os.path.join(
    DATASET_FOLDER,
    "uploaded_sales_data.csv"
)

MODEL_FILE = "model.pkl"
ENCODER_FILE = "encoders.pkl"


# ==============================
# LOAD MODEL
# ==============================

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


# ==============================
# LOAD ENCODERS
# ==============================

with open(ENCODER_FILE, "rb") as file:
    encoders = pickle.load(file)


# ==============================
# HELPER FUNCTION
# GET ACTIVE DATASET
# ==============================

def get_active_data():

    # If uploaded CSV exists,
    # use uploaded data
    if os.path.exists(UPLOADED_DATASET):
        return pd.read_csv(UPLOADED_DATASET)

    # Otherwise use default dataset
    return pd.read_csv(DEFAULT_DATASET)


# ==============================
# SAFE ENCODING FUNCTION
# Handles Unknown Categories
# ==============================

def safe_encode(column_name, value):

    encoder = encoders[column_name]

    value = str(value)

    # If value exists in trained encoder
    if value in encoder.classes_:
        return encoder.transform([value])[0]

    # For unknown/new value
    # use first known class safely
    return 0


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():

    return render_template("index.html")


# ==============================
# PREDICTION PAGE
# ==============================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # Load active dataset
    df = get_active_data()

    # Get unique products
    products = sorted(
        df["Product"]
        .dropna()
        .astype(str)
        .unique()
    )

    # Default values
    prediction = None
    stock_status = None
    reorder_quantity = None
    recommendation = None

    if request.method == "POST":

        try:

            # ==============================
            # GET USER INPUT
            # ==============================

            product = request.form["product"]
            category = request.form["category"]

            price = float(request.form["price"])
            discount = float(request.form["discount"])
            current_stock = float(
                request.form["current_stock"]
            )
            lead_time = float(
                request.form["lead_time"]
            )

            season = request.form["season"]


            # ==============================
            # SAFE ENCODING
            # ==============================

            product_encoded = safe_encode(
                "Product",
                product
            )

            category_encoded = safe_encode(
                "Category",
                category
            )

            season_encoded = safe_encode(
                "Season",
                season
            )


            # ==============================
            # CREATE MODEL INPUT
            # ==============================

            input_data = pd.DataFrame(
                [[
                    product_encoded,
                    category_encoded,
                    price,
                    discount,
                    current_stock,
                    lead_time,
                    season_encoded
                ]],
                columns=[
                    "Product",
                    "Category",
                    "Price",
                    "Discount",
                    "Current_Stock",
                    "Lead_Time",
                    "Season"
                ]
            )


            # ==============================
            # AI PREDICTION
            # ==============================

            prediction = round(
                float(model.predict(input_data)[0])
            )


            # Prevent negative prediction
            if prediction < 0:
                prediction = 0


            # ==============================
            # INVENTORY RECOMMENDATION
            # ==============================

            if current_stock < prediction:

                stock_status = "Low Stock"

                reorder_quantity = round(
                    prediction
                    - current_stock
                    + (prediction * 0.20)
                )

                recommendation = (
                    "Inventory is lower than predicted demand. "
                    "Reorder stock immediately."
                )


            elif current_stock < prediction * 1.5:

                stock_status = "Monitor"

                reorder_quantity = round(
                    prediction * 0.20
                )

                recommendation = (
                    "Stock is sufficient but should be monitored. "
                    "Consider ordering additional inventory soon."
                )


            else:

                stock_status = "Stock Safe"

                reorder_quantity = 0

                recommendation = (
                    "Inventory is sufficient. "
                    "No immediate reorder required."
                )


        except Exception as e:

            recommendation = (
                f"Prediction Error: {str(e)}"
            )


    return render_template(

        "predict.html",

        products=products,

        prediction=prediction,

        stock_status=stock_status,

        reorder_quantity=reorder_quantity,

        recommendation=recommendation
    )


# ==============================
# DASHBOARD
# ==============================

@app.route("/dashboard")
def dashboard():

    # Load active dataset
    df = get_active_data()


    # ==============================
    # CLEAN DATA
    # ==============================

    df["Product"] = (
        df["Product"]
        .fillna("Unknown")
        .astype(str)
    )

    df["Category"] = (
        df["Category"]
        .fillna("Unknown")
        .astype(str)
    )

    df["Current_Stock"] = pd.to_numeric(
        df["Current_Stock"],
        errors="coerce"
    ).fillna(0)

    df["Demand"] = pd.to_numeric(
        df["Demand"],
        errors="coerce"
    ).fillna(0)


    # ==============================
    # BASIC STATISTICS
    # ==============================

    total_products = int(len(df))


    # Low Stock
    low_stock = int(
        len(
            df[
                df["Current_Stock"]
                < df["Demand"]
            ]
        )
    )


    # Monitor
    monitor = int(
        len(
            df[
                (
                    df["Current_Stock"]
                    >= df["Demand"]
                )
                &
                (
                    df["Current_Stock"]
                    < df["Demand"] * 1.5
                )
            ]
        )
    )


    # Safe Stock
    safe_stock = int(
        len(
            df[
                df["Current_Stock"]
                >= df["Demand"] * 1.5
            ]
        )
    )


    # ==============================
    # PRODUCT STATUS
    # ==============================

    product_data = []

    for _, row in df.iterrows():

        current_stock = float(
            row["Current_Stock"]
        )

        demand = float(
            row["Demand"]
        )


        # Determine Status
        if current_stock < demand:

            status = "Low Stock"

        elif current_stock < demand * 1.5:

            status = "Monitor"

        else:

            status = "Stock Safe"


        # JSON-safe data
        product_data.append({

            "Product": str(
                row["Product"]
            ),

            "Category": str(
                row["Category"]
            ),

            "Current_Stock": float(
                current_stock
            ),

            "Demand": float(
                demand
            ),

            "Status": str(
                status
            )
        })


    # ==============================
    # CHART DATA
    # JSON SAFE CONVERSION
    # ==============================

    products = [
        str(x)
        for x in df["Product"].tolist()
    ]

    stocks = [
        float(x)
        for x in df["Current_Stock"].tolist()
    ]

    demands = [
        float(x)
        for x in df["Demand"].tolist()
    ]


    # ==============================
    # RENDER DASHBOARD
    # ==============================

    return render_template(

        "dashboard.html",

        total_products=total_products,

        low_stock=low_stock,

        monitor=monitor,

        safe_stock=safe_stock,

        product_data=product_data,

        products=products,

        stocks=stocks,

        demands=demands
    )


# ==============================
# UPLOAD PAGE
# ==============================

@app.route("/upload", methods=["GET", "POST"])
def upload_file():

    if request.method == "POST":

        # Check file
        if "file" not in request.files:

            return "No file selected"


        file = request.files["file"]


        # Check filename
        if file.filename == "":

            return "No file selected"


        # Check CSV extension
        if not file.filename.lower().endswith(
            ".csv"
        ):

            return (
                "Please upload only CSV file."
            )


        try:

            # Save uploaded file
            file.save(
                UPLOADED_DATASET
            )


            # Read uploaded data
            df = pd.read_csv(
                UPLOADED_DATASET
            )


            # ==============================
            # REQUIRED COLUMNS
            # ==============================

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


            # ==============================
            # CHECK MISSING COLUMNS
            # ==============================

            missing_columns = [

                column

                for column
                in required_columns

                if column
                not in df.columns
            ]


            if missing_columns:

                # Remove invalid file
                if os.path.exists(
                    UPLOADED_DATASET
                ):

                    os.remove(
                        UPLOADED_DATASET
                    )


                return (

                    "Missing required columns: "

                    + ", ".join(
                        missing_columns
                    )
                )


            # ==============================
            # SUCCESS
            # ==============================

            return redirect(
                url_for("dashboard")
            )


        except Exception as e:

            return (
                f"Upload Error: {str(e)}"
            )


    return render_template(
        "upload.html"
    )


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
    )