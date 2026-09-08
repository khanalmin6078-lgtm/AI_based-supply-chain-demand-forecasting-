import pandas as pd
import random
from datetime import datetime, timedelta

# Number of records
num_records = 1000

# Products
products = [
    "Laptop",
    "Wireless Mouse",
    "Keyboard",
    "Monitor",
    "Headphones",
    "Smartphone",
    "Gaming Keyboard",
    "Printer",
    "External Hard Drive"
]

# Product price ranges
product_prices = {
    "Laptop": (40000, 80000),
    "Wireless Mouse": (300, 1500),
    "Keyboard": (800, 3000),
    "Monitor": (8000, 25000),
    "Headphones": (1000, 5000),
    "Smartphone": (10000, 50000),
    "Gaming Keyboard": (2000, 8000),
    "Printer": (5000, 20000),
    "External Hard Drive": (3000, 10000)
}

# Base demand for each product
base_demand = {
    "Laptop": 40,
    "Wireless Mouse": 120,
    "Keyboard": 90,
    "Monitor": 60,
    "Headphones": 110,
    "Smartphone": 70,
    "Gaming Keyboard": 80,
    "Printer": 50,
    "External Hard Drive": 75
}

# Categories
categories = {
    "Laptop": "Electronics",
    "Wireless Mouse": "Accessories",
    "Keyboard": "Accessories",
    "Monitor": "Electronics",
    "Headphones": "Accessories",
    "Smartphone": "Electronics",
    "Gaming Keyboard": "Gaming",
    "Printer": "Electronics",
    "External Hard Drive": "Storage"
}

# Seasons
seasons = ["Summer", "Winter", "Monsoon"]

data = []

# Generate dataset
for i in range(num_records):

    product = random.choice(products)

    category = categories[product]

    # Product-specific price
    min_price, max_price = product_prices[product]
    price = random.randint(min_price, max_price)

    # Random discount
    discount = random.choice([0, 5, 10, 15, 20])

    # Current inventory stock
    current_stock = random.randint(30, 500)

    # Supplier lead time
    lead_time = random.randint(1, 10)

    # Season
    season = random.choice(seasons)

    # Base demand
    demand = base_demand[product]

    # Discount increases demand
    demand += discount * 2

    # Seasonal effect
    if season == "Winter":
        demand += random.randint(5, 20)

    elif season == "Summer":
        demand += random.randint(0, 15)

    else:
        demand += random.randint(-5, 10)

    # Price effect
    if price < (min_price + max_price) / 2:
        demand += random.randint(5, 20)
    else:
        demand -= random.randint(0, 10)

    # Add random variation
    demand += random.randint(-15, 15)

    # Demand should not be negative
    demand = max(10, demand)

    data.append([
        product,
        category,
        price,
        discount,
        current_stock,
        lead_time,
        season,
        demand
    ])

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "Product",
        "Category",
        "Price",
        "Discount",
        "Current_Stock",
        "Lead_Time",
        "Season",
        "Demand"
    ]
)

# Save CSV file
df.to_csv("dataset/sales_data.csv", index=False)

print("Dataset generated successfully!")
print("Total Records:", len(df))
print("\nProducts included:")
print(df["Product"].unique())

print("\nFirst 5 records:")
print(df.head())