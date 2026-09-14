import pandas as pd
import random

# Existing dataset load karo
file_path = "dataset/sales_data.csv"

df = pd.read_csv(file_path)

# Random historical dates generate karo
# 2025 se 14 September 2026 tak
start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2026-09-14")

date_range = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D"
)

# Har existing row ko ek date assign karo
random.seed(42)

df["Date"] = [
    random.choice(date_range).strftime("%Y-%m-%d")
    for _ in range(len(df))
]

# Date ko Product ke baad rakho
columns = [
    "Product",
    "Category",
    "Date",
    "Price",
    "Discount",
    "Current_Stock",
    "Lead_Time",
    "Season",
    "Demand"
]

df = df[columns]

# Same CSV file update karo
df.to_csv(file_path, index=False)

print("Date column added successfully!")
print("Total rows:", len(df))
print("\nFirst 10 records:")
print(df.head(10))