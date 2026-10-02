import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("cleaned_train.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

print("=" * 50)
print("TASK 3 - DATA VISUALIZATION")
print("=" * 50)

# Create output folder
import os
os.makedirs("task3_outputs", exist_ok=True)

# --------------------------------------------------
# 1. Monthly Sales Trend - Line Chart
# --------------------------------------------------

df["month"] = df["date"].dt.to_period("M")

monthly_sales = df.groupby("month")["sales"].sum()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index.astype(str), monthly_sales.values)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("task3_outputs/monthly_sales_trend.png")
plt.close()

# --------------------------------------------------
# 2. Sales by Store - Bar Chart
# --------------------------------------------------

store_sales = df.groupby("store")["sales"].sum()

plt.figure(figsize=(10, 6))
plt.bar(store_sales.index.astype(str), store_sales.values)
plt.title("Total Sales by Store")
plt.xlabel("Store")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("task3_outputs/store_sales_bar.png")
plt.close()

# --------------------------------------------------
# 3. Sales Distribution - Histogram
# --------------------------------------------------

plt.figure(figsize=(10, 6))
plt.hist(df["sales"], bins=30)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("task3_outputs/sales_distribution.png")
plt.close()

# --------------------------------------------------
# 4. Top 10 Items - Bar Chart
# --------------------------------------------------

item_sales = df.groupby("item")["sales"].sum()
top_items = item_sales.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.bar(top_items.index.astype(str), top_items.values)
plt.title("Top 10 Items by Total Sales")
plt.xlabel("Item")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("task3_outputs/top_10_items.png")
plt.close()

# --------------------------------------------------
# 5. Store vs Average Sales - Bar Chart
# --------------------------------------------------

average_store_sales = df.groupby("store")["sales"].mean()

plt.figure(figsize=(10, 6))
plt.bar(
    average_store_sales.index.astype(str),
    average_store_sales.values
)
plt.title("Average Sales by Store")
plt.xlabel("Store")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("task3_outputs/average_sales_by_store.png")
plt.close()

# --------------------------------------------------
# Save summary data
# --------------------------------------------------

monthly_sales.to_csv("task3_outputs/monthly_sales.csv")
store_sales.to_csv("task3_outputs/store_sales.csv")
top_items.to_csv("task3_outputs/top_10_items.csv")

print("\nVisualization completed successfully.")
print("5 charts created.")
print("Summary CSV files created.")
print("All files saved inside: task3_outputs/")