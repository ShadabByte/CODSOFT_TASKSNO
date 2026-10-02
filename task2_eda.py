import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("cleaned_train.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

print("=" * 50)
print("TASK 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 1. Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

# 2. Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# 3. Unique stores and items
print("\nNumber of Stores:", df["store"].nunique())
print("Number of Items:", df["item"].nunique())

# 4. Total and average sales
print("\nTotal Sales:", df["sales"].sum())
print("Average Sales:", round(df["sales"].mean(), 2))
print("Maximum Daily Sales:", df["sales"].max())
print("Minimum Daily Sales:", df["sales"].min())

# 5. Sales by store
store_sales = df.groupby("store")["sales"].agg(
    ["sum", "mean", "count"]
).sort_values("sum", ascending=False)

print("\nSales by Store:")
print(store_sales)

# 6. Sales by item
item_sales = df.groupby("item")["sales"].agg(
    ["sum", "mean", "count"]
).sort_values("sum", ascending=False)

print("\nTop 10 Items by Total Sales:")
print(item_sales.head(10))

# 7. Monthly sales trend
df["month"] = df["date"].dt.to_period("M")

monthly_sales = df.groupby("month")["sales"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

# 8. Outlier detection using IQR
Q1 = df["sales"].quantile(0.25)
Q3 = df["sales"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["sales"] < lower_limit) |
    (df["sales"] > upper_limit)
]

print("\nOutlier Analysis:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("Number of Outliers:", len(outliers))

# 9. Create output folder
import os
os.makedirs("eda_outputs", exist_ok=True)

# Save store analysis
store_sales.to_csv("eda_outputs/store_sales_summary.csv")

# Save item analysis
item_sales.to_csv("eda_outputs/item_sales_summary.csv")

# Save monthly analysis
monthly_sales.to_csv("eda_outputs/monthly_sales_summary.csv")

# 10. Sales distribution
plt.figure(figsize=(10, 6))
plt.hist(df["sales"], bins=30)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eda_outputs/sales_distribution.png")
plt.close()

# 11. Store sales chart
plt.figure(figsize=(10, 6))
store_sales["sum"].plot(kind="bar")
plt.title("Total Sales by Store")
plt.xlabel("Store")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("eda_outputs/store_sales.png")
plt.close()

# 12. Monthly sales trend
plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_outputs/monthly_sales_trend.png")
plt.close()

# 13. Top 10 items
plt.figure(figsize=(10, 6))
item_sales.head(10)["sum"].plot(kind="bar")
plt.title("Top 10 Items by Total Sales")
plt.xlabel("Item")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("eda_outputs/top_10_items.png")
plt.close()

print("\nEDA analysis completed successfully.")
print("Charts and summary files saved in: eda_outputs/")