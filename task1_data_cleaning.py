import pandas as pd

# Load dataset
df = pd.read_csv("data/train.csv")

print("Original Shape:", df.shape)

# 1. Check structure
print("\nDataset Info:")
print(df.info())

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Check duplicate records
print("\nDuplicate Rows:", df.duplicated().sum())

# 4. Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 5. Check invalid dates created during conversion
print("\nInvalid Dates:", df["date"].isnull().sum())

# 6. Remove duplicate rows
df = df.drop_duplicates()

# 7. Remove rows with invalid/missing dates
df = df.dropna(subset=["date"])

# 8. Check data types after cleaning
print("\nData Types After Cleaning:")
print(df.dtypes)

# 9. Final missing values check
print("\nFinal Missing Values:")
print(df.isnull().sum())

# 10. Save cleaned dataset
df.to_csv("cleaned_train.csv", index=False)

print("\nCleaned Shape:", df.shape)
print("\nCleaned dataset saved as: cleaned_train.csv")