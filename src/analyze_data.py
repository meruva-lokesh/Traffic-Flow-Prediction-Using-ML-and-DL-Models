import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('traffic_data.csv')

print("="*60)
print("DATASET ANALYSIS REPORT")
print("="*60)

print("\n1. DATASET SHAPE:")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n2. MISSING VALUES:")
print(df.isnull().sum())

print("\n3. DATA TYPES:")
print(df.dtypes)

print("\n4. STATISTICAL SUMMARY:")
print(df.describe())

print("\n5. TRAFFIC SITUATION DISTRIBUTION:")
print(df['TrafficSituation'].value_counts())
print(f"\nPercentage Distribution:")
print(df['TrafficSituation'].value_counts(normalize=True) * 100)

print("\n6. WEATHER DISTRIBUTION:")
print(df['Weather'].value_counts())

print("\n7. JUNCTION DISTRIBUTION:")
print(df['Junction'].value_counts())

print("\n8. CORRELATION WITH TRAFFIC SITUATION:")
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.corr())

print("\n9. AVERAGE VEHICLE COUNTS BY TRAFFIC SITUATION:")
print(df.groupby('TrafficSituation')[['CarCount', 'BusCount', 'BikeCount', 'TruckCount', 'TotalVehicles']].mean())

print("\n10. TRAFFIC SITUATION BY WEATHER:")
print(pd.crosstab(df['Weather'], df['TrafficSituation']))

print("\n11. DATA QUALITY ISSUES:")
# Check for duplicate rows
print(f"    Duplicate rows: {df.duplicated().sum()}")

# Check for outliers
print(f"\n    Vehicles exceeding 200: {(df['TotalVehicles'] > 200).sum()}")
print(f"    Negative values: {(numeric_df < 0).any().sum()} columns")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
