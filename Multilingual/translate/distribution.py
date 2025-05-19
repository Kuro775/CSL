import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# Load CSV files for Type 1 and Type 2 translations
# ---------------------------------------
df_type1 = pd.read_csv("translated_type1.csv")
df_type2 = pd.read_csv("translated_type2.csv")

# ---------------------------------------
# Type 1 - Count and percentage of each target language
# ---------------------------------------
type1_counts = df_type1["target_language"].value_counts()
type1_percent = df_type1["target_language"].value_counts(normalize=True) * 100

# Merge into a single DataFrame
type1_stats = pd.DataFrame({
    "Language": type1_counts.index,
    "Type1_Count": type1_counts.values,
    "Type1_Percentage": type1_percent.values
})

print("📊 Type 1 - Language Distribution:")
print(type1_stats)

# ---------------------------------------
# Type 2 - Count, percentage, and mean translated_ratio for each language
# ---------------------------------------
df_type2["translated_ratio"] = pd.to_numeric(df_type2["translated_ratio"], errors='coerce')

type2_counts = df_type2["target_language"].value_counts()
type2_percent = df_type2["target_language"].value_counts(normalize=True) * 100
type2_mean_ratio = df_type2.groupby("target_language")["translated_ratio"].mean()

# Merge into a single DataFrame
type2_stats = pd.DataFrame({
    "Language": type2_counts.index,
    "Type2_Count": type2_counts.values,
    "Type2_Percentage": type2_percent.values,
    "Type2_Mean_Ratio": type2_mean_ratio.values
})

print("\n📊 Type 2 - Language Count, Percentage, and Average Translated Ratio:")
print(type2_stats)

# ---------------------------------------
# Type 2 - Overall translated_ratio statistics
# ---------------------------------------
ratio_summary = df_type2["translated_ratio"].describe()
print("\n📊 Type 2 - Overall Translated Ratio Summary:")
print(ratio_summary)

# ---------------------------------------
# Plot 1: Type 1 Language Percentage (Pie Chart)
# ---------------------------------------
plt.figure(figsize=(10, 6))
plt.pie(type1_stats["Type1_Percentage"], labels=type1_stats["Language"], autopct='%1.1f%%')
plt.title("Type 1 - Language Percentage Distribution")
plt.tight_layout()
plt.show()

# ---------------------------------------
# Plot 2: Type 2 Translated Ratio Distribution (Histogram)
# ---------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df_type2["translated_ratio"], bins=15, edgecolor='black')
plt.title("Type 2 - Translated Ratio Distribution")
plt.xlabel("Translated Ratio")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
