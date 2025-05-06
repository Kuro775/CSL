import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# Load CSV files for Type 1 and Type 2 translations
# ---------------------------------------
df_type1 = pd.read_csv("translated_type1.csv")
df_type2 = pd.read_csv("translated_type2.csv")

# ---------------------------------------
# Part 1: Type 1 - Language distribution analysis
# ---------------------------------------
lang_counts = df_type1["target_language"].value_counts()

# Print language distribution counts
print("📊 Type 1 - Target Language Distribution:")
print(lang_counts)

# Plot bar chart for language distribution
lang_counts.plot(kind='bar', title="Type 1 - Target Language Distribution", ylabel="Count")
plt.tight_layout()
plt.show()

# ---------------------------------------
# Part 2: Type 2 - Translated ratio distribution analysis
# ---------------------------------------
ratios = df_type2["translated_ratio"]

# Print summary statistics for translation ratios
print("\n📊 Type 2 - Translated Ratio Statistics:")
print(ratios.describe())

# Plot histogram of ratio distribution
plt.hist(ratios, bins=10, edgecolor='black')
plt.title("Type 2 - Translated Ratio Distribution")
plt.xlabel("Ratio")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
