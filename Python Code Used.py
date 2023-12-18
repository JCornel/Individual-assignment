# ---- IMPORTING THE LIBRARIES ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from fancyimpute import IterativeImputer
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# ---- LOADING THE DATASET --------------------------------------------------------------------------------------------------------------------------
file_path = r"C:\Users\Swagm\OneDrive - HAN\002 School\002 Minor Data Driven Decision Making\- Individual project\Data\data.csv"
df = pd.read_csv(file_path)


# ---- IMPUTING MISSING VALUES ----------------------------------------------------------------------------------------------------------------------
missing_values = df.isnull().sum()

if missing_values.any():
    print("\nMissing values found. Removing missing values.")

    imputer = IterativeImputer()
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    print("Missing Values after Imputation:")
    print(df.isnull().sum())
    
else:
    print("\nNo missing values in the dataset. Skipping imputation.")


# ---- REMOVING IMPOSSIBLE VALUES -------------------------------------------------------------------------------------------------------------------
plausible_ranges = {
    'battery_power': (0, None),
    'blue': (0, 1),
    'clock_speed': (0, None),
    'dual_sim': (0, 1),
    'fc': (0, None),
    'four_g': (0, 1),
    'int_memory': (0, None),
    'm_dep': (0, None),
    'mobile_wt': (0, None),
    'n_cores': (0, None),
    'pc': (0, None),
    'px_height': (0, None),
    'px_width': (0, None),
    'ram': (0, None),
    'sc_h': (0, None),
    'sc_w': (0, None),
    'talk_time': (0, None),
    'three_g': (0, 1),
    'touch_screen': (0, 1),
    'wifi': (0, 1),
    'price_range': (0, 4),
}

impossible_values_count = sum(
    len(df[(df[variable] < min_val) | (df[variable] > max_val)])
    for variable, (min_val, max_val) in plausible_ranges.items()
)

if impossible_values_count > 0:
    print("\nImpossible values found. Removed impossible values.")

    for variable, (min_val, max_val) in plausible_ranges.items():
        df = df[(df[variable] >= min_val) & (df[variable] <= max_val)]
else:
    print("No impossible values found. Skipping removal.")


# ---- DESCRIPTIVE ANALYSIS ------------------------------------------------------------------------------------------------------------------------
descriptive_stats = df.describe(include='all').transpose()
descriptive_stats = descriptive_stats.round(1)
print("\nDescriptive Statistics Analysis:")
print(descriptive_stats)


# ---- CORRELATION MATRIX --------------------------------------------------------------------------------------------------------------------------
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".1f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()


# ---- REGRESSION MODEL ----------------------------------------------------------------------------------------------------------------------------
selected_features = df.drop(['battery_power', 'price_range'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(selected_features, df['battery_power'], test_size=0.2, random_state=42)
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
model_summary = model.summary()
model_summary_text = model_summary.as_text()
model_summary_text = '\n'.join([line for line in model_summary_text.split('\n') if 'const' not in line and 'price_range' not in line])
print(model_summary_text)