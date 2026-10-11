
import numpy as np
from pathlib import Path
from load_data import load_raw

# Load the original dataset
df = load_raw()

print("Original dataset shape:", df.shape)

# Check missing values before cleaning
print("\nMissing values in each column:")
print(df.isnull().sum())

# Remove columns that are completely empty
df = df.dropna(axis=1, how="all")

# Rename the label column
df = df.rename(columns={"Output (S)": "fault"})

# Check the cleaned dataset
print("\nCleaned dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check the number of fault and no-fault cases
print("\nFault counts:")
print(df["fault"].value_counts().sort_index())

print("\nFault percentages:")
print((df["fault"].value_counts(normalize=True).sort_index() * 100).round(2))

# Separate the six measurement features
feature_columns = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Split the data using the required random seed
rng = np.random.default_rng(42)
indices = rng.permutation(len(df))

n_train = int(0.6 * len(df))
n_val = int(0.2 * len(df))

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

# Keep copies of the raw data for later feature engineering
train_raw = df.iloc[train_idx].copy()
val_raw = df.iloc[val_idx].copy()
test_raw = df.iloc[test_idx].copy()

# Calculate mean and standard deviation from training data only
train_mean = train_raw[feature_columns].mean()
train_std = train_raw[feature_columns].std()

# Avoid division by zero if a feature has no variation
train_std = train_std.replace(0, 1)

# Standardise the features using training statistics
train_data = train_raw.copy()
val_data = val_raw.copy()
test_data = test_raw.copy()

train_data.loc[:, feature_columns] = (
    train_raw[feature_columns] - train_mean
) / train_std

val_data.loc[:, feature_columns] = (
    val_raw[feature_columns] - train_mean
) / train_std

test_data.loc[:, feature_columns] = (
    test_raw[feature_columns] - train_mean
) / train_std

# Display split sizes and fault fractions
print("\nSplit sizes:")
print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))

print("\nFault fraction in each split:")
print("Training:", round(train_raw["fault"].mean(), 4))
print("Validation:", round(val_raw["fault"].mean(), 4))
print("Test:", round(test_raw["fault"].mean(), 4))

# Check standardisation on training features
print("\nStandardised training feature means:")
print(train_data[feature_columns].mean().round(4))

print("\nStandardised training feature standard deviations:")
print(train_data[feature_columns].std().round(4))

# Save the prepared datasets
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

train_raw.to_csv(data_dir / "train_raw.csv", index=False)
val_raw.to_csv(data_dir / "validation_raw.csv", index=False)
test_raw.to_csv(data_dir / "test_raw.csv", index=False)

train_data.to_csv(data_dir / "train.csv", index=False)
val_data.to_csv(data_dir / "validation.csv", index=False)
test_data.to_csv(data_dir / "test.csv", index=False)

print("\nPrepared datasets saved in the data folder.")
