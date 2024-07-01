import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Create new features (example)
# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# Add polynomial features to the dataframe
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df.select_dtypes(include=['float64', 'int64']).columns))
df = pd.concat([df, poly_df], axis=1)

# Separate features and target variable
X = df.drop('target_column', axis=1)  # Replace 'target_column' with your target column name
y = df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Feature Selection

# PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Preserve 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f'Original number of features: {X_train.shape[1]}')
print(f'Number of features after PCA: {X_train_pca.shape[1]}')

# Feature Importance using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")

# Select important features based on a threshold
model = SelectFromModel(rf, threshold="median")
X_train_selected = model.fit_transform(X_train, y_train)
X_test_selected = model.transform(X_test)

print(f'Number of features after feature importance selection: {X_train_selected.shape[1]}')

# You can also use XGBoost for feature importance
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)

# Get feature importances from XGBoost
xgb_importances = xgb.feature_importances_
xgb_indices = np.argsort(xgb_importances)[::-1]

# Print feature ranking from XGBoost
print("Feature ranking (XGBoost):")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {xgb_indices[f]} ({xgb_importances[xgb_indices[f]]})")

# Select important features based on a threshold (XGBoost)
xgb_model = SelectFromModel(xgb, threshold="median")
X_train_xgb_selected = xgb_model.fit_transform(X_train, y_train)
X_test_xgb_selected = xgb_model.transform(X_test)

print(f'Number of features after feature importance selection (XGBoost): {X_train_xgb_selected.shape[1]}')
