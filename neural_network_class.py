import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load data from CSV
df = pd.read_csv("storedata.csv")

# a little bit of feature engineering
df["Total Population"] = df["40min population"] + df["30 min population"] + df["20 min population"] + df["10 min population"]

# Clean the "Staff" column
df.loc[df['Staff'] < 0, 'Staff'] = np.nan
df = df[df['Staff'] <= 50]
mean_staff = df['Staff'].mean()
df['Staff'].fillna(mean_staff, inplace=True)

# Convert Car park column to binary
df["Car park"] = df["Car park"].apply(lambda x: 1 if x=="Yes" else 0)

# Encode target column
df['Performance'] = pd.factorize(df['Performance'])[0]

# Select features and target
features = ["Staff", "Floor Space", "Window", "Car park", "Demographic score", "Clearance space", "Competition number", "Competition score", "Total Population"]
target = "Performance"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="relu", random_state=42, max_iter=2000)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Generate predictions and confusion matrix
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix in a table
cm_df = pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"])
print(cm_df)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
