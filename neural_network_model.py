import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

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
label_encoder = LabelEncoder()
df['Performance'] = label_encoder.fit_transform(df['Performance'])

# Select features and target
features = ["Staff", "Floor Space", "Window", "Car park", "Demographic score", "Clearance space", "Competition number", "Competition score", "Total Population"]
target = "Performance"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(64, input_shape=(len(features),), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {:.2f}%".format(score[1] * 100))

# Generate predictions and confusion matrix
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix in a table
cm_df = pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"])
print(cm_df)
