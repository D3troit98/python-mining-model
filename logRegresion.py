import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Select features and target
features = ["Staff", "Floor Space", "Window", "Car park", "Demographic score", "Clearance space", "Competition number", "Competition score", "Total Population"]
target = "Performance"

# Convert Car park column to binary
df["Car park"] = df["Car park"].apply(lambda x: 1 if x=="Yes" else 0)

# Select only the numerical features
num_features = ["Staff", "Floor Space", "Demographic score", "Clearance space", "Competition number", "Competition score", "Total Population"]
X = df[num_features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a logistic regression classifier
classifier = LogisticRegression(solver="lbfgs", random_state=0)

# Fit the classifier to the training data
classifier.fit(X_train, y_train)

# Make predictions on test set
y_pred = classifier.predict(X_test)

# Evaluate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Create a histogram of the "Staff" column after cleaning
plt.hist(df['Staff'], bins=20)
plt.xlabel('Staff')
plt.ylabel('Frequency')
plt.title('Distribution of Staff(after cleaning)')
plt.show()
