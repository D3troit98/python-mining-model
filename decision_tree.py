import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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

# Select features and target
features = ["Staff", "Floor Space", "Window", "Car park", "Demographic score", "Clearance space", "Competition number", "Competition score", "Total Population"]
target = "Performance"

# Convert Car park column to binary
df["Car park"] = df["Car park"].apply(lambda x: 1 if x=="Yes" else 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=42)

# Define parameter grid to search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

# Create decision tree classifier and use GridSearchCV to find the best hyperparameters
dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, param_grid, cv=5)
clf.fit(X_train, y_train)

# Print the best hyperparameters and accuracy on the test set
# print("Best hyperparameters: ", clf.best_params_)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)