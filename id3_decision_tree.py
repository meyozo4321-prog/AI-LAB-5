import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Training dataset
data = {
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain",
        "Rain", "Rain", "Overcast", "Sunny",
        "Sunny", "Rain", "Sunny", "Overcast",
        "Overcast", "Rain"
    ],

    "Temperature": [
        "Hot", "Hot", "Hot", "Mild",
        "Cool", "Cool", "Cool", "Mild",
        "Cool", "Mild", "Mild", "Mild",
        "Hot", "Mild"
    ],

    "Humidity": [
        "High", "High", "High", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "High"
    ],

    "Wind": [
        "Weak", "Strong", "Weak", "Weak",
        "Weak", "Strong", "Strong", "Weak",
        "Weak", "Weak", "Strong", "Strong",
        "Weak", "Strong"
    ],

    "Play Tennis": [
        "No", "No", "Yes", "Yes",
        "Yes", "No", "Yes", "No",
        "Yes", "Yes", "Yes", "Yes",
        "Yes", "No"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

print("========================================")
print("       ID3 DECISION TREE")
print("========================================")

print("\nTraining Dataset:")
print(df)

# Convert categorical values into numerical values
X = pd.get_dummies(df[["Outlook", "Temperature", "Humidity", "Wind"]])
y = df["Play Tennis"].map({"No": 0, "Yes": 1})

# Create Decision Tree using Entropy
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

# Train the model
model.fit(X, y)

# Display accuracy
accuracy = model.score(X, y)

print("\n========================================")
print("Model Accuracy:", accuracy * 100, "%")
print("========================================")

# Display feature importance
print("\nFeature Information Gain / Importance:")

for feature, importance in zip(X.columns, model.feature_importances_):
    if importance > 0:
        print(f"{feature}: {importance:.4f}")

# Plot the decision tree
plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("ID3 Decision Tree - Play Tennis")

plt.tight_layout()

# Show tree
plt.show()