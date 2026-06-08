from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump

# Build your first AI Model

iris = load_iris()

# Storing features in x and target in Y

x = iris.data
y = iris.target

# Divide Training and Test data set

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)

#Train the model

model = DecisionTreeClassifier(random_state=42)
trained_model=model.fit(x_train,y_train)


# Predict test data using newly trained model

y_pred=model.predict(x_test)
 
# Model accuracy check

accuracy=accuracy_score(y_pred,y_test)
print("Accuracy:",accuracy)

# Calculate confussion matrix

conf =confusion_matrix(y_pred,y_test)

print("Confusion Matrix:",conf)


# Compute confusion matrix
cm = confusion_matrix(y_pred,y_test)
# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

#output_dir = Path.cwd()

parent_dir = Path.cwd().parent
output_dir = parent_dir / "outputs"

output_dir.mkdir(exist_ok=True)

# save the model 
dump(model, output_dir/"model.joblib")

# Save as PNG
plt.savefig(output_dir.parent/"outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")

# Optional: display on screen
plt.show()

# Optional: free memory
plt.close()