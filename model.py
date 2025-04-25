from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

model_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(model_dir, 'iris_model.pkl')

iris = load_iris()
X, y = iris.data, iris.target

print(y)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, model_file_path)