from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os


def test_model():
    if not os.path.exists("savedmodel.pth"):
        raise FileNotFoundError("savedmodel.pth not found. Please run train.py first.")

    print("Loading saved model...")

    model = joblib.load("savedmodel.pth")

    print("Loading Olivetti faces dataset...")

    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    print("Splitting dataset into 70% train and 30% test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print("Running prediction on test dataset...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    test_model()