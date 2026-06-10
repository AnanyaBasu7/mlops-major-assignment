from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib


def train_model():
    print("Loading Olivetti faces dataset...")

    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    print("Dataset loaded successfully.")
    print(f"Total samples: {X.shape[0]}")
    print(f"Total features: {X.shape[1]}")
    print(f"Total classes: {len(set(y))}")

    print("Splitting dataset into 70% train and 30% test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print("Training DecisionTreeClassifier model...")

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("Saving model as savedmodel.pth...")

    joblib.dump(model, "savedmodel.pth")

    print("Model training completed successfully.")
    print("Model saved as savedmodel.pth")


if __name__ == "__main__":
    train_model()